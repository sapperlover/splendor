"""CPU world parallelism for interactive, deterministic recommendations."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import multiprocessing

import torch

from .mcts_policy import MCTSConfig, MCTSSearchResult
from .native_policy import NativeBatchRunner, NativePUCTSearchPolicy, NativeChancePUCTSearchPolicy
from .recommend import LocalAdvisor
from .root_q_selection import (RootQSettings, root_coverage_shortlist,
                               select_root_by_world_mean_q)

_WORKER_ADVISOR = None
_WORKER_PHASE = None
_MASK = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15


def world_base_seed(base_seed: int, world: int) -> int:
    """Map a global world to local index zero, retaining its exact old seed.

    Native derived_seed(word, base, j) = word ^ base ^ ((j+1)*gamma).
    Each independent job has one world, so compensate for its local j=0.
    """
    return (base_seed ^ _GAMMA ^ ((world + 1) * _GAMMA)) & _MASK


def _initialize_worker(checkpoint, threads):
    global _WORKER_ADVISOR
    torch.set_num_threads(threads)
    _WORKER_ADVISOR = LocalAdvisor(checkpoint,device='cpu',mode='greedy')


def _world_jobs(request, indices, config):
    advisor = _WORKER_ADVISOR
    jobs=[]
    for world in indices:
        local=replace(config,determinizations=1,seed=world_base_seed(config.seed,world))
        policy=NativePUCTSearchPolicy(advisor.model,config=local,catalog=advisor.catalog,
                                      evaluator=advisor.evaluator)
        jobs.append(policy.start_job(request,add_root_noise=False,temperature=0.))
    return jobs,NativeBatchRunner(advisor.evaluator,advisor.catalog,capacity=len(jobs),
                                  buckets=(16,32,48,64))


def _run_to_pause(jobs,runner):
    while not all(job.core.root_phase_complete() for job in jobs):
        if runner.run(jobs)==0:
            raise RuntimeError('initial root search stalled before its simulation limit')


def _run_to_finish(jobs,runner):
    while not all(job.done for job in jobs):
        if runner.run([job for job in jobs if not job.done])==0:
            raise RuntimeError('root search stalled before exhausting its budget')


def _search_worlds(request, indices, config):
    jobs,runner=_world_jobs(request,indices,config)
    _run_to_finish(jobs,runner)
    return [(world,job.result) for world,job in zip(indices,jobs)]


def _start_q_worlds(request,indices,config,settings):
    global _WORKER_PHASE
    if _WORKER_PHASE is not None:raise RuntimeError('previous root Q phase has not finished')
    limit=settings.initial_limit(config.simulations,len(request.legal_actions))
    jobs,runner=_world_jobs(request,indices,config)
    for job in jobs:job.core.set_root_search_limit(limit)
    _run_to_pause(jobs,runner)
    _WORKER_PHASE=(indices,jobs,runner)
    return [(world,job.core.root_statistics()) for world,job in zip(indices,jobs)]


def _finish_q_worlds(candidates,minimum):
    global _WORKER_PHASE
    if _WORKER_PHASE is None:raise RuntimeError('root Q continuation has no initial trees')
    indices,jobs,runner=_WORKER_PHASE
    try:
        for job in jobs:
            job.core.resume_with_root_coverage(candidates,minimum)
            job.collect_if_done()
        _run_to_finish(jobs,runner)
        return [(world,job.result) for world,job in zip(indices,jobs)]
    finally:
        _WORKER_PHASE=None


def merge_world_results(indexed, world_count):
    """Sum raw visits/Q numerators once; never average local visit shares."""
    indexed=sorted(indexed,key=lambda item:item[0])
    if [world for world,_ in indexed]!=list(range(world_count)):
        raise ValueError('parallel search returned missing or duplicated worlds')
    results=[result for _,result in indexed]
    prior=results[0].root_network_prior
    rows=tuple(result.per_world_root_visit_counts[0] for result in results)
    counts=tuple(sum(row[i] for row in rows) for i in range(len(prior)))
    total=sum(result.simulations for result in results)
    if total!=sum(counts):raise ValueError('parallel root visits do not sum to the simulation budget')
    chosen=max(range(len(prior)),key=lambda i:(counts[i],prior[i],-i))
    target=tuple(n/total for n in counts)
    return MCTSSearchResult(chosen,True,prior,prior,
        tuple(result.simulations for result in results),rows,counts,target,
        tuple(float(i==chosen) for i in range(len(prior))),total,
        max(result.max_tree_depth for result in results),sum(result.transition_cap_hits for result in results),
        tuple(result.per_world_policy_target_counts[0] for result in results),
        tuple(result.per_world_root_value_sums[0] for result in results),
        tuple(result.per_world_forced_playouts[0] for result in results))


class WorldParallelSearch:
    """Persistent workers; each drives a batch of independent one-world jobs."""

    def __init__(self,checkpoint,*,simulations_per_world,worlds,workers,seed=0,threads=1,
                 q_settings: RootQSettings | None=None):
        if not 1<=workers<=worlds:raise ValueError('workers must be between 1 and worlds')
        self.worlds=worlds
        self.workers=workers
        self.q_settings=q_settings
        self.config=MCTSConfig(simulations=simulations_per_world,determinizations=1,
                               seed=seed,root_noise_fraction=0.,forced_playouts=0.,policy_target_pruning=False)
        pool_args=dict(mp_context=multiprocessing.get_context('spawn'),
                       initializer=_initialize_worker,initargs=(checkpoint,threads))
        if q_settings is None:
            self.pool=ProcessPoolExecutor(max_workers=workers,**pool_args)
            self.pools=()
        else:
            # One executor per worker preserves tree ownership across the global
            # pause/shortlist/resume barrier. A shared pool cannot ensure affinity.
            self.pool=None
            self.pools=tuple(ProcessPoolExecutor(max_workers=1,**pool_args) for _ in range(workers))

    def search(self,request,*,add_root_noise=False,temperature=0.):
        if add_root_noise or temperature!=0.:
            raise ValueError('interactive parallel search requires no noise and deterministic selection')
        if len(request.legal_actions)==1:
            # Same short circuit as NativePUCTSearchPolicy: zero simulations.
            return MCTSSearchResult(0,False,(1.,),(1.,),(),(),(0,),(1.,),(1.,),0,0,0)
        groups=[tuple(range(worker,self.worlds,self.workers)) for worker in range(self.workers)]
        if self.q_settings is not None:
            self.q_settings.initial_limit(self.config.simulations,len(request.legal_actions))
            futures=[pool.submit(_start_q_worlds,request,group,self.config,self.q_settings)
                     for pool,group in zip(self.pools,groups)]
            initial=sorted((item for f in futures for item in f.result()),key=lambda item:item[0])
            if [world for world,_ in initial]!=list(range(self.worlds)):
                raise ValueError('initial root Q phase returned missing or duplicated worlds')
            visits=tuple(tuple(stats['visits'][0]) for _,stats in initial)
            candidates=root_coverage_shortlist(visits,initial[0][1]['prior'],self.q_settings)
            futures=[pool.submit(_finish_q_worlds,candidates,self.q_settings.min_visits) for pool in self.pools]
            merged=merge_world_results([item for f in futures for item in f.result()],self.worlds)
            return select_root_by_world_mean_q(merged,self.q_settings,candidates,visits)
        futures=[self.pool.submit(_search_worlds,request,group,self.config) for group in groups]
        indexed=[item for future in futures for item in future.result()]
        return merge_world_results(indexed,self.worlds)

    def close(self):
        for pool in ((self.pool,) if self.pool is not None else self.pools):
            pool.shutdown(wait=True,cancel_futures=True)


class BatchedWorldSearch:
    """Drive independent worlds or one sampled-chance tree through native batches."""

    def __init__(self,advisor,*,simulations_per_world,worlds,seed=0,q_settings: RootQSettings | None=None,
                 search_backend='native',leaf_batch_size=1):
        if search_backend not in ('native','native_chance'):
            raise ValueError('interactive search backend must be native or native_chance')
        if search_backend=='native_chance' and q_settings is not None:
            raise ValueError('native_chance uses one shared tree; world_mean_q coverage is not applicable')
        if leaf_batch_size!=1 and search_backend!='native_chance':
            raise ValueError('leaf batching requires native_chance')
        self.q_settings=q_settings
        self.budget=simulations_per_world
        self.last_search_diagnostics=None
        config=MCTSConfig(simulations=simulations_per_world*worlds,determinizations=worlds,
                          seed=seed,root_noise_fraction=0.,forced_playouts=0.,policy_target_pruning=False)
        policy_cls=NativeChancePUCTSearchPolicy if search_backend=='native_chance' else NativePUCTSearchPolicy
        kwargs={'leaf_batch_size':leaf_batch_size} if search_backend=='native_chance' else {}
        self.policy=policy_cls(advisor.model,config=config,catalog=advisor.catalog,evaluator=advisor.evaluator,**kwargs)
        self.runner=NativeBatchRunner(advisor.evaluator,advisor.catalog,capacity=worlds*leaf_batch_size,
                                       buckets=(16,32,48,64))

    def search(self,request,*,add_root_noise=False,temperature=0.):
        if self.q_settings is not None and (add_root_noise or temperature!=0.):
            raise ValueError('equal-world Q selection requires no noise and deterministic selection')
        before=(self.runner.batches,self.runner.rows,self.runner.fill_seconds,self.runner.consume_seconds)
        limit=(self.q_settings.initial_limit(self.budget,len(request.legal_actions))
               if self.q_settings is not None and len(request.legal_actions)>1 else None)
        job=self.policy.start_job(request,add_root_noise=add_root_noise,temperature=temperature)
        if limit is not None:
            job.core.set_root_search_limit(limit)
            _run_to_pause([job],self.runner)
            initial=job.core.root_statistics()
            candidates=root_coverage_shortlist(initial['visits'],initial['prior'],self.q_settings)
            job.core.resume_with_root_coverage(candidates,self.q_settings.min_visits)
            job.collect_if_done()
            _run_to_finish([job],self.runner)
            return select_root_by_world_mean_q(job.result,self.q_settings,candidates,initial['visits'])
        _run_to_finish([job],self.runner)
        if self.policy.stochastic:
            self.last_search_diagnostics={'search_backend':'native_chance',
                'search_config':asdict(self.policy.config),'stats':job.core.stats(),
                'chance':job.core.chance_diagnostics(),
                'leaf_batch_size':self.policy.leaf_batch_size,
                'runner':{'batches':self.runner.batches-before[0],
                          'rows':self.runner.rows-before[1],
                          'fill_seconds':self.runner.fill_seconds-before[2],
                          'consume_seconds':self.runner.consume_seconds-before[3]},
                'inference_cumulative':self.policy.evaluator.stats.to_dict()}
        return job.result

    def close(self):
        pass
