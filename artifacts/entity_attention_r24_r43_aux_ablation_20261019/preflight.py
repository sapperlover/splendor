"""Bounded real-R43 initialization, update/load and compiled S8 checks."""
from dataclasses import asdict, replace
from pathlib import Path
import argparse
import json
import time

import torch

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import environment_identity, read_json, sha256, write_json
from splendor.alphazero_replay import ReplayGame, ReplayIndex
from splendor.benchmark_search import fixed_requests
from splendor.catalog import load_default_catalog
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.offline_objectives import OfflineObjectiveConfig, initialize_learner
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.selfplay_data import collate_selfplay_samples
from splendor.value_auxiliary import ValueAuxiliaryConfig, auxiliary_targets
from run import ART, CONFIGS, OLD, ROOT, RUNTIME, prepare, evaluation_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")
    started = time.perf_counter(); plan = prepare()
    parent = load_alphazero_checkpoint(ROOT / plan["initial_checkpoint"]["path"])
    old = load_alphazero_checkpoint(OLD / "checkpoints/initial.pt")
    with (ROOT / read_json(CONFIGS['plain'])["manifest"]).open() as stream:
        row = next(x for line in stream if (x := json.loads(line))["round_index"] == 23 and x["split"] == "train")
    game = ReplayGame(**{k: row[k] for k in ('round_index','run_id','game_id','split','decision_count','policy_samples')}, path=Path(row['path']))
    index = ReplayIndex([game], cache_bytes=64 * 1024**2)
    samples = [index.cache.sample(game, i) for i in range(4)]
    batch = collate_selfplay_samples(samples); results = {}
    cases = fixed_requests(); encoded = [encode_decision(r) for _, r, _ in cases]
    initial_policy = None
    for name, path in CONFIGS.items():
        config = OfflineObjectiveConfig.from_dict(read_json(path))
        config = replace(config, learner=replace(config.learner, device=args.device))
        learner = initialize_learner(config, {'digest':'preflight_only'}, 22681)
        for key, value in parent.model_state_dict.items():
            torch.testing.assert_close(learner.model.state_dict()[key].cpu(), value.cpu(), rtol=0, atol=0)
        old_names = [k for k, _ in parent.build_model().named_parameters()]
        named = dict(learner.model.named_parameters())
        for key, old_id in zip(old_names, parent.optimizer_state_dict['param_groups'][0]['params'], strict=True):
            for field, value in parent.optimizer_state_dict['state'].get(old_id, {}).items():
                actual = learner.optimizer.state[named[key]][field]
                if isinstance(value, torch.Tensor): torch.testing.assert_close(actual.cpu(), value.cpu(), rtol=0, atol=0)
                else: assert actual == value
        if name == 'turns_only':
            assert not hasattr(learner.model, 'score_head')
            for key, value in learner.model.state_dict().items():
                if key.startswith('remaining_turns_head.'):
                    torch.testing.assert_close(value.cpu(), old.model_state_dict[key].cpu(), rtol=0, atol=0)
            assert all(not learner.optimizer.state.get(p) for p in learner.model.remaining_turns_head.parameters())
        learner.model.eval()
        with torch.inference_mode():
            initial = tuple(x.cpu() for x in learner.model(batch.decisions))
        if initial_policy is None: initial_policy = initial
        else:
            for x,y in zip(initial_policy, initial): torch.testing.assert_close(x,y,rtol=0,atol=0)
        auxiliary = None if config.value_auxiliary is None else ValueAuxiliaryConfig(**config.value_auxiliary)
        kwargs = {} if auxiliary is None else {'value_auxiliary':auxiliary, 'auxiliary_labels':auxiliary_targets(samples,auxiliary)}
        metrics = learner.train_step(batch, references=[], **kwargs)
        checkpoint = RUNTIME / f'preflight_{args.device}' / f'{name}.pt'
        checkpoint.parent.mkdir(parents=True, exist_ok=True); learner.save(checkpoint)
        frozen = load_frozen_actor_critic_checkpoint(checkpoint, device=args.device)
        for key,value in learner.model.state_dict().items():
            torch.testing.assert_close(value, frozen.state_dict()[key], rtol=0, atol=0)
        eager = FrozenSearchInference(frozen, device=args.device, config=InferenceConfig(actor_projection=False))
        evaluator = eager if args.device == 'cpu' else FrozenSearchInference(frozen, device=args.device,
            config=InferenceConfig(mode='compiled',actor_projection=False,native_transport='reused',compile_failure_policy='fail_fast'))
        for expected,actual in zip(eager.policy_value_encoded(encoded),evaluator.policy_value_encoded(encoded),strict=True):
            torch.testing.assert_close(torch.tensor(expected[0]),torch.tensor(actual[0]),atol=2e-5,rtol=2e-4)
            torch.testing.assert_close(torch.tensor(expected[1]),torch.tensor(actual[1]),atol=2e-5,rtol=2e-4)
        policy = NativePUCTSearchPolicy(frozen,device=args.device,evaluator=evaluator,
            config=MCTSConfig(simulations=8,determinizations=1,root_noise_fraction=0,seed=39150000))
        jobs = [policy.start_job(request) for _,request,_ in cases]
        runner = NativeBatchRunner(evaluator,load_default_catalog(),capacity=4,buckets=(16,32,48,64))
        while not all(j.done for j in jobs): assert runner.run([j for j in jobs if not j.done])>0
        assert all(j.result.simulations==8 for j in jobs)
        results[name] = {'passed':True,'parameters':sum(p.numel() for p in frozen.parameters()),
            'model_config':asdict(frozen.config),'base_weights_and_moments_preserved':True,
            'legacy_turns_initialization_matches':name=='turns_only','step_metrics':metrics,
            'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_sha256':sha256(checkpoint),
            'native_S8_roots':len(jobs),'inference_stats':evaluator.stats.to_dict()}
    # Freeze actual planned evaluation settings against the original panel domains.
    result = {'passed':True,'device':args.device,'plan_sha256':sha256(ART/'plan.json'),
              'validator_sha256':sha256(Path(__file__)),'environment':environment_identity(),
              'branches':results,'initial_policy_and_value_identical':True,'seconds':time.perf_counter()-started,
              'scope':'Four real archived positions per branch and three S8 roots; engineering verification, excluded from training/strength results.'}
    write_json(ART/f'{args.device}_preflight.json',result)
    print({'passed':True,'device':args.device,'branches':list(results),'seconds':result['seconds']},flush=True)


if __name__=='__main__': main()
