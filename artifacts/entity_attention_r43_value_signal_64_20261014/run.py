"""R43 conditional terminal-return signal: 64 independent roots, 16 continuations."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import torch

from splendor.alphazero_recording import environment_identity, read_json, sha256, source_identity, write_json
from splendor.compact_trace import decode_request
from splendor.contracts import DecisionRequest
from splendor.features import encode_decision
from splendor.mcts_policy import MCTSConfig
from splendor.native_policy import NativeBatchRunner, NativePUCTSearchPolicy
from splendor.opponent_pool import load_frozen_actor_critic_checkpoint
from splendor.policy_averaging import replay_roots
from splendor.replay import state_digest, to_primitive
from splendor.result import TerminalResult
from splendor.search_inference import FrozenSearchInference, InferenceConfig
from splendor.search_policy import determinize_request
from splendor.selfplay import derive_seed
from splendor.selfplay_data import STATE_FIELDS, load_game

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime" / ART.name
CHECKPOINT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"
CHECKPOINT_SHA = "b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14"
POOL_CONFIG = ROOT / "configs/entity_attention_v1/offline_r24_r43_value_auxiliary_20261013.json"
SELFPLAY_CONFIG = ROOT / "configs/entity_attention_v1/hybrid_r9_online_lr2e5_1000_forever_20261011.json"
SEEDS = {"selection":37000000, "world":37100000, "search":37200000, "noise":37300000, "bootstrap":37400000}
STAGES = ("early", "middle", "late")
REPEATS = 16


def identity(path):
    path = Path(path)
    return {"path": str(path.relative_to(ROOT)), "sha256":sha256(path)}


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Frozen experiment changed: {path}")
    else:
        write_json(path, value)


def prepare():
    if (ART / "plan.json").exists():
        verify_identity(read_json(ART / "plan.json"))
        return
    config = read_json(POOL_CONFIG)
    manifest = ROOT / config["manifest"]
    if sha256(manifest) != config["manifest_sha256"] or sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Historical pool or R43 identity changed")
    pool = [json.loads(s) for s in manifest.read_text().splitlines()]
    pool = [g for g in pool if 23 <= g["round_index"] <= 42 and g["split"] == "validation"]
    rng = np.random.default_rng(SEEDS["selection"])
    extra_rounds = set(map(int, rng.choice(np.arange(23,43),4,replace=False)))
    games = []
    for round_index in range(23,43):
        candidates = sorted((g for g in pool if g["round_index"] == round_index), key=lambda g:g["game_id"])
        games.extend(candidates[int(i)] for i in rng.choice(len(candidates),3+(round_index in extra_rounds),replace=False))
    stages = ["early"]*22 + ["middle"]*21 + ["late"]*21
    rng.shuffle(stages)
    bank = []
    for game, stage in zip(games,stages,strict=True):
        path = Path(game["path"])
        if sha256(path) != game["archive_sha256"]:
            raise ValueError("Archived game identity changed")
        arrays, meta = load_game(path)
        roots = [(i,r) for i,r in replay_roots(arrays,meta) if int(arrays["phase"][i]) == 0]
        fraction = {"early":.15,"middle":.5,"late":.85}[stage]
        index, request = roots[int((len(roots)-1)*fraction)]
        encoded = encode_decision(request)
        for name in STATE_FIELDS:
            if not np.array_equal(getattr(encoded,name),arrays[name][index]):
                raise ValueError(f"Replayed feature differs: {name}")
        lo,hi=map(int,arrays["candidate_offsets"][index:index+2])
        if not np.array_equal(encoded.candidate_features,arrays["candidate_features"][lo:hi]):
            raise ValueError("Replayed candidate features differ")
        bank.append({"id":len(bank),"stage":stage,"absolute_round":game["round_index"]+1,
            "game_id":game["game_id"],"archive":identity(path),"decision_index":index,
            "archived_player_id":int(arrays["player_id"][index]),"archived_z":int(arrays["value_target"][index]),
            "archived_teacher_sha256":meta["checkpoint_sha256"],"request":to_primitive(request),
            "completed_player_turns":int(arrays["completed_player_turns"][index]),
            "max_visible_score":max(request.observation.self_public.score,request.observation.opponent_public.score),
            "opponent_blind_reserved":sum(r.card is None for r in request.observation.opponent_public.reserved_cards)})
    if len(bank) != 64 or len({(r["absolute_round"],r["game_id"]) for r in bank}) != 64:
        raise ValueError("Root population is not 64 independent source games")
    write_json(ART / "root_bank.json",{"roots":bank,"selection_seed":SEEDS["selection"],
        "scope":"64 distinct validation games from absolute R24-R43, 3 per round plus four randomly assigned extras; one main-action root per game, stage fractions .15/.5/.85 of archived main decisions; stage-balanced, not position-frequency representative."})
    search = read_json(SELFPLAY_CONFIG)["selfplay"]["search_config"]
    if search["simulations"] != 1024 or search["determinizations"] != 1 or search["root_noise_fraction"] != .25:
        raise ValueError("Unexpected original collection search recipe")
    plan={"protocol":"r43-terminal-value-signal-64-v1","client_date":"2026-10-14", "roots":64,"repeats":REPEATS,
        "checkpoint":identity(CHECKPOINT),"pool_config":identity(POOL_CONFIG),"source_manifest":identity(manifest),
        "root_bank":identity(ART / "root_bank.json"),"runner":identity(Path(__file__)),
        "search":search,"add_root_noise":True,"temperature":0.,"seeds":SEEDS,
        "inference":InferenceConfig(mode="compiled",actor_projection=False,candidate_buckets=(16,32,48,64),
                                     compile_failure_policy="fail_fast").to_dict(),
        "active_continuations":128,"native_batch_capacity":64,"max_completed_turns":400,
        "sampling":"Existing determinize_request uniform hidden-card/deck sampler conditional on current actor's safe request; actual world sampled once per continuation; search samples its own independent worlds at every decision.",
        "estimand":"Current actor's terminal z under frozen R43 S1024/N1 self-play with root noise .25 and T0, averaging hidden starting worlds and independent search randomness. Not exact conditional expectation of historical mixed-teacher trajectories, not a proven Bayes floor under the real game posterior.",
        "statistics":"Root-equal metrics; unbiased within-root outcome variance; mean-target squared error minus s2/16; between-root variance minus mean(s2/16); stage-stratified independent-root bootstrap10000; Monte Carlo corrections are not clipped.",
        "engineering":"400 total completed-player-turn cap fails explicitly; never map truncation or exception to draw. One formal duplicate of root0/repeat0 tests fixed-seed reproducibility and is excluded from signal statistics."}
    write_json(ART / "plan.json",plan)
    print({"prepared_roots":64,"repeats":REPEATS,"stages":dict(Counter(stages))},flush=True)


def verify_identity(plan):
    for key in ("checkpoint","pool_config","source_manifest","root_bank","runner"):
        item=plan[key]
        if sha256(ROOT/item["path"]) != item["sha256"]:
            raise ValueError(f"Frozen {key} changed")


def rollout_seeds(root_id, repeat):
    return {name:derive_seed(base,0,root_id,"value_signal_"+name,repeat) for name,base in SEEDS.items()
            if name in ("world","search","noise")}


@dataclass
class Continuation:
    root: dict
    repeat: int
    model: object
    evaluator: object
    search: dict
    seeds: dict = field(init=False)
    engine: object = field(init=False)
    policies: tuple = field(init=False)
    response: object = field(init=False)
    job: object = None
    actions: list = field(default_factory=list)
    simulations: int = 0
    searchable_decisions: int = 0
    transition_cap_hits: int = 0
    first_target: list | None = None

    def __post_init__(self):
        self.seeds=rollout_seeds(self.root["id"],self.repeat)
        self.engine=determinize_request(decode_request(self.root["request"]),seed=self.seeds["world"],engine_mode="fast")
        self.initial_digest=state_digest(self.engine._debug_state())
        self.response=self.engine.decision()
        # determinize_request maps the original root actor to player0.
        if self.engine.current_player != 0:
            raise ValueError("Root perspective changed")
        self.policies=tuple(NativePUCTSearchPolicy(self.model,evaluator=self.evaluator,
            config=MCTSConfig(**{**self.search,"seed":derive_seed(self.seeds["search"],0,p,"player")})) for p in (0,1))

    def ensure_job(self):
        while isinstance(self.response,DecisionRequest) and self.job is None:
            if sum(self.engine._debug_state().completed_turns) >= 400:
                raise RuntimeError("Engineering turn cap reached; no terminal label assigned")
            self.job=self.policies[self.engine.current_player].start_job(self.response,add_root_noise=True,temperature=0.,
                sampling_seed=derive_seed(self.seeds["noise"],0,len(self.actions),"decision"))
            if self.job.done:
                self.consume()

    def consume(self):
        result=self.job.result
        if result.searchable and (result.simulations != self.search["simulations"]
                or tuple(result.simulation_budgets) != (self.search["simulations"],)
                or sum(result.root_visit_counts) != self.search["simulations"]):
            raise ValueError("Actual continuation search budget differs")
        if self.first_target is None:
            self.first_target=list(result.policy_target)
        self.simulations+=result.simulations
        self.searchable_decisions+=int(result.searchable)
        self.transition_cap_hits+=result.transition_cap_hits
        index=result.chosen_candidate_index
        self.actions.append(index)
        self.response=self.engine.apply_candidate(index,state_version=self.engine.state_version)
        self.job=None

    def record(self):
        if not isinstance(self.response,TerminalResult):
            raise ValueError("Continuation is not naturally terminal")
        winners=self.response.winner_ids
        z=0 if len(winners)==2 else (1 if 0 in winners else -1)
        return {"root_id":self.root["id"],"repeat":self.repeat,"seeds":self.seeds,"z":z,
            "terminal":to_primitive(self.response),"status":"natural_terminal","root_player":0,
            "initial_state_digest":self.initial_digest,"final_state_digest":state_digest(self.engine._debug_state()),
            "chosen_candidate_indices":self.actions,"first_policy_target":self.first_target,
            "searchable_decisions":self.searchable_decisions,"simulations":self.simulations,
            "transition_cap_hits":self.transition_cap_hits}


def drive(tasks, model, evaluator, search, *, active_limit, commit):
    waiting=deque(tasks);active=[]
    catalog=model.catalog if hasattr(model,"catalog") else None
    from splendor.catalog import load_default_catalog
    runner=NativeBatchRunner(evaluator,catalog or load_default_catalog(),capacity=64,buckets=(16,32,48,64),slots=1)
    started=time.perf_counter();last_update=started;completed=0
    while waiting or active:
        while waiting and len(active)<active_limit:
            root,repeat,key=waiting.popleft()
            active.append((Continuation(root,repeat,model,evaluator,search),key))
        remaining=[]
        for session,key in active:
            session.ensure_job()
            if isinstance(session.response,TerminalResult):
                commit(key,session.record());completed+=1
            else:
                remaining.append((session,key))
        active=remaining
        if active:
            jobs=[s.job for s,_ in active]
            if runner.run(jobs)<=0:
                raise RuntimeError("Continuation search made no progress")
            for session,_ in active:
                if session.job.done:
                    session.consume()
        if time.perf_counter()-last_update>=30:
            print(json.dumps({"event":"progress","completed_this_session":completed,"queued":len(waiting),
                "active":len(active),"seconds":time.perf_counter()-started}),flush=True)
            last_update=time.perf_counter()
    return {"seconds":time.perf_counter()-started,"completed":completed,"native_batches":runner.batches,"native_rows":runner.rows}


def raw_path(root_id, repeat):
    return ART / "raw" / f"root-{root_id:02d}-repeat-{repeat:02d}.json"


def run(smoke=False):
    plan=read_json(ART / "plan.json");verify_identity(plan)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this authorized continuation experiment")
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME / "run.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        name=("smoke-" if smoke else "formal-")+str(time.time_ns())
        source=source_identity(ROOT,RUNTIME/name)
        source["native_source"]=identity(ROOT / "src/splendor/native_search.cpp")
        shutil.copyfile(__file__,RUNTIME/name/"run.py")
        execution={"source":source,"environment":environment_identity(),"command":sys.argv,"plan":identity(ART/"plan.json"),
            "status":"running","start_unix":time.time(),"tf32":False}
        execution_path=ART / "execution" / f"{name}.json"
        write_json(execution_path,execution)
        try:
            roots=read_json(ART / "root_bank.json")["roots"]
            model=load_frozen_actor_critic_checkpoint(CHECKPOINT,device="cuda")
            evaluator=FrozenSearchInference(model,device="cuda",config=InferenceConfig(**plan["inference"]))
            values=[float(value) for _,value in evaluator.policy_value([decode_request(r["request"]) for r in roots])]
            if not smoke:
                freeze(ART / "predictions.json",{"checkpoint":plan["checkpoint"],"root_bank":plan["root_bank"],"values":values})
            collected={}
            if smoke:
                root=next(r for r in roots if r["stage"]=="late")
                tasks=[(root,0,"first"),(root,0,"duplicate")]
                search={**plan["search"],"simulations":16}
                timing=drive(tasks,model,evaluator,search,active_limit=2,commit=lambda key,row:collected.update({key:row}))
                if collected["first"] != collected["duplicate"]:
                    raise ValueError("Fixed-world fixed-search-seed duplicate differs")
                write_json(ART / "smoke.json",{"passed":True,"simulations":16,"identical_rollouts":True,"timing":timing,
                    "control":collected["first"],"scope":"Two identical-seed full continuations at S16; engineering check, excluded from formal statistics."})
            else:
                tasks=[]
                for repeat in range(REPEATS):
                    for root in roots:
                        path=raw_path(root["id"],repeat)
                        if path.exists():
                            row=read_json(path)
                            if (row["root_id"],row["repeat"],row["seeds"]) != (root["id"],repeat,rollout_seeds(root["id"],repeat)):
                                raise ValueError("Completed continuation identity differs")
                        else:
                            tasks.append((root,repeat,path))
                control=ART / "fixed_seed_control.json"
                if not control.exists():
                    tasks.append((roots[0],0,control))
                timing=drive(tasks,model,evaluator,plan["search"],active_limit=plan["active_continuations"],commit=write_json)
                if read_json(control) != read_json(raw_path(0,0)):
                    raise ValueError("Formal fixed-seed continuation differs")
                execution["timing"]=timing
            for path,digest in source["source_files"].items():
                if sha256(ROOT/path) != digest:
                    raise ValueError(f"Source changed during execution: {path}")
            for path,digest in source["native_extensions"].items():
                if sha256(Path(path)) != digest:
                    raise ValueError("Native binary changed during execution")
            verify_identity(plan)
            execution.update(status="complete",end_unix=time.time(),inference_stats=evaluator.stats.to_dict())
        except BaseException as error:
            execution.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "failed",end_unix=time.time(),
                error=f"{type(error).__name__}: {error}")
            raise
        finally:
            write_json(execution_path,execution)


def signal_metrics(values, outcomes):
    values=np.asarray(values,dtype=np.float64);y=np.asarray(outcomes,dtype=np.float64)
    if y.ndim!=2 or len(values)!=len(y) or y.shape[1]<2 or not np.isin(y,[-1,0,1]).all() or not np.isfinite(values).all():
        raise ValueError("Expected finite values and at least two terminal outcomes per root")
    n=y.shape[1];means=y.mean(axis=1);variance=y.var(axis=1,ddof=1)
    single=float(np.mean((values[:,None]-y)**2));mean_mse=float(np.mean((values-means)**2))
    noise=float(variance.mean());mc=float(variance.mean()/n)
    signal=float(means.var(ddof=1)-mc) if len(values)>1 else None
    return {"roots":len(values),"repeats":n,"single_return_mse":single,"zero_baseline_mse":float(np.mean(y*y)),
        "within_root_variance":noise,"mean_target_mse":mean_mse,"mean_target_mc_variance":mc,
        "corrected_expected_value_mse":mean_mse-mc,"noise_fraction_of_model_mse":noise/single if single>0 else None,
        "between_root_signal_variance":signal,
        "signal_fraction_of_return_variance":signal/(signal+noise) if signal is not None and signal+noise>0 else None,
        "correlation_with_mean":float(np.corrcoef(values,means)[0,1]) if values.std()>0 and means.std()>0 else None,
        "mean_prediction":float(values.mean()),"mean_return":float(means.mean()),
        "mixed_win_loss_roots":int(np.sum((y==1).any(axis=1)&(y==-1).any(axis=1)))}


def bootstrap(values,y,stages,resamples=10000):
    rng=np.random.default_rng(SEEDS["bootstrap"])
    groups=[np.flatnonzero(np.asarray(stages)==stage) for stage in STAGES]
    indices=np.concatenate([rng.choice(g,(resamples,len(g)),replace=True) for g in groups if len(g)],axis=1)
    means=y.mean(axis=1);s2=y.var(axis=1,ddof=1);n=y.shape[1]
    single=np.mean((values[:,None]-y)**2,axis=1)
    mse=(values-means)**2
    noise=s2[indices].mean(axis=1);single=single[indices].mean(axis=1)
    corrected=mse[indices].mean(axis=1)-noise/n
    signal=means[indices].var(axis=1,ddof=1)-noise/n
    quantities={"single_return_mse":single,"within_root_variance":noise,"corrected_expected_value_mse":corrected,
        "noise_fraction_of_model_mse":noise/single,"between_root_signal_variance":signal,
        "signal_fraction_of_return_variance":signal/(signal+noise)}
    return {key:[float(x) for x in np.quantile(val,[.025,.975])] for key,val in quantities.items()}


def verify_rollout(root,row,simulations,*,checked=False):
    if row["seeds"] != rollout_seeds(root["id"],row["repeat"]) or row["root_id"] != root["id"]:
        raise ValueError("Rollout seeds/root differ")
    engine=determinize_request(decode_request(root["request"]),seed=row["seeds"]["world"],engine_mode="checked" if checked else "fast")
    if state_digest(engine._debug_state()) != row["initial_state_digest"]:
        raise ValueError("Initial sampled world differs")
    for index in row["chosen_candidate_indices"]:
        request=engine.decision()
        if not isinstance(request,DecisionRequest) or not 0<=index<len(request.legal_actions):
            raise ValueError("Continuation contains invalid action")
        engine.apply(request.legal_actions[index])
    terminal=engine.decision()
    if not isinstance(terminal,TerminalResult) or to_primitive(terminal) != row["terminal"]:
        raise ValueError("Continuation terminal replay differs")
    z=0 if len(terminal.winner_ids)==2 else (1 if 0 in terminal.winner_ids else -1)
    if row["status"]!="natural_terminal" or z!=row["z"] or state_digest(engine._debug_state())!=row["final_state_digest"]:
        raise ValueError("Continuation terminal label/digest differs")
    if row["simulations"]!=simulations*row["searchable_decisions"]:
        raise ValueError("Continuation total search budget differs")


def analyze():
    plan=read_json(ART/"plan.json");verify_identity(plan)
    roots=read_json(ART/"root_bank.json")["roots"]
    values=np.asarray(read_json(ART/"predictions.json")["values"],dtype=np.float64)
    rows=[[read_json(raw_path(root["id"],j)) for j in range(REPEATS)] for root in roots]
    for root,repeats in zip(roots,rows,strict=True):
        for j,row in enumerate(repeats):
            verify_rollout(root,row,1024,checked=j==0)
    if read_json(ART/"fixed_seed_control.json")!=rows[0][0]:
        raise ValueError("Formal fixed-seed control differs")
    y=np.asarray([[row["z"] for row in repeats] for repeats in rows],dtype=np.float64)
    stages=[r["stage"] for r in roots]
    summary={"status":"complete","plan":identity(ART/"plan.json"),"overall":signal_metrics(values,y),
        "ci95":bootstrap(values,y,stages),"by_stage":{},"outcomes":dict(Counter(map(int,y.flat))),
        "validation":{"natural_terminals":int(y.size),"replayed_terminals":int(y.size),"checked_invariant_replays":64,
            "fixed_seed_control_identical":True,"unique_source_games":64,"raw_sha256":{p.name:sha256(p) for p in sorted((ART/"raw").glob('*.json'))}},
        "scope":plan["estimand"]}
    per_root=[]
    for i,root in enumerate(roots):
        per_root.append({k:v for k,v in root.items() if k!="request"}|{"prediction":float(values[i]),"outcomes":y[i].astype(int).tolist(),
            "mean_z":float(y[i].mean()),"sample_variance":float(y[i].var(ddof=1)),"mean_se":float(np.sqrt(y[i].var(ddof=1)/REPEATS)),
            "mean_target_mse":float((values[i]-y[i].mean())**2),"corrected_expected_value_mse":float((values[i]-y[i].mean())**2-y[i].var(ddof=1)/REPEATS)})
    for stage in STAGES:
        mask=np.asarray(stages)==stage
        summary["by_stage"][stage]=signal_metrics(values[mask],y[mask])
    write_json(ART/"per_root.json",per_root);write_json(ART/"summary.json",summary)
    m=summary["overall"];ci=summary["ci95"]
    lines=["# R43：64局面×16独立续局的终局信号诊断","", "冻结原R43小模型；原R24–R43整局留出中64个不同对局各取一个主行动局面，早／中／晚22／21／21。每次从相同当前玩家可见请求重新采样隐藏世界及牌堆，再由双方R43 S1024/N1、根噪声0.25、T0续至自然终局。", "",
        "| 指标 | 估计 | 95%按根分层bootstrap区间 |","| --- | ---: | --- |"]
    for key,label in (("single_return_mse","模型对单次终局z的MSE"),("within_root_variance","同局面终局结果方差（无偏估计）"),
        ("corrected_expected_value_mse","模型对期望结果的MSE（扣除16次均值的抽样方差）"),
        ("noise_fraction_of_model_mse","结果方差／模型单次MSE"),("between_root_signal_variance","局面之间期望结果方差（扣除均值抽样方差）"),
        ("signal_fraction_of_return_variance","局面间信号方差／总结果方差")):
        lines.append(f"| {label} | {m[key]:.6f} | [{ci[key][0]:.6f}, {ci[key][1]:.6f}] |")
    lines += ["",f"模型对16次均值的原始MSE={m['mean_target_mse']:.6f}，均值抽样方差估计={m['mean_target_mc_variance']:.6f}；常数0对单次z的MSE={m['zero_baseline_mse']:.6f}。模型与均值相关系数={m['correlation_with_mean']:.6f}。64个局面中{m['mixed_win_loss_roots']}个同时出现过赢和输。", "",
        "| 阶段 | 局面 | 单次MSE | 结果方差 | 对期望结果MSE（修正） |","| --- | ---: | ---: | ---: | ---: |"]
    for stage,row in summary["by_stage"].items():
        lines.append(f"| {stage} | {row['roots']} | {row['single_return_mse']:.6f} | {row['within_root_variance']:.6f} | {row['corrected_expected_value_mse']:.6f} |")
    lines += ["", "1024个正式续局全部自然终局并回放核对；每根第0次另做checked不变量回放，共64局。额外一局同世界同搜索种子重复与正式root0/repeat0逐动作及终局完全一致，排除重复运行控制失败；不计入统计。", "",
        "这不是对旧训练池不可约误差的精确估计：续局统一使用R43，旧标签来自多个历史教师；隐藏牌采用现有均匀重建采样器，未利用历史动作推断对手暗牌后验；当前样本按教师轮次和阶段配额选取，非全部局面频率加权。方差包含起始世界、未来牌序及搜索随机性，未拆分这些因素。64根×16次仍有明显Monte Carlo误差，修正项及其区间不裁剪到0或1。早／中／晚由原轨迹相对位置定义。", "",
        "逐根结果见per_root.json；所有动作索引、种子、终局及状态digest见raw/；原始请求与NPZ身份见root_bank.json；配置、代码版本及环境见plan.json与execution/。"]
    (ART/"results.md").write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({"overall":m,"ci95":ci},ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("prepare","smoke","run","analyze"))
    args=parser.parse_args()
    if args.action=="prepare":prepare()
    elif args.action=="analyze":analyze()
    else:run(smoke=args.action=="smoke")


if __name__=="__main__":main()
