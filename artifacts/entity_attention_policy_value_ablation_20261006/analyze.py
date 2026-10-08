"""Compare completed paired panels on their common seeds and fixed fitting curves."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from splendor.alphazero_recording import sha256, write_json
from splendor.paired_statistics import serialized_pair_scores, paired_bootstrap_difference

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent


def difference(left,right):
    ls,lv=serialized_pair_scores(left)
    rs,rv=serialized_pair_scores(right)
    lm,rm=dict(zip(ls,lv)),dict(zip(rs,rv))
    seeds=tuple(s for s in ls if s in rm)
    assert len(seeds) in (50,100)
    a=np.array([lm[s] for s in seeds])
    b=np.array([rm[s] for s in seeds])
    result=paired_bootstrap_difference(seeds,a,seeds,b,resamples=10000,seed=13530000).to_dict()
    return result|{"left_common_seed_score":float(a.mean()),"right_common_seed_score":float(b.mean())}


def main(partial=False):
    spec=json.loads((ART/"specification.json").read_text())
    inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha256(path)
        return json.loads(path.read_text())
    fitting=read(ART/"fitting.json")
    panels={}
    for panel in spec["panels"]:
        path=ART/"evaluations"/f"{panel['id']}.json"
        if not path.exists():
            assert partial
            continue
        value=read(path)
        if not value["passed"]:
            assert partial
            continue
        assert value["valid_games"]==panel["pairs"]*2
        assert value["checkpoint_identity"]["policy_b"]["sha256"]=="009d17d2e9bc0d879142d7ca3ff1777a6f8caa9dda9243c6a2d5eb3538b5d5b0"
        panels[panel["id"]]=value
    summaries={key:{name:p[name] for name in ("win_rate_score","a_wins","a_losses","draws","valid_games",
                  "paired_bootstrap","policy_cycle_draws","elapsed_seconds_this_invocation","checkpoint_identity")}
               for key,p in panels.items()}
    differences={}
    for mode in ("greedy","mcts"):
        initial=f"initial_{mode}"
        if initial not in panels:
            continue
        for key in panels:
            if key!=initial and key.endswith(f"_{mode}"):
                differences[f"{key}_minus_initial"]=difference(panels[key],panels[initial])
        for step in (100,500):
            for arm in ("policy_only","value_detached"):
                left,right=f"{arm}_{step}_{mode}",f"joint_{step}_{mode}"
                if left in panels and right in panels:
                    differences[f"{left}_minus_joint"]=difference(panels[left],panels[right])
            left,right=f"value_detached_{step}_{mode}",f"policy_only_{step}_{mode}"
            if left in panels and right in panels:
                differences[f"value_detached_minus_policy_only_{step}_{mode}"]=difference(panels[left],panels[right])
        for arm in spec["arms"]:
            left,right=f"{arm}_500_{mode}",f"{arm}_100_{mode}"
            if left in panels and right in panels:
                differences[f"{arm}_500_minus_100_{mode}"]=difference(panels[left],panels[right])
    for arm in spec["arms"]:
        fixed,standard=f"{arm}_500_mcts_fixed_value",f"{arm}_500_mcts"
        if fixed in panels and standard in panels:
            differences[f"{arm}_500_fixed_value_minus_own_value"]=difference(panels[fixed],panels[standard])
            differences[f"{arm}_500_fixed_value_minus_initial"]=difference(panels[fixed],panels["initial_mcts"])
    if "policy_only_500_mcts_fixed_value" in panels and "joint_500_mcts_fixed_value" in panels:
        differences["policy_only_minus_joint_500_fixed_value"]=difference(
            panels["policy_only_500_mcts_fixed_value"],panels["joint_500_mcts_fixed_value"])
    for right_arm in ("joint","policy_only"):
        left,right="value_detached_500_mcts_fixed_value",f"{right_arm}_500_mcts_fixed_value"
        if left in panels and right in panels:
            differences[f"value_detached_minus_{right_arm}_500_fixed_value"]=difference(panels[left],panels[right])
    fitting_summary={}
    for arm,result in fitting["arms"].items():
        rows=[]
        for point in result["curve"]:
            grads=point["gradient_probe"]
            row={"step":point["step"],"gradient_cosine_mean":float(np.mean([g["cosine"] for g in grads])),
                 "negative_gradient_cosines":sum(g["cosine"]<0 for g in grads),"gradient_probe_batches":len(grads),
                 "value_to_policy_shared_gradient_norm_ratio_mean":float(np.mean([g["value_norm"]/g["policy_norm"] for g in grads]))}
            for split in ("train","validation"):
                row[split]={key:point[split][key] for key in ("policy_kl","value_loss","policy_top1_agreement",
                    "policy_target_mass_on_greedy","value_function_drift_rms","policy_entropy","target_entropy","value_zero_mse")}
            rows.append(row)
        metrics_path=ROOT/spec["output"]/arm/"training_metrics.jsonl"
        inputs[str(metrics_path.relative_to(ROOT))]=sha256(metrics_path)
        metrics=[json.loads(line) for line in metrics_path.read_text().splitlines()]
        fitting_summary[arm]={"curve":rows,"elapsed_seconds":result["elapsed_seconds"],
            "clipped_steps":sum(row["grad_clipped"] for row in metrics),"updates":len(metrics),
            "critic_parameter_max_abs_difference":result["critic_parameter_max_abs_difference"]}
    external=read(ROOT/"artifacts/entity_attention_strong_anchor_20261006/evaluations/round50_vs_ppo.json")
    if "joint_100_mcts" in panels:
        differences["fresh_only_joint_100_minus_formal_uniform_round50"]=difference(panels["joint_100_mcts"],external)
    result={"protocol":"entity-attention-policy-value-ablation-analysis-v1","complete":len(panels)==len(spec["panels"]),
            "completed_panels":len(panels),"expected_panels":len(spec["panels"]),"fitting":fitting_summary,
            "panels":summaries,"paired_differences":differences,"inputs_sha256":inputs,
            "limitations":["One initial checkpoint, one data/sampling seed and repeated DEV panels.",
                           "All main training arms use latest-only fixed labels, not the planned25% online mixture.",
                           "Shared gradient cosine probes are hypothetical unblocked joint gradients; few batches and inherited AdamW, not proof of a sole cause.",
                           "Global norm clipping remains shared: detached critic gradients can still change policy clipping scale.",
                           "Fixed-value composed panels require this artifact runner, not the stock config CLI."]}
    capacity=ART/"capacity_probe.json"
    if capacity.exists():
        result["exploratory_capacity_probe"]=read(capacity)
    uncertainty=ART/"holdout_uncertainty.json"
    if uncertainty.exists():
        result["holdout_game_bootstrap"]=read(uncertainty)["differences"]
    value_profile=ART/"value_profile.json"
    if value_profile.exists():
        result["search_clipped_holdout_value_profile"]=read(value_profile)["profiles"]
    longer=ART/"policy_only_longer.json"
    if longer.exists():
        record=read(longer)
        result["exploratory_full_pool_policy_longer"]={key:record[key] for key in
            ("additional_updates","batch_size","curve","checkpoints","elapsed_seconds","scope")}
    write_json(ART/("partial_analysis.json" if partial else "analysis.json"),result)
    print(json.dumps({"completed_panels":len(panels),"panels":{key:p["win_rate_score"] for key,p in panels.items()},
                      "paired_differences":differences},ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--partial",action="store_true")
    args=parser.parse_args()
    main(args.partial)
