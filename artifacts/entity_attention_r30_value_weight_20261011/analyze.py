"""Summarize the fixed fitting sets and paired opponent-panel differences."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ART = Path(__file__).resolve().parent
BRANCHES = ("baseline", "value025", "value0")
POINTS = (0, 100, 250, 500, 1000)


def fitting_row(raw):
    val = raw["online_validation_full"]
    train = raw["online_train_fixed"]
    return {
        "window_validation": val["pool_weighted"]["validation"],
        "latest_validation": {stage: val["metrics"][f"g030/validation/{stage}"]
                              for stage in ("all", "early_mid", "late")},
        "window_train_fixed": train["pool_weighted"]["train"],
        "latest_train_fixed": train["metrics"]["g030/train/all"],
    }


def fitting_change(before, after, *, latest=False, stage="all"):
    b = before["online_validation_full"]["per_game"]
    a = after["online_validation_full"]["per_game"]
    keys = sorted(k for k in b if k.endswith("/" + stage)
                  and (not latest or k.startswith("29:")))
    if not keys or any(k not in a for k in keys):
        raise ValueError("validation game populations differ")
    result = {"games": len(keys), "stage": stage, "latest_only": latest,
              "resamples": 10000, "seed": 32620000, "unit": "whole validation game"}
    rng = np.random.default_rng(result["seed"])
    for name, numerator, denominator in (("policy_kl", "kl", "policy_weight"),
                                          ("value_mse", "sq_error", "positions")):
        weights = np.array([b[k][denominator] for k in keys], dtype=np.float64)
        if any(b[k][denominator] != a[k][denominator] for k in keys):
            raise ValueError("validation position weights differ")
        diff = np.array([a[k][numerator] - b[k][numerator] for k in keys])
        samples = []
        for _ in range(40):
            indices = rng.integers(0, len(keys), size=(250, len(keys)))
            samples.extend((diff[indices].sum(1) / weights[indices].sum(1)).tolist())
        result[name] = {"delta": float(diff.sum() / weights.sum()),
                        "ci95": np.quantile(samples, [.025, .975]).tolist()}
    return result


def arena_row(raw, plan, branch, mode):
    if not raw["passed"] or raw["completed_pairs"] != 200 or raw["valid_games"] != 400:
        raise ValueError("completed arena must contain 200 valid pairs")
    config = json.loads(json.dumps(raw["config"]))
    expected = json.loads(json.dumps(plan["arena_templates"][mode]))
    # Only the candidate checkpoint/name changes across this fixed opponent panel.
    config["policy_a"]["checkpoint"] = expected["policy_a"]["checkpoint"]
    config["policy_a"]["name"] = expected["policy_a"]["name"]
    if config != expected:
        raise ValueError("arena settings differ from original panel")
    if branch == "baseline":
        expected_sha = plan["baseline_final"]["sha256"]
    else:
        expected_sha = read_json(ART / branch / "training_verification.json")["checkpoint"]["sha256"]
    if raw["checkpoint_identity"]["policy_a"]["sha256"] != expected_sha:
        raise ValueError("evaluated candidate checkpoint differs")
    seeds, scores = serialized_pair_scores(raw)
    ci = paired_bootstrap_mean(scores, resamples=10000, seed=config["bootstrap_seed"]).to_dict()
    if ci != raw["paired_bootstrap"]:
        raise ValueError("arena confidence interval mismatch")
    if mode == "search":
        for pair in raw["pairs"]:
            for game in pair["games"]:
                for side in ("policy_a_stats", "policy_b_stats"):
                    stat = game[side]
                    if (stat["simulations"] != 1024 * stat["searchable_decisions"]
                            or stat["determinizations"] != stat["searchable_decisions"]):
                        raise ValueError("actual S1024/N1 budget differs")
    return {key: raw[key] for key in (
        "completed_pairs", "valid_games", "win_rate_score", "paired_bootstrap",
        "a_wins", "draws", "a_losses", "a_first", "a_second", "exceptions",
        "illegal_actions", "engineering_truncations", "elapsed_seconds_this_invocation")}


def main():
    plan = read_json(ART / "plan.json")
    result = {"plan_sha256": sha256(ART / "plan.json"), "analyzer_sha256": sha256(Path(__file__)),
              "scope": plan["scope"], "branches": {}, "arena_differences": {},
              "parameter_drift_groups": {
                  "critic": "model.critic: complete value head",
                  "actor": "model.actor: final policy MLP only",
                  "shared_encoder": "runner label for all other state tensors; includes shared state encoder AND policy action-query modules, not a pure shared-encoder norm"},
              "statistics": "Whole-game ratio bootstrap for fitting; whole-pair bootstrap for strength. Fixed DEV panel, one training trajectory; exploratory, no multiplicity adjustment."}
    raws, arenas = {}, {}
    for branch in BRANCHES:
        row = {"value_loss_weight": {"baseline": 1.0, "value025": .25, "value0": 0.0}[branch],
               "fitting": [], "arenas": {}}
        result["branches"][branch] = row
        for step in POINTS:
            path = ART / branch / "fitting" / f"step-{step:09d}.json"
            if not path.exists():
                continue
            raw = read_json(path)
            raws[branch, step] = raw
            row["fitting"].append({"step": step, **fitting_row(raw), "result_sha256": sha256(path)})
        if (branch, 1000) in raws:
            final = raws[branch, 1000]
            row["validation_deltas"] = {}
            for population in ("window", "latest"):
                for stage in ("all", "early_mid", "late"):
                    comparisons = {"minus_start": raws["baseline", 0], "minus_weight1": raws["baseline", 1000]}
                    row["validation_deltas"][f"{population}/{stage}"] = {
                        name: fitting_change(before, final, latest=population == "latest", stage=stage)
                        for name, before in comparisons.items()}
        for mode in ("greedy", "search"):
            path = ART / branch / "evaluation" / f"{mode}.json"
            if not path.exists():
                continue
            raw = read_json(path)
            if not raw["passed"] or raw["completed_pairs"] != 200:
                row["arenas"][mode] = {"status": "running", "completed_pairs": raw["completed_pairs"]}
                continue
            arenas[branch, mode] = raw
            row["arenas"][mode] = {"status": "complete", **arena_row(raw, plan, branch, mode),
                                    "result_sha256": sha256(path)}
    for mode in ("greedy", "search"):
        for after, before in (("value025", "baseline"), ("value0", "baseline"), ("value0", "value025")):
            if (after, mode) not in arenas or (before, mode) not in arenas:
                continue
            sa, a = serialized_pair_scores(arenas[after, mode])
            sb, b = serialized_pair_scores(arenas[before, mode])
            if sa != sb:
                raise ValueError("paired opener panel differs")
            result["arena_differences"][f"{mode}/{after}-minus-{before}"] = paired_bootstrap_mean(
                a - b, resamples=10000, seed=32630000).to_dict()
    result["status"] = "complete" if all(
        (branch, 1000) in raws and all((branch, mode) in arenas for mode in ("greedy", "search"))
        for branch in BRANCHES) else "incomplete"
    write_json(ART / "summary.json", result)
    lines = ["# 第29轮起点：value权重实验结果", "", f"状态：`{result['status']}`。",
             "", "各分支从同一第29轮终点独立训练1000步；其余设置和抽样序列相同。权重1复用原第30轮。", "",
             "| value权重 | 最新代完整留出KL | 最新代完整留出MSE | 窗口留出KL | 窗口留出MSE | greedy对P4得分 | S1024/N1对P4得分 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for branch, row in result["branches"].items():
        final = next((p for p in row["fitting"] if p["step"] == 1000), None)
        if final is None:
            continue
        v, w = final["latest_validation"]["all"], final["window_validation"]
        cells = []
        for mode in ("greedy", "search"):
            arena = row["arenas"].get(mode)
            if not arena or arena["status"] != "complete":
                cells.append("待完成")
                continue
            ci = arena["paired_bootstrap"]
            cells.append(f"{100*arena['win_rate_score']:.3f}% [{100*ci['lower']:.3f}, {100*ci['upper']:.3f}]")
        lines.append(f"| {row['value_loss_weight']:g} | {v['policy_kl']:.6f} | {v['value_mse']:.6f} | {w['policy_kl']:.6f} | {w['value_mse']:.6f} | {' | '.join(cells)} |")
    lines += ["", "区间为95%整pair bootstrap；得分含和棋半分。以下是同开局、同对手面板的得分差，不是两个候选的直接对战。", ""]
    for key, ci in result["arena_differences"].items():
        lines.append(f"- `{key}`：{100*ci['estimate']:+.3f}个百分点，95% CI [{100*ci['lower']:+.3f}, {100*ci['upper']:+.3f}]。")
    lines += ["", "## 最新第30代完整留出学习曲线", "",
              "| 权重 | 步数 | KL | MSE |", "| --- | --- | --- | --- |"]
    for row in result["branches"].values():
        for point in row["fitting"]:
            v = point["latest_validation"]["all"]
            lines.append(f"| {row['value_loss_weight']:g} | {point['step']} | {v['policy_kl']:.6f} | {v['value_mse']:.6f} |")
    lines += ["", "各代、前中期／后期、完整窗口与最新代的原始统计及配对变化区间见[summary.json](summary.json)。训练步数、相同抽样序列及参数漂移验证见各分支`training_verification.json`。", "",
              "权重0保留AdamW动量及参数衰减，并继续更新共享encoder；取消value loss不等于冻结value网络。这里只比较一个固定窗口的一轮更新，不能证明连续自迭代长期收益；固定DEV对战面板也不能代替独立面板复测。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": result["status"], "branches": {
        name: {"fitting_steps": [r["step"] for r in row["fitting"]],
               "arenas": {k: v.get("win_rate_score", v.get("completed_pairs")) for k, v in row["arenas"].items()}}
        for name, row in result["branches"].items()}}), flush=True)


if __name__ == "__main__":
    main()
