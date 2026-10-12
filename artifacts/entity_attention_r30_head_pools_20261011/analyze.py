"""Summarize the fixed-label B/C/D curves and matched greedy panel."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
ORIGINAL = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-020"


def change(before, after, generations):
    keys = sorted(k for k in before if k.endswith("/all") and int(k.split(":")[0]) in generations)
    if any(k not in after for k in keys):
        raise ValueError("validation games differ")
    rng = np.random.default_rng(33450000)
    result = {"games": len(keys), "resamples": 10000, "seed": 33450000, "unit": "whole validation game"}
    for metric, numerator, denominator in (("policy_kl", "kl", "policy_weight"), ("value_mse", "sq_error", "positions")):
        weights = np.array([before[k][denominator] for k in keys])
        if any(before[k][denominator] != after[k][denominator] for k in keys):
            raise ValueError("validation denominators differ")
        diffs = np.array([after[k][numerator]-before[k][numerator] for k in keys])
        means = []
        for _ in range(40):
            selected = rng.integers(0, len(keys), size=(250, len(keys)))
            means.extend((diffs[selected].sum(1)/weights[selected].sum(1)).tolist())
        result[metric] = {"delta": float(diffs.sum()/weights.sum()), "ci95": np.quantile(means, [.025, .975]).tolist()}
    return result


def main():
    plan = read_json(ART / "plan.json")
    dataset = read_json(ART / "dataset.json")["games"]
    baseline = ART / "common/fitting/step-000000000.json"
    zero = read_json(baseline) if baseline.exists() else None
    rows, matrix, greedy = [], [], {}
    for branch in ("A_existing", "B", "C", "D"):
        for step in plan["steps"]:
            path = (ORIGINAL if branch == "A_existing" else ART / branch) / "fitting" / f"step-{step:09d}.json"
            if not path.exists():
                continue
            raw = read_json(path)
            row = {"branch": branch, "step": step, "global_step": raw["total_train_step"]}
            for split, field in (("train", "online_train_fixed"), ("validation", "online_validation_full")):
                metrics = raw[field]["metrics"]
                for key, values in metrics.items():
                    if key.startswith("g"):
                        generation, _, stage = key.split("/")
                        matrix.append({"branch": branch, "step": step, "absolute_round": int(generation[1:]),
                            "split": split, "stage": stage, **{k: values[k] for k in
                            ("positions", "policy_weight", "policy_kl", "value_mse", "teacher_top1_agreement", "value_bias", "value_correlation")}})
                for window in (1000, 3000, 6000, 9000):
                    generations = range(31-window//1000, 31)
                    if not all(f"g{g:03d}/{split}/all" in metrics for g in generations):
                        continue
                    for metric, count in (("policy_kl", "policy_samples"), ("value_mse", "decision_count")):
                        weights = {g: sum(r[count] for r in dataset if r["round_index"]+1 == g and r["split"] == split) for g in generations}
                        row[f"w{window}_{split}_{metric}"] = sum(metrics[f"g{g:03d}/{split}/all"][metric]*n for g,n in weights.items())/sum(weights.values())
            if step == 1000 and zero is not None:
                row["latest_minus_start"] = change(zero["online_validation_full"]["per_game"], raw["online_validation_full"]["per_game"], {29})
                a = read_json(ORIGINAL / "fitting/step-000001000.json")
                row["latest_minus_A"] = change(a["online_validation_full"]["per_game"], raw["online_validation_full"]["per_game"], {29})
                row["window6k_minus_A"] = change(a["online_validation_full"]["per_game"], raw["online_validation_full"]["per_game"], set(range(24, 30)))
            rows.append(row)
        path = ORIGINAL / "evaluation/greedy_vs_start.json" if branch == "A_existing" else ART / branch / "greedy.json"
        if path.exists():
            raw = read_json(path)
            if raw["passed"] and raw["completed_pairs"] == 200:
                seeds, scores = serialized_pair_scores(raw)
                if seeds != tuple(plan["greedy_template"]["game_seeds"]):
                    raise ValueError("greedy panel differs")
                if branch != "A_existing":
                    checkpoint = ROOT / read_json(ART / branch / "training_verification.json")["checkpoint"]["path"]
                    if raw["checkpoint_identity"]["policy_a"]["sha256"] != sha256(checkpoint):
                        raise ValueError("evaluated checkpoint differs")
                greedy[branch] = {"score": raw["win_rate_score"], "ci95": raw["paired_bootstrap"],
                    **{k: raw[k] for k in ("a_wins", "draws", "a_losses", "engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "policy_cycle_draws")}}
                if branch == "A_existing":
                    a_seeds, a_scores = seeds, scores
                else:
                    greedy[branch]["minus_A"] = paired_bootstrap_difference(seeds, scores, a_seeds, a_scores, resamples=10000, seed=33460000).to_dict()
    complete = all(any(r["branch"] == b and r["step"] == 1000 for r in rows) and b in greedy for b in ("B", "C", "D"))
    if complete:
        hashes = {b: read_json(ART / b / "head_reference_hashes.json") for b in ("B", "C", "D")}
        if hashes["B"]["policy"] != hashes["D"]["policy"] or hashes["C"]["value"] != hashes["D"]["value"]:
            raise ValueError("unchanged head streams differ across branches")
    result = {"status": "complete" if complete else "incomplete", "plan_sha256": sha256(ART / "plan.json"),
        "points": rows, "greedy": greedy, "matched_common_head_streams": True if complete else None,
        "scope": plan["evaluation"], "uncertainty": "Conditional DEV, one training seed, exploratory comparisons without multiplicity adjustment."}
    write_json(ART / "summary.json", result)
    if matrix:
        with (ART / "generation_stage_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(matrix[0]))
            writer.writeheader()
            writer.writerows(matrix)
    lines = ["# 第29轮终点：policy缩池／value扩池BCD", "", f"状态：`{result['status']}`。",
        "", "均从第29轮结束模型（global58898）出发，使用截至第30轮的已采数据；1000步、lr2e-5、每头batch256、loss权重1:1、共享encoder联合反传、统一clip1。A仅引用原第30轮，无新A训练或对战。",
        "", "B=P3k/V6k，C=P6k/V9k，D=P3k/V9k。3k=第28–30轮，6k=25–30轮，9k=22–30轮。原整局划分不变；指标均在相同局面与原标签上比较。",
        "", "最新第30轮：固定2048个Train局面与完整Validation。KL=KL(搜索教师标签∥学生)，MSE对原终局标签，越低越好。", "",
        "| 分支 | 步数 | Train KL | 留出 KL | Train MSE | 留出 MSE |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f"| {r['branch']} | {r['step']} | {r['w1000_train_policy_kl']:.6f} | {r['w1000_validation_policy_kl']:.6f} | {r['w1000_train_value_mse']:.6f} | {r['w1000_validation_value_mse']:.6f} |")
    lines += ["", "Greedy均对P4V4，同原A的200 pair开局，和棋半分；差值为同一开局面板上的得分差，不是直接对A的胜率。", "",
        "| 分支 | 胜／和／负 | 得分率 | 95% CI | 相对A差值及95% CI |", "| --- | --- | ---: | --- | --- |"]
    for b, r in greedy.items():
        ci = r["ci95"]
        d = r.get("minus_A")
        delta = f"{100*d['estimate']:+.3f}pp [{100*d['lower']:+.3f}, {100*d['upper']:+.3f}]" if d else "参考"
        lines.append(f"| {b} | {r['a_wins']}/{r['draws']}/{r['a_losses']} | {100*r['score']:.3f}% | [{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%] | {delta} |")
    lines += ["", "终点相对原A的最新轮完整留出差值（整局配对bootstrap10000次，负值为改善）：", "",
        "| 分支 | ΔKL及95% CI | ΔMSE及95% CI |", "| --- | --- | --- |"]
    for r in rows:
        if r["step"] != 1000 or r["branch"] == "A_existing" or "latest_minus_A" not in r:
            continue
        cells = []
        for metric in ("policy_kl", "value_mse"):
            d = r["latest_minus_A"][metric]
            cells.append(f"{d['delta']:+.6f} [{d['ci95'][0]:+.6f}, {d['ci95'][1]:+.6f}]")
        lines.append(f"| {r['branch']} | {' | '.join(cells)} |")
    lines += ["", "全窗口、逐代、前中期／后期数据见summary.json和generation_stage_matrix.csv。Train指标是固定子集，不是完整Train；summary中的窗口Train按原池局面数加权。后期定义双方最高分≥10。",
        "", "6k未改变的头已逐步核对原A抽样序列；B/D的policy、C/D的value序列也相同。断点保存AdamW、Torch和采样RNG，每100步可恢复，250步另存。",
        "", "这是固定第30轮数据、单训练seed的趋势筛查。Greedy只使用policy输出；value池可通过共享encoder影响policy。不能据此确认带搜索棋力、长期自迭代收益或value改进本身的棋力效应。旧留出／对战面板已多次使用，均属DEV。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    path = ROOT / "artifacts/experiment_index.json"
    index = read_json(path)
    entry_id = "entity-attention-r30-head-pools-2026-10-11"
    entry = next((e for e in index["entries"] if e["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    files = [p for p in ART.rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".py", ".csv") and ".json.games" not in str(p)]
    entry.update(status="valid" if complete else "incomplete", report=str((ART / "results.md").relative_to(ROOT)),
        scope=plan["evaluation"], evidence=[str(p.relative_to(ROOT)) for p in sorted(files)],
        evidence_hashes={str(p.relative_to(ROOT)): sha256(p) for p in files})
    write_json(path, index)
    print(result["status"], flush=True)


if __name__ == "__main__":
    main()
