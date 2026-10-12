"""Consolidate existing round-nine metrics; no model loading or evaluation."""

import hashlib
import json
import math
from pathlib import Path


BASE = Path(__file__).resolve().parent
ROOT = BASE.parent.parent
STEPS = (100, 250, 500, 1000, 2000)
LABELS = {
    "baseline": "原配置",
    "lr_5e-05": "lr=5e-5",
    "lr_2e-05": "lr=2e-5",
    "lr_1e-05": "lr=1e-5",
    "pool_9000": "池=9k局",
    "pool_3000": "池=3k局",
    "pool_1000": "池=1k局",
    "batch_64": "B=64",
    "batch_128": "B=128",
    "batch_512": "B=512",
    "batch_1024": "B=1024",
}
METRICS = {
    "policy_kl": ("validation", "g009/validation/all", "policy_kl"),
    "value_mse": ("validation", "g009/validation/all", "value_mse"),
    "latest_train_kl": ("latest_train", "g009/train/all", "policy_kl"),
    "latest_train_mse": ("latest_train", "g009/train/all", "value_mse"),
}


def read(path):
    return json.loads(path.read_text())


def identity(path):
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def check_close(actual, expected, tolerance=1e-12):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f"Metric mismatch: {actual} != {expected}")


def main():
    plan = read(BASE / "plan.json")
    summary = read(BASE / "summary.json")
    plan_identity = identity(BASE / "plan.json")
    assert summary["status"] == "complete"
    assert summary["plan_sha256"] == plan_identity["sha256"]
    plan_panels = {p["id"]: p for p in plan["panels"]}
    configs = [{
        "id": "baseline", "label": LABELS["baseline"], "axis": "baseline",
        "learner": plan["base_learner"], "pool_games": plan["baseline_pool_games"],
        "generations": [4, 5, 6, 7, 8, 9],
    }]
    configs += [dict(plan_panels[p["id"]], label=LABELS[p["id"]])
                for p in summary["panels"]]
    point_sets = {"baseline": summary["baseline"]}
    point_sets.update({p["id"]: p["points"] for p in summary["panels"]})
    rows = []
    train_digests = set()
    for config in configs:
        points = {p["step"]: p for p in point_sets[config["id"]]}
        for step in STEPS:
            point = points[step]
            raw_path = BASE / "fitting" / config["id"] / f"step-{step:09d}.json"
            raw = read(raw_path)
            assert raw["step"] == step
            assert raw["global_step"] == plan["start"]["total_train_step"] + step
            assert raw["checkpoint"] == point["checkpoint"]
            assert raw["plan_sha256"] == plan_identity["sha256"]
            for name, (split, key, metric) in METRICS.items():
                check_close(point[name], raw[split]["metrics"][key][metric])
            val = raw["validation"]["metrics"]["g009/validation/all"]
            train = raw["latest_train"]["metrics"]["g009/train/all"]
            assert (val["positions"], val["policy_weight"]) == (8174, 6688)
            assert (train["positions"], train["policy_weight"]) == (2048, 1652)
            train_digests.add(raw["latest_train"]["set_digest"])
            difference_key = "delta_vs_500" if config["id"] == "baseline" else "delta_vs_same_step"
            rows.append({
                "config_id": config["id"], "step": step,
                "global_step": raw["global_step"],
                **{name: point[name] for name in METRICS},
                "checkpoint": point["checkpoint"], "raw_evaluation": identity(raw_path),
                "existing_comparison": {"reference": difference_key, **point[difference_key]},
            })
    assert len(configs) == 11 and len(rows) == 55 and len(train_digests) == 1
    canonical = {r["step"]: r for r in rows if r["config_id"] == "baseline"}
    duplicate_audit = []
    # Original evaluations and extension evaluations reuse these same checkpoints.
    for p in plan["baselines"]:
        if p["step"] not in STEPS:
            continue
        path = ROOT / p["original_evaluation"]["path"]
        assert identity(path) == p["original_evaluation"]
        duplicate_audit.append(audit_duplicate(path, canonical[p["step"]]))
    extension_500 = ROOT / "artifacts/entity_attention_r9_step_extension_20261010/fitting/step-000000500.json"
    duplicate_audit.append(audit_duplicate(extension_500, canonical[500]))

    matrices = {
        metric: {c["id"]: [next(r[metric] for r in rows
                               if r["config_id"] == c["id"] and r["step"] == step)
                          for step in STEPS] for c in configs}
        for metric in METRICS
    }
    data = {
        "scope": "Existing round-nine OFAT data only; same round-eight endpoint and AdamW state; no new training or evaluation",
        "steps": list(STEPS), "configurations": configs, "unique_points": len(rows),
        "inputs": [plan_identity, identity(BASE / "summary.json")],
        "start": plan["start"], "initial_checkpoint": plan["initial_checkpoint"],
        "validation": plan["validation"],
        "latest_train": {"positions": 2048, "policy_positions": 1652,
                         "set_digest": next(iter(train_digests)),
                         "references": plan["latest_train_references"]},
        "rows": rows, "matrices": matrices,
        "duplicate_baseline_audit": duplicate_audit,
        "excluded_steps": {"baseline": [0, 5000], "all_new_branches": [0], "batch_1024": [125]},
        "caveat": "Single starting model and sampling seed; reused fixed DEV; row minima descriptive; no new strength results",
    }
    (BASE / "steps_100_2000.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_report(configs, matrices, duplicate_audit)
    print(json.dumps({"unique_points": len(rows), "configurations": len(configs),
                      "duplicate_evaluations_checked": len(duplicate_audit),
                      "max_duplicate_metric_difference": max(a["max_abs_difference"] for a in duplicate_audit)},
                     ensure_ascii=False))


def audit_duplicate(path, canonical):
    raw = read(path)
    if "online_validation_full" in raw:
        checkpoint_sha = raw["checkpoint_sha256"]
        val = raw["online_validation_full"]["metrics"]["g009/validation/all"]
        train = raw["online_train_fixed"]["metrics"]["g009/train/all"]
        observed = {"policy_kl": val["policy_kl"], "value_mse": val["value_mse"],
                    "latest_train_kl": train["policy_kl"], "latest_train_mse": train["value_mse"]}
    else:
        checkpoint_sha = raw["checkpoint"]["sha256"]
        val = raw["evaluation"]["metrics"]["g009/validation/all"]
        observed = {"policy_kl": val["policy_kl"], "value_mse": val["value_mse"]}
    assert checkpoint_sha == canonical["checkpoint"]["sha256"]
    for metric, value in observed.items():
        check_close(value, canonical[metric], tolerance=1e-6)
    return {"step": canonical["step"], "source": identity(path),
            "checkpoint_sha256": checkpoint_sha, "metrics_checked": list(observed),
            "max_abs_difference": max(abs(value - canonical[metric]) for metric, value in observed.items())}


def write_report(configs, matrices, duplicates):
    lines = [
        "# 第九轮实验：100／250／500／1000／2000步完整数据汇总", "",
        "2026-10-10。汇总本次同起点单变量实验的全部11组配置、5个指定节点，共55个唯一配置—节点记录，无缺项。仅整理已有结果，没有新增训练、重新评估模型或搜索对战。", "",
        "共同起点为第八轮结束模型（全局38,398步），继承原AdamW状态与RNG。下表步数均从该起点计；全局步数分别为38,498／38,648／38,898／39,398／40,398。", "",
        "原配置：**lr=1e-4，在线池6k局（第4–9轮），B=256**。每行仅改变行名指定的变量；权重衰减1e-4、policy/value等权联合训练、梯度裁剪1保持一致。在线池9k／3k／1k分别为第1–9轮／第7–9轮／仅第9轮，训练只抽原Train局面，29k旧离线池不参与。", "",
        "留出为第九轮固定完整100局，8174个value局面、6688个有效policy局面；Train为第九轮固定2048局面子集，其中1652个有效policy局面，**不是整个6k窗口的训练loss**。KL=KL(原行为搜索教师标签∥学生)，MSE对原终局标签，两项均越低越好。", "",
        "粗体表示该行在这五个节点中的最低点，只是描述性排序。所有数值保留6位小数，完整精度、checkpoint身份、原始评估路径与已有配对区间保存在[JSON汇总](steps_100_2000.json)。", "",
    ]
    for metric, title in (("policy_kl", "完整留出集 policy KL"),
                          ("value_mse", "完整留出集 value MSE"),
                          ("latest_train_kl", "最新代固定Train子集 policy KL"),
                          ("latest_train_mse", "最新代固定Train子集 value MSE")):
        lines.extend([f"## {title}", "", "| 配置 | 100步 | 250步 | 500步 | 1000步 | 2000步 |",
                      "| --- | ---: | ---: | ---: | ---: | ---: |"])
        for c in configs:
            values = matrices[metric][c["id"]]
            cells = [f"**{v:.6f}**" if v == min(values) else f"{v:.6f}" for v in values]
            lines.append("| " + " | ".join([c["label"], *cells]) + " |")
        lines.append("")
    lines.extend([
        "## 来源与重复记录", "",
        "统一数值取自[已完成汇总](summary.json)，55个节点的四项指标均逐一与`fitting/<配置>/step-XXXXXXXXX.json`核对；逐点checkpoint SHA与汇总一致，所有点共用相同局面分母及Train引用集合。冻结配置见[plan.json](plan.json)。", "",
        f"原配置旧实验的{len(duplicates)}份重复评估也已核对checkpoint SHA及可比指标：100／250／500步的原第九轮诊断，1000／2000步的扩展实验，以及扩展实验中再次记录的500步。它们属于同一训练轨迹，不另计为独立样本。最大绝对数值差为{max(d['max_abs_difference'] for d in duplicates):.3g}，完整审计见JSON。", "",
        "用户本次指定范围外的0步、B1024的125步、原配置5000步保留在原结果中，本表不混入。配对区间在JSON中区分两种参照：原配置各节点相对原500步；其他配置相对原配置的同一步数。不同参照的Δ不能直接混比，详见[完整区间表](results.md)。", "",
        "## 读表要点", "",
        "原配置的五个节点中，留出KL和MSE都在500步取最低值；100／250步的policy回退同时出现在Train和留出。较小学习率减轻早期policy回退，其中lr=1e-5的五点最低KL在1000步。", "",
        "1k池随更新增加表现出强烈的value训练—留出分化：2000步Train MSE约0.254506，留出MSE约1.015097。扩大batch时，固定步数也增加了抽样量，不能只据本表判断同等数据预算下的优劣。", "",
        "本表来自同一起点、单采样随机流与固定DEV，未作多重比较校正。最低点不能当作已确认的最优配置，较低KL/MSE也不自动代表棋力提升；具体区间和实验解释见[完成分析](analysis.md)。", "",
        "复现本次整理（仅CPU读取已有JSON）：", "", "```bash",
        "conda run --no-capture-output -n splendor python artifacts/entity_attention_r9_ofat_20261010/consolidate.py",
        "```", "",
    ])
    (BASE / "steps_100_2000.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
