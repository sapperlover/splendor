"""Policy/search alignment, with whole-game bootstrap and two-seed search variation."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def policy_metrics(visits, log_policy):
    """One row per decision. Top1 ties use original candidate order for strict agreement."""
    visits, lp = np.asarray(visits), np.asarray(log_policy, dtype=np.float64)
    q = visits/visits.sum(axis=-1, keepdims=True)
    logq = np.zeros_like(q)
    np.log(q, out=logq, where=q > 0)
    safe_lp = np.where(np.isfinite(lp), lp, 0.)
    search_action, policy_action = np.argmax(visits, -1), np.argmax(lp, -1)
    rows = np.arange(len(q))
    ordered = np.sort(q, axis=-1)
    return {"top1_strict": (search_action == policy_action).astype(float),
        "top1_tie_aware": (visits[rows, policy_action] == visits.max(-1)).astype(float),
        "search_top1_tied": ((visits == visits.max(-1, keepdims=True)).sum(-1) > 1).astype(float),
        "visit_kl": (q*(logq-safe_lp)).sum(-1),
        "greedy_visit_mass": q[rows, policy_action], "search_top1_policy_probability": np.exp(lp[rows, search_action]),
        "search_entropy": -(q*logq).sum(-1), "top1_visit_mass": ordered[:, -1],
        "top1_margin": ordered[:, -1]-ordered[:, -2]}


def search_variation(a, b, log_policy):
    q1 = a/a.sum(-1, keepdims=True)
    q2 = b/b.sum(-1, keepdims=True)
    mean = (q1+q2)/2
    def entropy(q):
        logs = np.zeros_like(q)
        np.log(q, out=logs, where=q > 0)
        return -(q*logs).sum(-1)
    js = entropy(mean)-(entropy(q1)+entropy(q2))/2
    lp = np.where(np.isfinite(log_policy), log_policy, 0.)
    logs = np.zeros_like(mean)
    np.log(mean, out=logs, where=mean > 0)
    return {"repeat_top1_agreement": (np.argmax(a, -1) == np.argmax(b, -1)).astype(float),
            "repeat_visit_jsd": js, "mean_two_targets_kl": (mean*(logs-lp)).sum(-1)}


def cluster_mean(values, games, seed, resamples=10000):
    values, games = np.asarray(values, dtype=float), np.asarray(games)
    unique, inverse = np.unique(games, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(resamples, len(unique)))
    means = sums[sampled].sum(1)/counts[sampled].sum(1)
    return {"estimate": float(values.mean()), "lower": float(np.quantile(means, .025)),
            "upper": float(np.quantile(means, .975)), "games": len(unique), "positions": len(values)}


def main():
    plan = read_json(ART / "plan.json")
    bank = read_json(ART / "root_references.json")
    refs, n = bank["references"], plan["positions"]
    width = max(r["candidate_count"] for r in refs)
    visits = np.zeros((2, n, width), dtype=np.int32)
    logp = np.full((2, n, width), -np.inf, dtype=np.float64)
    extras = {k: np.zeros((2, n), dtype=float) for k in ("terminal_fraction", "cap_hits", "max_depth")}
    evidence = []
    for repeat, seed in enumerate(plan["world_seeds"]):
        for start in range(0, n, plan["capacity"]):
            stop = min(start+plan["capacity"], n)
            path = ART / "raw" / f"repeat-{repeat}-roots-{start:06d}-{stop:06d}.npz"
            with np.load(path, allow_pickle=False) as raw:
                if not np.array_equal(raw["root_ids"], np.arange(start, stop)) or int(raw["seed"]) != seed:
                    raise ValueError("chunk identity differs")
                w = raw["visits"].shape[1]
                visits[repeat, start:stop, :w] = raw["visits"]
                logp[repeat, start:stop, :w] = raw["log_policy"]
                for key in extras:
                    extras[key][repeat, start:stop] = raw[key]
            evidence.append({"path": str(path.relative_to(ROOT)), "sha256": sha256(path)})
    if not np.all(visits.sum(-1) == 1024) or np.any(visits < 0):
        raise ValueError("incomplete or invalid S1024 visits")
    if not np.allclose(np.exp(logp[0]), np.exp(logp[1]), rtol=2e-5, atol=2e-7):
        raise ValueError("policy prior differs across repeats")
    individual = [policy_metrics(visits[i], logp[i]) for i in range(2)]
    metrics = {key: (individual[0][key]+individual[1][key])/2 for key in individual[0]}
    metrics.update(search_variation(visits[0], visits[1], logp[0]))
    metrics.update({key: value.mean(0) for key, value in extras.items()})
    residual = metrics["visit_kl"]-metrics["mean_two_targets_kl"]-metrics["repeat_visit_jsd"]
    if np.max(np.abs(residual)) > 1e-6:
        raise ValueError("KL/JSD decomposition differs")
    games = np.array([r["game_id"] for r in refs])
    groups = {"all": np.ones(n, dtype=bool),
        "early_mid": np.array([r["stage"] == "early_mid" for r in refs]),
        "late": np.array([r["stage"] == "late" for r in refs]),
        "main_action": np.array([r["phase"] == 0 for r in refs]),
        "both_repeats_margin_ge_0.1": np.minimum(individual[0]["top1_margin"], individual[1]["top1_margin"]) >= .1}
    summary = {"status": "complete", "plan_sha256": sha256(ART / "plan.json"), "scope": plan["scope"],
        "positions": n, "games": plan["games"], "by_group": {}, "per_repeat": {},
        "kl_jsd_max_residual": float(np.max(np.abs(residual))), "raw_evidence": evidence,
        "ci": {"unit": "whole source game with multiplicity; position-weighted metric", "resamples": 10000,
               "seed": plan["bootstrap_seed"], "scope": "conditional on model, historical self-play distribution and these two world seeds; no training-seed or general playing-strength claim"}}
    for name, mask in groups.items():
        if mask.any():
            summary["by_group"][name] = {key: cluster_mean(value[mask], games[mask], plan["bootstrap_seed"]) for key, value in metrics.items()}
    for i, seed in enumerate(plan["world_seeds"]):
        summary["per_repeat"][str(seed)] = {key: float(value.mean()) for key, value in individual[i].items()}
    write_json(ART / "summary.json", summary)
    with (ART / "per_root.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ("root_id", "game_id", "decision_index", "phase", "candidate_count", "stage", *metrics)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, ref in enumerate(refs):
            writer.writerow({**{key: ref[key] for key in fields if key in ref}, **{key: float(value[i]) for key,value in metrics.items()}})
    lines = ["# 第30轮policy与无根噪声S1024的一致性", "", "已完成。原第30轮完整模型（global59898），第30轮完整留出集的全部有效policy决策；原对局由第29轮教师采集，未用于模型训练。只有一个合法动作的决策不计入。",
        "", f"{plan['games']}局、{n}个局面，每局面做两个独立世界seed的S1024/N1，无Dirichlet根噪声、温度0。搜索使用当前第30轮同一完整网络。指标先对每个局面的两次搜索取均值，再对局面平均；不把两次搜索称为N2。", "",
        "Top-1严格一致率比较policy logits的argmax与根访问次数argmax；平局时均取原候选序首个。另记录并列容忍一致率。KL为KL(归一化根访问分布∥policy softmax)，自然对数；不使用温度0的one-hot行为分布。95% CI按来源整局重采样10000次。", "",
        "| 范围 | 局面数 | Top-1一致率及95% CI | 访问KL及95% CI | 搜索重复Top-1一致率 |", "| --- | ---: | --- | --- | ---: |"]
    for name in ("all", "early_mid", "late", "main_action", "both_repeats_margin_ge_0.1"):
        g = summary["by_group"].get(name)
        if g is None:
            continue
        a, k, r = g["top1_strict"], g["visit_kl"], g["repeat_top1_agreement"]
        lines.append(f"| {name} | {a['positions']} | {100*a['estimate']:.3f}% [{100*a['lower']:.3f}, {100*a['upper']:.3f}] | {k['estimate']:.6f} [{k['lower']:.6f}, {k['upper']:.6f}] | {100*r['estimate']:.3f}% |")
    g = summary["by_group"]["all"]
    lines += ["", f"并列容忍Top-1一致率{100*g['top1_tie_aware']['estimate']:.3f}%；搜索最大访问数并列比例{100*g['search_top1_tied']['estimate']:.3f}%。Greedy动作平均获得搜索访问份额{g['greedy_visit_mass']['estimate']:.6f}，搜索首选动作平均获得policy概率{g['search_top1_policy_probability']['estimate']:.6f}。", "",
        f"两次搜索的访问JSD={g['repeat_visit_jsd']['estimate']:.6f}。两目标均值对policy的KL={g['mean_two_targets_kl']['estimate']:.6f}；单次目标平均KL=均值目标KL+两目标JSD，最大恒等式残差{summary['kl_jsd_max_residual']:.3g}。两次重复只提供有限随机性参照；经验均值不是已知正确策略，差异也不是已证明不可约噪声。", "",
        "前中期／后期以双方最高分<10／≥10划分；主行动组排除其他选择阶段。高margin组要求两次搜索的Top-1访问份额均比Top-2至少高0.1，只反映搜索决策明确程度，不是已知真实价值优势。", "",
        "这些指标测搜索策略与网络的剩余差距。单个当前checkpoint不能证明历史训练是否学会了搜索提升，也不能把与自身搜索的一致当作正确决策或棋力。完全同策略的第30轮新自对弈分布、训练前后固定独立教师比较均未在本实验测试。", "",
        "模型、输入NPZ、公开请求、候选顺序／特征、S1024预算、根噪声关闭、搜索prior与独立网络前向均已核对。逐局面数据见[per_root.csv](per_root.csv)，全部访问数与policy log概率见raw/，配置与来源见plan.json和root_references.json。", ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    path = ROOT / "artifacts/experiment_index.json"
    index = read_json(path)
    entry_id = "entity-attention-r30-search-alignment-2026-10-11"
    entry = next((e for e in index["entries"] if e["id"] == entry_id), None)
    if entry is None:
        entry = {"id": entry_id}
        index["entries"].append(entry)
    files = [p for p in ART.rglob("*") if p.is_file() and p.suffix in (".json", ".md", ".py", ".csv", ".npz")]
    entry.update(status="valid", report=str((ART / "results.md").relative_to(ROOT)), scope=plan["scope"],
        evidence=[str(p.relative_to(ROOT)) for p in sorted(files)],
        evidence_hashes={str(p.relative_to(ROOT)): sha256(p) for p in files})
    write_json(path, index)
    print({"status": "complete", "positions": n, "top1": g["top1_strict"], "KL": g["visit_kl"]}, flush=True)


if __name__ == "__main__":
    main()
