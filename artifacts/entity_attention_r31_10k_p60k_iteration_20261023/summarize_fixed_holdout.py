"""Archive the six-round backfill and compute paired whole-game intervals."""
from pathlib import Path
import csv
import shutil

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.fixed_holdout import COLUMNS, check_disjoint

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
REPORT = ART / "fixed_holdout_20261023"
RUN = ROOT / "runs" / ART.name
SEED, RESAMPLES = 36970000, 4000


def interval(before, after, manifest):
    # Paired resampling of complete games, stratified by original source round.
    rng = np.random.default_rng(SEED)
    strata = [np.array([i for i, g in enumerate(manifest["games"]) if g["round_index"] == r])
              for r in sorted({g["round_index"] for g in manifest["games"]})]
    columns = [COLUMNS.index(k) for k in ("kl", "policy_weight", "sq_error", "positions")]
    a, b = before[:, columns], after[:, columns]
    deltas = []
    for start in range(0, RESAMPLES, 32):
        n = min(32, RESAMPLES-start)
        ids = np.concatenate([rng.choice(s, size=(n, len(s))) for s in strata], axis=1)
        x, y = a[ids].sum(axis=1), b[ids].sum(axis=1)
        deltas.append(np.stack((y[:, 0]/y[:, 1]-x[:, 0]/x[:, 1],
                               y[:, 2]/y[:, 3]-x[:, 2]/x[:, 3]), axis=1))
    values = np.concatenate(deltas)
    x, y = a.sum(axis=0), b.sum(axis=0)
    point = [y[0]/y[1]-x[0]/x[1], y[2]/y[3]-x[2]/x[3]]
    return {key: {"delta": float(point[i]), "ci95": np.quantile(values[:, i], [.025, .975]).tolist()}
            for i, key in enumerate(("policy_kl", "value_mse"))}


def main():
    manifest = read_json(REPORT / "manifest.json")
    config = read_json(ROOT / "configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json")
    state_path = RUN / "state.json"
    state = read_json(state_path)
    if len(state["completed_rounds"]) != 6:
        raise ValueError("this report freezes precisely local rounds1-6, R32-R37")
    nodes, raw_manifest, arrays = [], [], {}
    for completed in state["completed_rounds"]:
        local = completed["iteration"]
        snapshot = Path(completed["snapshot"]["path"])
        if sha256(snapshot) != completed["snapshot"]["sha256"]:
            raise ValueError("replay snapshot changed")
        check_disjoint(manifest, read_json(snapshot)["games"])
        for step in config["validation_steps"]:
            source = RUN / f"rounds/round-{local-1:03d}/fixed_holdout/step-{step:09d}.json"
            result = read_json(source)
            stats = Path(result["per_game"]["path"])
            original_validation = RUN / f"rounds/round-{local-1:03d}/validation/step-{step:09d}.json"
            if (result["panel_digest"] != manifest["digest"]
                    or sha256(Path(result["checkpoint"]["path"])) != result["checkpoint"]["sha256"]
                    or read_json(original_validation)["checkpoint_sha256"] != result["checkpoint"]["sha256"]
                    or sha256(stats) != result["per_game"]["sha256"]):
                raise ValueError("backfill result identity changed")
            with np.load(stats) as archive:
                values = archive["per_game"]
            if values.shape != (len(manifest["games"]), len(COLUMNS)):
                raise ValueError("incomplete per-game result")
            arrays[local, step] = values
            for original in (source, stats):
                target = REPORT / "raw" / f"round-{local:03d}-step-{step:04d}{original.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, target)
                raw_manifest.append({"original_path": str(original), "path": str(target.relative_to(ROOT)),
                                     "sha256": sha256(target)})
            nodes.append({"local_round": local, "lineage": 31+local, "step": step,
                "metrics": result["metrics"], "seconds": result["seconds"], "checkpoint": result["checkpoint"]})
    comparisons = {f"R{31+r}_within_round": interval(arrays[r, 0], arrays[r, 3000], manifest) for r in range(1, 7)}
    comparisons["R32_initial_to_R37_final"] = interval(arrays[1, 0], arrays[6, 3000], manifest)
    summary = {"panel_digest": manifest["digest"], "games": len(manifest["games"]),
        "policy_positions": manifest["policy_positions"], "value_positions": manifest["value_positions"],
        "completed_nodes": len(nodes), "expected_nodes": 42, "nodes": nodes, "comparisons": comparisons,
        "checkpoint_identities_match_original_validation": len(nodes),
        "bootstrap": {"seed": SEED, "resamples": RESAMPLES, "unit": "paired whole games stratified by source round",
                      "scope": "pointwise95%; exploratory DEV; no multiple-comparison correction"},
        "state_unchanged_from_before_integration": sha256(state_path) == read_json(REPORT / "before_integration.json")["state"]["sha256"],
        "evaluation_seconds_sum": sum(n["seconds"] for n in nodes)}
    write_json(REPORT / "summary.json", summary)
    write_json(REPORT / "raw_manifest.json", {"files": raw_manifest})
    with (REPORT / "checkpoints.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=("lineage", "local_round", "step", "policy_kl", "value_mse", "seconds"))
        writer.writeheader()
        for n in nodes:
            writer.writerow({k: n[k] for k in ("lineage", "local_round", "step", "seconds")} |
                            {k: n["metrics"][k] for k in ("policy_kl", "value_mse")})
    lines = ["# R26–R31固定留出：R32–R37六轮回补", "",
        f"全量原始validation共{len(manifest['games'])}局、{manifest['policy_positions']}个有效policy位置、{manifest['value_positions']}个value位置。",
        "42/42节点完成；每轮0／100／250／500／1000／2000／3000步，同一面板、原始标签、FP32前向。六轮训练快照均无留出泄漏。", "",
        "| 模型轮 | KL：0→3000 | ΔKL 95% CI | MSE：0→3000 | ΔMSE 95% CI |", "| --- | --- | --- | --- | --- |"]
    for r in range(1, 7):
        a = next(n["metrics"] for n in nodes if n["local_round"] == r and n["step"] == 0)
        b = next(n["metrics"] for n in nodes if n["local_round"] == r and n["step"] == 3000)
        d = comparisons[f"R{31+r}_within_round"]
        ci = lambda key: "[" + ", ".join(f"{v:+.6f}" for v in d[key]["ci95"]) + "]"
        lines.append(f"| R{31+r} | {a['policy_kl']:.6f}→{b['policy_kl']:.6f} | {ci('policy_kl')} | {a['value_mse']:.6f}→{b['value_mse']:.6f} | {ci('value_mse')} |")
    total = comparisons["R32_initial_to_R37_final"]
    lines += ["", f"累计变化（R32训练前→R37训练后）：KL {total['policy_kl']['delta']:+.6f}，95% CI [{total['policy_kl']['ci95'][0]:+.6f}, {total['policy_kl']['ci95'][1]:+.6f}]；MSE {total['value_mse']['delta']:+.6f}，95% CI [{total['value_mse']['ci95'][0]:+.6f}, {total['value_mse']['ci95'][1]:+.6f}]。",
        "", "固定旧面板上，value有累计改善证据，但5/6轮末尾下降不等于逐轮稳定收益：只有R32单轮的MSE点区间不跨0。policy KL三轮下降、三轮上升，累计小幅增加。不能继续把跨轮2048小面板的绝对值当收敛曲线；也不能把对旧teacher的KL上升解释为棋力下降。",
        "", "区间以整局配对、按六个来源轮分层，4000次bootstrap，seed36970000；逐项95%区间未经多重比较校正。此为固定旧teacher标签上的DEV拟合诊断，不能等同棋力或用于证明学习率最优。",
        "", "[逐节点CSV](checkpoints.csv) · [指标及区间](summary.json) · [数据清单](manifest.json) · [原始逐局统计与身份](raw_manifest.json)", ""]
    (REPORT / "results.md").write_text("\n".join(lines))
    print({"completed_nodes": len(nodes), "overall": total}, flush=True)


if __name__ == "__main__":
    main()
