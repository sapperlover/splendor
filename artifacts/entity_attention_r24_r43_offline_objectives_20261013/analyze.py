"""Summarize existing offline logs and the completed greedy panels; no inference."""
from pathlib import Path
import json

from splendor.alphazero_recording import read_json, sha256, write_json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def training_window(rows):
    sums = {k: sum(r[k] for r in rows) for k in ("policy_ce_sum", "policy_weight_sum", "target_entropy_sum", "value_sq_sum", "value_weight_sum")}
    result = {"steps": len(rows), "policy_ce": sums["policy_ce_sum"] / sums["policy_weight_sum"],
              "policy_kl": (sums["policy_ce_sum"] - sums["target_entropy_sum"]) / sums["policy_weight_sum"],
              "value_mse": sums["value_sq_sum"] / sums["value_weight_sum"],
              "mean_total_loss": sum(r["total_loss"] for r in rows) / len(rows)}
    for k in ("score_loss", "remaining_turns_loss"):
        if k in rows[0]:
            result[k] = sum(r[k] for r in rows) / len(rows)
    return result


def main():
    branches = {}
    ledger_hashes = []
    reference_validation = None
    for branch in ("policy_surprise", "value_auxiliary"):
        root = ROOT / f"runs/entity_attention_r24_r43_{branch}_offline_20261013"
        progress = read_json(root / "progress.json")
        if progress["status"] != "complete" or progress["step"] != 22681:
            raise ValueError(f"training incomplete: {branch}")
        curve = [json.loads(line) for line in (root / "validation_curve.jsonl").read_text().splitlines()]
        rows = [json.loads(line) for line in (root / "training_metrics.jsonl").read_text().splitlines()]
        rows = [r for r in rows if r.get("event") == "train_step"]
        assert len(rows) == 22681
        ledgers = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((root / "references").glob("*.bin"))}
        ledger_hashes.append(list(ledgers.values()))
        fixed = read_json(root / "fixed_validation.json")
        if reference_validation is not None:
            assert fixed == reference_validation
        reference_validation = fixed
        best = {}
        for metric in ("policy_kl", "value_loss", "total_loss"):
            p = min(curve, key=lambda r: r["metrics"][metric])
            best[metric] = {"step": p["step"], "value": p["metrics"][metric]}
        common = min(curve, key=lambda r: r["metrics"]["policy_loss"] + r["metrics"]["value_loss"])
        best["common_ce_plus_value_mse"] = {"step": common["step"], "value": common["metrics"]["policy_loss"] + common["metrics"]["value_loss"]}
        branches[branch] = {"progress": {k: progress[k] for k in ("status", "step", "steps_target")},
            "fixed_initial": curve[0]["metrics"], "fixed_final": curve[-1]["metrics"],
            "full_final": read_json(root / "full_validation.json"), "best_fixed": best,
            "first_1000_training_steps": training_window(rows[:1000]),
            "last_1000_training_steps": training_window(rows[-1000:]), "reference_ledgers": ledgers,
            "final_checkpoint": str((root / "checkpoints/final.pt").relative_to(ROOT)),
            "final_checkpoint_sha256": sha256(root / "checkpoints/final.pt"),
            "training_seconds": progress["training"]["training_seconds"]}
    assert ledger_hashes[0] == ledger_hashes[1]
    greedy = read_json(HERE / "greedy_vs_r43_1000pair/summary.json")
    result = {"status": "training_and_greedy_complete", "branches": branches, "greedy": greedy,
              "matched_sampling_ledger_sha256": ledger_hashes[0], "same_fixed_validation": True,
              "scope": "single training seed; no ordinary same-budget control and no search arena",
              "command": "conda run --no-capture-output -n splendor env PYTHONPATH=src python artifacts/entity_attention_r24_r43_offline_objectives_20261013/analyze.py"}
    write_json(HERE / "summary.json", result)
    psw, aux = branches["policy_surprise"], branches["value_auxiliary"]
    lines = ["# R24–43离线目标实验：训练与greedy结果", "",
        "两组22681步离线训练完成；两个final分别对原R43扩至1000pair greedy，每组2000局，前200pair复用原始记录，各新增800pair。两组使用相同完整开局库，未测搜索棋力。完整抽样账本SHA相同，两个训练分支使用同一逐步抽样与4096个固定留出局面。", "",
        "## 同一4096个固定留出局面", "",
        "| 指标 | 起点R43 | PSW最终 | 双辅助最终 |", "| --- | ---: | ---: | ---: |"]
    for name, key in (("Policy CE", "policy_loss"), ("Policy KL", "policy_kl"), ("胜负Value MSE", "value_loss")):
        lines.append(f"| {name} | {psw['fixed_initial'][key]:.6f} | {psw['fixed_final'][key]:.6f} | {aux['fixed_final'][key]:.6f} |")
    values = [r["policy_loss"] + r["value_loss"] for r in (psw["fixed_initial"], psw["fixed_final"], aux["fixed_final"])]
    lines += [f"| 原始CE＋Value MSE | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} |", "",
        f"辅助分差MAE从随机新头的{aux['fixed_initial']['score_mae_points']:.3f}降至{aux['fixed_final']['score_mae_points']:.3f}分；剩余回合MAE从{aux['fixed_initial']['remaining_turns_mae']:.3f}降至{aux['fixed_final']['remaining_turns_mae']:.3f}个玩家回合。",
        "辅助目标学进去了，但未见胜负value的净改善。辅助组的总loss含两个额外项，不与PSW总loss直接比较；主任务共同指标应使用原始CE＋Value MSE。", "",
        "## 全量留出与训练走势", "",
        "全量留出为原2029局、163595局面；没有起点的全量评估，不能将固定子集起点与全量终点直接相减。", "",
        "| 分支 | 全量Policy KL | 全量胜负Value MSE |", "| --- | ---: | ---: |"]
    for branch, b in branches.items():
        lines.append(f"| {branch} | {b['full_final']['policy_kl']:.6f} | {b['full_final']['value_loss']:.6f} |")
    lines += ["", "两组Train value MSE从首1000步到末1000步继续下降，固定留出value却后期回升，符合过拟合迹象。PSW训练policy指标使用surprise权重，与辅助组普通训练policy指标不能直接横比。", ""]
    for branch, b in branches.items():
        lines.append(f"- {branch}：Train value MSE {b['first_1000_training_steps']['value_mse']:.6f}→{b['last_1000_training_steps']['value_mse']:.6f}；固定留出value最小节点为{b['best_fixed']['value_loss']['step']}步，原CE＋Value MSE最小节点为{b['best_fixed']['common_ce_plus_value_mse']['step']}步。")
    lines += ["", "## 对原R43的greedy棋力", "",
        "| 候选 | 胜／和／负 | 得分率 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for branch, p in greedy["panels"].items():
        ci = p["paired_bootstrap"]
        lines.append(f"| {branch} | {p['a_wins']}／{p['draws']}／{p['a_losses']} | {p['win_rate_score']:.3%} | [{ci['lower']:.3%}, {ci['upper']:.3%}] |")
    diff = greedy["auxiliary_minus_psw_against_r43"]
    lines += ["", f"辅助组减PSW对共同R43的得分差{diff['estimate']*100:+.3f}pp，95%配对区间[{diff['lower']*100:+.3f}, {diff['upper']*100:+.3f}]pp；这是对共同对手的得分差，不是候选直接互打。", ""]
    for branch, p in greedy["panels"].items():
        ci = p["paired_bootstrap"]
        finding = "得分区间高于50%" if ci["lower"] > .5 else "得分区间低于50%" if ci["upper"] < .5 else "得分区间包含50%，未确认相对R43的提升"
        lines.append(f"- {branch}：{finding}。")
    lines += ["", "greedy选招不使用胜负value。训练时共享encoder受value／辅助梯度影响仍可能相关，尚未独立消融。没有普通同预算训练对照，不能单独确认辅助目标或PSW相对普通训练的因果收益。扩样复用了已查看的前200pair，不能称为新的独立确认；区间未作顺序检验校正，未显著也不代表所有小收益为零。", "",
        "[1000pair逐pair原始结果、配置和恢复命令](greedy_vs_r43_1000pair/results.md)，[原200pair记录](greedy_vs_r43/results.md)，[机器可读训练与评测汇总](summary.json)。原始训练曲线、终点、full_validation和逐步账本位于对应runs目录；本次不修改原件。原verification.json的prepared状态是实现交付时的有限验证记录，不代表当前正式训练状态。", ""]
    (HERE / "results.md").write_text("\n".join(lines))
    print({"status": result["status"], "same_training_draws": True, "greedy_scores": {b: p["win_rate_score"] for b, p in greedy["panels"].items()}})


if __name__ == "__main__":
    main()
