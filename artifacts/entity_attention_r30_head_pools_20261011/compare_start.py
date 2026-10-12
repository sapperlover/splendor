"""Compare already measured greedy scores with the actual round29 start."""
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, write_json
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
ROUNDS = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds"


def main():
    path = ROUNDS / "round-019/evaluation/greedy_vs_start.json"
    start = read_json(path)
    seeds, scores = serialized_pair_scores(start)
    summary = read_json(ART / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("BCD not complete")
    result = {"start": {"absolute_round": 29, "score": start["win_rate_score"],
        "result": str(path.relative_to(ROOT)), "sha256": sha256(path)}, "comparisons": {},
        "command": f"conda run -n splendor env PYTHONPATH=src python {Path(__file__).relative_to(ROOT)}",
        "source_sha256": sha256(Path(__file__)),
        "scope": "Reuses existing round29 and all final greedy vs P4V4 results, identical DEV opener/policy seeds and opponent; no new matches."}
    for branch in ("A_existing", "B", "C", "D"):
        path = ROUNDS / "round-020/evaluation/greedy_vs_start.json" if branch == "A_existing" else ART / branch / "greedy.json"
        raw = read_json(path)
        if (raw["config"]["policy_seed_base"] != start["config"]["policy_seed_base"]
                or raw["checkpoint_identity"]["policy_b"] != start["checkpoint_identity"]["policy_b"]):
            raise ValueError("greedy start comparison differs in panel or opponent")
        current_seeds, current = serialized_pair_scores(raw)
        result["comparisons"][branch] = {"score": raw["win_rate_score"],
            "result": str(path.relative_to(ROOT)), "sha256": sha256(path),
            "minus_start": paired_bootstrap_difference(current_seeds, current, seeds, scores,
                resamples=10000, seed=33470000).to_dict()}
    write_json(ART / "greedy_start_comparison.json", result)
    lines = ["# BCD趋势结论", "", "本次已完成。起点是第29轮结束，数据为第30轮及其在线历史；三个分支各1000步，greedy各200pair，无新采集、重标签或搜索对战。",
        "", "**目前不支持把policy缩到3k、value扩到9k当作已有效的改进方案。B有微弱greedy正向数值，尚未确认；D没有显示两种改动叠加的优势。**",
        "", "| 分支 | 最新轮Train KL | 最新轮留出KL | 最新轮留出MSE | greedy对P4V4 | 相对A得分差95% CI |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for branch in ("A_existing", "B", "C", "D"):
        row = next(r for r in summary["points"] if r["branch"] == branch and r["step"] == 1000)
        g = summary["greedy"][branch]
        d = g.get("minus_A")
        ci = f"[{100*d['lower']:+.3f}, {100*d['upper']:+.3f}]pp" if d else "参考"
        lines.append(f"| {branch} | {row['w1000_train_policy_kl']:.6f} | {row['w1000_validation_policy_kl']:.6f} | {row['w1000_validation_value_mse']:.6f} | {100*g['score']:.3f}% | {ci} |")
    lines += ["", "B/D缩policy池后，最新轮Train KL从0.421197降至0.416350/0.416998，优于原A终点0.422533；但留出没有同步改善，且三分支终点的最新轮留出KL均高于训练前0.413672。相对原A，最近6k窗口完整留出KL三组均变差（条件bootstrap区间均大于0），有把拟合转向近期Train而损失窗口内旧局面拟合的迹象。不能把这概括为所有局面都无法泛化：原A的6k整体留出KL本轮仍从0.424481降至0.422988。",
        "", "C/D的最新轮留出MSE数值低于A，但差值区间均跨0；仍高于训练前0.696947。扩大value池也没有改善共同9k留出MSE：起点0.759165，B/C/D终点为0.760733/0.762934/0.763860。这更像拟合分配改变，尚无全窗口泛化改善证据。Greedy不使用value评估，因此本轮也未验证critic的带搜索价值。",
        "", "起点第29轮已有同面板greedy得分79.500%，以下仅复用已有对局，比较本轮训练前后：", "",
        "| 分支 | 相对第29轮起点得分差 | 95% CI |", "| --- | ---: | --- |"]
    for branch, r in result["comparisons"].items():
        d = r["minus_start"]
        lines.append(f"| {branch} | {100*d['estimate']:+.3f}pp | [{100*d['lower']:+.3f}, {100*d['upper']:+.3f}]pp |")
    lines += ["", "建议暂保留原配置作为主线。若继续复核，优先复验B的greedy微弱信号，在其他轮次或训练seed上检查最新留出与旧代损失的取舍；不要因为Train KL下降就切换池策略。当前不启动下一阶段。",
        "", "单训练seed、固定历史标签和反复使用的DEV面板；区间未校正多重比较。所有训练更新触发统一clip1，损失权重相同不等于两头实际梯度影响相同。本次证据不足以确定长期最优池大小、最优更新步数或带搜索棋力。",
        "", "实现与恢复验证：17项相关测试通过，真实1000步引用账本确认未改变的6k头逐步匹配原A、B/D policy一致、C/D value一致，三组greedy均完成400个有效对局且无异常／非法动作／不变量失败／工程截断。完整曲线与逐代阶段数据见[结果](results.md)、[summary.json](summary.json)、[矩阵](generation_stage_matrix.csv)。", ""]
    (ART / "conclusions.md").write_text("\n".join(lines), encoding="utf-8")
    print(result["comparisons"], flush=True)


if __name__ == "__main__":
    main()
