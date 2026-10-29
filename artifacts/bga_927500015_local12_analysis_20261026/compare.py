"""Compare the completed local12 and original R20 matrices, without inference."""
import hashlib
import json
from pathlib import Path

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
OLD = ROOT / "artifacts/bga_927500015_r20_analysis_20261020"
LABELS = {
    "greedy": "Greedy",
    "native_S1024_N1": "确定化 S1024N1（总1024）",
    "native_S1024_N4": "确定化 S256N4（总1024）",
    "chance_S1024_N1": "Chance（总1024）",
    "cpw_S1024_N1": "CPW（总1024）",
    "native_S16384_N1": "确定化 S16384N1（总16384）",
    "native_S16384_N4": "确定化 S4096N4（总16384）",
    "chance_S16384_N1": "Chance（总16384）",
    "cpw_S16384_N1": "CPW（总16384）",
}


def read(path):
    return json.loads(path.read_text())


def main():
    old, new = [read(p / "statistics/summary.json") for p in (OLD, ART)]
    old_plan, new_plan = [read(p / "plan.json") for p in (OLD, ART)]
    assert old["complete"] and new["complete"]
    for key in ("panels", "seeds", "search_common", "cpw", "chunk_roots", "capacity", "pipeline_slots", "inference", "catalog_fingerprint"):
        assert old_plan[key] == new_plan[key], key
    assert old["searches"] == new["searches"] == 1824
    comparisons = {}
    for name, label in LABELS.items():
        a, b = old["panels"][name]["main_action"]["all"], new["panels"][name]["main_action"]["all"]
        comparisons[name] = {"label": label, "old": a, "local12": b,
            "top1_delta_count": b["human_top1"]-a["human_top1"],
            "top3_delta_count": b["human_top3"]-a["human_top3"],
            "human_probability_delta_pp": b["mean_human_probability_pct"]-a["mean_human_probability_pct"],
            "human_rank_delta": b["mean_human_rank"]-a["mean_human_rank"]}
    search_rows = [v for k, v in comparisons.items() if k != "greedy"]
    changes = {"higher": sum(v["top1_delta_count"] > 0 for v in search_rows),
               "lower": sum(v["top1_delta_count"] < 0 for v in search_rows),
               "equal": sum(v["top1_delta_count"] == 0 for v in search_rows)}
    paths = [OLD / "plan.json", OLD / "statistics/summary.json", ART / "plan.json",
             ART / "statistics/summary.json", ART / "baseline_check.json", Path(__file__)]
    result = {"scope": "Same 56 observed human main actions, 8 configurations, 3 matched seeds; descriptive single-game comparison.",
              "search_top1_changes": changes, "panels": comparisons,
              "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (ART / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    lines = ["# 第12轮与旧R20：同一人类棋谱的搜索矩阵比较", "",
             "双方均为56次主行动、20次有选择支付，12次强制决策排除。每组3个相同种子，按平均根访问份额确定共识首选；模型分别冻结为旧混合预算R20与当前本地第12轮final。", "",
             f"8组搜索的主行动首选吻合：提高{changes['higher']}组、下降{changes['lower']}组、持平{changes['equal']}组。", "",
             "| 方法 | 旧R20首选/56 | 第12轮首选/56 | 变化步数 | 旧前三/56 | 新前三/56 | 旧实际平均概率 | 新实际平均概率 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for v in comparisons.values():
        a, b = v["old"], v["local12"]
        lines.append(f"| {v['label']} | {a['human_top1']} | {b['human_top1']} | {v['top1_delta_count']:+d} | {a['human_top3']} | {b['human_top3']} | {a['mean_human_probability_pct']:.2f}% | {b['mean_human_probability_pct']:.2f}% |")
    lines += ["", "Policy概率与搜索访问份额分别按原口径统计；主行动、双方和支付的全部排名、Q覆盖、低访问敏感性和种子变化见[完整统计](statistics/results.md)，逐步候选见[HTML报告](report.html)。", "",
              "同一盘棋的吻合变化不等于棋力变化，8种搜索配置也不是8次独立的人类对局。Q差为各模型自己的搜索意见，不能解释为校准后的真实胜率损失。新旧源码版本不同，16个旧模型根的有限执行栈复核见baseline_check.json；所有算法参数、种子及冻结目录已逐项相等检查。"]
    (ART / "comparison.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"search_top1_changes": changes, "panels": {k: {"old_top1": v['old']['human_top1'], "local12_top1": v['local12']['human_top1'], "probability_delta_pp": v['human_probability_delta_pp']} for k, v in comparisons.items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
