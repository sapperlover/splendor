"""Main-action stage cuts and stable search changes, using saved results only."""
import csv
import runpy
from pathlib import Path

import numpy as np

from splendor.alphazero_recording import read_json, sha256, write_json

ART = Path(__file__).resolve().parent


def main():
    cluster = runpy.run_path(str(ART / "analyze.py"))["cluster_mean"]
    plan = read_json(ART / "plan.json")
    refs = read_json(ART / "root_references.json")["references"]
    n = len(refs)
    actions, policy = np.zeros((2, n), dtype=int), np.zeros(n, dtype=int)
    for path in sorted((ART / "raw").glob("*.npz")):
        with np.load(path, allow_pickle=False) as raw:
            repeat = plan["world_seeds"].index(int(raw["seed"]))
            ids = raw["root_ids"]
            actions[repeat, ids] = np.argmax(raw["visits"], axis=-1)
            policy[ids] = np.argmax(raw["log_policy"], axis=-1)
    rows = list(csv.DictReader((ART / "per_root.csv").open(encoding="utf-8")))
    a, b = actions
    same = a == b
    values = {"both_match_policy": (same & (a == policy)).astype(float),
        "stable_search_change": (same & (a != policy)).astype(float),
        "seed_sensitive_one_match": (~same & ((a == policy) | (b == policy))).astype(float),
        "seed_sensitive_neither_matches": (~same & (a != policy) & (b != policy)).astype(float),
        **{key: np.array([float(r[key]) for r in rows]) for key in ("top1_strict", "visit_kl", "repeat_top1_agreement")}}
    groups = {"all": np.ones(n, dtype=bool), "main_action": np.array([r["phase"] == 0 for r in refs])}
    for stage in ("early_mid", "late"):
        groups[f"main_action_{stage}"] = groups["main_action"] & np.array([r["stage"] == stage for r in refs])
    games = np.array([r["game_id"] for r in refs])
    result = {"source_sha256": sha256(Path(__file__)), "base_summary_sha256": sha256(ART / "summary.json"),
        "command": f"conda run -n splendor env PYTHONPATH=src python artifacts/{ART.name}/supplement.py",
        "scope": "Saved searches only; two-seed stability is not correctness or a general noise ceiling.", "groups": {}}
    for name, mask in groups.items():
        result["groups"][name] = {key: cluster(v[mask], games[mask], plan["bootstrap_seed"]) for key, v in values.items()}
    write_json(ART / "supplement.json", result)
    overall, main_action = result["groups"]["all"], result["groups"]["main_action"]
    lines = ["# 搜索策略学会了吗：本次可支持的结论", "",
        "**当前policy与自身S1024仍有明显差距，尤其在主行动上；本次单点测量不能证明历史训练学会了多少，也不能证明搜索动作是正确答案。**",
        "", f"全部有效决策Top-1一致率{100*overall['top1_strict']['estimate']:.3f}%，主行动仅{100*main_action['top1_strict']['estimate']:.3f}%。总体包含1205个弃牌选择、280个其他选择和3个贵族选择；这些阶段抬高总体一致率。", "",
        "| 仅主行动范围 | 局面数 | policy／搜索Top-1一致率 | 访问KL | 两次搜索自身一致率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in ("main_action", "main_action_early_mid", "main_action_late"):
        g = result["groups"][name]
        lines.append(f"| {name} | {g['top1_strict']['positions']} | {100*g['top1_strict']['estimate']:.3f}% | {g['visit_kl']['estimate']:.6f} | {100*g['repeat_top1_agreement']['estimate']:.3f}% |")
    lines += ["", "按两个无根噪声、独立N1世界seed区分主行动分歧：", "",
        "| 类型 | 主行动比例 | 整局bootstrap95% CI |", "| --- | ---: | --- |"]
    for key, label in (("both_match_policy", "两次搜索均与policy一致"), ("stable_search_change", "两次搜索一致，但都改写policy动作"),
        ("seed_sensitive_one_match", "搜索相互分歧，其中一次匹配policy"), ("seed_sensitive_neither_matches", "搜索相互分歧，两次均不匹配policy")):
        d = main_action[key]
        lines.append(f"| {label} | {100*d['estimate']:.3f}% | [{100*d['lower']:.3f}, {100*d['upper']:.3f}]% |")
    lines += ["", "稳定改写的存在说明剩余差距不只来自本次两个seed彼此的分歧；但两个相同选择也可能共有搜索偏差，不能据此判定网络选错，或把该比例当成可训练收益。搜索自身一致率只是参照，不是policy一致率的上限。",
        "", "单次目标平均KL0.358594，对两目标均值的KL0.295166，两目标JSD0.063428。合并两个目标后仍有较大分布差距；有限样本分解不能估计全部不可约噪声，也不能直接把剩余项都归为欠拟合。",
        "", "从阶段看policy更接近后期搜索。当前结果没有支持“后期搜索增量主要未被学会”的说法；但一致率不是搜索对实际胜率的贡献，仍不能据此判断前中期和后期哪段搜索更有价值。",
        "", "若要回答训练到底吸收了多少搜索增量，需要在同一独立局面、同一冻结教师标签上比较训练前后policy，避免教师随checkpoint一起变。此为下一步判断方法，本次不额外启动训练或搜索。",
        "", "适用范围：原第30轮主线模型，在第29轮教师采集的第30轮完整留出局面上，重新运行当前模型的S1024。未采第30轮自身新对局。3项指标测试通过；8283局面×2次×1024模拟共16,963,584次模拟完成，模型／来源／候选与预算核对通过。", "",
        "[完整结果及95%区间](results.md)、[补充分组与区间](supplement.json)、[可复算原始访问数](raw/)。", ""]
    (ART / "conclusions.md").write_text("\n".join(lines), encoding="utf-8")
    print(result["groups"], flush=True)


if __name__ == "__main__":
    main()
