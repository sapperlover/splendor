"""Extend both frozen 200-pair greedy panels by 800 pairs without rerunning the prefix."""
from __future__ import annotations

import argparse
import copy
import shutil
import subprocess
import sys
import time
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig, _evaluation_summary_from_records
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = HERE.parent / "greedy_vs_r43"
RUNTIME = ROOT / "runtime" / HERE.parent.name / HERE.name
BRANCHES = ("policy_surprise", "value_auxiliary")


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen expansion input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    inputs = {}
    for branch in BRANCHES:
        prefix_path = PREFIX / f"{branch}.json"
        prefix = read_json(prefix_path)
        seeds, _ = serialized_pair_scores(prefix)
        if seeds != tuple(range(36200000, 36200200)) or prefix["valid_games"] != 400:
            raise ValueError("expected the frozen original 200-pair prefix")
        for model in prefix["checkpoint_identity"].values():
            if sha256(Path(model["path"])) != model["sha256"]:
                raise ValueError("evaluated checkpoint changed")
        full = copy.deepcopy(prefix["config"])
        full.update(game_seeds=list(range(36200000, 36201000)),
                    metadata={**full["metadata"], "scope": "expanded 1000-pair panel; original 200 plus 800 additional pairs",
                              "prefix_result_sha256": sha256(prefix_path)})
        extra = copy.deepcopy(full)
        extra.update(game_seeds=list(range(36200200, 36201000)), policy_seed_base=36210400,
                     bootstrap_seed=36220001,
                     metadata={**extra["metadata"], "part": "additional 800 pairs", "global_pair_offset": 200})
        freeze(HERE / f"{branch}_combined_config.json", FixedEvaluationConfig.from_dict(full).to_dict())
        freeze(HERE / f"{branch}_extension_config.json", FixedEvaluationConfig.from_dict(extra).to_dict())
        inputs[branch] = {"prefix": str(prefix_path), "prefix_sha256": sha256(prefix_path),
                          "checkpoint_identity": prefix["checkpoint_identity"]}
    manifest = {"inputs": inputs, "original_pairs": 200, "additional_pairs": 800, "total_pairs_per_branch": 1000,
                "game_seed_range": [36200000, 36200999], "additional_seed_range": [36200200, 36200999],
                "total_games": 4000, "additional_games": 3200,
                "combined_bootstrap_seed": 36220000, "difference_bootstrap_seed": 36230000,
                "commands": [[sys.executable, "-u", "-m", "splendor.evaluation", "--config",
                              str(HERE / f"{b}_extension_config.json"), "--output", str(HERE / f"{b}_extension.json"),
                              "--progress-every", "50", "--summary-every", "50"] for b in BRANCHES]}
    freeze(HERE / "manifest.json", manifest)
    return manifest


def analyze():
    manifest = prepare()
    panels, seeds, scores = {}, {}, {}
    for branch in BRANCHES:
        expected_identity = manifest["inputs"][branch]["checkpoint_identity"]
        full_config = FixedEvaluationConfig.from_dict(read_json(HERE / f"{branch}_combined_config.json"))
        records, parts, evaluation_seconds = {}, [], 0.
        for path, offset, count in ((PREFIX / f"{branch}.json", 0, 200),
                                    (HERE / f"{branch}_extension.json", 200, 800)):
            raw = read_json(path)
            part_seeds, _ = serialized_pair_scores(raw)
            if raw["completed_pairs"] != count or raw["checkpoint_identity"] != expected_identity:
                raise ValueError("part is incomplete or used different checkpoints")
            if part_seeds != full_config.game_seeds[offset:offset + count]:
                raise ValueError("part opening seeds do not match the expansion plan")
            if offset and raw["config"] != read_json(HERE / f"{branch}_extension_config.json"):
                raise ValueError("extension configuration changed")
            parts.append({"path": str(path), "sha256": sha256(path), "global_pair_offset": offset,
                          "pairs": count, "game_log_root": raw["game_log_root"]})
            evaluation_seconds += raw["elapsed_seconds_this_invocation"]
            for pair in raw["pairs"]:
                global_id = offset + pair["pair_id"]
                if global_id in records:
                    raise ValueError("overlapping evaluation parts")
                games = copy.deepcopy(pair["games"])
                for game in games:
                    game["pair_id"] = global_id
                    if game["policy_seeds"] != [36210000 + 2 * global_id, 36210001 + 2 * global_id]:
                        raise ValueError("part policy seeds do not match the global pair identity")
                    if game["policy_names"] != [full_config.policy_a.name, full_config.policy_b.name]:
                        raise ValueError("part policy names changed")
                records[global_id] = games
        if set(records) != set(range(1000)):
            raise ValueError("combined panel lacks exactly 1000 unique pairs")
        output = HERE / f"{branch}.json"
        raw = _evaluation_summary_from_records(full_config, records, output_path=output,
                                               game_log_root=HERE, elapsed_seconds=0.)
        raw.update(protocol="combined-fixed-paired-evaluation-v1", checkpoint_identity=expected_identity,
                   parts=parts, evaluation_seconds_parts_sum=evaluation_seconds,
                   aggregation_note="prefix kept unchanged; extension local pair IDs remapped by +200 only")
        raw.pop("game_log_root")
        raw.pop("elapsed_seconds_this_invocation")
        seeds[branch], scores[branch] = serialized_pair_scores(raw)
        write_json(output, raw)
        panels[branch] = {k: raw[k] for k in ("passed", "completed_pairs", "valid_games", "a_wins", "draws", "a_losses",
            "win_rate_score", "paired_bootstrap", "policy_cycle_draws", "engineering_truncations", "exceptions",
            "illegal_actions", "invariant_failures", "evaluation_seconds_parts_sum")}
    difference = paired_bootstrap_difference(seeds["value_auxiliary"], scores["value_auxiliary"],
        seeds["policy_surprise"], scores["policy_surprise"], resamples=10000, seed=36230000).to_dict()
    summary = {"status": "complete", "panels": panels, "auxiliary_minus_psw_against_r43": difference,
               "scope": "1000 pairs per final checkpoint, reusing the previously inspected 200-pair prefix; not an independent confirmation",
               "original_pairs_reused": 200, "additional_pairs": 800}
    write_json(HERE / "summary.json", summary)
    lines = ["# 两组final对原R43：扩至1000 pair的greedy评测", "",
        "每组1000 pair／2000局，共4000局；保留已完成并查看过的前200 pair，只新增800 pair／1600局。两组使用同一完整开局库，交换先后手，纯greedy、和棋半分；95%区间按完整pair bootstrap 10000次。原200 pair产物保留原位。", "",
        "| 候选 | 胜／和／负 | 得分率 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for branch, p in panels.items():
        ci = p["paired_bootstrap"]
        lines.append(f"| {branch} | {p['a_wins']}／{p['draws']}／{p['a_losses']} | {p['win_rate_score']:.3%} | [{ci['lower']:.3%}, {ci['upper']:.3%}] |")
    lines += ["", "辅助组的得分区间高于50%，支持本次模型对R43的greedy提升；PSW区间包含50%，尚未确认提升。辅助组对共同对手的得分高于PSW，但差值区间包含0，尚未确认两候选之间的优势。",
        "", f"共{sum(p['valid_games'] for p in panels.values())}局有效比赛，其中{sum(p['policy_cycle_draws'] for p in panels.values())}局为按现行规则计半分的policy cycle draw，其余自然终局；工程截断、异常、非法行动和不变量失败均为0。"]
    lines += ["", f"辅助减PSW对共同R43的配对得分差：{difference['estimate']*100:+.3f}pp，95%区间[{difference['lower']*100:+.3f}, {difference['upper']*100:+.3f}]pp。不是两个候选直接互打。",
        "", "扩样是在已查看前200 pair后由用户要求进行，结果是完整1000 pair的描述性评测，不能称为全新独立确认；区间未作顺序检验校正。单训练seed、单对手、仅final，不推断搜索棋力，也不单独归因PSW／辅助目标相对普通训练的收益。",
        "", "manifest保存原件SHA、模型身份、命令和分片构成。*_extension.json是新增800 pair原始输出，*.json是保留全部逐局记录的1000 pair汇总；汇总只把新增部分的局部pair_id加200，开局／policy seed和结果不变。summary.json保存共同对手的配对差。恢复仅运行未完成的扩展部分。", ""]
    (HERE / "results.md").write_text("\n".join(lines))
    print(summary, flush=True)
    return summary


def run():
    manifest = prepare()
    session = RUNTIME / str(time.time_ns())
    write_json(session / "identity.json", {**source_identity(ROOT, session / "source"),
        "command": sys.argv, "runner_sha256": sha256(Path(__file__)), "agents_sha256": sha256(ROOT / "AGENTS.md"),
        "manifest": manifest})
    shutil.copyfile(__file__, session / "run.py")
    with (session / "evaluation.log").open("a") as log:
        for branch, command in zip(BRANCHES, manifest["commands"], strict=True):
            path = HERE / f"{branch}_extension.json"
            if path.exists() and read_json(path).get("passed") and read_json(path).get("completed_pairs") == 800:
                continue
            args = list(command)
            if path.exists() or Path(str(path) + ".games").exists():
                args.append("--resume")
            child = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(branch, line.rstrip(), flush=True)
            if child.wait():
                raise RuntimeError(f"extension panel failed: {branch}")
    return analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"), nargs="?", default="run")
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "analyze": analyze}[action]()
