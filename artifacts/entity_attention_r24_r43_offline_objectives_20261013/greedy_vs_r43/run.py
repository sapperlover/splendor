"""Two preselected final checkpoints, paired greedy against original R43."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256, source_identity, write_json
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import paired_bootstrap_difference, serialized_pair_scores

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNTIME = ROOT / "runtime/entity_attention_r24_r43_offline_objectives_20261013/greedy_vs_r43"
BRANCHES = ("policy_surprise", "value_auxiliary")
OPPONENT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"


def freeze(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"frozen evaluation input changed: {path}")
    else:
        write_json(path, value)


def prepare():
    opponent_sha = sha256(OPPONENT)
    if opponent_sha != "b11f12c2ef1d48d86c8970eaae795f5139ae4c282476a52585a3fa5dff784f14":
        raise ValueError("original R43 checkpoint changed")
    models = {}
    for branch in BRANCHES:
        root = ROOT / f"runs/entity_attention_r24_r43_{branch}_offline_20261013"
        state = read_json(root / "progress.json")
        if state["status"] != "complete" or state["step"] != 22681:
            raise ValueError(f"offline training incomplete: {branch}")
        checkpoint = root / "checkpoints/final.pt"
        models[branch] = {"path": str(checkpoint), "sha256": sha256(checkpoint),
                          "training_config_sha256": sha256(root / "config.json")}
        config = FixedEvaluationConfig.from_dict({
            "policy_a": {"kind": "greedy_checkpoint", "name": branch, "checkpoint": str(checkpoint)},
            "policy_b": {"kind": "greedy_checkpoint", "name": "original_R43", "checkpoint": str(OPPONENT)},
            "game_seeds": list(range(36200000, 36200200)), "policy_seed_base": 36210000,
            "bootstrap_seed": 36220000, "bootstrap_resamples": 10000,
            "workers": 8, "device": "cpu", "max_completed_turns": 400,
            "adjudicate_deterministic_policy_cycles": True,
            "metadata": {"experiment": HERE.parent.name, "panel": "final_checkpoint_greedy_vs_original_R43",
                         "candidate_sha256": models[branch]["sha256"], "opponent_sha256": opponent_sha,
                         "selection": "final 22681-update checkpoint requested before evaluation",
                         "scope": "shared fresh 200-pair panel; no search, no checkpoint selection"}})
        freeze(HERE / f"{branch}_config.json", config.to_dict())
    manifest = {"models": models, "opponent": {"path": str(OPPONENT), "sha256": opponent_sha},
                "game_seed_range": [36200000, 36200199], "pairs_per_panel": 200,
                "total_games": 800, "policy_seed_base": 36210000,
                "bootstrap_seed": 36220000, "difference_bootstrap_seed": 36230000,
                "commands": [[sys.executable, "-u", "-m", "splendor.evaluation", "--config",
                              str(HERE / f"{b}_config.json"), "--output", str(HERE / f"{b}.json"),
                              "--progress-every", "10", "--summary-every", "10"] for b in BRANCHES]}
    freeze(HERE / "manifest.json", manifest)
    return manifest


def analyze():
    manifest = read_json(HERE / "manifest.json")
    panels, score_arrays, seeds = {}, {}, {}
    for branch in BRANCHES:
        raw = read_json(HERE / f"{branch}.json")
        config = read_json(HERE / f"{branch}_config.json")
        if raw["config"] != config or raw["completed_pairs"] != 200 or raw["valid_games"] != 400:
            raise ValueError(f"evaluation incomplete or changed: {branch}")
        expected = {"policy_a": {k: manifest["models"][branch][k] for k in ("path", "sha256")},
                    "policy_b": manifest["opponent"]}
        if raw["checkpoint_identity"] != expected:
            raise ValueError("evaluated checkpoint identity differs from frozen plan")
        seeds[branch], score_arrays[branch] = serialized_pair_scores(raw)
        panels[branch] = {k: raw[k] for k in ("passed", "completed_pairs", "valid_games", "a_wins", "draws", "a_losses",
            "win_rate_score", "paired_bootstrap", "policy_cycle_draws", "engineering_truncations",
            "exceptions", "illegal_actions", "invariant_failures", "elapsed_seconds_this_invocation")}
    diff = paired_bootstrap_difference(seeds["value_auxiliary"], score_arrays["value_auxiliary"],
                                      seeds["policy_surprise"], score_arrays["policy_surprise"],
                                      resamples=10000, seed=36230000).to_dict()
    result = {"status": "complete", "panels": panels, "auxiliary_minus_psw_against_r43": diff,
              "scope": "one final checkpoint per branch, one shared fresh 200-pair panel; difference is not a direct head-to-head"}
    write_json(HERE / "summary.json", result)
    lines = ["# 两组离线最终模型对原 R43：greedy 棋力", "",
             "两组均固定使用追加22681步的final，原R43为共同对手；相同200个新开局、交换先后手，每组400局。CPU八worker，纯greedy、不使用value或搜索，和棋半分；95%区间按完整pair重采样10000次。", "",
             "| 候选 | 胜／和／负 | 得分率 | 95%配对区间 |", "| --- | ---: | ---: | --- |"]
    for branch, p in panels.items():
        ci = p["paired_bootstrap"]
        lines.append(f"| {branch} | {p['a_wins']}／{p['draws']}／{p['a_losses']} | {p['win_rate_score']:.3%} | [{ci['lower']:.3%}, {ci['upper']:.3%}] |")
    lines += ["", f"辅助组减PSW对共同R43的配对得分差：{diff['estimate']*100:+.3f}个百分点，95%区间[{diff['lower']*100:+.3f}, {diff['upper']*100:+.3f}]。不是两候选直接对打。",
              "", "逐pair结果、配置、模型SHA及命令保存在本目录；恢复sidecar为相邻被忽略的*.json.games/，源码会话在runtime。范围限定本对手、新开局库和这两个最终模型，不外推搜索棋力或单独归因辅助目标／PSW的收益。", ""]
    (HERE / "results.md").write_text("\n".join(lines))
    print(result, flush=True)
    return result


def run():
    manifest = prepare()
    session = RUNTIME / str(time.time_ns())
    write_json(session / "identity.json", {**source_identity(ROOT, session / "source"),
        "runner_sha256": sha256(Path(__file__)), "command": sys.argv, "manifest": manifest})
    with (session / "evaluation.log").open("a") as log:
        for branch, command in zip(BRANCHES, manifest["commands"], strict=True):
            result_path = HERE / f"{branch}.json"
            if result_path.exists() and read_json(result_path).get("passed") and read_json(result_path).get("completed_pairs") == 200:
                continue
            args = list(command)
            if result_path.exists() or Path(str(result_path) + ".games").exists():
                args.append("--resume")
            child = subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(branch, line.rstrip(), flush=True)
            if child.wait():
                raise RuntimeError(f"greedy panel failed: {branch}")
    return analyze()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"), nargs="?", default="run")
    action = parser.parse_args().action
    {"prepare": prepare, "run": run, "analyze": analyze}[action]()
