"""Early 1000-pair greedy panel: 256x4 step-34,021 (50%) vs original R43.

Uses the same opening-seed domain as the training driver's own greedy panel
(36600000+, 1000 pairs), so the early node and the final selected node can be
compared pairwise by seed later. CPU-only greedy; safe to run beside training.

Usage (from the repository root):

    python -u artifacts/entity_attention_256x4_aux_r24_r43_early34021_20261014/run.py run
    python -u .../run.py status
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
DRIVER = ROOT / "artifacts/entity_attention_256x4_aux_r24_r43_20261014/run.py"
CANDIDATE = ROOT / "runs/entity_attention_256x4_aux_r24_r43_20261014/checkpoints/step-000034021.pt"
OPPONENT = ROOT / "runs/entity_attention_r9_online_lr2e5_1000_forever_20261011/rounds/round-033/checkpoints/final.pt"
PANEL = {"id": "greedy_early34021_vs_R43", "kind": "greedy", "pairs": 1000, "seed_start": 36600000}
CONFIG = ART / "evaluation_config.json"
OUTPUT = ART / "evaluation.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def prepare():
    driver = load_module("driver_256x4", DRIVER)
    plan = json.loads((ROOT / "artifacts/entity_attention_256x4_aux_r24_r43_20261014/plan.json").read_text())
    selection = {"checkpoint": str(CANDIDATE), "checkpoint_sha256": sha256(CANDIDATE),
                 "selected_step": 34021}
    config = driver.evaluation_config(plan, selection, PANEL)
    record = {"protocol": "early-greedy-panel-v1", "client_date": "2026-10-14",
              "note": "mid-training node (50%); seeds shared with the driver's future greedy panel",
              "candidate": {"path": str(CANDIDATE.relative_to(ROOT)), "sha256": selection["checkpoint_sha256"]},
              "opponent": plan["opponent"], "panel": PANEL}
    for path, payload in ((ART / "plan.json", record), (CONFIG, config)):
        if path.exists() and json.loads(path.read_text()) != payload:
            raise ValueError(f"frozen file changed: {path}")
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return plan


def run():
    prepare()
    command = [sys.executable, "-u", "-m", "splendor.evaluation", "--config", str(CONFIG),
               "--output", str(OUTPUT), "--progress-every", "50", "--summary-every", "50"]
    if OUTPUT.exists():
        command.append("--resume")
    subprocess.run(command, cwd=ROOT, check=True)
    report()


def report():
    data = json.loads(OUTPUT.read_text())
    ci = data["paired_bootstrap"]
    lines = ["# 256×4 第 34,021 步（50%）对原 R43：1000pair greedy", "",
             f"状态：{data['completed_pairs']}/{data['total_pairs']} pair；passed={data['passed']}",
             "",
             f"- 候选：`{CANDIDATE.relative_to(ROOT)}`（sha256 `{sha256(CANDIDATE)}`）",
             f"- 对手：`{OPPONENT.relative_to(ROOT)}`",
             f"- 开局种子 36600000–36600999（与训练驱动的正式 greedy 面板同域，可逐 pair 比较）；双方 greedy、无搜索、交换先后手、和棋半分",
             "",
             f"**得分：{100*data['win_rate_score']:.3f}%（{data['a_wins']}胜/{data['draws']}和/{data['a_losses']}负），95% 配对区间 "
             f"[{100*ci['lower']:.3f}%, {100*ci['upper']:.3f}%]**",
             "",
             "| 项 | 值 |", "| --- | --- |",
             f"| 有效局 | {data['valid_games']} |",
             f"| 异常/非法动作/工程截断 | {data['exceptions']}/{data['illegal_actions']}/{data['engineering_truncations']} |",
             f"| 用时（本次调用） | {data['elapsed_seconds_this_invocation']:.1f} 秒 |",
             "",
             "本面板只回答“50% 节点的裸网是否已超过 R43”，不覆盖搜索棋力与训练终点；结果受单训练 seed 与固定 DEV 开局限制。",
             ""]
    (ART / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"score": data["win_rate_score"], "ci": ci,
                      "pairs": f"{data['completed_pairs']}/{data['total_pairs']}"}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "status"), nargs="?", default="run")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
        print("prepared")
    elif args.action == "run":
        run()
    else:
        report()


if __name__ == "__main__":
    main()
