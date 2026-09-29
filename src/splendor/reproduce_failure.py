"""Replay compact real actions and an optional sampled search branch in checked mode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .compact_trace import decode_request, replay_trace
from .contracts import DecisionRequest
from .replay import _decode_action
from .result import TerminalResult
from .search_policy import determinize_request


def reproduce(scene: Mapping[str, Any]) -> dict[str, Any]:
    trace = scene.get("trace")
    current = replay_trace(trace, require_terminal=False) if isinstance(trace, Mapping) else None
    root_raw = scene.get("root_request")
    if root_raw is not None:
        root = decode_request(root_raw)
        if current is not None and current != root:
            raise ValueError(f"real trace diverges before root decision {scene.get('root_decision_index')}")
    elif isinstance(current, DecisionRequest):
        root = current
    else:
        root = None
    result: dict[str, Any] = {
        "real_actions_replayed": len(trace["actions"]) if isinstance(trace, Mapping) else 0,
        "root_phase": None if root is None else root.observation.phase.value,
        "original_error_type": scene.get("error_type"),
    }
    search = scene.get("search_context")
    if isinstance(search, Mapping) and search.get("determinization_seed") is not None:
        request = decode_request(search["root_request"])
        if root is not None and request != root:
            raise ValueError("search root disagrees with real trace")
        engine = determinize_request(request, seed=int(search["determinization_seed"]),
                                     engine_mode="checked")
        branch = engine.decision()
        for index, raw in enumerate(search.get("branch_actions", ())):
            if not isinstance(branch, DecisionRequest):
                raise ValueError(f"search branch action {index}: sampled world is terminal")
            action = _decode_action(raw)
            if action not in branch.legal_actions:
                raise ValueError(f"search branch action {index}: action is illegal")
            branch = engine.apply(action)
        for batched in scene.get("batch_requests", ()):
            if batched.get("request_id") == search.get("request_id") and batched.get("request") is not None:
                expected = decode_request(batched["request"])
                if branch != expected:
                    raise ValueError("search branch diverges before pending evaluator request")
                break
        result.update({
            "world_index": search.get("world_index"),
            "simulation_index": search.get("simulation_index"),
            "branch_actions_replayed": len(search.get("branch_actions", ())),
            "branch_phase": branch.observation.phase.value if isinstance(branch, DecisionRequest) else "terminal",
            "reconstructed_to": "search_branch_after_successful_actions",
        })
    else:
        result["reconstructed_to"] = "real_request_or_terminal"
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    scene = json.loads(args.input.read_text(encoding="utf-8"))
    print(json.dumps(reproduce(scene), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
