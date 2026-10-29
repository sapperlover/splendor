"""Verify the completed local12/R20 arena and retain compact paired scores."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from splendor.catalog import load_default_catalog
from splendor.evaluation import FixedEvaluationConfig
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores


ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]


def read(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    plan = read(ART / "execution_plan.json")
    raw = ROOT / plan["output"]
    result = read(raw)
    config = FixedEvaluationConfig.from_dict(read(ART / "config.json"))
    assert digest(ART / "config.json") == plan["config_sha256"]
    assert result["passed"] and result["completed_pairs"] == 1000
    assert result["valid_games"] == result["completed_games"] == 2000
    assert result["config"] == config.to_dict()
    catalog = load_default_catalog()
    assert len(catalog.nobles) == 12
    assert result["catalog_fingerprint"] == plan["catalog_fingerprint"] == catalog.data_fingerprint
    logs = Path(str(raw) + ".games")
    header = read(logs / "header.json")
    assert header["config"] == config.to_dict()
    assert header["catalog_fingerprint"] == catalog.data_fingerprint
    for key, identity in plan["checkpoint_identity"].items():
        assert digest(Path(identity["path"])) == identity["sha256"]
        assert header["checkpoint_identity"][key]["sha256"] == identity["sha256"]
    assert len(list(logs.glob("pair-*.jsonl"))) == 1000
    rows = []
    outcomes = Counter()
    score = {"a_win": 1.0, "draw": 0.5, "a_loss": 0.0}
    for pair_id, pair in enumerate(result["pairs"]):
        games = [json.loads(line) for line in (logs / f"pair-{pair_id:07d}.jsonl").read_text().splitlines()]
        games.sort(key=lambda g: g["game_in_pair"])
        assert games == pair["games"]
        assert pair["pair_id"] == pair_id and pair["game_seed"] == config.game_seeds[pair_id]
        assert [g["first_player"] for g in games] == [0, 1]
        for index, game in enumerate(games):
            assert game["pair_id"] == pair_id and game["game_in_pair"] == index
            assert game["game_seed"] == pair["game_seed"]
            assert game["policy_names"] == [config.policy_a.name, config.policy_b.name]
            assert game["policy_seeds"] == [config.policy_seed_base + 2 * pair_id, config.policy_seed_base + 2 * pair_id + 1]
            assert game["status"] in ("natural_terminal", "policy_cycle_draw")
            assert not game["error_type"] and not game["forced_skips"]
            outcomes[game["outcome_for_a"]] += 1
        rows.append([pair["game_seed"], *[score[g["outcome_for_a"]] for g in games]])
    assert [outcomes[k] for k in ("a_win", "draw", "a_loss")] == [result[k] for k in ("a_wins", "draws", "a_losses")]
    seeds, scores = serialized_pair_scores(result)
    assert seeds == config.game_seeds
    bootstrap = paired_bootstrap_mean(scores, resamples=config.bootstrap_resamples, seed=config.bootstrap_seed).to_dict()
    assert bootstrap == result["paired_bootstrap"]
    assert float(scores.mean()) == result["win_rate_score"]
    for key in ("engineering_truncations", "exceptions", "illegal_actions", "invariant_failures", "forced_skips"):
        assert result[key] == 0
    recording = raw.parent / "recording/evaluation/recording"
    identity_paths = list((recording / "identities").glob("*.json"))
    assert len(identity_paths) == 1
    identity = read(identity_paths[0])
    assert identity["config"] == config.to_dict()
    for path, expected in identity["source_files"].items():
        assert digest(ROOT / path) == expected, path
    sessions = [read(p) for p in (recording / "sessions").glob("*.json")]
    assert len(sessions) == 1 and sessions[0]["status"] == "complete"
    wall = read(recording.parent / "wall_summary.json")
    assert wall["completed_pairs"] == 1000
    compact = {k: v for k, v in result.items() if k != "pairs"}
    compact.update(status="complete", checkpoint_identity=plan["checkpoint_identity"],
                   config_sha256=plan["config_sha256"], result_sha256=digest(raw),
                   recorded_wall_seconds=wall["recorded_wall_seconds"],
                   source_identity={k: identity[k] for k in ("git_head", "source_sha256", "native_extensions")})
    write(ART / "summary.json", compact)
    write(ART / "pair_scores.json", {"columns": ["game_seed", "a_first_score", "a_second_score"], "rows": rows})
    write(ART / "verification.json", {
        "passed": True, "pairs_verified": 1000, "games_verified": 2000,
        "checks": ["checkpoint SHA unchanged", "config/catalog/source identity unchanged", "all pair sidecars exactly match result", "opening/policy seeds and swapped seats", "all games scored without engineering errors", "outcome totals and paired bootstrap recomputed", "single complete recorded invocation"],
        "raw_result": str(raw.relative_to(ROOT)), "raw_result_sha256": digest(raw),
        "recording_identity": str(identity_paths[0].relative_to(ROOT)),
        "recording_identity_sha256": digest(identity_paths[0]),
        "summary_sha256": digest(ART / "summary.json"),
        "pair_scores_sha256": digest(ART / "pair_scores.json"),
    })
    print(json.dumps({k: compact[k] for k in ("a_wins", "draws", "a_losses", "win_rate_score", "paired_bootstrap", "a_first", "a_second", "policy_cycle_draws", "recorded_wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
