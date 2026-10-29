"""Read existing evaluations only; preserve the interrupted R12 panel as partial."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from splendor.mcts_policy import MCTSConfig
from splendor.paired_statistics import (
    outcome_score, paired_bootstrap_difference, paired_bootstrap_mean,
)

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025"
BASE = ROOT / "artifacts/entity_attention_128x2_opponent_policy_r21_r37_20261024/panels"
BOOT = dict(resamples=10000, seed=20261025, batch_size=256)
sources = {}


def load(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def scores(panel):
    result = {}
    for pair in panel["pairs"]:
        games = sorted(pair["games"], key=lambda game: game["game_in_pair"])
        for game in games:
            assert game["status"] in ("natural_terminal", "policy_cycle_draw")
            assert game["game_seed"] == pair["game_seed"]
        if len(games) == 2:
            assert [game["game_in_pair"] for game in games] == [0, 1]
            result[pair["game_seed"]] = sum(outcome_score(g["outcome_for_a"]) for g in games) / 2
    assert len(result) == panel["completed_pairs"]
    return result


def check_protocol(left, right):
    assert left["protocol"] == right["protocol"]
    assert left["catalog_fingerprint"] == right["catalog_fingerprint"]
    assert left["checkpoint_identity"]["policy_b"]["sha256"] == right["checkpoint_identity"]["policy_b"]["sha256"]
    for key in ("game_seeds", "policy_seed_base", "max_completed_turns", "adjudicate_deterministic_policy_cycles"):
        assert left["config"].get(key) == right["config"].get(key), key
    for side in ("policy_a", "policy_b"):
        a, b = left["config"][side], right["config"][side]
        assert a["kind"] == b["kind"]
        assert asdict(MCTSConfig(**(a.get("search_config") or {}))) == asdict(MCTSConfig(**(b.get("search_config") or {})))


def difference(left, right):
    check_protocol(left, right)
    a, b = scores(left), scores(right)
    common = sorted(a.keys() & b.keys())
    ci = paired_bootstrap_difference(common, [a[s] for s in common], common, [b[s] for s in common], **BOOT)
    return {**ci.to_dict(), "unit": "score fraction; left minus right",
            "partial_completed_pair_selection": not (left["passed"] and right["passed"])}


def brief(panel):
    paired = scores(panel)
    ci = paired_bootstrap_mean(list(paired.values()), **BOOT)
    return {key: panel[key] for key in ("win_rate_score", "completed_pairs", "completed_games", "passed")} | {
        "completed_pairs_only": ci.to_dict(), "updated_at": panel["updated_at"]}


def main():
    greedy = {0: load(BASE / "greedy_vs_R37.json")}
    search = {0: load(BASE / "search_S1024N1_vs_R37.json")}
    for r in range(1, 13):
        stage = RUN / f"rounds/round-{r-1:03d}"
        greedy[r] = load(stage / "greedy.json")
        if r % 3 == 0:
            search[r] = load(stage / "search.json")
    for panels in (greedy, search):
        for p in panels.values():
            check_protocol(panels[0], p)
    current = search[12]
    known_score = sum(outcome_score(game["outcome_for_a"]) for pair in current["pairs"] for game in pair["games"])
    target = current["total_games"]
    missing = target - current["completed_games"]
    assert np.isclose(known_score/current["completed_games"], current["win_rate_score"])
    latest = {
        "known_score_points": known_score, "missing_games": missing,
        "final_score_bounds": [known_score/target, (known_score+missing)/target],
        "reason": "Every missing game contributes between zero and one point; no missing-at-random assumption.",
    }
    holdout = []
    for r in range(7, 13):
        folder = RUN / f"fixed_holdout/blocks/iterations-007-012/rounds/round-{r-1:03d}/fixed_holdout"
        row = {"round": r, "steps": {}}
        for step in (0, 1000, 2000, 3000):
            doc = load(folder / f"step-{step:09d}.json")
            row["steps"][step] = {k: doc["metrics"][k] for k in ("policy_kl", "value_mse")}
        holdout.append(row)
    history = []
    for name in ("entity_attention_37423_random_budget_p6k_v24k_20261019",
                 "entity_attention_r20_s256n4_noble12_iteration_20261022",
                 "entity_attention_r31_10k_p60k_iteration_20261023"):
        state = load(ROOT / "runs" / name / "state.json")
        history.append({"run": name, "rows": [{"iteration": row["iteration"],
            **{k: {"score": v["win_rate_score"], "pairs": v["completed_pairs"]}
               for k, v in row["evaluations"].items() if k in ("greedy", "search")}}
            for row in state["completed_rounds"]]})
    result = {
        "scope": "Read-only analysis; R12 search remains interrupted and is not promoted to passed.",
        "greedy": {r: brief(p) for r, p in greedy.items()},
        "search": {r: brief(p) for r, p in search.items()},
        "greedy_differences": {f"R{a}-R{b}": difference(greedy[a], greedy[b]) for a, b in ((12, 0), (12, 6), (12, 8), (12, 10))},
        "search_differences": {f"R{a}-R{b}": difference(search[a], search[b]) for a, b in ((9, 0), (12, 0), (12, 6), (12, 9))},
        "r12_search_bounds": latest, "fixed_holdout_local7_12": holdout,
        "historical_branches_against_R43": history,
        "limits": ["One training trajectory; no independent training seeds or no-policy-redesign control.",
                   "Completed-pair R12 intervals are conditional on the 475 pairs that finished; long games may be censored.",
                   "Checkpoint comparisons are exploratory and share the same opponent and opening panel.",
                   "Fixed holdout cohorts differ across local6/7; absolute values are not a common-panel comparison."],
        "source_sha256": sources,
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    for key in ("greedy_differences", "search_differences", "r12_search_bounds"):
        print(key, json.dumps(result[key], ensure_ascii=False))


if __name__ == "__main__":
    main()
