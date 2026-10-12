"""Paired whole-game uncertainty for the three early heldout checkpoints."""
import argparse
from pathlib import Path

from analyze_fit import paired_change
from splendor.alphazero_recording import read_json, sha256, write_json

ART = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extended", action="store_true", help="include the exposure5-to8 fitting extension")
    args = parser.parse_args()
    selection_path = ART / ("extended_selection.json" if args.extended else "early_selection.json")
    selection = read_json(selection_path)
    full = {row["step"]: read_json(ART / Path(row["result"]).name) for row in selection["rows"]}
    result = {"protocol": "recent-pool-early-fit-review-v1",
        "selection_sha256": sha256(selection_path),
        "statistics_implementation_sha256": sha256(ART / "analyze_fit.py"),
        "source_hashes": {str(step): sha256(ART / f"full_holdout_step_{step:09d}.json") for step in full},
        "paired_changes": {},
        "scope": "conditional on these 2935 heldout games, one training seed, adaptive DEV checkpoint selection; exploratory intervals without multiplicity correction"}
    steps = sorted(full)
    for i, (a, b) in enumerate(zip(steps, steps[1:])):
        before, after = [full[step]["evaluation"]["per_game"] for step in (a, b)]
        result["paired_changes"][f"{b}_minus_{a}"] = {
            "policy_kl": paired_change(before, after, numerator="kl", denominator="policy_weight", seed=30400000+2*i),
            "value_mse": paired_change(before, after, numerator="sq_error", denominator="positions", seed=30400001+2*i)}
    write_json(ART / ("extended_fit_review.json" if args.extended else "early_fit_review.json"), result)
    print(__import__("json").dumps(result["paired_changes"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
