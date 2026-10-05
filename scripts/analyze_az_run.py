"""Aggregate AZ records without launching games, training or evaluation."""
import argparse
from pathlib import Path
from splendor.alphazero_analysis import analyze_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_run(args.input, args.output)
    print(f"{result['formal_experiment_state']}: {args.output / 'summary.json'}")


if __name__ == "__main__":
    main()
