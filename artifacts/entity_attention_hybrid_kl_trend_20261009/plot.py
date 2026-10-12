"""Plot the frozen-holdout KL trend from trend.json (no model loading)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent


def series(checkpoints):
    steps = [row["total_train_step"] for row in checkpoints]
    offline = [row["offline_fixed"]["policy_kl"] for row in checkpoints]
    online = [row["online_fixed"]["policy_kl"] for row in checkpoints]
    return steps, offline, online


def main():
    data = json.loads((ART / "trend.json").read_text(encoding="utf-8"))
    steps, offline, online = series(data["checkpoints"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)

    ax = axes[0]
    ax.plot(steps, offline, "o-", label="offline_fixed (old corpus)")
    ax.plot(steps, online, "s-", label="online_fixed (round-1 holdout)")
    ax.axvline(5000, color="grey", ls=":", lw=1)
    ax.set_title("5000-step start -> 9 self-iteration rounds")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("KL(target || student), weighted")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    fork = data.get("fork_series", {}).get("n1", {}).get("rows")
    if fork:
        fsteps = [row["total_train_step"] for row in fork]
        foffline = [row["offline_fixed"]["policy_kl"] for row in fork]
        fonline = [row["online_fixed"]["policy_kl"] for row in fork]
        ax.plot(steps + fsteps, offline + foffline, "o-", ms=3,
                label="offline_fixed")
        ax.plot(steps + fsteps, online + fonline, "s-", ms=3, label="online_fixed")
        ax.axvline(9500, color="grey", ls=":", lw=1)
        ax.text(9600, ax.get_ylim()[0], "N1 fork (100 steps/round)", fontsize=8)
    ax.set_title("stored series to step 14500 (N1 fork)")
    ax.set_xlabel("optimizer step")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(ART / "trend.png", dpi=150)
    print("wrote", ART / "trend.png")


if __name__ == "__main__":
    main()
