"""Plot the 1024-position capacity curves (KL and value MSE, log scale)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, color in (("small_128x2", "tab:orange"), ("big_256x4", "tab:blue")):
        data = json.loads((ART / "curves" / f"{name}.json").read_text())
        steps = [row["step"] for row in data["curve"]]
        kl = [row["policy_kl"] for row in data["curve"]]
        mse = [row["value_mse"] for row in data["curve"]]
        axes[0].plot(steps, kl, color=color, label=f"{name} ({data['parameters']:,} params)")
        axes[1].plot(steps, mse, color=color, label=name)
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    axes[0].set_title("Train policy KL (1024 fixed positions)")
    axes[1].set_title("Train value MSE (constant-0 baseline 0.992)")
    for ax in axes:
        ax.set_xlabel("optimizer step"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ART / "capacity_curves.png", dpi=150)
    print("wrote", ART / "capacity_curves.png")


if __name__ == "__main__":
    main()
