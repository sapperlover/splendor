"""Plot the generation x checkpoint online-holdout KL matrix."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent


def main():
    data = json.loads((ART / "online_matrix.json").read_text(encoding="utf-8"))
    rows = data["checkpoints"]
    generations = [str(i) for i in range(1, 10)]
    parent = [row for row in rows if row["series"] == "parent"]
    steps_all = [row["total_train_step"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                             gridspec_kw={"width_ratios": [1, 1.35]})

    ax = axes[0]
    matrix = np.asarray([[row["holdouts"][g]["policy_kl"] for g in generations]
                         for row in parent])
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(generations)))
    ax.set_xticklabels([f"g{g}" for g in generations])
    ax.set_yticks(range(len(parent)))
    ax.set_yticklabels([str(row["total_train_step"]) for row in parent], fontsize=8)
    ax.set_xlabel("frozen online holdout (generation)")
    ax.set_ylabel("checkpoint step")
    ax.set_title("parent run 5000->9500: KL per holdout")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center",
                    color="white" if matrix[i, j] < matrix.max() - 0.06 else "black",
                    fontsize=6.5)
    fig.colorbar(im, ax=ax, fraction=0.046, label="KL(target || student)")

    ax = axes[1]
    base = {g: rows[0]["holdouts"][g]["policy_kl"] for g in generations}
    for g in generations:
        values = [row["holdouts"][g]["policy_kl"] - base[g] for row in rows]
        ax.plot(steps_all, values, "-", lw=1.1, label=f"g{g}")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(9500, color="grey", ls=":", lw=1)
    ax.text(9550, ax.get_ylim()[1], "N1 fork", fontsize=8, va="top")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("KL change vs step-5000 (negative = better)")
    ax.set_title("every generation's holdout, through step 14500")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)

    fig.tight_layout()
    fig.savefig(ART / "online_matrix.png", dpi=150)
    print("wrote", ART / "online_matrix.png")


if __name__ == "__main__":
    main()
