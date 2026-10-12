"""Render the fixed-holdout OFAT curves; no model or training dependencies.

From repository root with a matplotlib environment:
MPLCONFIGDIR=/tmp/splendor-r9-ofat-mpl python artifacts/entity_attention_r9_ofat_20261010/plot.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART = Path(__file__).resolve().parent
data = json.loads((ART / "summary.json").read_text())
if data["status"] != "complete":
    raise ValueError("render final curves only after all panels complete")
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "r9-ofat-20261010", "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True, sharey="row")
groups = (("learning_rate", "Learning rate", "1e-4"), ("pool_games", "Online pool", "6k games"), ("batch_size", "Batch size", "B256"))
colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
baseline = [p for p in data["baseline"] if p["step"] <= 2000]
for col, (axis, title, base_label) in enumerate(groups):
    panels = [p for p in data["panels"] if p["axis"] == axis]
    for row, metric in enumerate(("policy_kl", "value_mse")):
        ax = axes[row, col]
        ax.plot([p["step"] for p in baseline], [p[metric] for p in baseline], color="#333333", linestyle="--", marker="o", markersize=4, linewidth=1.7, label=f"Baseline: {base_label}")
        for color, panel in zip(colors, panels):
            value = panel["override"][axis]
            label = f"{value:g}" if axis == "learning_rate" else f"{value//1000}k games" if axis == "pool_games" else f"B{value}"
            points = [p for p in panel["points"] if p["step"] != 125]
            ax.plot([p["step"] for p in points], [p[metric] for p in points], color=color, marker="o", markersize=4, linewidth=1.5, label=label)
        ax.grid(alpha=.18)
        ax.set_xticks((0, 250, 500, 1000, 1500, 2000))
        ax.legend(fontsize=8, framealpha=.9)
        if row == 0:
            ax.set_title(title, weight="bold")
        else:
            ax.set_xlabel("Optimizer updates")
        if col == 0:
            ax.set_ylabel("Teacher-to-student KL" if row == 0 else "Value MSE")
fig.suptitle("Round 9 fixed holdout: one variable changed from the same model and AdamW state", fontsize=14, weight="bold")
fig.text(.5, .02, "100 held-out games | 8,174 value positions / 6,688 policy positions | Lower is better | Point estimates; paired intervals in report", ha="center", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, .05, 1, .95))
fig.savefig(ART / "holdout_curves.svg", metadata={"Date": None})
fig.savefig("/tmp/splendor_r9_ofat_curves.png", dpi=150)
