"""Render existing analysis JSON only; no model or training dependencies."""
import hashlib
import json
import platform
from pathlib import Path

ART = Path(__file__).resolve().parent
ROUNDS = (3, 9)

def read_json(path):
    return json.loads(path.read_text())

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def input_record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}

def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = read_json(ART / "validation_curves.json")["rows"]
    strength = read_json(ART / "strength_comparison.json")
    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    labels = {"current": "Current round holdout", "offline_fixed": "Fixed offline (4096)",
              "online_fixed": "Fixed first online round (2048)"}
    for col, human in enumerate(ROUNDS):
        subset = [r for r in rows if r["round"] == human]
        steps = [r["step"] for r in subset]
        for row, metric in enumerate(("policy_kl", "value_loss")):
            ax = axes[row, col]
            for source, label in labels.items():
                base = subset[0][source][metric]
                ax.plot(steps, [r[source][metric] - base for r in subset], marker="o", label=label)
            ax.axhline(0, color="grey", lw=0.8)
            ax.axvline(100, color="grey", lw=0.8, ls=":")
            ax.set(xlabel="Updates within round", ylabel=f"Change in {metric} from step 0",
                   title=f"Round {human}: {'policy KL' if row == 0 else 'value MSE'} (lower is better)")
            ax.grid(alpha=0.2)
            if row == 0:
                ax.legend(fontsize=8)
        panel = next(r for r in strength if r["round"] == human)
        scores, lo, hi = [], [], []
        for step in (100, 500):
            summary = panel[f"summary_{step}"]
            score = summary["win_rate_score"]
            scores.append(score)
            lo.append(score - summary["paired_bootstrap"]["lower"])
            hi.append(summary["paired_bootstrap"]["upper"] - score)
        ax = axes[2, col]
        ax.errorbar([100, 500], scores, yerr=[lo, hi], fmt="o", capsize=5, color="tab:blue")
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set(xlim=(0, 600), ylim=(0.3, 0.9), xticks=[100, 500], xlabel="Updates within round",
               ylabel="Score against theta_5000", title=f"Round {human}: MCTS S1024, 50 paired seeds")
        for step, score in zip((100, 500), scores):
            ax.annotate(f"{score:.1%}", (step, score), xytext=(8, 0), textcoords="offset points")
        ax.grid(alpha=0.2)
    fig.savefig(ART / "comparison.png", dpi=160)
    fig.savefig(ART / "comparison.pdf")
    plt.close(fig)
    manifest = read_json(ART / "manifest.json")
    manifest["analysis_script"] = input_record(ART / "analyze.py")
    manifest["plotting"] = {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "environment": "base", "model_inference": False}
    manifest["outputs"] = [input_record(path) for path in sorted(ART.rglob("*"))
                           if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts]
    write_json(ART / "manifest.json", manifest)


if __name__ == "__main__":
    plot()
