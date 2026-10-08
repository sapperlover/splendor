"""Plot completed screening JSON only, without model dependencies."""
import hashlib
import json
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def main():
    scores = read(ART / "screening_summary.json")
    costs = {r["case"]: r for r in read(ART / "cost_comparison.json")["cases"]}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
    colors = {"s512_n1": "tab:blue", "s1024_n1": "tab:orange", "s512_n2": "tab:green", "s1024_n2": "tab:red"}
    for row in scores["strengths"]:
        case = row["case"]
        cost = costs[case]
        x, y = cost["mean_speed_ratio_vs_s1024_n1"], row["score"]
        axes[0].errorbar(x, y,
                        xerr=[[x - cost["ratio_range"][0]], [cost["ratio_range"][1] - x]],
                        yerr=[[y - row["score_ci"]["lower"]], [row["score_ci"]["upper"] - y]],
                        fmt="o", capsize=4, color=colors[case], label=case.upper().replace("_", "/"))
    axes[0].axhline(0.5, color="grey", ls="--", lw=0.8)
    axes[0].axvline(1.0, color="grey", ls=":", lw=0.8)
    axes[0].set(xlabel="Short-window search throughput / S1024/N1", ylabel="Score against theta_5000",
                title="Frozen theta_9500: strength and collection cost", ylim=(0.3, 0.9))
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.2)
    cases = [r["case"] for r in scores["strengths"]]
    for offset, noise, label in ((-0.18, False, "No root noise"), (0.18, True, "Training root noise (fixed)")):
        values = [next(r for r in scores["root_sensitivity"] if r["case"] == c and r["root_noise"] == noise)
                  ["mean_js_across_world_seeds"] for c in cases]
        axes[1].bar(np.arange(len(cases)) + offset, values, width=0.36, label=label)
    axes[1].set(xticks=np.arange(len(cases)), xticklabels=[c.upper().replace("_", "/") for c in cases],
                ylabel="Mean target JS across determinization seeds",
                title="24 fixed roots, 3 world seeds (lower = more stable)")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(ART / "screening.png", dpi=160)
    fig.savefig(ART / "screening.pdf")
    plt.close(fig)
    manifest = read(ART / "manifest.json")
    manifest["plotting"] = {"environment": "base", "python": platform.python_version(), "matplotlib": matplotlib.__version__}
    manifest["outputs"] = [record(p) for p in sorted(ART.rglob("*")) if p.is_file() and p.name != "manifest.json" and "__pycache__" not in p.parts]
    (ART / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
