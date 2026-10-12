"""Render static report figures from retained statistics; no model inference."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
evidence = json.loads(gzip.decompress((ROOT / "completed_evidence.json.gz").read_bytes()))
analysis = json.loads((ROOT / "completed_analysis.json").read_text())
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), layout="constrained")
ax = axes[0]
for mode, rounds, color, label in [("greedy_vs_start",range(1,10),"#2166ac","Greedy vs P4"),
                                   ("mcts_vs_fixed_anchor",[3,6,9],"#c65a15","S1024/N1 vs P4")]:
    rows = [evidence["battles"][f"rounds/round-{i-1:03d}/evaluation/{mode}.json"] for i in rounds]
    means = np.array([b["win_rate_score"] for b in rows])*100
    ci = np.array([[b["paired_bootstrap"]["lower"],b["paired_bootstrap"]["upper"]] for b in rows])*100
    ax.errorbar(list(rounds),means,yerr=np.array([means-ci[:,0],ci[:,1]-means]),fmt="o-",
                capsize=3,color=color,label=label,lw=1.7,ms=5)
ax.axhline(50,color="#666666",ls="--",lw=1)
ax.axvspan(6.5,9.45,color="#eeeeee",zorder=-1)
ax.text(7.5,44,"Online only",ha="center",fontsize=9,color="#555555")
ax.set(xlabel="Completed round",ylabel="Score (%) with paired 95% CI",xlim=(.6,9.4),ylim=(42,80),xticks=range(1,10))
ax.set_title("Policy strength and search strength")
ax.legend(loc="upper left",frameon=False)
ax.grid(axis="y",alpha=.18)
ax = axes[1]
keys = ["rounds/round-008/evaluation/mcts_vs_fixed_anchor.json",
        "final_evaluation/online_vs_offline_control.json","final_evaluation/online_vs_P12V4.json"]
labels = ["vs P4 start","vs old-pool control","vs P12 + V4"]
for i,(key,label) in enumerate(zip(keys,labels)):
    b = evidence["battles"][key]
    mean = b["win_rate_score"]*100
    low,high = [b["paired_bootstrap"][k]*100 for k in ("lower","upper")]
    ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt="o",color="#2166ac",capsize=5,ms=7)
    ax.text(high+.6,i,f"{mean:.3f}%",va="center",fontsize=10)
ax.axvline(50,color="#666666",ls="--",lw=1)
ax.set(yticks=range(3),yticklabels=labels,xlabel="Round 9 score (%) with paired 95% CI",xlim=(47,71),ylim=(2.6,-.6))
ax.set_title("Final S1024/N1 panels: 200 pairs each")
ax.grid(axis="x",alpha=.18)
fig.savefig(ROOT / "strength_completed.svg")
plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(12,5.5),layout="constrained")
for ax,name,title,limit in zip(axes,["delta_kl","delta_mse"],
                              ["Heldout policy KL change","Heldout value MSE change"],[.035,.065]):
    data = np.array([[np.nan if v is None else v for v in row] for row in analysis["matrices"][name]])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(data,cmap=cmap,vmin=-limit,vmax=limit)
    ax.set(xticks=range(9),xticklabels=range(1,10),yticks=range(9),yticklabels=range(1,10),
           xlabel="Data generation (fixed labels within each column)",ylabel="Completed student round")
    ax.set_title(title)
    for i in range(9):
        for j in range(9):
            if np.isfinite(data[i,j]):
                ax.text(j,i,f"{data[i,j]:+.3f}",ha="center",va="center",fontsize=8,
                        color="white" if abs(data[i,j])>limit*.67 else "#222222")
    fig.colorbar(im,ax=ax,shrink=.72,label="End of round minus generation's first pre-training metric")
fig.suptitle("Blue = lower error; gray = outside measured active window. Stored-label DEV.",fontsize=11)
fig.savefig(ROOT / "fitting_changes_completed.svg")
plt.close(fig)
print("Rendered strength_completed.svg and fitting_changes_completed.svg")
