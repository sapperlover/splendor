"""Plot completed 64-root statistics; no model execution or new sampling."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ART=Path(__file__).resolve().parent
summary=json.loads((ART/"summary.json").read_text())
roots=json.loads((ART/"per_root.json").read_text())
fig,(ax,bx)=plt.subplots(1,2,figsize=(11,4.6),layout="constrained")
for stage,color in (("early","#276FBF"),("middle","#E58F22"),("late","#38986C")):
    group=[r for r in roots if r["stage"]==stage]
    ax.errorbar([r["prediction"] for r in group],[r["mean_z"] for r in group],
        yerr=[r["mean_se"] for r in group],fmt="o",ms=4.5,alpha=.75,color=color,label=stage,capsize=2,lw=.8)
ax.plot([-1,1],[-1,1],"--",color="#777777",lw=1)
ax.set(xlabel="R43 value prediction",ylabel="Mean terminal z over 16 continuations",
       title="Same visible root, independent continuations",xlim=(-1.15,1.15),ylim=(-1.2,1.2))
ax.legend(title="Archived trajectory stage",fontsize=9)
ax.text(.03,.97,"Error bars: +/-1 Monte Carlo SE",transform=ax.transAxes,va="top",fontsize=9)
keys=("single_return_mse","within_root_variance","corrected_expected_value_mse")
labels=("MSE vs single z","Outcome variance","MSE vs expectation\n(MC corrected)")
means=np.array([summary["overall"][k] for k in keys])
cis=np.array([summary["ci95"][k] for k in keys])
bx.bar(np.arange(3),means,color=["#59697B","#E58F22","#276FBF"],width=.6,alpha=.9)
bx.errorbar(np.arange(3),means,yerr=np.stack((means-cis[:,0],cis[:,1]-means)),fmt="none",color="#222222",capsize=4)
for i,x in enumerate(means):
    bx.text(i,cis[i,1]+.018,f"{x:.3f}",ha="center",va="bottom",fontsize=10)
bx.axhline(0,color="#777777",lw=.8)
bx.set(xticks=np.arange(3),xticklabels=labels,ylabel="Squared terminal-return units",
       title="Single-return error decomposition",ylim=(min(0.,cis[:,0].min()-.025),cis[:,1].max()+.15))
bx.text(.02,.97,"95% stage-stratified root bootstrap",transform=bx.transAxes,va="top",fontsize=9)
for panel in (ax,bx):panel.grid(axis="y",alpha=.18);panel.set_axisbelow(True)
fig.suptitle("Frozen R43, native S1024/N1, root noise 0.25, T=0 | 64 roots x 16 continuations",fontsize=11)
fig.savefig(ART/"value_signal.png",dpi=180)
plt.close(fig)
