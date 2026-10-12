"""Render the published small learning-curve JSON with base matplotlib."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
data=json.loads((root/"training_summary.json").read_text())["curve"]
fig,axes=plt.subplots(1,2,figsize=(11,4))
for axis,metric,title in zip(axes,("policy_kl","value_mse"),("Stored-label policy KL","Terminal-value MSE")):
    for split in ("train","validation"):
        axis.plot([r["draw_exposure"] for r in data],[r[split][metric] for r in data],label=split)
    axis.axvline(8,color="gray",linestyle=":",label="LR 1e-4 -> 2.5e-5")
    axis.set(xlabel="Average draws per train position",ylabel=metric,title=title)
    axis.grid(alpha=.2)
    axis.legend()
fig.tight_layout()
fig.savefig(root/"learning_curve.png",dpi=150)
plt.close(fig)
