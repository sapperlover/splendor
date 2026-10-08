"""Run the predeclared strength panels serially in separate GPU processes."""
import json
import subprocess
import sys
from pathlib import Path

ART=Path(__file__).resolve().parent
spec=json.loads((ART/"specification.json").read_text())
panels={p["id"]:p for p in spec["panels"]}
order=[p["id"] for p in spec["panels"] if p["mode"]=="greedy"]
order += ["initial_mcts","joint_500_mcts","policy_only_500_mcts","value_detached_500_mcts",
          "joint_100_mcts","policy_only_100_mcts","value_detached_100_mcts"]
order += [p["id"] for p in spec["panels"] if p["fixed_value"]]
assert set(order)==set(panels) and len(order)==len(panels)
for index,case in enumerate(order):
    output=ART/"evaluations"/f"{case}.json"
    if output.exists() and json.loads(output.read_text())["passed"]:
        print(json.dumps({"event":"panel_already_completed","case":case}),flush=True)
        continue
    print(json.dumps({"event":"suite_panel","number":index+1,"total":len(order),"case":case}),flush=True)
    subprocess.run([sys.executable,"-u",str(ART/"evaluate.py"),"--case",case],check=True)
print(json.dumps({"event":"suite_completed","panels":len(order)}),flush=True)
