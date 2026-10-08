"""Freeze the finite ablation, common batch order and actual input identities."""
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np

from splendor.alphazero_checkpoint import load_alphazero_checkpoint
from splendor.alphazero_recording import sha256, write_json
from splendor.rng import DeterministicRng
from splendor.offline_train import make_index

from core import ARMS

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent
RUN = ROOT / "runs/entity_attention_hybrid_s1024_n1_from_round9_seed1"
OUT = ROOT / "runs/entity_attention_policy_value_ablation_20261006"


def identity(path):
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "bytes": path.stat().st_size}


def main():
    assert not (ART / "specification.json").exists()
    stage = RUN / "rounds/round-049"
    snap_path = stage / "replay_snapshot.json"
    snapshot = json.loads(snap_path.read_text())
    latest = [g for g in snapshot["games"] if g["round_index"] == snapshot["round_index"]]
    assert len(latest) == 200
    initial = stage / "checkpoints/initial.pt"
    payload = load_alphazero_checkpoint(initial)
    assert payload.total_train_step == 14400
    teacher_identity = json.loads((stage / "selfplay/manifest.json").read_text())["checkpoint_identity"]
    teacher = Path(teacher_identity["path"])
    assert sha256(teacher) == teacher_identity["sha256"]
    teacher_payload = load_alphazero_checkpoint(teacher)
    assert payload.model_sha256 == teacher_payload.model_sha256
    assert teacher_payload.total_train_step == 14400
    index = make_index({"games": latest}, 1024**3)
    train = index.split_games("train")
    starts = np.cumsum([0] + [g.decision_count for g in train])
    rng = DeterministicRng.restore(payload.sampling_rng_state)
    refs = np.empty((500, 256, 2), dtype="<i8")
    for step in range(500):
        for row in range(256):
            position = rng.randbelow(int(starts[-1]))
            which = int(np.searchsorted(starts, position, side="right") - 1)
            refs[step, row] = train[which].game_id, position - int(starts[which])
    OUT.mkdir(parents=True, exist_ok=False)
    ref_path = OUT / "batch_references.npy"
    np.save(ref_path, refs)
    probe_rng = DeterministicRng(14850000)
    probe = []
    for _ in range(8*256):
        position = probe_rng.randbelow(int(starts[-1]))
        which = int(np.searchsorted(starts, position, side="right") - 1)
        probe.append([train[which].game_id, position-int(starts[which])])
    probe_path = OUT / "gradient_probe_references.json"
    write_json(probe_path, {"seed": 14850000, "references": probe})
    inputs = [identity(p) for p in (initial, teacher, snap_path, RUN/"config.json", RUN/"hybrid_state.json",
               stage/"selfplay/manifest.json", stage/"selfplay/summary.json",
               ROOT/"configs/entity_attention_v1/hybrid_fork_s1024_n1.json",
               ROOT/"runs/5c_main_seed1/checkpoints/local_005000000_lifetime_005512112.pt",
               ROOT/"artifacts/entity_attention_strong_anchor_20261006/evaluations/round50_vs_ppo.json")]
    inputs.extend(identity(Path(game["path"])) for game in latest)
    checkpoints = {"initial": str(initial.relative_to(ROOT))}
    for arm in ARMS:
        for step in (100,500):
            checkpoints[f"{arm}_{step}"] = str((OUT/arm/f"step-{step:06d}.pt").relative_to(ROOT))
    search = json.loads((ROOT/"artifacts/entity_attention_strong_anchor_20261006/configs/round50_vs_ppo.json").read_text())
    greedy = json.loads((ROOT/"configs/entity_attention_v1/greedy_vs_ppo_5_5m_100pairs.json").read_text())
    panels=[]
    for candidate, checkpoint in checkpoints.items():
        for mode, template in (("greedy",greedy),("mcts",search)):
            raw = json.loads(json.dumps(template))
            raw["policy_a"]["checkpoint"] = checkpoint
            raw["policy_a"]["name"] = candidate
            raw["pairs"] = 100 if mode=="greedy" or candidate in ("initial","joint_500","policy_only_500") else 50
            key = f"{candidate}_{mode}"
            config_path = ART/"configs"/f"{key}.json"
            write_json(config_path, raw)
            panels.append({"id":key,"candidate":candidate,"mode":mode,"pairs":raw["pairs"],
                           "fixed_value":False,"config":str(config_path.relative_to(ART))})
    for arm in ARMS:
        candidate=f"{arm}_500"
        raw=json.loads(json.dumps(search))
        raw["policy_a"].update(checkpoint=checkpoints[candidate],name=f"{candidate}_fixed_value")
        raw["pairs"]=50
        raw["metadata"]={"diagnostic":"policy from candidate plus complete value function from initial",
                          "value_checkpoint_identity":identity(initial),"adapter":"core.FixedValueModel"}
        key=f"{candidate}_mcts_fixed_value"
        config_path=ART/"configs"/f"{key}.json"
        write_json(config_path,raw)
        panels.append({"id":key,"candidate":candidate,"mode":"mcts","pairs":50,"fixed_value":True,
                       "config":str(config_path.relative_to(ART))})
    spec={"protocol":"entity-attention-policy-value-ablation-v1","initial":identity(initial),
          "output":str(OUT.relative_to(ROOT)),"data_round":snapshot["round_index"],"games":latest,
          "train_games":len(train),"train_positions":int(starts[-1]),
          "validation_games":len(index.split_games("validation")),
          "validation_positions":sum(g.decision_count for g in index.split_games("validation")),
          "arms":list(ARMS),"steps":500,"batch_size":256,"observations":[0,100,300,500],
          "sampling_rng_start":asdict(payload.sampling_rng_state),"sampling_rng_end":asdict(rng.snapshot()),
          "teacher_checkpoint":teacher_identity,"initial_model_sha256":payload.model_sha256,
          "batch_references":identity(ref_path),"gradient_probe":identity(probe_path),
          "learner_config":payload.learner_config,"inputs":inputs,"panels":panels,
          "panel_seed_scope":"Repeated DEV; first50 or full100 seeds from13500000, swapped seats.",
          "clip_rule":"All arms retain one global clip norm1; detached critic can still affect clipping scale.",
          "optimizer_rule":"Restore identical full AdamW state. Policy-only critic requires_grad=false and has grad=None: no moment/decay updates.",
          "limits":"Fixed current-round labels, one checkpoint/data/sampling seed; no self-iteration or proof of optimal long-run window."}
    write_json(ART/"specification.json",spec)
    print(json.dumps({"event":"prepared","train_games":len(train),"train_positions":int(starts[-1]),
                      "validation_positions":spec["validation_positions"],"panels":len(panels),
                      "total_evaluation_games":sum(2*p["pairs"] for p in panels)},ensure_ascii=False))


if __name__=="__main__":
    main()
