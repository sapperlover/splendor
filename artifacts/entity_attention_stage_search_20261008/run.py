"""Two frozen stage-gated searches vs same-weight greedy, with pair resume."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

import torch

from splendor.alphazero_recording import (environment_identity, read_json, sha256,
                                        source_identity, write_json)
from splendor.evaluation import FixedEvaluationConfig, run_fixed_evaluation
from splendor.mcts_policy import MCTSStats
from splendor.native_policy import NativePUCTSearchPolicy, NativeSearchJob
from splendor.neural_policy import GreedyActorCriticPolicy
from splendor.paired_statistics import paired_bootstrap_mean, serialized_pair_scores

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "entity-attention-stage-search-v1"


def stage_for(request, threshold):
    observation = request.observation
    return "late" if max(observation.self_public.score,
                         observation.opponent_public.score) >= threshold else "early"


@dataclass(slots=True)
class StageStats(MCTSStats):
    early_decisions: int = 0
    late_decisions: int = 0
    early_greedy_decisions: int = 0
    late_greedy_decisions: int = 0
    early_search_decisions: int = 0
    late_search_decisions: int = 0
    early_simulations: int = 0
    late_simulations: int = 0
    early_tree_depth_sum: int = 0
    late_tree_depth_sum: int = 0
    early_terminal_hits: int = 0
    late_terminal_hits: int = 0
    late_entry_count: int = 0
    late_entry_completed_turns_sum: int = 0


class StageSearchJob(NativeSearchJob):
    def __init__(self, *args, stage, **kwargs):
        self.stage = stage
        super().__init__(*args, **kwargs)

    def _collect_result(self):
        if self._finished:
            return
        counts = dict(self.core.stats())
        super()._collect_result()
        for key in ("simulations", "tree_depth_sum", "terminal_hits"):
            name = f"{self.stage}_{key}"
            setattr(self.policy.stats, name, getattr(self.policy.stats, name) + counts[key])


def make_stage_policy(mode, threshold):
    if mode not in ("late_only", "early_only") or type(threshold) is not int or not 1 <= threshold <= 15:
        raise ValueError("stage search requires late_only/early_only and a score threshold in 1..15")

    class StagePolicy(NativePUCTSearchPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stats = StageStats()
            self.greedy = GreedyActorCriticPolicy(self.model, device=self.device)
            self.entered_late = False

        def start_job(self, request, *, add_root_noise=False, temperature=0.,
                      sampling_seed=None, root_encoded=None):
            if add_root_noise or temperature != 0:
                raise ValueError("this frozen experiment uses zero root noise and temperature")
            stage = stage_for(request, threshold)
            setattr(self.stats, f"{stage}_decisions", getattr(self.stats, f"{stage}_decisions") + 1)
            if stage == "late" and not self.entered_late:
                self.entered_late = True
                self.stats.late_entry_count += 1
                observation = request.observation
                self.stats.late_entry_completed_turns_sum += (
                    observation.own_completed_turns + observation.opponent_completed_turns)
            use_search = (stage == "late") == (mode == "late_only")
            route = "search" if use_search else "greedy"
            counter = f"{stage}_{route}_decisions"
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)
            if use_search:
                return StageSearchJob(self, request, stage=stage,
                    add_root_noise=False, temperature=0., sampling_seed=sampling_seed,
                    root_encoded=root_encoded)
            action = self.greedy.choose(request)
            self.stats.decisions += 1
            self.stats.single_candidate_decisions += len(request.legal_actions) == 1
            # The cooperative owner consumes completed jobs before native batching.
            # Greedy jobs therefore never allocate a tree or enter NativeBatchRunner.
            return SimpleNamespace(done=True, result=SimpleNamespace(
                chosen_candidate_index=request.legal_actions.index(action)))

    return StagePolicy


@contextmanager
def install_stage_policy(mode, threshold):
    # Only this isolated experiment process replaces the existing owner factory.
    # The opponent is constructed through its unchanged greedy_checkpoint branch.
    import splendor.evaluation_batch as batch
    original = batch.NativePUCTSearchPolicy
    batch.NativePUCTSearchPolicy = make_stage_policy(mode, threshold)
    try:
        yield
    finally:
        batch.NativePUCTSearchPolicy = original


def evaluation_config(plan, mode, code_fingerprint):
    checkpoint = str((ROOT / plan["checkpoint"]).resolve())
    inference = {"mode": plan["execution"]["inference_mode"], "actor_projection": False,
                 "candidate_buckets": plan["execution"]["candidate_buckets"],
                 "native_transport": "reused", "compile_failure_policy": "fail_fast",
                 **plan.get("inference", {})}
    return FixedEvaluationConfig.from_dict({
        "policy_a": {"name": mode, "kind": "mcts_checkpoint", "checkpoint": checkpoint,
                     "search_backend": "native", "inference": inference,
                     "search_config": dict(plan["search_config"], seed=plan["search_seed"])},
        "policy_b": {"name": "same_weight_greedy", "kind": "greedy_checkpoint",
                     "checkpoint": checkpoint, "inference": inference},
        "pairs": plan["pairs"], "seed_start": plan["game_seed_start"],
        "policy_seed_base": plan["policy_seed_base"], "workers": 1,
        "device": plan["device"], "max_completed_turns": plan["max_completed_turns"],
        "bootstrap_resamples": plan["bootstrap_resamples"],
        "bootstrap_seed": plan["bootstrap_seed"], "execution": plan["execution"],
        "metadata": {"protocol": PROTOCOL, "stage_search_mode": mode,
                     "late_score_threshold": plan["late_score_threshold"],
                     "stage_rule": "max(self_public.score, opponent_public.score) >= threshold",
                     "required_runner": str(Path(__file__).relative_to(ROOT)),
                     "code_fingerprint": code_fingerprint}})


def prepare(plan, config_path):
    if plan["protocol"] != PROTOCOL or plan["arms"] != ["late_only", "early_only"]:
        raise ValueError("this experiment runs late_only then early_only")
    if type(plan["pairs"]) is not int or plan["pairs"] <= 0:
        raise ValueError("pairs must be positive")
    make_stage_policy("late_only", plan["late_score_threshold"])
    checkpoint = ROOT / plan["checkpoint"]
    if sha256(checkpoint) != plan["checkpoint_sha256"]:
        raise ValueError("frozen checkpoint differs from the experiment plan")
    artifact, runtime = ROOT / plan["artifact_root"], ROOT / plan["runtime_root"]
    # Record execution code once and compare hashes on resume; no game/replay
    # ledger audit or zero-round self-match is required.
    paths = [Path(__file__), *(ROOT / "src/splendor").glob("*.py"),
             ROOT / "src/splendor/native_search.cpp",
             *(ROOT / "src/splendor").glob("_native_search*.so")]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest_path = artifact / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["plan"] != plan or manifest["code_fingerprint"] != fingerprint:
            raise ValueError("plan or execution code changed; use a new output directory")
    else:
        snapshot = runtime / "source"
        source = source_identity(ROOT, snapshot)
        for p in (Path(__file__), ROOT / "src/splendor/native_search.cpp",
                  *(ROOT / "src/splendor").glob("_native_search*.so")):
            target = snapshot / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, target)
        manifest = {"protocol": PROTOCOL, "plan": plan, "code_fingerprint": fingerprint,
                    "code_hashes": hashes, "source": source,
                    "checkpoint": {"path": str(checkpoint), "sha256": plan["checkpoint_sha256"]},
                    "command": sys.argv, "config_path": str(config_path.resolve()),
                    "environment": environment_identity()}
        write_json(manifest_path, manifest)
        write_json(artifact / "configuration_snapshot.json", plan)
    for mode in plan["arms"]:
        config = evaluation_config(plan, mode, fingerprint)
        write_json(artifact / "configs" / f"{mode}.json", config.to_dict())
    return artifact, runtime, fingerprint


def analyze(plan, artifact):
    panels, values, seeds = {}, {}, {}
    for mode in plan["arms"]:
        path = artifact / "evaluations" / f"{mode}.json"
        if not path.exists():
            continue
        summary = read_json(path)
        if summary["completed_pairs"] != plan["pairs"]:
            continue
        seeds[mode], values[mode] = serialized_pair_scores(summary)
        stats = summary["policy_a_stats"]
        wrong_stage = "early" if mode == "late_only" else "late"
        if stats[f"{wrong_stage}_simulations"] != 0:
            raise ValueError("search ran outside its assigned stage")
        if stats["simulations"] != plan["search_config"]["simulations"] * stats["searchable_decisions"]:
            raise ValueError("searched decisions did not use the declared budget")
        if summary["policy_b_stats"]:
            raise ValueError("the greedy opponent unexpectedly collected search statistics")
        panels[mode] = {"score": summary["win_rate_score"], "ci95": summary["paired_bootstrap"],
            "wins": summary["a_wins"], "draws": summary["draws"], "losses": summary["a_losses"],
            "policy_cycle_draws": summary["policy_cycle_draws"], "stats": stats,
            "mean_search_depth": stats["tree_depth_sum"] / stats["simulations"] if stats["simulations"] else None,
            "terminal_fraction": stats["terminal_hits"] / stats["simulations"] if stats["simulations"] else None,
            "late_entry_mean_completed_turns": (stats["late_entry_completed_turns_sum"] / stats["late_entry_count"]
                                                if stats["late_entry_count"] else None)}
    result = {"protocol": PROTOCOL, "status": "complete" if len(panels) == 2 else "incomplete",
              "panels": panels, "scope": "Frozen16500 weights; score10 gate; same-weight greedy opponent. "
              "Stage policies change trajectories. This does not measure the fraction of full-search benefit "
              "or isolate exact-terminal backups from other search effects."}
    if len(values) == 2:
        if seeds["late_only"] != seeds["early_only"]:
            raise ValueError("stage panels must have identical paired game seeds")
        result["late_minus_early"] = paired_bootstrap_mean(
            values["late_only"] - values["early_only"], resamples=plan["bootstrap_resamples"],
            seed=plan["bootstrap_seed"] + 1).to_dict()
    write_json(artifact / "analysis.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("plan_fast.json"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    plan = read_json(args.config)
    artifact, runtime, fingerprint = prepare(plan, args.config)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "pairs_per_arm": plan["pairs"],
                          "arms": plan["arms"], "checkpoint": plan["checkpoint"]}), flush=True)
        return 0
    if args.analyze_only:
        print(json.dumps(analyze(plan, artifact), ensure_ascii=False), flush=True)
        return 0
    write_json(artifact / "invocations" / f"{time.time_ns()}.json",
               {"command": sys.argv, "environment": environment_identity()})
    for mode in plan["arms"]:
        path = artifact / "evaluations" / f"{mode}.json"
        config = evaluation_config(plan, mode, fingerprint)
        print(json.dumps({"event": "arm_start", "arm": mode, "pairs": plan["pairs"]}), flush=True)

        def progress(n, total):
            rss = next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines()
                       if line.startswith("VmRSS:"))
            row = {"event": "progress", "arm": mode, "completed_pairs": n, "pairs": total,
                   "rss_bytes": int(rss) * 1024}
            if plan["device"] == "cuda":
                row.update(cuda_allocated_bytes=torch.cuda.memory_allocated(),
                           cuda_reserved_bytes=torch.cuda.memory_reserved())
            print(json.dumps(row), flush=True)
            write_json(runtime / f"{mode}_memory_latest.json", row)

        with install_stage_policy(mode, plan["late_score_threshold"]):
            run = run_fixed_evaluation(config, path, resume=Path(str(path) + ".games").exists(),
                failure_dir=runtime / "failures" / mode, progress_every=25, summary_every=25,
                progress_callback=progress)
        if not run.summary["passed"] or run.summary["completed_pairs"] != plan["pairs"]:
            raise RuntimeError(f"{mode} contains incomplete/invalid games; inspect the retained failure records")
        analyze(plan, artifact)
        print(json.dumps({"event": "arm_done", "arm": mode, "score": run.summary["win_rate_score"],
                          "ci95": run.summary["paired_bootstrap"]}), flush=True)
        del run
        gc.collect()
        if plan["device"] == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
