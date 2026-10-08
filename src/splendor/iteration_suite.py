"""Serial, resource-bounded and resumable continuous-iteration experiments (Linux)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

PROTOCOL = "entity-attention-iteration-trend-v1"
GIB = 1024**3
ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def available_memory():
    return next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def cgroup_path():
    lines = Path("/proc/self/cgroup").read_text().splitlines()
    relative = next(line.split("::", 1)[1] for line in lines if line.startswith("0::"))
    return Path("/sys/fs/cgroup") / relative.lstrip("/")


def arm_parent_death(parent_pid):
    """Linux sends SIGINT even if the supervisor is killed without running cleanup."""
    import ctypes
    if os.getppid() != parent_pid:
        raise RuntimeError("systemd scope worker is not a direct child of its supervisor")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGINT), 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        raise OSError(ctypes.get_errno(), "cannot arm parent-death signal")
    if os.getppid() != parent_pid:
        raise KeyboardInterrupt


def validate_plan(plan):
    if plan.get("protocol") != PROTOCOL:
        raise ValueError("use the current iteration_trend config; old comparison protocols are retired")
    for key in ("rounds", "fork_round", "replay_cache_mib"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for rows, kind in ((plan["arms"], "arm"), (plan["replicates"], "replicate")):
        ids = [row["id"] for row in rows]
        if not ids or len(set(ids)) != len(ids) or any(not v.replace("_", "").isalnum() for v in ids):
            raise ValueError(f"{kind} ids must be nonempty, distinct and alphanumeric with underscores")
    for arm in plan["arms"]:
        if type(arm["updates_per_round"]) is not int or arm["updates_per_round"] <= 0:
            raise ValueError("arm updates must be positive")
    evaluation = plan["evaluation"]
    nodes = evaluation["rounds"]
    if (not nodes or any(type(n) is not int or not 0 < n <= plan["rounds"] for n in nodes)
            or nodes != sorted(set(nodes)) or nodes[-1] != plan["rounds"]):
        raise ValueError("evaluation rounds must increase and include the final round")
    for key in ("pairs", "bootstrap_resamples"):
        if type(evaluation[key]) is not int or evaluation[key] <= 0:
            raise ValueError(f"{key} must be positive")
    limits = plan["resources"]
    if not 0 < limits["memory_high_gib"] < limits["memory_max_gib"]:
        raise ValueError("invalid cgroup memory bounds")
    if plan["replay_cache_mib"] / 1024 >= limits["memory_high_gib"]:
        raise ValueError("cache must leave memory for models and workers")
    for key in ("min_available_gib", "min_gpu_free_mib", "poll_seconds"):
        if not math.isfinite(limits[key]) or limits[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")


def arm_configs(plan):
    base = read(ROOT / plan["base_config"])
    configs = {}
    for arm in plan["arms"]:
        for rep in plan["replicates"]:
            raw = json.loads(json.dumps(base))
            name = f"{rep['id']}_{arm['id']}"
            raw["run_id"] = f"{plan['suite_id']}_{name}"
            raw["updates_per_round"] = arm["updates_per_round"]
            raw["evaluation_every_rounds"] = 0
            raw["evaluation"] = None
            raw.pop("evaluation_anchor", None)
            raw["seeds"]["collection"] = rep["collection_seed"]
            raw["seeds"]["split"] = rep["split_seed"]
            raw["selfplay"]["search_config"]["seed"] = rep["search_seed"]
            raw["selfplay"]["execution"].update(plan["collection_execution"])
            if arm.get("learning_rate_override") is not None:
                raw["learning_rate_override"] = arm["learning_rate_override"]
            if arm.get("policy_recent_rounds") is not None:
                raw["split_training"]["policy_recent_rounds"] = arm["policy_recent_rounds"]
            configs[name] = raw
    return configs


def evaluation_template(plan):
    raw = read(ROOT / plan["base_config"])["evaluation"]
    raw["execution"].update(plan["evaluation_execution"])
    raw["policy_b"] = json.loads(json.dumps(raw["policy_a"]))
    raw["policy_b"].update(checkpoint=str(ROOT / plan["start_checkpoint"]), name="frozen_start")
    for spec in (raw["policy_a"], raw["policy_b"]):
        spec["inference"]["mode"] = plan["evaluation_execution"]["inference_mode"]
    return raw


def stop_child(process, unit):
    # Only this invocation's new session and uniquely named scope are addressed.
    try:
        os.killpg(process.pid, signal.SIGINT)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        subprocess.run(["systemctl", "--user", "stop", unit], timeout=20,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
    # A worker may outlive its leader. Stopping our scope also covers such descendants.
    subprocess.run(["systemctl", "--user", "stop", unit], timeout=20,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_stage(plan, task, key, *, round_number=None):
    runtime = ROOT / plan["runtime_root"]
    artifacts = ROOT / plan["artifact_root"]
    token = f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    unit = f"splendor-suite-{os.getpid()}-{token}.scope"
    identity = runtime / "processes" / f"{token}.json"
    log = runtime / "logs" / f"{task}-{key}-{token}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    limits = plan["resources"]
    expected_max = int(limits["memory_max_gib"] * GIB)
    command = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "--unit", unit,
               "-p", f"MemoryMax={expected_max}", "-p", f"MemoryHigh={int(limits['memory_high_gib'] * GIB)}",
               "-p", "MemorySwapMax=0", sys.executable, "-u", "-m", "splendor.iteration_suite",
               "--worker", task, "--manifest", str(artifacts / "manifest.json"),
               "--key", key, "--process-identity", str(identity), "--memory-max", str(expected_max),
               "--parent-pid", str(os.getpid())]
    if round_number is not None:
        command.extend(["--round", str(round_number)])
    if available_memory() < limits["min_available_gib"] * GIB:
        raise RuntimeError("available RAM below reserve; close other work and rerun the same command")
    started = time.monotonic()
    record = {"task": task, "key": key, "round": round_number, "command": command,
              "log": str(log), "unit": unit, "status": "running", "start_unix": time.time(),
              "peak_cgroup_bytes": 0, "min_available_bytes": available_memory()}
    invocation_path = artifacts / "resources" / f"{task}-{key}-{token}.json"
    write(invocation_path, record)
    emit("stage_start", task=task, key=key, round=round_number, log=str(log))
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TORCHINDUCTOR_COMPILE_THREADS="2")
    process = None
    last_notice = 0.
    try:
        with log.open("w", encoding="utf-8") as handle:
            process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            with (runtime / "resources.jsonl").open("a", encoding="utf-8") as samples:
                while process.poll() is None:
                    available = available_memory()
                    sample = {"time": time.time(), "task": task, "key": key, "round": round_number,
                              "available_bytes": available}
                    if identity.exists():
                        cg = Path(read(identity)["cgroup"])
                        try:
                            sample["cgroup_bytes"] = int((cg / "memory.current").read_text())
                            peak = int((cg / "memory.peak").read_text())
                            record["peak_cgroup_bytes"] = max(record["peak_cgroup_bytes"], peak)
                        except FileNotFoundError:
                            pass  # The scope disappears normally when its last process exits.
                    record["min_available_bytes"] = min(record["min_available_bytes"], available)
                    samples.write(json.dumps(sample) + "\n")
                    samples.flush()
                    if available < limits["min_available_gib"] * GIB:
                        raise RuntimeError("resource reserve reached: available RAM below limit; "
                                           "stage stopped, rerun after releasing memory")
                    if time.monotonic() - last_notice >= 30:
                        emit("stage_running", task=task, key=key, round=round_number,
                             elapsed_seconds=round(time.monotonic() - started),
                             peak_memory_gib=round(record["peak_cgroup_bytes"] / GIB, 2))
                        last_notice = time.monotonic()
                    time.sleep(limits["poll_seconds"])
            if process.returncode != 0:
                tail = log.read_bytes()[-6000:].decode("utf-8", errors="replace")
                raise RuntimeError(f"stage exit={process.returncode}; log={log}\n{tail}")
        if not identity.exists():
            raise RuntimeError("worker did not verify its cgroup hard limit")
        record["status"] = "complete"
    except BaseException as error:
        record["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        record["error"] = str(error)
        if process is not None:
            stop_child(process, unit)
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        if identity.exists():
            info = read(identity)
            record["worker"] = info
            record["peak_cgroup_bytes"] = max(record["peak_cgroup_bytes"], info.get("peak_cgroup_bytes", 0))
        write(invocation_path, record)
    emit("stage_complete", task=task, key=key, round=round_number,
         elapsed_seconds=round(record["elapsed_seconds"], 1))


def checkpoint(plan, key, round_number):
    return ROOT / plan["run_root"] / key / "rounds" / f"round-{round_number - 1:03d}" / "checkpoints/final.pt"


def evaluation_jobs(plan):
    for arm in plan["arms"]:
        for rep in plan["replicates"]:
            key = f"{rep['id']}_{arm['id']}"
            for n in plan["evaluation"]["rounds"]:
                yield {"id": f"{key}_r{n:03d}", "round": n, "replicate": rep["id"],
                       "arm": arm["id"], "a": str(checkpoint(plan, key, n)),
                       "pairs": plan["evaluation"]["pairs"]}


def evaluation_config(manifest, job):
    plan = manifest["plan"]
    rep = next(r for r in plan["replicates"] if r["id"] == job["replicate"])
    raw = json.loads(json.dumps(manifest["evaluation_template"]))
    raw.update(pairs=job["pairs"], seed_start=rep["eval_game_seed"],
               policy_seed_base=rep["eval_policy_seed"], bootstrap_seed=rep["eval_bootstrap_seed"],
               bootstrap_resamples=plan["evaluation"]["bootstrap_resamples"], metadata={"suite_job": job})
    raw["policy_a"].update(checkpoint=job["a"], name=job["id"])
    for spec in (raw["policy_a"], raw["policy_b"]):
        spec["search_config"]["seed"] = rep["eval_search_seed"]
    return raw


def stage_schedule(plan, jobs):
    """Finish B and its anchor evaluations, then C, then D; no extra panels."""
    for arm in plan["arms"]:
        for n in range(1, plan["rounds"] + 1):
            for rep in plan["replicates"]:
                yield "train", f"{rep['id']}_{arm['id']}", n
            if n in plan["evaluation"]["rounds"]:
                for job in jobs:
                    if job["arm"] == arm["id"] and job["round"] == n:
                        yield "evaluate", job["id"], None
                yield "analyze", f"{arm['id']}_round{n}", None


def completion_path(plan, task, key, round_number=None):
    suffix = f"_{round_number:03d}" if round_number is not None else ""
    return ROOT / plan["runtime_root"] / "completed" / f"{task}_{key}{suffix}.json"


def stratified_interval(groups, *, confidence=.95, resamples=10000, seed=0):
    """Keep the trained trajectories fixed, resampling paired game seeds within each."""
    import numpy as np
    arrays = [np.asarray(g, dtype=np.float64) for g in groups]
    if not arrays or any(a.ndim != 1 or not len(a) or not np.isfinite(a).all() for a in arrays):
        raise ValueError("bootstrap requires nonempty finite trajectories")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples)
    for start in range(0, resamples, 256):
        stop = min(start + 256, resamples)
        means[start:stop] = sum(a[rng.integers(len(a), size=(stop-start, len(a)))].mean(axis=1)
                               for a in arrays) / len(arrays)
    tail = (1-confidence)/2
    low, high = np.quantile(means, (tail, 1-tail))
    return {"estimate": float(np.mean([a.mean() for a in arrays])), "lower": float(low), "upper": float(high),
            "confidence": confidence, "pair_count": sum(len(a) for a in arrays), "strata": len(arrays),
            "resamples": resamples, "seed": seed}


def analyze(manifest):
    from .paired_statistics import serialized_pair_scores
    plan = manifest["plan"]
    root = ROOT / plan["artifact_root"]
    panels, scores = {}, {}
    for job in manifest["jobs"]:
        path = root / "evaluations" / f"{job['id']}.json"
        if not completion_path(plan, "evaluate", job["id"]).exists():
            continue
        summary = read(path)
        seeds, values = serialized_pair_scores(summary)
        panels[job["id"]] = {"job": job, "score": float(values.mean()),
                              "ci95": summary["paired_bootstrap"], "path": str(path)}
        scores[job["id"]] = (seeds, values)
    trends = {}
    common = dict(resamples=plan["evaluation"]["bootstrap_resamples"], seed=plan["analysis_seed"])
    for arm in plan["arms"]:
        points = []
        for n in plan["evaluation"]["rounds"]:
            keys = [f"{r['id']}_{arm['id']}_r{n:03d}" for r in plan["replicates"]]
            if any(k not in scores for k in keys):
                continue
            values = [scores[k][1] for k in keys]
            point = {"round": n, "ci95": stratified_interval(values, **common),
                     "per_replicate": {r["id"]: panels[k] for r, k in zip(plan["replicates"], keys)}}
            if points:
                previous = points[-1]["round"]
                previous_keys = [f"{r['id']}_{arm['id']}_r{previous:03d}" for r in plan["replicates"]]
                if any(scores[a][0] != scores[b][0] for a, b in zip(keys, previous_keys)):
                    raise ValueError("trend panels have unaligned paired seeds")
                differences = [scores[a][1] - scores[b][1] for a, b in zip(keys, previous_keys)]
                point["change_from_previous"] = {"previous_round": previous,
                                                  "ci95": stratified_interval(differences, **common)}
            points.append(point)
        trends[arm["id"]] = points
    diagnostics = []
    for key in manifest["configs"]:
        for n in range(1, plan["rounds"] + 1):
            if completion_path(plan, "train", key, n).exists():
                stage = checkpoint(plan, key, n).parent.parent
                diagnostics.append({"key": key, "round": n,
                    "before": read(stage / "validation_before.json"),
                    "after": read(stage / "validation_after.json"),
                    "source_usage": str(stage / "source_usage.json")})
    result = {"protocol": PROTOCOL,
              "status": "complete" if len(panels) == len(manifest["jobs"])
              and len(diagnostics) == len(manifest["configs"]) * plan["rounds"] else "incomplete",
              "panels": panels, "trends": trends, "training_diagnostics": diagnostics,
              "training_replicates": len(plan["replicates"]),
              "inference_scope": "exploratory trends conditional on these trained trajectories; no optimum or population CI claim"}
    write(root / "analysis.json", result)
    return result


def worker(args):
    if getattr(args, "parent_pid", None) is not None:
        arm_parent_death(args.parent_pid)
    cg = cgroup_path()
    if int((cg / "memory.max").read_text()) != args.memory_max or int((cg / "memory.swap.max").read_text()) != 0:
        raise RuntimeError("worker requires the exact cgroup RAM hard limit and zero swap")
    identity = {"pid": os.getpid(), "cgroup": str(cg), "memory_max": args.memory_max, "swap_max": 0}
    write(args.process_identity, identity)
    manifest = read(args.manifest)
    plan = manifest["plan"]
    try:
        if args.worker == "prepare":
            import torch
            from .hybrid_train import HybridConfig
            from .alphazero_recording import environment_identity
            for raw in manifest["configs"].values():
                HybridConfig.from_dict(raw)
            if not torch.cuda.is_available():
                raise RuntimeError("suite requires usable CUDA")
            free, total = torch.cuda.mem_get_info()
            if free < plan["resources"]["min_gpu_free_mib"] * 1024**2:
                raise RuntimeError("insufficient free GPU memory; release other GPU jobs before resuming")
            write(ROOT / plan["artifact_root"] / "preflight.json", {
                "status": "ready", "environment": environment_identity(),
                "free_gpu_bytes": free, "total_gpu_bytes": total})
        elif args.worker == "train":
            from .hybrid_train import HybridConfig, run_hybrid
            cfg = HybridConfig.from_dict(manifest["configs"][args.key])
            output = ROOT / plan["run_root"] / args.key
            resume = (output / "config.json").exists()
            state = run_hybrid(cfg, output, resume=resume, online_rounds=args.round,
                fork_from=None if resume else ROOT / plan["parent_run"],
                fork_round=None if resume else plan["fork_round"],
                replay_cache_bytes_override=plan["replay_cache_mib"] * 1024**2)
            if state["completed_rounds"] != list(range(args.round)):
                raise ValueError("training did not commit exactly the target round")
            write(completion_path(plan, "train", args.key, args.round),
                  {"completed_round": args.round, "checkpoint": str(checkpoint(plan, args.key, args.round))})
        elif args.worker == "evaluate":
            from .evaluation import FixedEvaluationConfig, run_fixed_evaluation
            job = next(j for j in manifest["jobs"] if j["id"] == args.key)
            path = ROOT / plan["artifact_root"] / "evaluations" / f"{job['id']}.json"
            raw = evaluation_config(manifest, job)
            write(ROOT / plan["artifact_root"] / "evaluation_configs" / f"{job['id']}.json", raw)
            result = run_fixed_evaluation(FixedEvaluationConfig.from_dict(raw), path,
                resume=path.exists() or Path(str(path) + ".games").exists(), require_scored_pairs=True,
                progress_callback=lambda done, total: emit("evaluation_progress", completed=done, total=total),
                summary_every=25)
            if not result.summary["passed"]:
                raise RuntimeError("evaluation has engineering failures; cannot score the panel")
            from .paired_statistics import serialized_pair_scores
            seeds, _ = serialized_pair_scores(result.summary)
            if len(seeds) != job["pairs"]:
                raise ValueError("evaluation does not contain exactly the required scored pairs")
            write(completion_path(plan, "evaluate", args.key), {"pairs": len(seeds), "summary": str(path)})
        elif args.worker == "analyze":
            result = analyze(manifest)
            emit("analysis_complete", status=result["status"])
        else:
            raise ValueError("unknown worker task")
    finally:
        identity["peak_cgroup_bytes"] = int((cg / "memory.peak").read_text())
        write(args.process_identity, identity)


def run_suite(plan, *, prepare_only=False):
    validate_plan(plan)
    artifacts = ROOT / plan["artifact_root"]
    runtime = ROOT / plan["runtime_root"]
    artifacts.mkdir(parents=True, exist_ok=True)
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".suite.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        configs = arm_configs(plan)
        expected = {"protocol": PROTOCOL, "plan": plan, "configs": configs,
                    "evaluation_template": evaluation_template(plan), "jobs": list(evaluation_jobs(plan))}
        path = artifacts / "manifest.json"
        if path.exists():
            if read(path) != expected:
                raise ValueError("frozen suite config changed; restore it or use a new output directory")
        else:
            if digest(ROOT / plan["start_checkpoint"]) != plan["start_sha256"]:
                raise ValueError("starting checkpoint SHA mismatch")
            for name, raw in configs.items():
                write(artifacts / "training_configs" / f"{name}.json", raw)
            git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
            write(artifacts / "code_version.json", {"git_head": git.stdout.strip(),
                  "entrypoint_sha256": digest(Path(__file__)),
                  "training_source_records": "per-run source/ and source_identity records from hybrid_train"})
            write(path, expected)
        state_path = runtime / "suite_state.json"
        if state_path.exists() and read(state_path)["status"] == "complete" and not prepare_only:
            emit("suite_already_complete", analysis=str(artifacts / "analysis.json"))
            return
        run_stage(plan, "prepare", "preflight")
        if prepare_only:
            emit("suite_prepared", manifest=str(path), training_started=False)
            return
        write(state_path, {"status": "running"})
        analysis_path = artifacts / "analysis.json"
        dirty = not analysis_path.exists() or any(
            p.stat().st_mtime_ns > analysis_path.stat().st_mtime_ns
            for p in (runtime / "completed").glob("*.json"))
        try:
            for task, key, n in stage_schedule(plan, expected["jobs"]):
                if task == "analyze":
                    if dirty:
                        run_stage(plan, task, key)
                        dirty = False
                    continue
                marker = completion_path(plan, task, key, n)
                if marker.exists():
                    output = (checkpoint(plan, key, n) if task == "train"
                              else artifacts / "evaluations" / f"{key}.json")
                    if not output.exists():
                        raise ValueError(f"completed stage output is missing: {output}")
                    continue
                run_stage(plan, task, key, round_number=n)
                dirty = True
            if read(artifacts / "analysis.json")["status"] != "complete":
                raise ValueError("suite analysis is incomplete")
            write(state_path, {"status": "complete"})
            emit("suite_complete", analysis=str(artifacts / "analysis.json"))
        except BaseException as error:
            write(state_path, {"status": "interrupted" if isinstance(error, KeyboardInterrupt)
                  else "failed", "error": str(error)})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--worker", choices=("prepare", "train", "evaluate", "analyze"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--key")
    parser.add_argument("--round", type=int)
    parser.add_argument("--process-identity", type=Path)
    parser.add_argument("--memory-max", type=int)
    parser.add_argument("--parent-pid", type=int)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    try:
        if args.worker:
            worker(args)
        else:
            if args.config is None:
                parser.error("--config is required")
            if Path.cwd().resolve() != ROOT:
                raise ValueError("run from the repository root")
            arm_parent_death(os.getppid())
            run_suite(read(args.config), prepare_only=args.prepare_only)
    except KeyboardInterrupt:
        emit("suite_interrupted", resume="rerun the same command")
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        emit("suite_failed", error=str(error), resume="fix the reported cause and rerun the same command")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
