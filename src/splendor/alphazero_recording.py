"""Opt-in, file-backed AZ experiment records. No model/sampling decisions."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def append_json(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unavailable(reason: str) -> dict[str, Any]:
    return {"available": False, "reason": reason}


def distribution(values: Any) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return unavailable("empty group")
    return {"count": len(array), "mean": float(array.mean()),
            **{f"p{p}": float(np.percentile(array, p)) for p in (50, 90, 99)}, "max": float(array.max())}


@dataclass(frozen=True, slots=True)
class DiagnosticsConfig:
    artifact_dir: str
    fixed_positions: int = 2048
    fixed_seed: int = 11340000
    resource_interval_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.artifact_dir or not 0 < self.fixed_positions <= 2048:
            raise ValueError("diagnostics requires artifact_dir and 1..2048 fixed_positions")
        if not np.isfinite(self.resource_interval_seconds) or self.resource_interval_seconds <= 0:
            raise ValueError("resource interval must be positive")


def source_identity(root: Path, destination: Path | None = None) -> dict[str, Any]:
    files = sorted([*root.glob("src/splendor/**/*.py"), *root.glob("scripts/*.py")])
    hashes = {str(path.relative_to(root)): sha256(path) for path in files}
    if destination is not None:
        for path in files:
            target = destination / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout
    diff = git("diff", "HEAD", "--", "src", "scripts", "configs", "tests", "docs", "artifacts")
    if destination is not None:
        (destination / "dirty.patch").write_text(diff, encoding="utf-8")
    return {"git_head": git("rev-parse", "HEAD").strip(), "git_status": git("status", "--short"),
            "dirty_diff_sha256": hashlib.sha256(diff.encode()).hexdigest(), "source_files": hashes,
            "source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
            "native_extensions": {str(p): sha256(p) for p in root.glob("src/splendor/_native_search*.so")}}


def environment_identity() -> dict[str, Any]:
    command = ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]
    gpu = subprocess.run(command, capture_output=True, text=True, check=False, timeout=10) if shutil.which("nvidia-smi") else None
    return {"python": sys.version, "executable": sys.executable, "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "torch_threads": torch.get_num_threads(), "cpu_count": os.cpu_count(),
            "platform": platform.platform(), "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
            "meminfo": Path("/proc/meminfo").read_text(),
            "gpu": gpu.stdout.strip() if gpu and gpu.returncode == 0 else unavailable("nvidia-smi unavailable in this execution environment"),
            "performance_environment": {k: v for k, v in os.environ.items() if k.startswith(("SPLENDOR_", "OMP_", "MKL_", "TORCH_", "CUDA_", "PYTORCH_")) or k in ("PYTHONPATH", "CONDA_DEFAULT_ENV")},
            "compile_cache": {"path": os.environ.get("TORCHINDUCTOR_CACHE_DIR", "default /tmp/torchinductor_<user>"),
                              "warm_state": "unknown; compilation hit/miss counters in owner status are authoritative"}}


class ReferenceWriter:
    """Batch hashes, with optional exact int64 references and attempt logs."""
    def __init__(self, root: Path, session_id: str, batch_size: int, identity: Mapping[str, Any],
                 *, persist_records: bool = True):
        root.mkdir(parents=True, exist_ok=True)
        self.persist_records = persist_records
        self.path = root / f"{session_id}{'.bin' if persist_records else '.json'}"
        self.handle = self.path.open("ab") if persist_records else None
        self.metrics_handle = (self.path.with_suffix(".steps.jsonl").open("a", encoding="utf-8")
                               if persist_records else None)
        write_json(self.path.with_suffix(".json"), {
                   "format": "az-reference-i64le-v1" if persist_records else "az-reference-sha256-only-v1",
                   "columns": ["step", "round", "game", "decision"],
                   "batch_size": batch_size, "session_id": session_id, **identity})

    def append(self, step: int, references: Any) -> str:
        rows = np.asarray([(step, r.round_index, r.game_id, r.decision_index) for r in references], dtype="<i8")
        data = rows.tobytes()
        if self.persist_records:
            self.handle.write(data)
        return hashlib.sha256(data).hexdigest()

    def flush(self) -> None:
        if not self.persist_records:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.metrics_handle.flush()
        os.fsync(self.metrics_handle.fileno())

    def record_step(self, record: Mapping[str, Any]) -> None:
        if self.persist_records:
            self.metrics_handle.write(json.dumps(dict(record), sort_keys=True, allow_nan=False) + "\n")

    def close(self) -> None:
        if self.persist_records:
            self.flush()
            self.handle.close()
            self.metrics_handle.close()

    def archive_metrics(self, path: Path) -> None:
        if self.persist_records:
            shutil.copyfile(path, self.path.with_suffix(".metrics.jsonl"))


def summarize_references(root: Path, committed_step: int, round_index: int, pools: Mapping[str, int]) -> dict[str, Any]:
    batches: dict[int, np.ndarray] = {}
    attempted = torn_bytes = 0
    for path in sorted(root.glob("*.bin")):
        meta = read_json(path.with_suffix(".json"))
        raw = path.read_bytes()
        torn_bytes += len(raw) % 32
        rows = np.frombuffer(raw[:len(raw) // 32 * 32], dtype="<i8").reshape(-1, 4)
        attempted += len(rows)
        for step in np.unique(rows[:, 0]):
            block = rows[rows[:, 0] == step]
            if len(block) != meta["batch_size"] or step > committed_step:
                continue
            if step in batches and not np.array_equal(batches[step], block):
                raise ValueError(f"resume reference sequence differs at committed step {step}")
            batches[int(step)] = block.copy()
    expected = set(range(1, committed_step + 1))
    if set(batches) != expected:
        return unavailable(f"reference records missing committed steps {sorted(expected - batches.keys())}; legacy data cannot recover exact coverage")
    rows = np.concatenate([batches[s] for s in sorted(batches)]) if batches else np.empty((0, 4), dtype="<i8")
    counts = Counter(tuple(map(int, row[1:])) for row in rows)
    keys = np.asarray(list(counts), dtype="<i8").reshape(-1, 3)
    uses = np.asarray(list(counts.values()), dtype="<i8")
    count_path = root / "committed_counts.npz"
    np.savez_compressed(count_path, references=keys, counts=uses)
    result: dict[str, Any] = {"committed_steps": committed_step, "committed_draws": len(rows), "attempted_draws": attempted,
        "repeated_or_uncommitted_draws": attempted - len(rows), "torn_bytes": torn_bytes,
        "sequence_sha256": hashlib.sha256(rows.tobytes()).hexdigest(), "counts_path": str(count_path.resolve()), "counts_sha256": sha256(count_path)}
    for name, current in (("current", True), ("history", False)):
        selected = uses[(keys[:, 0] == round_index) if current else (keys[:, 0] < round_index)]
        size = pools[name]
        total = int(selected.sum())
        # Distribution includes uncovered positions, without allocating a huge zero array.
        histogram = Counter(map(int, selected))
        if size:
            histogram[0] = size - len(selected)
        result[name] = {"pool_positions": size, "draws": total, "covered_positions": len(selected),
            "mean_uses": total / size if size else None, "unseen_fraction": (size - len(selected)) / size if size else None,
            "use_count_histogram": dict(sorted(histogram.items()))}
    return result


class InvocationRecorder:
    def __init__(self, output: Path, config: DiagnosticsConfig, state: Mapping[str, Any], frozen_config: Mapping[str, Any]):
        self.session_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:12]}"
        self.root = output / "recording"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "sessions" / f"{self.session_id}.json"
        self.started = time.perf_counter()
        self.phase = "initialize"
        self.round_index = None
        self.stop = threading.Event()
        self.error: BaseException | None = None
        self.record = {"session_id": self.session_id, "start_unix": time.time(), "command": [sys.executable, *sys.argv],
                       "status": "running", "recovery_start": dict(state), "phases": [],
                       "config_sha256": hashlib.sha256(json.dumps(frozen_config, sort_keys=True).encode()).hexdigest()}
        write_json(self.path, self.record)
        source_root = Path(__file__).resolve().parents[2]
        identity = source_identity(source_root, self.root / "source" / self.session_id)
        identity["config_file_sha256"] = {str(output / "config.json"): sha256(output / "config.json")} if (output / "config.json").exists() else {}
        if "--config" in sys.argv:
            index = sys.argv.index("--config")
            path = Path(sys.argv[index + 1]).resolve()
            identity["config_file_sha256"][str(path)] = sha256(path)
        write_json(self.root / "identities" / f"{self.session_id}.json", {**identity, "environment": environment_identity(), "config": frozen_config})
        self.interval = config.resource_interval_seconds
        self.thread = threading.Thread(target=self._sample_resources, daemon=True)
        self.thread.start()

    @contextmanager
    def stage(self, name: str, round_index: int | None = None):
        old_phase, old_round = self.phase, self.round_index
        self.phase, self.round_index = name, round_index
        phase = {"phase": name, "round_index": round_index, "start_unix": time.time(), "status": "running"}
        self.record["phases"].append(phase)
        write_json(self.path, self.record)
        started = time.perf_counter()
        try:
            yield phase
        except BaseException as error:
            phase.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error_type=type(error).__name__, error=str(error))
            raise
        else:
            phase["status"] = "complete"
        finally:
            phase["seconds"] = time.perf_counter() - started
            phase["end_unix"] = time.time()
            write_json(self.path, self.record)
            self.phase, self.round_index = old_phase, old_round

    def event(self, record: Mapping[str, Any]) -> None:
        append_json(self.root / "events.jsonl", {"session_id": self.session_id, "unix_time": time.time(), **record})

    def _sample_resources(self) -> None:
        try:
            previous_cpu: dict[int, int] = {}
            previous_time = None
            while not self.stop.is_set():
                processes = {}
                for entry in Path("/proc").iterdir():
                    if not entry.name.isdigit():
                        continue
                    try:
                        stat = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                        processes[int(entry.name)] = (int(stat[1]), int(stat[11]) + int(stat[12]), entry)
                    except (OSError, ValueError, IndexError):
                        continue  # Process exited between enumeration and read.
                pids = {os.getpid()}
                while True:
                    expanded = pids | {pid for pid, (parent, _, _) in processes.items() if parent in pids}
                    if expanded == pids:
                        break
                    pids = expanded
                rss = pss = ticks = observed = 0
                for pid in pids:
                    if pid not in processes:
                        continue
                    _, cpu, entry = processes[pid]
                    ticks += cpu
                    try:
                        lines = (entry / "smaps_rollup").read_text().splitlines()
                        fields = {line.split(":")[0]: line.split()[1] for line in lines if ":" in line}
                        rss += int(fields["Rss"]) * 1024
                        pss += int(fields["Pss"]) * 1024
                        observed += 1
                    except (OSError, KeyError):
                        continue
                row = {"session_id": self.session_id, "unix_time": time.time(), "phase": self.phase, "round_index": self.round_index,
                       "processes": len(pids), "memory_observed_processes": observed, "tree_rss_bytes": rss, "tree_pss_bytes": pss,
                       "alive_tree_cpu_seconds": ticks / os.sysconf("SC_CLK_TCK"), "cpu_scope": "cumulative alive processes; exited children are not retained"}
                current_cpu = {pid: processes[pid][1] for pid in pids if pid in processes}
                now = time.monotonic()
                row["tree_cpu_percent_one_core"] = (sum(max(0, ticks - previous_cpu.get(pid, 0)) for pid, ticks in current_cpu.items())
                    / os.sysconf("SC_CLK_TCK") / (now - previous_time) * 100) if previous_time is not None else unavailable("first sample has no interval")
                row["cpu_interval_scope"] = "alive process deltas; newly observed process counted since birth; exited-process tail is unavailable; one core=100%"
                previous_cpu, previous_time = current_cpu, now
                if shutil.which("nvidia-smi"):
                    gpu = subprocess.run(["nvidia-smi", "--query-gpu=uuid,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                    row["gpu_csv"] = gpu.stdout.strip() if gpu.returncode == 0 else unavailable(gpu.stderr.strip())
                else:
                    row["gpu_csv"] = unavailable("nvidia-smi missing")
                if torch.cuda.is_initialized():
                    row["torch_allocator"] = {"allocated_bytes": torch.cuda.memory_allocated(), "reserved_bytes": torch.cuda.memory_reserved(),
                                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                append_json(self.root / "resources.jsonl", row)
                self.stop.wait(self.interval)
        except Exception as error:
            self.error = error

    def finish(self, state: Mapping[str, Any], error: BaseException | None = None) -> None:
        self.stop.set()
        self.thread.join(timeout=10)
        self.record.update(end_unix=time.time(), wall_seconds=time.perf_counter() - self.started,
                           status="complete" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                           end_state=dict(state), error_type=type(error).__name__ if error else None, error=str(error) if error else None,
                           resource_error=str(self.error) if self.error else None)
        write_json(self.path, self.record)


def human_event(record: Mapping[str, Any]) -> str:
    event = record.get("event", "event")
    prefix = f"[第 {int(record['round_index']) + 1} 轮] " if "round_index" in record else ""
    if event == "collect_progress":
        return prefix + f"采集 {record['completed_games']}/{record['total_games']} ({record['completed_games']/record['total_games']:.1%}) {record['games_per_minute']:.2f} 局/分 本段 {record['elapsed_seconds']:.1f}s"
    if event == "train_progress":
        cache = record['replay_cache']
        delta = record['cache_delta']
        requests = delta['hits'] + delta['misses']
        hit = f"{delta['hits']/requests:.1%}" if requests else "无请求"
        return prefix + f"训练 {record['round_train_step']}/{record['steps_target']} 累计 {record['total_train_step']} | CE {record['policy_loss']:.4f} H目标 {record['target_entropy'] if record['target_entropy'] is not None else '无有效策略样本'} KL {record['policy_kl']} value {record['value_loss']:.4f} | {record['steps_per_second']:.2f} 步/s ETA {record['eta_seconds']}s | cache {cache['cache_bytes']/1024**3:.2f}/{cache['capacity_bytes']/1024**3:.2f}GiB 本批命中 {hit}"
    if event == "evaluate_progress":
        return prefix + f"评测 {record['panel']} {record['completed_pairs']}/{record['total_pairs']} pairs {record['pairs_per_minute']:.2f} pairs/分"
    if event == "collect_start":
        return prefix + f"采集开始，共 {record['total_games']} 局，已有提交 {record['already_completed']} 局"
    if event == "train_start":
        return prefix + f"训练开始，目标 {record['steps_target']} 步，设备 {record['device']}，继承optimizer={record['continued']}"
    if event == "evaluate_complete":
        panels = []
        for panel in ("mcts_vs_parent", "mcts_vs_initial", "greedy_vs_initial"):
            if panel not in record:
                continue
            values = record[panel]
            score = values['win_rate_score']
            bootstrap = values.get('paired_bootstrap')
            ci = f" CI95% [{bootstrap['lower']:.1%}, {bootstrap['upper']:.1%}]" if bootstrap else " CI不可用"
            panels.append(f"{panel}: {score:.1%}{ci} ({values['completed_pairs']} pairs)")
        return prefix + "评测完成 | " + " | ".join(panels)
    if event == "loop_finished":
        return f"循环完成 {len(record['completed_rounds'])}/{record['num_rounds']} 轮，最终模型 {record['latest_checkpoint']['path']}，输出 {record['output']}"
    if event == "round_complete":
        return prefix + f"完成 {record['rounds_completed']}/{record['num_rounds']} 轮，本段 {record['wall_seconds_this_session']:.1f}s，模型 {record['theta_out']}"
    if event == "diagnostics_start":
        return prefix + "离线诊断开始：全量checked replay、同集前后与固定集checkpoint曲线"
    if event == "diagnostics_complete":
        return prefix + f"离线诊断完成，本段 {record['seconds_this_session']:.1f}s，当轮/固定局面 {record['current_positions']}/{record['fixed_positions']}"
    return prefix + json.dumps(dict(record), ensure_ascii=False, sort_keys=True)
