"""AlphaZero-style policy/value loop: collect -> train -> evaluate -> repeat.

One round is:

``freeze theta_r -> MCTS self-play collection D_r -> update replay buffer ->
continue training from theta_r to theta_(r+1) -> short fixed evaluation ->
next round collects with theta_(r+1)``

This is an AlphaZero-shaped loop adapted to imperfect-information Splendor
(single determinization per root, frozen search semantics untouched); it is
not a re-implementation of the original perfect-information paper.  The loop
owns scheduling, state and recovery only: collection stays in
``selfplay_fast``, training stays in ``alphazero_train``, and evaluation stays
in ``evaluation``.

Recovery rules:

* collection resumes through ``selfplay_fast.generate`` and skips games that
  were already committed;
* a round's training is considered complete when its final AlphaZero
  checkpoint matches the round, parent identity, frozen replay digest and
  step target, so an interrupted evaluation never restarts training;
* a partially trained round resumes from its last committed step checkpoint
  with the saved optimizer, sampling RNG and step counters;
* the optimizer and the global step counter continue across round boundaries,
  so every checkpoint carries one monotonically increasing training step;
* a round counts as complete only when both evaluation panels finished
  without engineering failures; a truncation, exception or illegal action
  stops the loop before the next round collects;
* ``loop_state.json`` (including the latest-checkpoint pointer) is committed
  atomically after each phase, and a finished loop resumes to a no-op.  When
  the file is missing, a round is recovered as complete only if its final
  checkpoint *and* both complete evaluation summaries exist.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from torch.utils.tensorboard import SummaryWriter

from .alphazero_checkpoint import load_alphazero_checkpoint
from .alphazero_replay import ReplayRun, build_replay_index
from .alphazero_train import (
    AlphazeroLearner, LearnerConfig, NonFiniteTrainingError, run_training_round,
)
from .evaluation import (
    FixedEvaluationConfig, SerializablePolicySpec, run_fixed_evaluation,
)
from .mcts_policy import MCTSConfig
from .selfplay import TemperatureSchedule, derive_seed
from .selfplay_fast import (
    ExecutionConfig, FastSelfPlayConfig, generate as generate_selfplay,
)
from .selfplay_data import SelfPlaySample

LOOP_PROTOCOL = "alphazero-loop-v1"
STATE_PROTOCOL = "alphazero-loop-state-v1"
SEED_PROTOCOL = "splendor-alphazero-loop-seed-v1"
SELFPLAY_PROTOCOL = "mcts-selfplay-v2"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


class EvaluationEngineeringFailure(RuntimeError):
    """A fixed evaluation finished with truncated or errored pairs."""

    def __init__(
        self, message: str, *, round_index: int, panel: str, details: Mapping[str, Any]
    ) -> None:
        super().__init__(message)
        self.round_index = round_index
        self.panel = panel
        self.details = dict(details)


class ProgressThrottle:
    """Rate-limit periodic progress lines; the first call always passes."""

    def __init__(
        self, interval_seconds: float, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not math.isfinite(interval_seconds) or interval_seconds < 0:
            raise ValueError("progress interval must be a finite nonnegative number")
        self.interval_seconds = float(interval_seconds)
        self._clock = clock
        self._last_emit: float | None = None

    def ready(self) -> bool:
        now = self._clock()
        if self._last_emit is not None and now - self._last_emit < self.interval_seconds:
            return False
        self._last_emit = now
        return True


def _print_event(record: Mapping[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _committed_games(run_root: Path) -> int:
    """Count committed archives without validating them (progress only)."""

    games_root = run_root / "games"
    return len(list(games_root.glob("game-*.npz"))) if games_root.is_dir() else 0


@dataclass(frozen=True, slots=True)
class LoopSeeds:
    """Independent seed domains for collection, split and training sampling.

    Evaluation panels carry their own ``seed_start`` / ``policy_seed_base`` so
    a fixed DEV panel can be reused across rounds to read trends.
    """

    collection: int
    split: int
    sampling: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"seed domain {name} must be an integer")

    def collection_for(self, round_index: int) -> int:
        return derive_seed(self.collection, round_index, 0, f"{SEED_PROTOCOL}:collection")

    def split_for(self, round_index: int) -> int:
        return derive_seed(self.split, round_index, 0, f"{SEED_PROTOCOL}:split")

    def sampling_for(self, round_index: int) -> int:
        return derive_seed(self.sampling, round_index, 0, f"{SEED_PROTOCOL}:sampling")


@dataclass(frozen=True, slots=True)
class SelfPlaySection:
    search_config: dict[str, Any]
    execution: dict[str, Any]
    add_root_noise: bool
    temperature_schedule: dict[str, Any]
    validation_fraction: float
    max_completed_turns: int
    protocol: str = SELFPLAY_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != SELFPLAY_PROTOCOL:
            raise ValueError("self-play protocol must be mcts-selfplay-v2")
        MCTSConfig(**self.search_config)
        ExecutionConfig(**self.execution)
        TemperatureSchedule(**self.temperature_schedule)
        if type(self.add_root_noise) is not bool:
            raise ValueError("add_root_noise must be bool")
        if not isinstance(self.validation_fraction, (int, float)) or not 0 <= self.validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if type(self.max_completed_turns) is not int or self.max_completed_turns <= 0:
            raise ValueError("max_completed_turns must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationPanel:
    pairs: int
    seed_start: int
    policy_seed_base: int
    search_config: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.pairs) is not int or self.pairs <= 0:
            raise ValueError("evaluation pairs must be a positive integer")
        for name in ("seed_start", "policy_seed_base"):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.search_config is not None:
            MCTSConfig(**self.search_config)

    @property
    def game_seeds(self) -> tuple[int, ...]:
        return tuple(range(self.seed_start, self.seed_start + self.pairs))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationSection:
    workers: int
    device: str
    adjudicate_deterministic_policy_cycles: bool
    mcts_vs_parent: EvaluationPanel
    greedy_vs_initial: EvaluationPanel
    mcts_backend: str = "python"
    mcts_execution: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if type(self.workers) is not int or self.workers <= 0:
            raise ValueError("evaluation workers must be a positive integer")
        if self.mcts_backend not in ("python", "native"):
            raise ValueError("mcts_backend must be python or native")
        if self.device not in ("cpu", "cuda"):
            raise ValueError("evaluation device must be cpu or cuda")
        if self.workers > 1 and self.device != "cpu":
            raise ValueError("parallel evaluation requires CPU inference")
        if self.mcts_vs_parent.search_config is None:
            raise ValueError("the MCTS-vs-parent panel requires a search configuration")
        if self.greedy_vs_initial.search_config is not None:
            raise ValueError("the greedy-vs-initial panel cannot carry search configuration")
        if type(self.adjudicate_deterministic_policy_cycles) is not bool:
            raise ValueError("adjudicate_deterministic_policy_cycles must be bool")
        if self.mcts_execution is not None:
            from .evaluation_batch import BatchEvaluationConfig
            BatchEvaluationConfig(**self.mcts_execution)
            if self.mcts_backend != "native" or self.workers != 1:
                raise ValueError("batched MCTS evaluation requires native backend and one owner")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.mcts_backend == "python":
            value.pop("mcts_backend")
        if self.mcts_execution is None:
            value.pop("mcts_execution")
        return value


@dataclass(frozen=True, slots=True)
class LoopConfig:
    run_id: str
    num_rounds: int
    new_games_per_round: int
    initial_checkpoint: str
    replay_max_games: int
    replay_cache_bytes: int
    seeds: LoopSeeds
    selfplay: SelfPlaySection
    learner: LearnerConfig
    evaluation: EvaluationSection
    protocol: str = LOOP_PROTOCOL

    def __post_init__(self) -> None:
        if self.protocol != LOOP_PROTOCOL or not self.run_id:
            raise ValueError("invalid AlphaZero loop protocol or run_id")
        for name in ("num_rounds", "new_games_per_round", "replay_max_games", "replay_cache_bytes"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not Path(self.initial_checkpoint).is_file():
            raise FileNotFoundError(self.initial_checkpoint)

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "run_id": self.run_id,
            "num_rounds": self.num_rounds,
            "new_games_per_round": self.new_games_per_round,
            "initial_checkpoint": str(Path(self.initial_checkpoint).resolve()),
            "replay_max_games": self.replay_max_games,
            "replay_cache_bytes": self.replay_cache_bytes,
            "seeds": asdict(self.seeds),
            "selfplay": self.selfplay.to_dict(),
            "learner": asdict(self.learner),
            "evaluation": asdict(self.evaluation),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LoopConfig:
        if raw.get("protocol") != LOOP_PROTOCOL:
            raise ValueError("unsupported AlphaZero loop protocol")
        evaluation = raw["evaluation"]
        return cls(
            run_id=str(raw["run_id"]),
            num_rounds=raw["num_rounds"],
            new_games_per_round=raw["new_games_per_round"],
            initial_checkpoint=str(Path(raw["initial_checkpoint"]).resolve()),
            replay_max_games=raw["replay_max_games"],
            replay_cache_bytes=raw["replay_cache_bytes"],
            seeds=LoopSeeds(**raw["seeds"]),
            selfplay=SelfPlaySection(**raw["selfplay"]),
            learner=LearnerConfig(**raw["learner"]),
            evaluation=EvaluationSection(
                workers=evaluation["workers"],
                device=evaluation["device"],
                adjudicate_deterministic_policy_cycles=evaluation["adjudicate_deterministic_policy_cycles"],
                mcts_vs_parent=EvaluationPanel(**evaluation["mcts_vs_parent"]),
                greedy_vs_initial=EvaluationPanel(**evaluation["greedy_vs_initial"]),
                mcts_backend=evaluation.get("mcts_backend", "python"),
                mcts_execution=evaluation.get("mcts_execution"),
            ),
        )


def _file_identity(path: Path, *, kind: str, round_index: int, total_train_step: int) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "kind": kind,
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "round_index": round_index,
        "total_train_step": total_train_step,
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _round_root(output: Path, round_index: int) -> Path:
    return output / "rounds" / f"round-{round_index:03d}"


def _read_checkpoint_identity(path: Path) -> dict[str, Any]:
    """Read an AlphaZero checkpoint's identity; used only at round boundaries."""

    payload = load_alphazero_checkpoint(path)
    return {
        "kind": "alphazero",
        "path": str(path.resolve()),
        "sha256": _sha256_file(path),
        "model_sha256": payload.model_sha256,
        "round_index": payload.round_index,
        "round_train_step": payload.round_train_step,
        "total_train_step": payload.total_train_step,
        "round_steps_target": payload.round_steps_target,
        "parent_checkpoint": payload.parent_checkpoint,
        "replay_identity": payload.replay_identity,
    }


def _final_identity(output: Path, round_index: int) -> dict[str, Any] | None:
    path = _round_root(output, round_index) / "checkpoints" / "final.pt"
    if not path.is_file():
        return None
    return _read_checkpoint_identity(path)


class LoopRunner:
    """Owns one output directory; methods keep the on-disk facts authoritative."""

    def __init__(
        self,
        config: LoopConfig,
        output: Path,
        *,
        progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
    ) -> None:
        if not math.isfinite(progress_interval_seconds) or progress_interval_seconds < 0:
            raise ValueError("progress interval must be a finite nonnegative number")
        self.config = config
        self.progress_interval_seconds = float(progress_interval_seconds)
        self.output = Path(output).resolve()
        self.rounds_root = self.output / "rounds"
        self.state_path = self.output / "loop_state.json"
        self.initial_identity = _file_identity(
            Path(config.initial_checkpoint), kind="initial", round_index=-1, total_train_step=0
        )
        fingerprint_payload = config.to_dict() | {
            "initial_checkpoint_sha256": self.initial_identity["sha256"]
        }
        self.config_fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.tensorboard: SummaryWriter | None = None
        self.state: dict[str, Any] = {}

    # ------------------------------------------------------------------ setup

    def initialize(self, *, resume: bool) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        config_path = self.output / "config.json"
        snapshot = {
            "protocol": LOOP_PROTOCOL,
            "fingerprint": self.config_fingerprint,
            "config": self.config.to_dict(),
        }
        if config_path.exists():
            if not resume:
                raise ValueError(f"{self.output} already contains a loop; pass --resume")
            existing = _load_json(config_path)
            if existing.get("fingerprint") != snapshot["fingerprint"]:
                raise ValueError("existing loop was created with a different configuration")
        else:
            if any(self.output.iterdir()):
                raise ValueError(f"nonempty output directory without config.json: {self.output}")
            if resume:
                raise ValueError("cannot resume a loop that has not been initialized")
            _atomic_write_json(config_path, snapshot)
        if self.state_path.exists():
            self.state = _load_json(self.state_path)
            if self.state.get("config_fingerprint") != snapshot["fingerprint"]:
                raise ValueError("loop_state.json belongs to a different configuration")
            if self.state.get("protocol") != STATE_PROTOCOL:
                raise ValueError("unsupported AlphaZero loop state protocol")
        else:
            self.state = self._recover_state()
            self._save_state()
        return self.state

    def _recover_state(self) -> dict[str, Any]:
        """Rebuild the state when the file is missing but round artifacts exist."""

        rounds: dict[str, Any] = {}
        completed: list[int] = []
        latest = dict(self.initial_identity)
        for round_index in range(self.config.num_rounds):
            final = _final_identity(self.output, round_index)
            # A committed final checkpoint only proves training finished; the
            # round is not complete until both evaluation panels passed.
            if final is None or not self._evaluation_complete(round_index):
                break
            entry = {
                "phase": "complete",
                "final_checkpoint": final,
                "train_step": final.get("round_train_step", 0),
                "steps_target": final.get("round_steps_target", 0),
                "completed_at": None,
                "recovered": True,
            }
            rounds[str(round_index)] = entry
            completed.append(round_index)
            latest = final | {"kind": "alphazero"}
        return {
            "protocol": STATE_PROTOCOL,
            "run_id": self.config.run_id,
            "config_fingerprint": self.config_fingerprint,
            "initial_checkpoint": self.initial_identity,
            "latest_checkpoint": latest,
            "completed_rounds": completed,
            "rounds": rounds,
            "updated_at": _now_utc(),
        }

    def _save_state(self) -> None:
        self.state["updated_at"] = _now_utc()
        _atomic_write_json(self.state_path, self.state)

    def _evaluation_complete(self, round_index: int) -> bool:
        """Both summaries must exist, cover every pair and pass engineering checks."""

        for panel in ("mcts_vs_parent", "greedy_vs_initial"):
            path = _round_root(self.output, round_index) / "evaluation" / f"{panel}.json"
            if not path.is_file():
                return False
            summary = _load_json(path)
            if (
                summary.get("passed") is not True
                or summary.get("completed_pairs") != summary.get("total_pairs")
            ):
                return False
        return True

    # ------------------------------------------------------------- round flow

    def run(self) -> dict[str, Any]:
        if len(self.state["completed_rounds"]) >= self.config.num_rounds:
            print(
                json.dumps(
                    {"event": "loop_no_work", "completed_rounds": self.config.num_rounds},
                    sort_keys=True,
                ),
                flush=True,
            )
            return self._summary()
        self.tensorboard = SummaryWriter(log_dir=str(self.output / "tensorboard"))
        try:
            for round_index in range(self.config.num_rounds):
                if round_index in self.state["completed_rounds"]:
                    continue
                theta_in = self._resolve_theta_in(round_index)
                self._run_round(round_index, theta_in)
        finally:
            self.tensorboard.flush()
            self.tensorboard.close()
            self.tensorboard = None
        return self._summary()

    def _resolve_theta_in(self, round_index: int) -> dict[str, Any]:
        if round_index == 0:
            return dict(self.initial_identity)
        recorded = self.state["rounds"].get(str(round_index - 1), {}).get("final_checkpoint")
        if recorded is not None and Path(recorded["path"]).is_file():
            return dict(recorded)
        recovered = _final_identity(self.output, round_index - 1)
        if recovered is None:
            raise RuntimeError(
                f"round {round_index} has no parent checkpoint from round {round_index - 1}"
            )
        return recovered | {"kind": "alphazero"}

    def _run_round(self, round_index: int, theta_in: dict[str, Any]) -> None:
        round_root = _round_root(self.output, round_index)
        started = time.perf_counter()
        phase = "collect"
        phases: list[str] = []
        try:
            collect = self._collect_round(round_root, round_index, theta_in)
            phases.append("collect(skipped)" if collect.get("resumed") else "collect")
            self._record_round_phase(round_index, "train", theta_in, collect)
            phase = "train"
            train = self._train_round(round_root, round_index, theta_in)
            phases.append("train(skipped)" if train.get("resumed") else "train")
            theta_out = _final_identity(self.output, round_index)
            if theta_out is None:
                raise RuntimeError("training finished without a final checkpoint")
            phase = "evaluate"
            evaluation = self._evaluate_round(round_root, round_index, theta_out, theta_in)
            phases.append("evaluate")
        except Exception as error:
            self._write_failure(round_index, phase, theta_in, error)
            raise
        summary = {
            "round_index": round_index,
            "theta_in": theta_in,
            "theta_out": theta_out,
            "collect": collect,
            "train": train,
            "evaluation": evaluation,
            # Scope matters: a resumed invocation only performs the phases it
            # actually ran, and this file is rewritten by the latest session.
            "session_phases": phases,
            "wall_seconds_this_session": time.perf_counter() - started,
            "completed_at": _now_utc(),
        }
        _atomic_write_json(round_root / "round_summary.json", summary)
        self.state["rounds"][str(round_index)] = {
            "phase": "complete",
            "theta_in": theta_in,
            "final_checkpoint": theta_out,
            "train_step": train["steps_completed"],
            "steps_target": train["steps_target"],
            "validation": train["validation"],
            "evaluation": {
                "mcts_vs_parent": self._evaluation_record(evaluation["mcts_vs_parent"]),
                "greedy_vs_initial": self._evaluation_record(evaluation["greedy_vs_initial"]),
            },
            "session_phases": phases,
            "wall_seconds_this_session": summary["wall_seconds_this_session"],
            "completed_at": summary["completed_at"],
        }
        if round_index not in self.state["completed_rounds"]:
            self.state["completed_rounds"].append(round_index)
        self.state["latest_checkpoint"] = theta_out
        self._save_state()
        print(
            json.dumps(
                {
                    "event": "round_complete",
                    "round_index": round_index,
                    "rounds_completed": len(self.state["completed_rounds"]),
                    "num_rounds": self.config.num_rounds,
                    "theta_out": theta_out["path"],
                    "session_phases": phases,
                    "wall_seconds_this_session": summary["wall_seconds_this_session"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    def _record_round_phase(
        self, round_index: int, phase: str, theta_in: Mapping[str, Any], collect: Mapping[str, Any]
    ) -> None:
        self.state["rounds"][str(round_index)] = {
            "phase": phase,
            "theta_in": dict(theta_in),
            "collect": dict(collect),
        }
        self._save_state()

    def _collect_round(
        self, round_root: Path, round_index: int, theta_in: Mapping[str, Any]
    ) -> dict[str, Any]:
        selfplay_dir = round_root / "selfplay"
        config = FastSelfPlayConfig(
            protocol=self.config.selfplay.protocol,
            run_id=f"{self.config.run_id}-round-{round_index:03d}",
            round_index=round_index,
            checkpoint=str(theta_in["path"]),
            num_games=self.config.new_games_per_round,
            seed=self.config.seeds.collection_for(round_index),
            search_config=MCTSConfig(**self.config.selfplay.search_config),
            execution=ExecutionConfig(**self.config.selfplay.execution),
            add_root_noise=self.config.selfplay.add_root_noise,
            temperature_schedule=TemperatureSchedule(**self.config.selfplay.temperature_schedule),
            validation_fraction=self.config.selfplay.validation_fraction,
            split_seed=self.config.seeds.split_for(round_index),
            max_completed_turns=self.config.selfplay.max_completed_turns,
        )
        summary_path = selfplay_dir / "summary.json"
        if summary_path.is_file():
            summary = _load_json(summary_path)
            if (
                summary.get("completed_games") == config.num_games
                and summary.get("missing_games") == 0
            ):
                summary = summary | {"resumed": True}
                _print_event(
                    {"event": "collect_skipped", "round_index": round_index, "games": config.num_games}
                )
                self._log_collection_metrics(theta_in, summary)
                return summary
        baseline = _committed_games(selfplay_dir)
        started = time.perf_counter()
        throttle = ProgressThrottle(self.progress_interval_seconds)
        _print_event(
            {
                "event": "collect_start",
                "round_index": round_index,
                "total_games": config.num_games,
                "already_completed": baseline,
                "resume": selfplay_dir.exists(),
            }
        )

        def on_collect_progress(completed: int, total: int) -> None:
            if not throttle.ready():
                return
            elapsed = max(time.perf_counter() - started, 1e-9)
            fresh = max(completed - baseline, 0)
            _print_event(
                {
                    "event": "collect_progress",
                    "round_index": round_index,
                    "completed_games": completed,
                    "total_games": total,
                    "new_games_this_session": fresh,
                    "elapsed_seconds": round(elapsed, 3),
                    "games_per_minute": round(fresh / elapsed * 60.0, 3),
                }
            )
            if self.tensorboard is not None:
                # Intra-round progress runs on its own elapsed-seconds axis, so
                # it is tagged per round instead of overwriting earlier rounds.
                prefix = f"collect/round-{round_index:03d}"
                self.tensorboard.add_scalar(f"{prefix}/games_completed", completed, int(elapsed))
                self.tensorboard.add_scalar(
                    f"{prefix}/games_per_minute", fresh / elapsed * 60.0, int(elapsed)
                )

        summary = generate_selfplay(
            config,
            selfplay_dir,
            resume=selfplay_dir.exists(),
            progress_callback=on_collect_progress,
        )
        summary = summary | {"resumed": False, "wall_seconds_this_session": time.perf_counter() - started}
        self._log_collection_metrics(theta_in, summary)
        _print_event(
            {
                "event": "collect_complete",
                "round_index": round_index,
                "completed_games": summary.get("completed_games"),
                "statistics_scope": summary.get("statistics_scope"),
            }
        )
        return summary

    def _log_collection_metrics(
        self, theta_in: Mapping[str, Any], summary: Mapping[str, Any]
    ) -> None:
        """Round-level collection counts and throughput against the entry theta."""

        if self.tensorboard is None:
            return
        step = int(theta_in.get("total_train_step", 0))
        scalars = {
            "collect/completed_games": summary.get("completed_games"),
            "collect/samples": summary.get("samples"),
            "collect/policy_samples": summary.get("policy_samples"),
            "collect/simulations": summary.get("simulations"),
            "collect/wall_seconds": summary.get("wall_seconds"),
            "collect/games_per_second": summary.get("games_per_wall_second"),
            "collect/policy_samples_per_second": summary.get("policy_samples_per_wall_second"),
            "collect/simulations_per_second": summary.get("simulations_per_wall_second"),
        }
        for tag, value in scalars.items():
            if value is not None:
                self.tensorboard.add_scalar(tag, float(value), step)

    def _train_round(
        self, round_root: Path, round_index: int, theta_in: Mapping[str, Any]
    ) -> dict[str, Any]:
        runs = [
            ReplayRun(
                round_index=r,
                run_id=f"{self.config.run_id}-round-{r:03d}",
                root=_round_root(self.output, r) / "selfplay",
            )
            for r in range(round_index + 1)
        ]
        index = build_replay_index(
            runs,
            max_games=self.config.replay_max_games,
            cache_bytes=self.config.replay_cache_bytes,
            state_path=self.output / "replay_index.json",
        )
        identity = index.identity()
        current_pool = index.position_pool(split="train", round_index=round_index)
        history_pool = index.position_pool(split="train", before_round=round_index)
        validation_games = [
            game for game in index.split_games("validation") if game.round_index == round_index
        ]
        validation_samples: list[SelfPlaySample] = [
            index.cache.sample(game, position)
            for game in validation_games
            for position in range(game.decision_count)
        ]
        steps_target = self.config.learner.steps_for_round(len(current_pool))
        metadata = {
            "round_index": round_index,
            "parent_checkpoint": dict(theta_in),
            "replay_identity": identity,
            "steps_target": steps_target,
            "current_positions": len(current_pool),
            "history_positions": len(history_pool),
            "validation_games": len(validation_games),
        }
        final_path = round_root / "checkpoints" / "final.pt"
        previous = self.state["rounds"].get(str(round_index), {})
        if self._final_checkpoint_matches(final_path, round_index, theta_in, identity, steps_target):
            final_identity = _read_checkpoint_identity(final_path)
            training_path = round_root / "training.json"
            if training_path.is_file():
                train = _load_json(training_path)
            else:
                train = {
                    "round_index": round_index,
                    "steps_target": steps_target,
                    "steps_completed": final_identity["round_train_step"],
                    "validation": previous.get("validation", {"available": False}),
                    "recovered": True,
                }
            train = dict(train) | {"resumed": True}
            _print_event(
                {"event": "train_skipped", "round_index": round_index, "steps_target": steps_target}
            )
            self._log_validation(final_identity, train.get("validation"))
            return train
        learner = self._resume_or_warm_start(round_root, round_index, theta_in, identity, steps_target)
        started = time.perf_counter()
        session_start_step = learner.round_train_step
        throttle = ProgressThrottle(self.progress_interval_seconds)

        def on_train_progress(record: Mapping[str, Any]) -> None:
            self._on_train_step(record)
            round_step = int(record["round_train_step"])
            if round_step < steps_target and not throttle.ready():
                return
            elapsed = max(time.perf_counter() - started, 1e-9)
            done_this_session = round_step - session_start_step
            rate = done_this_session / elapsed
            _print_event(
                {
                    "event": "train_progress",
                    "round_index": round_index,
                    "round_train_step": round_step,
                    "steps_target": steps_target,
                    "total_train_step": int(record["total_train_step"]),
                    "total_loss": round(float(record["total_loss"]), 6),
                    "steps_per_second": round(rate, 3),
                    "eta_seconds": round((steps_target - round_step) / rate, 1) if rate > 0 else None,
                }
            )

        train = run_training_round(
            learner,
            steps_target=steps_target,
            current_pool=current_pool,
            history_pool=history_pool,
            validation_samples=validation_samples,
            validation_games=len(validation_games),
            checkpoint_dir=round_root / "checkpoints",
            metrics_path=round_root / "training_metrics.jsonl",
            progress_callback=on_train_progress,
        )
        train["metadata"] = metadata
        train["resumed"] = False
        # The final checkpoint is byte-identical to the round's last step
        # checkpoint, so an interrupted evaluation can be resumed against an
        # unchanged file after a recovery that only re-commits the checkpoint.
        final_step_path = round_root / "checkpoints" / f"step-{steps_target:09d}.pt"
        if final_step_path.is_file():
            temporary = final_path.with_suffix(final_path.suffix + ".tmp")
            shutil.copyfile(final_step_path, temporary)
            temporary.replace(final_path)
        else:
            learner.save(
                final_path,
                metadata={"steps_target": steps_target, "replay_digest": identity["digest"]},
            )
        final_identity = _read_checkpoint_identity(final_path)
        _atomic_write_json(final_path.with_name("final.json"), final_identity)
        _atomic_write_json(round_root / "training.json", train)
        _print_event(
            {
                "event": "train_complete",
                "round_index": round_index,
                "steps_completed": train["steps_completed"],
                "validation_available": train["validation"]["available"],
            }
        )
        self._log_validation(final_identity, train["validation"])
        return train

    def _log_validation(
        self, theta_out: Mapping[str, Any], validation: Mapping[str, Any] | None
    ) -> None:
        """Validation CE/MSE of one round against its final checkpoint step."""

        if self.tensorboard is None or not validation or not validation.get("available"):
            return
        step = int(theta_out["total_train_step"])
        self.tensorboard.add_scalar("validation/policy_ce", float(validation["policy_loss"]), step)
        self.tensorboard.add_scalar("validation/value_mse", float(validation["value_loss"]), step)
        if validation.get("policy_entropy") is not None:
            self.tensorboard.add_scalar(
                "validation/policy_entropy", float(validation["policy_entropy"]), step
            )
        self.tensorboard.add_scalar("validation/games", float(validation.get("games") or 0), step)
        self.tensorboard.add_scalar(
            "validation/positions", float(validation.get("positions") or 0), step
        )

    def _resume_or_warm_start(
        self,
        round_root: Path,
        round_index: int,
        theta_in: Mapping[str, Any],
        identity: Mapping[str, Any],
        steps_target: int,
    ) -> AlphazeroLearner:
        checkpoint_dir = round_root / "checkpoints"
        candidates = sorted(
            checkpoint_dir.glob("step-*.pt"),
            key=lambda path: int(path.stem.removeprefix("step-")),
        )
        if candidates:
            return AlphazeroLearner.resume(
                candidates[-1],
                self.config.learner,
                replay_identity=identity,
                round_index=round_index,
                round_steps_target=steps_target,
            )
        if round_index > 0:
            # Round r continues the learner that produced theta_r: AdamW
            # moments and the global step counter carry over.
            learner = AlphazeroLearner.continue_from(
                theta_in["path"],
                self.config.learner,
                sampling_seed=self.config.seeds.sampling_for(round_index),
                replay_identity=identity,
                parent_checkpoint=dict(theta_in),
                round_index=round_index,
                round_steps_target=steps_target,
            )
            continued = True
        else:
            learner = AlphazeroLearner.warm_start(
                theta_in["path"],
                self.config.learner,
                sampling_seed=self.config.seeds.sampling_for(round_index),
                replay_identity=identity,
                parent_checkpoint=dict(theta_in),
                round_index=round_index,
                round_steps_target=steps_target,
            )
            continued = False
        _print_event(
            {
                "event": "train_start",
                "round_index": round_index,
                "steps_target": steps_target,
                "device": self.config.learner.device,
                "continued": continued,
                "total_train_step": learner.total_train_step,
            }
        )
        return learner

    def _final_checkpoint_matches(
        self,
        path: Path,
        round_index: int,
        theta_in: Mapping[str, Any],
        identity: Mapping[str, Any],
        steps_target: int,
    ) -> bool:
        if not path.is_file():
            return False
        record = _read_checkpoint_identity(path)
        return (
            record["round_index"] == round_index
            and record["round_steps_target"] == steps_target
            and record["parent_checkpoint"].get("sha256") == theta_in["sha256"]
            and record["replay_identity"].get("digest") == identity["digest"]
        )

    def _on_train_step(self, record: Mapping[str, Any]) -> None:
        if self.tensorboard is None:
            return
        step = int(record["total_train_step"])
        for tag in ("policy_loss", "value_loss", "total_loss", "policy_entropy", "grad_norm", "learning_rate"):
            self.tensorboard.add_scalar(f"train/{tag}", record[tag], step)
        self.tensorboard.add_scalar("train/current_positions", record["current_positions"], step)
        self.tensorboard.add_scalar("train/history_positions", record["history_positions"], step)
        if int(record["round_train_step"]) % 20 == 0:
            self.tensorboard.flush()

    # ------------------------------------------------------------- evaluation

    def _evaluate_round(
        self,
        round_root: Path,
        round_index: int,
        theta_new: Mapping[str, Any],
        theta_parent: Mapping[str, Any],
    ) -> dict[str, Any]:
        evaluation_root = round_root / "evaluation"
        parent_panel = self.config.evaluation.mcts_vs_parent
        greedy_panel = self.config.evaluation.greedy_vs_initial
        common = {
            "workers": self.config.evaluation.workers,
            "device": self.config.evaluation.device,
            "max_completed_turns": self.config.selfplay.max_completed_turns,
            "adjudicate_deterministic_policy_cycles": self.config.evaluation.adjudicate_deterministic_policy_cycles,
        }
        started = time.perf_counter()
        search_config = dict(parent_panel.search_config or {})
        mcts_run = run_fixed_evaluation(
            FixedEvaluationConfig(
                policy_a=SerializablePolicySpec.mcts_checkpoint(
                    "theta_new_mcts", theta_new["path"], search_config=search_config,
                    search_backend=self.config.evaluation.mcts_backend
                ),
                policy_b=SerializablePolicySpec.mcts_checkpoint(
                    "theta_parent_mcts", theta_parent["path"], search_config=search_config,
                    search_backend=self.config.evaluation.mcts_backend
                ),
                game_seeds=parent_panel.game_seeds,
                execution=self.config.evaluation.mcts_execution,
                policy_seed_base=parent_panel.policy_seed_base,
                metadata={
                    "loop_protocol": LOOP_PROTOCOL,
                    "round_index": round_index,
                    "panel": "mcts_vs_parent",
                    "theta_new": theta_new["path"],
                    "theta_parent": theta_parent["path"],
                },
                **common,
            ),
            evaluation_root / "mcts_vs_parent.json",
            store_dir=evaluation_root / "mcts_vs_parent.games",
            failure_dir=evaluation_root / "failures" / "mcts_vs_parent",
            resume=True,
            progress_callback=self._evaluation_progress(
                round_index, "mcts_vs_parent", started
            ),
        )
        mcts_seconds = time.perf_counter() - started
        self._require_evaluation_passed(round_index, "mcts_vs_parent", mcts_run.summary)
        started = time.perf_counter()
        greedy_run = run_fixed_evaluation(
            FixedEvaluationConfig(
                policy_a=SerializablePolicySpec.greedy_checkpoint("theta_new_greedy", theta_new["path"]),
                policy_b=SerializablePolicySpec.greedy_checkpoint(
                    "initial_greedy", self.initial_identity["path"]
                ),
                game_seeds=greedy_panel.game_seeds,
                policy_seed_base=greedy_panel.policy_seed_base,
                metadata={
                    "loop_protocol": LOOP_PROTOCOL,
                    "round_index": round_index,
                    "panel": "greedy_vs_initial",
                    "theta_new": theta_new["path"],
                    "theta_initial": self.initial_identity["path"],
                },
                **common,
            ),
            evaluation_root / "greedy_vs_initial.json",
            store_dir=evaluation_root / "greedy_vs_initial.games",
            failure_dir=evaluation_root / "failures" / "greedy_vs_initial",
            resume=True,
            progress_callback=self._evaluation_progress(
                round_index, "greedy_vs_initial", started
            ),
        )
        greedy_seconds = time.perf_counter() - started
        self._require_evaluation_passed(round_index, "greedy_vs_initial", greedy_run.summary)
        result = {
            "mcts_vs_parent": mcts_run.summary | {"wall_seconds_this_session": mcts_seconds},
            "greedy_vs_initial": greedy_run.summary | {"wall_seconds_this_session": greedy_seconds},
        }
        if self.tensorboard is not None:
            step = int(theta_new.get("total_train_step", 0))
            self.tensorboard.add_scalar(
                "eval/mcts_vs_parent_score_rate", mcts_run.summary["win_rate_score"] or 0.0, step
            )
            self.tensorboard.add_scalar(
                "eval/greedy_vs_initial_score_rate", greedy_run.summary["win_rate_score"] or 0.0, step
            )
            self.tensorboard.add_scalar("system/eval_mcts_seconds_this_session", mcts_seconds, step)
            self.tensorboard.add_scalar("system/eval_greedy_seconds_this_session", greedy_seconds, step)
            self.tensorboard.flush()
        _print_event(
            {
                "event": "evaluate_complete",
                "round_index": round_index,
                "mcts_vs_parent": self._evaluation_record(result["mcts_vs_parent"]),
                "greedy_vs_initial": self._evaluation_record(result["greedy_vs_initial"]),
            }
        )
        return result

    def _evaluation_progress(
        self, round_index: int, panel: str, started: float
    ) -> Callable[[int, int], None]:
        throttle = ProgressThrottle(self.progress_interval_seconds)

        def report(completed: int, total: int) -> None:
            if completed < total and not throttle.ready():
                return
            elapsed = max(time.perf_counter() - started, 1e-9)
            _print_event(
                {
                    "event": "evaluate_progress",
                    "round_index": round_index,
                    "panel": panel,
                    "completed_pairs": completed,
                    "total_pairs": total,
                    "elapsed_seconds": round(elapsed, 3),
                    "pairs_per_minute": round(completed / elapsed * 60.0, 3),
                }
            )

        return report

    @staticmethod
    def _require_evaluation_passed(
        round_index: int, panel: str, summary: Mapping[str, Any]
    ) -> None:
        """Stop the loop instead of collecting the next round on a broken panel."""

        if summary.get("passed"):
            return
        raise EvaluationEngineeringFailure(
            f"evaluation panel {panel} failed engineering checks in round {round_index}",
            round_index=round_index,
            panel=panel,
            details={
                "completed_pairs": summary.get("completed_pairs"),
                "total_pairs": summary.get("total_pairs"),
                "engineering_truncations": summary.get("engineering_truncations"),
                "exceptions": summary.get("exceptions"),
                "illegal_actions": summary.get("illegal_actions"),
                "invariant_failures": summary.get("invariant_failures"),
                "output": summary.get("output"),
            },
        )

    @staticmethod
    def _evaluation_record(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "win_rate_score": summary.get("win_rate_score"),
            "a_wins": summary.get("a_wins"),
            "draws": summary.get("draws"),
            "a_losses": summary.get("a_losses"),
            "completed_pairs": summary.get("completed_pairs"),
            "total_pairs": summary.get("total_pairs"),
            "passed": summary.get("passed"),
            "engineering_truncations": summary.get("engineering_truncations"),
            "exceptions": summary.get("exceptions"),
            "a_first": summary.get("a_first"),
            "a_second": summary.get("a_second"),
            "output": summary.get("output"),
        }

    # ----------------------------------------------------------------- output

    def _write_failure(
        self, round_index: int, phase: str, theta_in: Mapping[str, Any], error: Exception
    ) -> None:
        scene = {
            "protocol": "alphazero-loop-failure-v1",
            "round_index": round_index,
            "phase": phase,
            "theta_in": dict(theta_in),
            "config_fingerprint": self.config_fingerprint,
            "error_type": type(error).__name__,
            "error_detail": str(error),
            "traceback": traceback.format_exc(),
        }
        if isinstance(error, NonFiniteTrainingError):
            scene["training_details"] = dict(error.details)
        if isinstance(error, EvaluationEngineeringFailure):
            scene["evaluation"] = {"panel": error.panel, "details": error.details}
        _atomic_write_json(
            self.output / "failures" / f"round-{round_index:03d}-{phase}.json", scene
        )

    def _summary(self) -> dict[str, Any]:
        completed = len(self.state["completed_rounds"])
        return {
            "protocol": LOOP_PROTOCOL,
            "run_id": self.config.run_id,
            "output": str(self.output),
            "num_rounds": self.config.num_rounds,
            "completed_rounds": list(self.state["completed_rounds"]),
            "latest_checkpoint": self.state["latest_checkpoint"],
            "loop_complete": completed >= self.config.num_rounds,
        }


def _lock_directory(root: Path) -> int:
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"an AlphaZero loop is already running in {root}") from error
    return descriptor


def _unlock_directory(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def run_loop(
    config: LoopConfig,
    output: Path,
    *,
    resume: bool = False,
    progress_interval_seconds: float = DEFAULT_PROGRESS_INTERVAL_SECONDS,
) -> dict[str, Any]:
    output = Path(output).resolve()
    descriptor = _lock_directory(output)
    try:
        runner = LoopRunner(
            config, output, progress_interval_seconds=progress_interval_seconds
        )
        runner.initialize(resume=resume)
        result = runner.run()
    finally:
        _unlock_directory(descriptor)
    print(json.dumps({"event": "loop_finished", **result}, sort_keys=True), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = LoopConfig.from_dict(_load_json(args.config))
    result = run_loop(config, args.output, resume=args.resume)
    return 0 if len(result["completed_rounds"]) >= config.num_rounds else 1


if __name__ == "__main__":
    raise SystemExit(main())
