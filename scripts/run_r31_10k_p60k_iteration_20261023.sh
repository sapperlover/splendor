#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_source="$task_repo_root/runtime/entity_attention_r31_10k_p60k_iteration_20261023/source_recording_checkpoint_20261023"
cd "$task_source"
exec conda run --no-capture-output -n splendor env \
  PYTHONPATH="$task_source/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.random_budget_iteration \
  --config "$task_repo_root/configs/entity_attention_v1/random_budget_r31_10k_p60k_20261023.json" \
  --output "$task_repo_root/runs/entity_attention_r31_10k_p60k_iteration_20261023" "$@"
