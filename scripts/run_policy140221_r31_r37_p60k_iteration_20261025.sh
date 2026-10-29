#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
task_source="$task_repo_root/runtime/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/source_cache_release_20261025"
cd "$task_source"
exec conda run --no-capture-output -n splendor env \
  PYTHONPATH="$task_source/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u -m splendor.random_budget_iteration \
  --config "$task_repo_root/configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json" \
  --output "$task_repo_root/runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025" "$@"
