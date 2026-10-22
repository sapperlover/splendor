#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_repo_root"
exec conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -u -m splendor.random_budget_iteration \
  --config configs/entity_attention_v1/random_budget_37423_p6k_v24k_20261019.json \
  --output runs/entity_attention_37423_random_budget_p6k_v24k_20261019 "$@"
