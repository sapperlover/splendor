#!/usr/bin/env bash
set -euo pipefail
task_repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$task_repo_root"
exec conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -u artifacts/value_outcome_60k_20261120/run.py scalar_lambda "$@"
