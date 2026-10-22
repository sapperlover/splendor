#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if (($# == 0)); then
  set -- run
fi
exec conda run --no-capture-output -n splendor env \
  PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/entity_set_mlp_256x4_offline_80k_20261021/run.py "$@"
