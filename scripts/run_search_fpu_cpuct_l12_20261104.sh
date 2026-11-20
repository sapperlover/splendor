#!/usr/bin/env bash
# FPU / c_puct screening 2a+2b: L12 new search vs L12 historical search, both S128N1, 2000 pairs per panel.
# Actions: prepare | smoke | run | status. Re-running `run` resumes from complete pairs.
set -euo pipefail
cd "$(dirname "$0")/.."
exec conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  python -u artifacts/search_fpu_cpuct_l12_20261104/run.py "$@"
