#!/usr/bin/env bash
set -u
cd /home/lc/下载/splendor
run() {
  name=$1; cfg=$2; shift 2
  echo "===== $name $(date +%H:%M:%S) ====="
  conda run --no-capture-output -n splendor env PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
    python -u scripts/benchmark_collect_throughput.py \
    --config "$cfg" --output "runtime/collection_tuning_20261022/results/$name.json" \
    --scratch runtime/collection_tuning_20261022/scratch "$@" 2>&1 | tail -4
}
run attention_a2000_mb128_60_r2 runtime/collection_tuning_20261022/configs/attention_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 128
run attention_a2000_mb256_60_r2 runtime/collection_tuning_20261022/configs/attention_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 256
run attention_a2000_mb128_60_r3 runtime/collection_tuning_20261022/configs/attention_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 128
run attention_a2000_mb256_60_r3 runtime/collection_tuning_20261022/configs/attention_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 256
run mlp_a2000_mb256_60_r2 runtime/collection_tuning_20261022/configs/mlp_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 256
run mlp_a2000_mb128_60_r2 runtime/collection_tuning_20261022/configs/mlp_gpu.json --games-per-worker 336 --seconds 60 --warmup-seconds 10 --search-workers 12 --total-active 2000 --local-microbatch 128
echo "DONE $(date +%H:%M:%S)"
