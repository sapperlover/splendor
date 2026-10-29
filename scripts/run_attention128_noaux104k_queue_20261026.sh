#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec conda run --no-capture-output -n splendor env PYTHONPATH=src \
  python -u artifacts/entity_attention_noaux104k_queue_20261026/serial_queue.py "$@"
