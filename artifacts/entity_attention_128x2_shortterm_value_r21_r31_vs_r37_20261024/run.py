"""Run one prepared continuation auxiliary ablation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "entity_attention_auxiliary_queue_20261024"))
from run_experiment import main

if __name__ == "__main__":
    raise SystemExit(main("value44k"))
