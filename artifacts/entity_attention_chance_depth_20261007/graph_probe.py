"""Use the existing fixed-address CUDA Graph transport for the frozen depth probe."""
import argparse
import importlib.util
from pathlib import Path

from splendor.alphazero_recording import read_json, sha256
from splendor.search_inference import InferenceConfig

ART = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("depth_probe", ART / "probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "native_chance"), required=True)
    parser.add_argument("--subset", choices=("pilot", "confirmation"), required=True)
    parser.add_argument("--sims", type=int, nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    args = parser.parse_args()
    assert args.batch_size > 0 and all(s > 0 for s in args.sims)
    manifest = read_json(ART / "graph/manifest.json")
    assert sha256(Path(__file__)) == manifest["graph_adapter"]["sha256"]
    probe.ART = ART / "graph"

    def configuration(**kwargs):
        kwargs.update(mode="compiled", compile_mode="native-cudagraph",
                      compile_failure_policy="fail_fast", native_transport="reused",
                      batch_buckets=(args.batch_size,))
        return InferenceConfig(**kwargs)

    probe.InferenceConfig = configuration
    probe.run(args)
