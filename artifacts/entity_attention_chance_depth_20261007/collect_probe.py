"""Experiment-only backend adapter for the existing timed self-play benchmark.

The environment flag is inherited by spawn workers; production source/configuration
is unchanged. Sampled-chance data carries its own search/RNG fingerprint.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def install_chance():
    import splendor.native_policy as native
    import splendor.selfplay_fast as selfplay

    native.NativePUCTSearchPolicy = native.NativeChancePUCTSearchPolicy
    original = selfplay.build_manifest

    def manifest(config):
        result = original(config)
        result["versions"]["search"] = "sampled-chance-puct-native-v1"
        result["versions"]["rng"] = "splitmix64-chance-root-resampling-v1"
        frozen = {key: result[key] for key in ("config", "checkpoint_identity", "versions")}
        result["semantic_fingerprint"] = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result

    selfplay.build_manifest = manifest


if os.environ.get("SPLENDOR_EXPERIMENT_CHANCE") == "1":
    install_chance()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--search-backend", choices=("native", "native_chance"), required=True)
    args, remaining = parser.parse_known_args()
    assert "SPLENDOR_EXPERIMENT_CHANCE" not in os.environ
    if args.search_backend == "native_chance":
        os.environ["SPLENDOR_EXPERIMENT_CHANCE"] = "1"
        install_chance()
    import torch
    import splendor.native_policy as native
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert native.NativePUCTSearchPolicy.stochastic == (args.search_backend == "native_chance")
    spec = importlib.util.spec_from_file_location(
        "collect_benchmark", ROOT / "scripts/benchmark_collect_throughput.py")
    benchmark = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = benchmark
    spec.loader.exec_module(benchmark)
    sys.argv = [str(ROOT / "scripts/benchmark_collect_throughput.py"), *remaining]
    benchmark.main()
    output = Path(remaining[remaining.index("--output") + 1])
    result = json.loads(output.read_text())
    result["search_backend"] = args.search_backend
    result["experiment_adapter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output.write_text(json.dumps(result, indent=2) + "\n")
