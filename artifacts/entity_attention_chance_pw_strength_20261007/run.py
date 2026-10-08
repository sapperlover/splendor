"""Fixed evaluation CLI with the same FP32/CPU-thread controls as the target diagnostic."""
import sys
from pathlib import Path

import torch

from splendor.evaluation import main
from splendor.alphazero_recording import read_json, sha256, environment_identity, write_json


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    artifact = Path(__file__).resolve().parent
    manifest = read_json(artifact / "manifest.json")
    assert sha256(Path(manifest["checkpoint"]["path"])) == manifest["checkpoint"]["sha256"]
    for path, expected in manifest["source"]["native_extensions"].items():
        assert sha256(Path(path)) == expected
    assert sha256(Path(__file__)) == next(x["sha256"] for x in manifest["inputs"] if x["path"] == str(Path(__file__).resolve()))
    environment_path = artifact / "environment.json"
    if not environment_path.exists():
        write_json(environment_path, environment_identity())
    raise SystemExit(main(sys.argv[1:]))
