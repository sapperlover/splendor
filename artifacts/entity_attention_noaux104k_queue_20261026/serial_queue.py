"""Wait for policy+score's successful full chain, then run noaux104k."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = Path(__file__).resolve().parent


def configured_queue():
    plan = json.loads((ART / "plan.json").read_text())
    path = ROOT / plan["shared_runner"]["path"]
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["shared_runner"]["sha256"]:
        raise ValueError("Shared serial queue implementation changed")
    script = ROOT / plan["successor_script"]["path"]
    if hashlib.sha256(script.read_bytes()).hexdigest() != plan["successor_script"]["sha256"]:
        raise ValueError("Successor launcher changed")
    spec = importlib.util.spec_from_file_location("noaux104k_serial_queue", path)
    queue = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(queue)
    queue.__doc__ = __doc__
    queue.ART, queue.RUNTIME = ART, ROOT / "runtime" / ART.name
    queue.CURRENT = plan["predecessor"]
    queue.JOBS = ((plan["successor"], plan["successor_script"]["path"]),)
    return queue


if __name__ == "__main__":
    raise SystemExit(configured_queue().main())
