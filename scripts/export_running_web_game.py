"""One-time, no-restart export of a game held by an older local web server.

GDB only queues a Python pending call; Python performs the export itself after
the debugger detaches. No Python evaluation is forced into a suspended thread.
Run with the permissions needed to attach to the specified server process.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import textwrap
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    command = Path(f"/proc/{args.pid}/cmdline").read_bytes().split(b"\0")
    if b"splendor.web_app" not in command:
        parser.error("PID must identify python -m splendor.web_app")
    output = args.output.resolve()
    status = output.with_suffix(".export-status.json")
    if output.exists() or status.exists():
        parser.error("output or export status already exists; choose a new output path")

    payload = "\n".join([
        "import gc, json, traceback",
        "from pathlib import Path",
        f"output = Path({str(output)!r})",
        f"status = Path({str(status)!r})",
        "try:",
        "    services = [obj for obj in gc.get_objects()",
        "                if type(obj).__name__ == 'WebGameService'",
        f"                and {args.game_id!r} in obj._games]",
        "    if len(services) != 1:",
        "        raise RuntimeError('expected exactly one service containing the game')",
        "    service = services[0]",
        "    with service._lock:",
        f"        game = service._games[{args.game_id!r}]",
        "        game.engine.save(output)",
        "        result = {'ok': True, 'path': str(output),",
        f"                  'game_id': {args.game_id!r},",
        "                  'events': len(game.engine.replay_record.events),",
        "                  'revision': game.revision,",
        "                  'opponent': game.opponent_label}",
        "except Exception:",
        "    output.parent.mkdir(parents=True, exist_ok=True)",
        "    result = {'ok': False, 'error': traceback.format_exc()}",
        "status.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\\n')",
    ])
    # The fixed-width address is filled after allocation. Compilation finishes
    # before exec runs, so the first statement can release the source buffer.
    source_body = (
        "import ctypes\n"
        "ctypes.pythonapi.PyMem_RawFree.argtypes = [ctypes.c_void_p]\n"
        "ctypes.pythonapi.PyMem_RawFree.restype = None\n"
        "ctypes.pythonapi.PyMem_RawFree(ctypes.c_void_p(0xADDRESS_PLACEHOLDER))\n"
        f"exec({payload!r}, {{'__name__': '__splendor_web_export__'}})\n"
    )
    source = f"exec({source_body!r}, {{'__name__': '__splendor_web_export_buffer__'}})\n"
    gdb_python = f"""
import gdb
source = {source!r}
placeholder = 'ADDRESS_PLACEHOLDER'
source = source.replace(placeholder, '0' * 16)
size = len(source.encode('utf-8')) + 1
pointer = int(gdb.parse_and_eval('(void *)PyMem_RawMalloc(%d)' % size))
if not pointer:
    raise RuntimeError('target allocation failed')
source = {source!r}.replace(placeholder, '%016x' % pointer)
gdb.selected_inferior().write_memory(pointer, source.encode('utf-8') + b'\\0')
rc = int(gdb.parse_and_eval('(int)Py_AddPendingCall((int (*)(void *))PyRun_SimpleString, (void *)%d)' % pointer))
if rc != 0:
    gdb.parse_and_eval('(void)PyMem_RawFree((void *)%d)' % pointer)
    raise RuntimeError('Python pending-call queue is full; no export was scheduled')
"""
    with tempfile.TemporaryDirectory(prefix="splendor-web-export-") as temporary:
        script = Path(temporary) / "export.gdb"
        script.write_text(
            "set pagination off\nset confirm off\nset scheduler-locking off\n"
            "python\ntry:\n" + textwrap.indent(gdb_python, "    ")
            + "\nexcept Exception:\n    import traceback\n    traceback.print_exc()\n"
            "    gdb.execute('detach')\n    gdb.execute('quit 1')\n"
            "end\ndetach\nquit\n"
        )
        completed = subprocess.run(
            ["gdb", "-q", "-nx", "-batch", "-p", str(args.pid), "-x", str(script)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    if completed.returncode:
        print(completed.stdout)
        raise SystemExit("Could not schedule export; the server was not restarted.")
    deadline = time.monotonic() + 10
    while not status.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not status.exists():
        raise SystemExit(f"Export is queued; check {status}. Do not retry or restart the server.")
    result = json.loads(status.read_text())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
