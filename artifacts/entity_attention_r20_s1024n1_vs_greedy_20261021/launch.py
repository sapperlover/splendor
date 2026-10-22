"""Launch the frozen experiment once, detached; no queue polling."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
ART=Path(__file__).resolve().parent
WORK=ROOT/'runtime'/ART.name
record_path=ART/'launch.json'
if record_path.exists():
    old=json.loads(record_path.read_text())
    if Path(f"/proc/{old['pid']}").exists():
        raise RuntimeError('Recorded process still exists; inspect before launching another instance')
source=WORK/'source'
environment={**os.environ,'PYTHONPATH':str(source/'src'),'SPLENDOR_ARENA_ROOT':str(ROOT),
    'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
    'TORCHINDUCTOR_COMPILE_THREADS':'1','TORCHINDUCTOR_CACHE_DIR':str(WORK/'compile_cache'),
    'TRITON_CACHE_DIR':str(WORK/'triton_cache')}
command=[sys.executable,'-u',str(source/'artifacts'/ART.name/'run.py'),'run']
with (WORK/'run.log').open('ab',buffering=0) as log:
    process=subprocess.Popen(command,cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,
        stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
record=dict(pid=process.pid,command=command,start_unix=time.time(),
            log=str((WORK/'run.log').relative_to(ROOT)),detached=True)
record_path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(record,ensure_ascii=False),flush=True)
