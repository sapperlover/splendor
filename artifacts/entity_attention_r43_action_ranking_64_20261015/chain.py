"""Run continuation, adaptive expansion and replay/analysis serially in background."""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time

from splendor.alphazero_recording import read_json,sha256,write_json

ART=Path(__file__).resolve().parent
ROOT=ART.parents[1]
RUNTIME=ROOT/'runtime'/ART.name
child=None
stopping=False


def stop(signum,frame):
    global stopping
    stopping=True
    if child is not None and child.poll() is None:child.send_signal(signal.SIGINT)


def main():
    global child
    RUNTIME.mkdir(parents=True,exist_ok=True)
    with (RUNTIME/'chain.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        session=str(time.time_ns())
        record={'status':'running','pid':os.getpid(),'start_unix':time.time(),'command':[sys.executable,*sys.argv],
            'runner_sha256':sha256(ART/'run.py'),'chain_sha256':sha256(Path(__file__)),'plan_sha256':sha256(ART/'plan.json')}
        try:
            for phase in ('run','analyze'):
                if stopping:raise KeyboardInterrupt
                record.update(phase=phase)
                write_json(RUNTIME/'live.json',record)
                print(f'START {phase}',flush=True)
                with (RUNTIME/f'{phase}.log').open('a') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(ART/'run.py'),phase],cwd=ROOT,
                        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    record['child_pid']=child.pid;write_json(RUNTIME/'live.json',record)
                    code=child.wait();child=None
                if stopping or code==130:raise KeyboardInterrupt
                if code:raise subprocess.CalledProcessError(code,phase)
                print(f'DONE {phase}',flush=True)
            result=read_json(ART/'summary.json')
            if result['status']!='complete':raise ValueError('Analysis did not complete')
            record.update(status='complete',end_unix=time.time(),counts=result['counts'])
            write_json(RUNTIME/'live.json',record)
            return 0
        except KeyboardInterrupt:
            record.update(status='interrupted',end_unix=time.time());write_json(RUNTIME/'live.json',record);return 130
        except Exception as error:
            record.update(status='failed',error=f'{type(error).__name__}: {error}',end_unix=time.time())
            write_json(RUNTIME/'live.json',record)
            raise
        finally:write_json(ART/'chains'/f'{session}.json',record)


if __name__=='__main__':raise SystemExit(main())
