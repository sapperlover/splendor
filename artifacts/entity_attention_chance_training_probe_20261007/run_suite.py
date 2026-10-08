"""Run the predeclared finite stages in separate processes with visible progress."""
import json
import subprocess
import sys
from pathlib import Path

ART=Path(__file__).resolve().parent
spec=json.loads((ART/'specification.json').read_text())

def run(*args):
    subprocess.run([sys.executable,'-u',str(ART/'experiment.py'),*args],check=True)

if __name__=='__main__':
    for arm in spec['arms']:
        if not (ART/f'smoke_{arm}_result.json').exists():run('collect','--arm',arm,'--smoke')
    for arm in spec['arms']:
        if not (ART/f'collect_{arm}_result.json').exists():run('collect','--arm',arm)
        if not (ART/f'train_{arm}.json').exists():run('train','--arm',arm)
    for panel in spec['panels']:
        path=ART/'evaluations'/f"{panel['id']}.json"
        if not path.exists() or not json.loads(path.read_text())['passed']:
            run('evaluate','--case',panel['id'])
    run('analyze')
