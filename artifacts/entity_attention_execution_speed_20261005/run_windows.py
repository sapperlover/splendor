"""Sequential, bounded S1024/N1 windows; never resumes a user run."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from splendor.alphazero_recording import environment_identity, source_identity

ROOT = Path('artifacts/entity_attention_execution_speed_20261005')


def gpu_state():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return result.stdout.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='+', required=True)
    parser.add_argument('--repeat', type=int, required=True)
    args = parser.parse_args()
    for name in args.variants:
        command = [sys.executable, 'scripts/benchmark_collect_throughput.py', '--config',
                   str(ROOT/'configs'/f'{name}.json'), '--games-per-worker','125',
                   '--warmup-seconds','3','--seconds','15','--scratch',
                   'runtime/entity_attention_execution_speed_20261005', '--output',str(ROOT/f'{name}_r{args.repeat}.json')]
        record = {'command':command,'repeat':args.repeat,'started_epoch_seconds':time.time(),'source':source_identity(Path('.').resolve()),
                  'environment':environment_identity(),'gpu_before':gpu_state()}
        completed = subprocess.run(command, capture_output=True, text=True)
        record.update(returncode=completed.returncode, stdout=completed.stdout,stderr=completed.stderr,gpu_after=gpu_state(),ended_epoch_seconds=time.time())
        (ROOT/f'{name}_r{args.repeat}.command.json').write_text(json.dumps(record,indent=2)+'\n')
        print(name, completed.stdout.strip(), completed.stderr[-1500:], flush=True)
        completed.check_returncode()

if __name__ == '__main__':
    main()
