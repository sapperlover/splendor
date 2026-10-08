"""Finite FP32 checks and four complete native games, isolated from performance windows."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from splendor.alphazero_recording import environment_identity, source_identity
from splendor.selfplay_fast import FastSelfPlayConfig, generate, validate_run

ROOT = Path('artifacts/entity_attention_execution_speed_20261005')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variants', nargs='*', default=['baseline', 'static'])
    parser.add_argument('--smoke-config', type=Path, default=ROOT/'configs/smoke.json')
    args = parser.parse_args()
    records = []
    for name in args.variants:
        command = [sys.executable,str(ROOT/'verify_gpu.py'),'--variant',name,'--output',str(ROOT/f'gpu_{name}.json')]
        result = subprocess.run(command,capture_output=True,text=True)
        records.append({'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr})
        (ROOT/('gpu_commands_'+'_'.join(args.variants)+'.json')).write_text(json.dumps(records,indent=2)+'\n')
        print(name,result.stdout,result.stderr[-1500:],flush=True)
        result.check_returncode()
    config = FastSelfPlayConfig.from_dict(json.loads(args.smoke_config.read_text()))
    root = Path('runtime/entity_attention_execution_speed_20261005')/('complete_'+args.smoke_config.stem)
    summary = generate(config,root)
    checked = validate_run(root,replay=True)
    assert checked['completed_games'] == 4 and checked['failed_games'] == 0
    status = json.loads((root/'status.json').read_text())
    output = {'config':config.to_dict(),'runtime_output':str(root),'generation':summary,
              'checked_replay':checked,'gpu_owner':status['gpu_owner'],
              'environment':environment_identity(),'source':source_identity(Path('.').resolve())}
    (ROOT/('gpu_complete_'+args.smoke_config.stem+'.json')).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'smoke':'passed','completed_games':checked['completed_games'],
                      'failed_games':checked['failed_games'],'batches_prepared_while_inflight':status['gpu_owner']['batches_prepared_while_inflight']}),flush=True)

if __name__ == '__main__':
    main()
