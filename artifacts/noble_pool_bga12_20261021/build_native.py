"""Build the production extension and atomically replace its file."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sysconfig
import torch

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime/noble_pool_bga12_20261021'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RUNTIME.mkdir(parents=True,exist_ok=True)
    name='_native_search'+sysconfig.get_config_var('EXT_SUFFIX')
    destination=ROOT/'src/splendor'/name
    previous=RUNTIME/('before-'+name)
    assert not previous.exists(),'Preserve the original pre-change extension.'
    shutil.copyfile(destination,previous)
    built=RUNTIME/name
    headers=Path(torch.__file__).parent/'include'
    assert (headers/'pybind11/pybind11.h').is_file()
    command=['c++','-O3','-DNDEBUG','-std=c++17','-shared','-fPIC','-fvisibility=hidden',
        '-I'+str(headers),'-I'+sysconfig.get_path('include'),
        str(ROOT/'src/splendor/native_search.cpp'),'-o',str(built)]
    subprocess.run(command,check=True)
    built.replace(destination)
    (OUT/'native_build.json').write_text(json.dumps({'command':command,
        'source_sha256':sha(ROOT/'src/splendor/native_search.cpp'),
        'old_binary':str(previous.relative_to(ROOT)),'old_binary_sha256':sha(previous),
        'new_binary':str(destination.relative_to(ROOT)),'new_binary_sha256':sha(destination),
        'atomic_replace':True},ensure_ascii=False,indent=2)+'\n')
    print('Production native extension rebuilt and atomically installed.',flush=True)


if __name__=='__main__':
    main()
