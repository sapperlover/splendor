"""User-approved 2026-11-01 cold-file archive; stream to T7, verify, then release."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tarfile
import time
from zoneinfo import ZoneInfo

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
STAGE = ROOT / 'archive/20261101'
RUNTIME = ROOT / 'runtime/repository_archive_20261101'
TARGET = Path('/media/lc/T7/splendor/archive-20261101.tar.zst')
PARTIAL = TARGET.with_name(TARGET.name + '.partial')
CURRENT = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'
OLD = ROOT / 'runs/entity_attention_r31_10k_p60k_iteration_20261023'
PARENT = ROOT / 'runs/entity_attention_r20_s256n4_noble12_iteration_20261022'
OFFLINE_NAMES = (
    'entity_attention_128x2_policy_score_r21_r37_20261026',
    'entity_attention_128x2_noaux_r21_r37_20261026',
    'entity_attention_r37_104k_lr1e4_20261027',
    'entity_attention_r37_104k_lr2e5_20261027',
)
COUNTS = dict(zip(OFFLINE_NAMES, (192, 210, 105, 227))) | {
    'old_online_ordinary': 324, 'policy_online_ordinary': 26,
    'artifacts_raw': 170323, 'runtime_smoke_checkpoints': 95,
    'completed_training_metrics': 3, 'retired_online_training_games': 63038,
}
FIXED_MODELS = {'initial.pt', 'final.pt'} | {
    f'step-{s:09d}.pt' for s in (100, 250, 500, 1000, 2000, 3000)
}


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write_rows(path, entries):
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for row in entries:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def record(path, category):
    st = path.lstat()
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f'Expected original regular file: {path}')
    row = dict(path=str(path.relative_to(ROOT)), category=category, size=st.st_size,
               allocated_bytes=st.st_blocks * 512, mtime_ns=st.st_mtime_ns,
               inode=[st.st_dev, st.st_ino], sha256=sha(path))
    after = path.lstat()
    if (st.st_size, st.st_mtime_ns, st.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed during hashing: {path}')
    return row


def unchanged(row, *, content=True):
    p = ROOT / row['path']; st = p.lstat()
    if (not stat.S_ISREG(st.st_mode)
            or (st.st_size, st.st_mtime_ns, [st.st_dev, st.st_ino])
            != (row['size'], row['mtime_ns'], row['inode'])):
        raise ValueError(f'Original metadata changed: {p}')
    if content and sha(p) != row['sha256']:
        raise ValueError(f'Original SHA changed: {p}')


def regular_files(root):
    for dp, ds, fs in os.walk(root, followlinks=False):
        ds[:] = [d for d in ds if not (Path(dp) / d).is_symlink() and d != '__pycache__']
        for name in fs:
            p = Path(dp) / name
            if stat.S_ISREG(p.lstat().st_mode):
                yield p


def original_scope():
    with gzip.open(ART / 'approved_candidates.json.gz', 'rt') as f:
        return json.load(f)


def protected_dependencies():
    """Keep initial64k, active60k, all holdouts, pending collection and offline104k."""
    keep = set()

    def add(games):
        keep.update(Path(g['path']) for g in games)

    original = read(CURRENT / 'historical_games.json')['games']
    state = read(CURRENT / 'state.json')
    if len(original) != 64000 or len(state['completed_rounds']) != 13:
        raise ValueError('Current online branch moved beyond the approved audit')
    add(original)
    active = read(Path(state['completed_rounds'][-1]['snapshot']['path']))['games']
    if len(active) != 60000:
        raise ValueError('Active online pool differs from approved60k')
    add(active)
    for p in (CURRENT / 'fixed_holdout').rglob('manifest.json'):
        add(read(p)['games'])
    keep.update((CURRENT / 'rounds/round-013/selfplay/games').glob('*.npz'))
    manifest = ROOT / 'runs/entity_attention_128x2_shortterm_value_r21_r37_20261023/inputs/pool_games.jsonl'
    offline = [json.loads(line) for line in manifest.read_text().splitlines()]
    if len(offline) != 104000:
        raise ValueError('Shared offline104k pool changed')
    add(offline)
    return keep


def approved():
    scope = original_scope(); groups = scope['groups']
    if {k: len(v) for k, v in groups.items()} != COUNTS:
        raise ValueError('Approved group census changed')
    selected = [dict(path=p, category=k) for k, files in groups.items() for p in files]
    paths = {ROOT / r['path'] for r in selected}
    if len(paths) != len(selected):
        raise ValueError('Duplicate approved source')
    for p in paths:
        if p.is_absolute() and p.is_relative_to(ROOT) and '..' not in p.parts:
            continue
        raise ValueError(f'Invalid candidate path: {p}')
    tracked = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0'))
    if {r['path'] for r in selected} & tracked:
        raise ValueError('A candidate is tracked in Git')
    if paths & protected_dependencies():
        raise ValueError('Candidates overlap hot training/holdout dependencies')
    for i, name in enumerate(OFFLINE_NAMES):
        run = ROOT / 'runs' / name; progress = read(run / 'progress.json')
        if progress['status'] != ('incomplete' if i == 3 else 'complete'):
            raise ValueError(f'Offline status changed: {name}')
        if i == 3 and progress['step'] != 228000:
            raise ValueError('Low-lr resume boundary changed')
        keep = {run / 'checkpoints' / n for n in ('initial.pt', 'final.pt')}
        keep.update(Path(read(p)['checkpoint']) for p in (run / 'full_validation').glob('*.json'))
        if (run / 'selection.json').exists():
            keep.add(Path(read(run / 'selection.json')['checkpoint']))
        if progress.get('checkpoint'):
            keep.add(Path(progress['checkpoint']))
        ordinary = {p for p in (run / 'checkpoints').glob('*.pt') if not p.is_symlink() and p not in keep}
        if ordinary != {ROOT / p for p in groups[name]}:
            raise ValueError(f'Ordinary checkpoint selection differs: {name}')
    for run, key in ((OLD, 'old_online_ordinary'), (CURRENT, 'policy_online_ordinary')):
        ordinary = {p for p in run.glob('rounds/*/checkpoints/*.pt')
                    if not p.is_symlink() and p.name not in FIXED_MODELS}
        if ordinary != {ROOT / p for p in groups[key]}:
            raise ValueError(f'Online fixed-node retention differs: {run}')
    cold = set()
    deps = protected_dependencies()
    for i in range(7):
        games = read(CURRENT / f'rounds/round-{i:03d}/collection_index.json')['games']
        cold.update(Path(g['path']) for g in games if g['split'] == 'train' and Path(g['path']) not in deps)
    if cold != {ROOT / p for p in groups['retired_online_training_games']}:
        raise ValueError('Retired train-only NPZ selection changed')
    return sorted(selected, key=lambda r: r['path'])


def freeze(paths, category):
    result = []
    for i, p in enumerate(paths, 1):
        result.append(record(p, category))
        if i % 10000 == 0:
            print(f'Frozen {category}: {i}', flush=True)
    return result


def plan():
    if (ART / 'manifest.json').exists() or STAGE.exists():
        raise ValueError('Refusing to overwrite an existing batch plan')
    approved_files = approved(); selected = {r['path'] for r in approved_files}
    entries = []
    for i, r in enumerate(approved_files, 1):
        p = ROOT / r['path']
        if p.lstat().st_nlink != 1:
            raise ValueError(f'Candidate has another hardlink: {p}')
        row = record(p, r['category']); row['archive_member'] = 'archive/20261101/' + row['path']
        entries.append(row)
        if i % 10000 == 0:
            print(f'Frozen candidates {i}/{len(approved_files)}', flush=True)
    allocated = sum(r['allocated_bytes'] for r in entries)
    if allocated != original_scope()['allocated_unique_bytes']:
        raise ValueError('Candidate allocation differs from approved audit')
    run_files = list(regular_files(ROOT / 'runs'))
    protected = freeze(sorted(p for p in run_files if p.suffix == '.npz' and str(p.relative_to(ROOT)) not in selected), 'retained_npz')
    models = freeze(sorted(p for top in ('runs', 'runtime', 'play_checkpoints')
                          for p in regular_files(ROOT / top)
                          if p.suffix in ('.pt', '.pth') and str(p.relative_to(ROOT)) not in selected), 'retained_model')
    guards = {p for p in run_files if p.suffix not in ('.npz', '.pt', '.pth', '.lock', '.tmp')
              and p.name not in ('.lock',) and str(p.relative_to(ROOT)) not in selected}
    for top in ('artifacts', 'src', 'tests', 'configs', 'scripts', 'docs', 'data'):
        guards.update(p for p in regular_files(ROOT / top)
                      if not p.is_relative_to(ART) and p.suffix not in ('.lock', '.tmp', '.pyc')
                      and str(p.relative_to(ROOT)) not in selected)
    for top in ('runtime/kirameki_suite_20261031', 'runtime/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'):
        guards.update(p for p in regular_files(ROOT / top)
                      if p.suffix not in ('.pt', '.pth', '.lock', '.tmp') and p.name != '.lock'
                      and str(p.relative_to(ROOT)) not in selected)
    guards.add(ROOT / 'README.md')
    guard_rows = freeze(sorted(guards), 'retained_metadata_source_evidence')
    if selected & {r['path'] for r in (*models, *protected, *guard_rows)}:
        raise ValueError('Candidates overlap protected files')
    for name, data in (('files.jsonl.gz', entries), ('retained_models.jsonl.gz', models),
                       ('protected_games.jsonl.gz', protected), ('guards.jsonl.gz', guard_rows)):
        write_rows(ART / name, data)
    summary = {k: dict(files=sum(r['category'] == k for r in entries),
                       payload_bytes=sum(r['size'] for r in entries if r['category'] == k),
                       allocated_bytes=sum(r['allocated_bytes'] for r in entries if r['category'] == k))
               for k in COUNTS}
    write(ART / 'scope_summary.json', summary)
    m = dict(status='planned', created_at=now(), repository=str(ROOT),
             git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             runner_sha256=sha(Path(__file__)), external_archive=str(TARGET), archive_root='archive/20261101',
             scope='User-approved234543 cold files from the2026-11-01 audit only. No cache cleanup. Keep all tracked files, key/fixed/full-validation/resume models, low-lr228000 metrics/state, shared104k, initial64k, active60k, all existing holdouts, pending collection and recent160 website games.',
             categories=COUNTS, total_files=len(entries), retained_models=len(models),
             protected_game_files=len(protected), guard_files=len(guard_rows),
             allocated_bytes_to_release=allocated, payload_bytes=sum(r['size'] for r in entries),
             records={n: dict(path=n, sha256=sha(ART / n)) for n in (
                 'approved_candidates.json.gz', 'files.jsonl.gz', 'retained_models.jsonl.gz',
                 'protected_games.jsonl.gz', 'guards.jsonl.gz', 'scope_summary.json')})
    write(ART / 'manifest.json', m)
    STAGE.mkdir()
    for name in (*m['records'], 'manifest.json', 'run.py'):
        shutil.copy2(ART / name, STAGE / name)
    with (STAGE / 'SHA256SUMS').open('w') as f:
        for r in entries:
            f.write(f"{r['sha256']}  {r['path']}\n")
    (STAGE / 'README.md').write_text(
        '# 2026-11-01 冷产物归档\n\n234543个原件按批准清单流式压缩到T7；本机原路径保留相对恢复链接。'
        'files.jsonl.gz记录逐文件SHA256、大小、纳秒mtime及包内路径；不去重、不改原字节。'
        'guard与retained清单记录保留文件身份；包内manifest为打包前冻结计划，完成记录见artifacts/repository_archive_20261101/。\n\n'
        f'从仓库根目录恢复全部本批数据：\n\n```bash\ntar --zstd -xf "{TARGET}" -C "{ROOT}"\n'
        'cd archive/20261101\nsha256sum -c SHA256SUMS\n```\n\n'
        '部分恢复可将archive/20261101/加在原路径前作为tar成员名；所需逐局sidecar也应一并恢复。'
        '要续写的模型或日志，先按files.jsonl.gz复制回原路径的真实文件并恢复mtime，避免通过链接修改冻结归档。'
        '本批不含初始64k、活动60k、104k共享池及所有固定留出；现在线与低学习率恢复入口所需文件保留。'
        '旧普通checkpoint、旧逐步日志、淘汰训练局或原始评测明细需按需恢复本包；精确范围见scope_summary.json。\n')
    package = [dict(path=r['archive_member'], size=r['size'], mtime_ns=r['mtime_ns'], sha256=r['sha256']) for r in entries]
    package.extend(dict(path=str(p.relative_to(ROOT)), size=p.stat().st_size,
                        mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)) for p in sorted(STAGE.iterdir()))
    write_rows(ART / 'package_inventory.jsonl.gz', package)
    write(ART / 'package_identity.json', dict(manifest_sha256=sha(ART / 'manifest.json'),
                                             inventory_sha256=sha(ART / 'package_inventory.jsonl.gz')))
    print(json.dumps(m, ensure_ascii=False), flush=True)


def validated_manifest():
    identity = read(ART / 'package_identity.json')
    if (sha(ART / 'manifest.json') != identity['manifest_sha256']
            or sha(ART / 'package_inventory.jsonl.gz') != identity['inventory_sha256']):
        raise ValueError('Frozen manifest/inventory changed')
    m = read(ART / 'manifest.json')
    if sha(Path(__file__)) != m['runner_sha256']:
        raise ValueError('Archive runner changed after planning')
    for row in m['records'].values():
        if sha(ART / row['path']) != row['sha256']:
            raise ValueError('Frozen per-file manifest changed')
    return m


def validate_guards():
    for name in ('retained_models.jsonl.gz', 'protected_games.jsonl.gz', 'guards.jsonl.gz'):
        data = rows(ART / name)
        for i, row in enumerate(data, 1):
            unchanged(row)
            if i % 20000 == 0:
                print(f'Rechecked {name}: {i}/{len(data)}', flush=True)


def verify_tar(path):
    expected = {r['path']: r for r in rows(ART / 'package_inventory.jsonl.gz')}; seen = set()
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                row = expected[member.name]
                ns = int(Decimal(member.pax_headers.get('mtime', str(member.mtime))) * 10**9)
                if member.size != row['size'] or ns != row['mtime_ns']:
                    raise ValueError(f'Archive size/mtime mismatch: {member.name}')
                digest = hashlib.sha256(); handle = archive.extractfile(member)
                for block in iter(lambda: handle.read(4 * 1024**2), b''):
                    digest.update(block)
                if digest.hexdigest() != row['sha256']:
                    raise ValueError(f'Archive SHA mismatch: {member.name}')
                seen.add(member.name)
                if len(seen) % 20000 == 0:
                    print(f'Verified external files {len(seen)}/{len(expected)}', flush=True)
        for _ in iter(lambda: process.stdout.read(4 * 1024**2), b''):
            pass
        if process.wait() != 0 or seen != set(expected):
            raise ValueError('Incomplete archive/decompression')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.kill(); process.wait()
    return dict(regular_files=len(seen), all_sha256_match=True, all_nanosecond_mtime_preserved=True)


def export():
    m = validated_manifest()
    if m['status'] != 'planned' or TARGET.exists() or PARTIAL.exists() or TARGET.with_name(TARGET.name + '.sha256').exists():
        raise ValueError('Refusing to overwrite existing archive/partial/checksum')
    if not Path('/media/lc/T7').is_mount() or TARGET.parent.stat().st_dev == ROOT.stat().st_dev:
        raise ValueError('T7 must be mounted on a different filesystem')
    if shutil.disk_usage(TARGET.parent).free <= m['payload_bytes']:
        raise ValueError('Insufficient T7 space')
    entries = rows(ART / 'files.jsonl.gz')
    for row in entries:
        unchanged(row, content=False)
    validate_guards()
    listing = RUNTIME / 'files.nul'
    listing.write_bytes(b''.join(p.encode() + b'\0' for p in
                               [r['path'] for r in entries] + [str(p.relative_to(ROOT)) for p in sorted(STAGE.iterdir())]))
    started = time.monotonic()
    write(ART / 'export_state.json', dict(status='compressing', started_at=now(), partial=str(PARTIAL)))
    with PARTIAL.open('xb') as output:
        packer = subprocess.Popen(['tar', '--format=pax', '--pax-option=delete=atime,delete=ctime',
                                   '--transform=s,^runs/,archive/20261101/runs/,',
                                   '--transform=s,^runtime/,archive/20261101/runtime/,',
                                   '--transform=s,^artifacts/,archive/20261101/artifacts/,',
                                   '-C', str(ROOT), '--null', '--verbatim-files-from', '--no-recursion',
                                   '-T', str(listing), '-cf', '-'], stdout=subprocess.PIPE)
        compressor = subprocess.Popen(['zstd', '-q', '-T2', '-3', '-c'], stdin=packer.stdout, stdout=output)
        packer.stdout.close()
        zcode, tcode = compressor.wait(), packer.wait()
        if zcode or tcode:
            raise RuntimeError(f'tar/zstd exit codes: {tcode}/{zcode}')
        output.flush(); os.fsync(output.fileno())
    write(ART / 'export_state.json', dict(status='verifying', updated_at=now(), partial=str(PARTIAL)))
    print(f'Compressed {PARTIAL.stat().st_size} bytes; verifying whole external stream', flush=True)
    result = verify_tar(PARTIAL); digest = sha(PARTIAL)
    PARTIAL.rename(TARGET)
    with TARGET.with_name(TARGET.name + '.sha256').open('x') as f:
        f.write(f'{digest}  {TARGET.name}\n'); f.flush(); os.fsync(f.fileno())
    result.update(status='verified_external_local_originals_retained', verified_at=now(),
                  target=str(TARGET), size=TARGET.stat().st_size, sha256=digest,
                  seconds=time.monotonic()-started)
    write(ART / 'external_verification.json', result); write(ART / 'export_state.json', result)
    print(json.dumps(result), flush=True)


def release():
    m = validated_manifest(); v = read(ART / 'external_verification.json')
    if m['status'] != 'planned' or v['status'] != 'verified_external_local_originals_retained':
        raise ValueError('No verified external archive awaiting release')
    if TARGET.stat().st_size != v['size'] or sha(TARGET) != v['sha256']:
        raise ValueError('External archive identity changed')
    validate_guards()
    entries = rows(ART / 'files.jsonl.gz')
    for i, row in enumerate(entries, 1):
        unchanged(row)
        if i % 20000 == 0:
            print(f'Rechecked release originals {i}/{len(entries)}', flush=True)
    free_before = shutil.disk_usage(ROOT).free
    with (RUNTIME / 'moves.jsonl').open('x') as journal:
        for i, row in enumerate(entries, 1):
            unchanged(row, content=False)
            source, destination = ROOT / row['path'], ROOT / row['archive_member']
            link = source.with_name(source.name + '.archive-link.tmp')
            if link.exists() or link.is_symlink():
                raise ValueError(f'Temporary archive link already exists: {link}')
            link.symlink_to(os.path.relpath(destination, source.parent)); link.replace(source)
            journal.write(json.dumps(dict(path=row['path'], target=row['archive_member'])) + '\n')
            if i % 1000 == 0:
                journal.flush(); os.fsync(journal.fileno())
            if i % 20000 == 0:
                print(f'Released {i}/{len(entries)}', flush=True)
        journal.flush(); os.fsync(journal.fileno())
    for row in entries:
        p = ROOT / row['path']
        if not p.is_symlink() or os.readlink(p) != os.path.relpath(ROOT / row['archive_member'], p.parent):
            raise ValueError(f'Archive link mismatch: {p}')
    validate_guards()
    shutil.copyfile(RUNTIME / 'moves.jsonl', STAGE / 'moves.jsonl')
    links = sum((ROOT / r['path']).lstat().st_blocks * 512 for r in entries)
    result = dict(status='complete', completed_at=now(), external_archive=str(TARGET),
                  external_size=v['size'], external_sha256=v['sha256'], externalized=len(entries),
                  categories=m['categories'], removed_original_allocated_bytes=m['allocated_bytes_to_release'],
                  allocated_symlink_bytes=links, net_payload_space_released=m['allocated_bytes_to_release']-links,
                  observed_local_free_delta=shutil.disk_usage(ROOT).free-free_before,
                  protected_game_files_verified=m['protected_game_files'], retained_models_verified=m['retained_models'],
                  guard_files_verified=m['guard_files'], online_resume_inputs_unchanged=True,
                  compiled_caches_untouched=True)
    write(ART / 'completion.json', result)
    m.update(status='externalized', completed_at=result['completed_at'], external_sha256=v['sha256'], external_size=v['size'])
    write(ART / 'manifest.json', m)
    write(STAGE / 'completion.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'export', 'release', 'finish', 'all'))
    args = parser.parse_args(); RUNTIME.mkdir(parents=True, exist_ok=True)
    selected = approved() if args.phase in ('plan', 'all') else rows(ART / 'files.jsonl.gz')
    groups = {ROOT / '/'.join(r['path'].split('/')[:2]) for r in selected}
    groups.update((PARENT, OLD, CURRENT, *(ROOT / 'runs' / n for n in OFFLINE_NAMES)))
    with ExitStack() as stack:
        own = stack.enter_context((RUNTIME / '.lock').open('a'))
        fcntl.flock(own, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group in sorted(groups):
            fd = os.open(group, os.O_RDONLY); stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for p in sorted(set(group.glob('*.lock')) | set(group.glob('.*lock'))):
                f = stack.enter_context(p.open('a')); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phases = ('plan', 'export', 'release') if args.phase == 'all' else (
            ('export', 'release') if args.phase == 'finish' else (args.phase,))
        for phase in phases:
            {'plan': plan, 'export': export, 'release': release}[phase]()


if __name__ == '__main__':
    main()
