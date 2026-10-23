"""Archive the authorized rounds21-30 and interrupted Cross64 offline cold files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
ONLINE = ROOT / 'runs/entity_attention_r20_s256n4_noble12_iteration_20261022'
OFFLINE = ROOT / 'runs/entity_set_mlp_cross64_role005_offline_80k_20261022'
ORIGINAL = ROOT / 'runs/entity_attention_37423_random_budget_p6k_v24k_20261019'
STAGE = ROOT / 'archive/20261022'
TARGET = Path('/media/lc/T7/splendor/archive-20261022.tar.zst')
PARTIAL = TARGET.with_name(TARGET.name + '.partial')


def now():
    return datetime.now().astimezone().isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for data in iter(lambda: handle.read(1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def record(path, category):
    assert path.is_file() and not path.is_symlink(), path
    st = path.stat()
    return dict(path=str(path.relative_to(ROOT)), category=category, size=st.st_size,
                allocated_bytes=st.st_blocks * 512, mtime_ns=st.st_mtime_ns,
                inode=st.st_ino, sha256=sha(path))


def unchanged(row):
    path = ROOT / row['path']
    assert not path.is_symlink() and path.is_file(), path
    st = path.stat()
    assert (st.st_size, st.st_mtime_ns, st.st_ino) == (row['size'], row['mtime_ns'], row['inode']), path
    assert sha(path) == row['sha256'], path


def protected_groups():
    state = read(ONLINE / 'state.json')
    latest = state['completed_rounds'][-1]
    assert latest['iteration'] == 31
    snapshot = read(Path(latest['snapshot']['path']))
    active = [Path(g['path']) for g in snapshot['games']]
    assert len(active) == 24000
    all_collections = sorted(ONLINE.glob('rounds/round-0[23][0-9]/selfplay/games/*.npz'))
    # The explicit latest instruction keeps every collected game, including
    # retired training games. The active24k is contained in this protected set.
    all_collections = [p for p in all_collections if 20 <= int(p.parts[-4].split('-')[-1]) <= 30]
    assert len(all_collections) == 44000 and set(active) <= set(all_collections)
    return {
        'original_80000_npz': sorted(ORIGINAL.glob('rounds/*/selfplay/games/*.npz')),
        'all_round21_31_npz': all_collections,
        'partial_round32_npz': sorted((ONLINE / 'rounds/round-031/selfplay/games').glob('*.npz')),
    }


def fingerprints(groups):
    result = {}
    for name, paths in groups.items():
        h = hashlib.sha256()
        size = 0
        for p in paths:
            r = record(p, name)
            size += r['size']
            h.update((json.dumps(r, sort_keys=True) + '\n').encode())
        result[name] = dict(count=len(paths), bytes=size, sha256_of_ordered_file_identities=h.hexdigest())
        print(f'Protected {name}: {len(paths)} files', flush=True)
    assert result['original_80000_npz']['count'] == 80000
    return result


def plan():
    assert not (ART / 'manifest.json').exists() and not STAGE.exists()
    groups = protected_groups()
    protected = {str(p) for paths in groups.values() for p in paths}
    candidates, keep = [], []
    for i in range(20, 30):
        r = ONLINE / f'rounds/round-{i:03d}'
        keep.append(record(r / 'checkpoints/final.pt', 'online_round_final'))
        candidates.extend(record(p, 'online_intermediate_checkpoint') for p in sorted((r / 'checkpoints').glob('*.pt')) if p.name != 'final.pt')
        candidates.append(record(r / 'training_metrics.jsonl', 'online_training_ledger'))
        candidates.extend(record(p, 'online_sampling_ledger') for p in sorted((r / 'usage').rglob('*')) if p.is_file() and (p.suffix == '.bin' or p.name.endswith('.steps.jsonl')))
    validation = sorted((OFFLINE / 'full_validation').glob('*.json'))
    assert len(validation) == 6
    keep_names = {'initial.pt'} | {p.stem + '.pt' for p in validation}
    for p in sorted((OFFLINE / 'checkpoints').glob('*.pt')):
        (keep if p.name in keep_names else candidates).append(record(p, 'offline_key_checkpoint' if p.name in keep_names else 'offline_ordinary_checkpoint'))
    assert len([r for r in keep if r['category'] == 'offline_key_checkpoint']) == 7
    candidates.append(record(OFFLINE / 'training_metrics.jsonl', 'offline_training_ledger'))
    candidates.extend(record(p, 'offline_sampling_ledger') for p in sorted((OFFLINE / 'references').rglob('*')) if p.is_file() and (p.suffix == '.bin' or p.name.endswith('.steps.jsonl')))
    assert len({r['path'] for r in candidates}) == len(candidates)
    assert all(str(ROOT / r['path']) not in protected for r in candidates)
    by_hash = defaultdict(list)
    for row in keep + candidates:
        by_hash[(row['size'], row['sha256'])].append(row)
    kept = {r['path'] for r in keep}
    entries, duplicates = [], []
    for rows in by_hash.values():
        rows.sort(key=lambda r: (r['path'] not in kept, not r['path'].endswith('/training_metrics.jsonl'), r['path']))
        canonical = rows[0]
        for row in rows:
            if row['path'] in kept:
                continue
            if row is canonical:
                entries.append(row | {'archive_member': 'archive/20261022/' + row['path']})
            else:
                duplicates.append(row | {'identical_to': canonical['path']})
    guard_paths = [ONLINE / 'state.json', ONLINE / 'config.json', OFFLINE / 'progress.json', OFFLINE / 'config.json',
                   Path(read(ONLINE / 'state.json')['latest_checkpoint']['path']),
                   ONLINE / 'rounds/round-030/replay_snapshot.json',
                   OFFLINE / 'inputs/pool_games.jsonl', OFFLINE / 'dataset.json', OFFLINE / 'fixed_validation.json']
    manifest = dict(status='planned', created_at=now(), repository=str(ROOT),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        external_archive=str(TARGET), archive_root='archive/20261022',
        scope='Only redundant models and heavy training/sampling ledgers from new online branch iterations21-30 (physical round020-029) and Cross64 role005 offline run. Latest user instruction explicitly retains ALL collected games for iterations21-31:44000NPZ, including retired training games, plus partial round32 and original80000NPZ. No NPZ or collection metadata moved. Keep all online finals and seven offline initial/percentage checkpoints; no cache deletion or other offline-run changes.',
        entries=sorted(entries, key=lambda r: r['path']), deduplications=sorted(duplicates, key=lambda r: r['path']),
        retained_models=keep, guards=[record(p, 'resume_guard') for p in guard_paths],
        protected_data=fingerprints(groups),
        categories=dict(Counter(r['category'] for r in candidates)),
        allocated_bytes_to_release=sum(r['allocated_bytes'] for r in candidates),
        unique_payload_bytes=sum(r['size'] for r in entries))
    write(ART / 'manifest.json', manifest)
    STAGE.mkdir(parents=True)
    shutil.copyfile(ART / 'manifest.json', STAGE / 'manifest.json')
    (STAGE / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in manifest['entries']))
    (STAGE / 'README.md').write_text(
        '# 2026-10-22 冷产物归档\n\n真实原件存于T7；本机仅留元数据。原路径保留相对链接。\n\n'
        f'恢复：从仓库根目录执行 `tar --zstd -xf "{TARGET}" -C "{ROOT}"`，然后在本目录运行 `sha256sum -c SHA256SUMS`。\n\n'
        '在线第21–31轮全部44000局、当前第32轮未完成采集、原80k训练池、各轮final及离线关键节点仍在本机，在线续跑不需要本包。'
        '旧轮训练复算／离线逐步分析需先解压；恢复离线训练前应将所需账本或模型复制回原位真实文件，避免通过归档链接修改冻结原件。'
        '重复别名恢复为独立文件的方法见长期清单。\n')
    package = []
    for r in manifest['entries']:
        package.append(dict(path=r['archive_member'], size=r['size'], mtime_ns=r['mtime_ns'], sha256=r['sha256'], payload=True))
    for p in sorted(STAGE.iterdir()):
        st = p.stat()
        package.append(dict(path=str(p.relative_to(ROOT)), size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha(p), payload=False))
    write(ART / 'package_inventory.json', package)
    print(json.dumps({k:manifest[k] for k in ['status','categories','allocated_bytes_to_release','unique_payload_bytes']}) ,flush=True)
    print(f'Unique payload files {len(entries)}; duplicates {len(duplicates)}; retained models {len(keep)}', flush=True)


def verify_tar(path):
    expected = {r['path']:r for r in read(ART / 'package_inventory.json')}
    seen = set()
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                assert member.isfile() and member.name in expected and member.name not in seen, member.name
                row = expected[member.name]
                assert member.size == row['size'], member.name
                assert int(Decimal(member.pax_headers.get('mtime', str(member.mtime))) * 10**9) == row['mtime_ns'], member.name
                h = hashlib.sha256()
                handle = archive.extractfile(member)
                for data in iter(lambda:handle.read(1024**2), b''):h.update(data)
                assert h.hexdigest() == row['sha256'], member.name
                seen.add(member.name)
                if len(seen) % 4000 == 0:print(f'Verified external files {len(seen)}/{len(expected)}', flush=True)
        for _ in iter(lambda:process.stdout.read(1024**2), b''):pass
        assert process.wait() == 0 and seen == set(expected)
    finally:
        process.stdout.close()
        if process.poll() is None:process.kill();process.wait()
    return dict(regular_file_count=len(seen), all_member_sha256_match=True, all_nanosecond_mtime_preserved=True, all_payload_real_files=True)


def export():
    m = read(ART / 'manifest.json')
    assert m['status'] == 'planned' and not TARGET.exists() and not PARTIAL.exists()
    assert TARGET.parent.is_dir() and os.access(TARGET.parent, os.W_OK)
    assert shutil.disk_usage(TARGET.parent).free > m['unique_payload_bytes']
    for r in m['guards'] + m['retained_models'] + m['entries'] + m['deduplications']:unchanged(r)
    paths = [r['path'] for r in m['entries']] + [str(p.relative_to(ROOT)) for p in sorted(STAGE.iterdir())]
    listing = ROOT / 'runtime/repository_archive_20261022/files.nul'
    listing.parent.mkdir(parents=True, exist_ok=True)
    listing.write_bytes(b''.join(p.encode() + b'\0' for p in paths))
    os.nice(10)
    started = time.monotonic()
    write(ART / 'export_state.json', dict(status='compressing', started_at=now(), partial=str(PARTIAL)))
    with PARTIAL.open('xb') as output:
        packer = subprocess.Popen(['tar','--format=pax','--pax-option=delete=atime,delete=ctime',
            '--transform=s,^runs/,archive/20261022/runs/,','-C',str(ROOT),'--null','--verbatim-files-from',
            '--no-recursion','-T',str(listing),'-cf','-'], stdout=subprocess.PIPE)
        compressor = subprocess.Popen(['zstd','-q','-T2','-3','-c'], stdin=packer.stdout, stdout=output)
        packer.stdout.close()
        zcode, tcode = compressor.wait(), packer.wait()
        assert zcode == tcode == 0, (zcode,tcode)
        output.flush();os.fsync(output.fileno())
    write(ART / 'export_state.json', dict(status='verifying', partial=str(PARTIAL), updated_at=now()))
    print(f'Compressed {PARTIAL.stat().st_size} bytes; verifying complete stream', flush=True)
    verification = verify_tar(PARTIAL)
    digest = sha(PARTIAL)
    PARTIAL.rename(TARGET)
    with TARGET.with_name(TARGET.name+'.sha256').open('x') as f:
        f.write(f'{digest}  {TARGET.name}\n');f.flush();os.fsync(f.fileno())
    result = dict(status='verified_external_local_originals_retained', verified_at=now(), target=str(TARGET),
                  size=TARGET.stat().st_size, sha256=digest, seconds=time.monotonic()-started, **verification)
    write(ART / 'external_verification.json', result)
    write(ART / 'export_state.json', result)
    print(json.dumps(result),flush=True)


def release():
    m = read(ART / 'manifest.json');v = read(ART / 'external_verification.json')
    assert m['status'] == 'planned' and v['status'] == 'verified_external_local_originals_retained'
    assert TARGET.stat().st_size == v['size'] and sha(TARGET) == v['sha256']
    assert fingerprints(protected_groups()) == m['protected_data']
    for r in m['guards'] + m['retained_models'] + m['entries'] + m['deduplications']:unchanged(r)
    before = shutil.disk_usage(ROOT).free
    with (ART / 'moves.jsonl').open('x') as journal:
        # Duplicates keep canonical originals as relative aliases; archive entries
        # retain original names as links into the separately recoverable package.
        for r in m['deduplications'] + m['entries']:
            source = ROOT / r['path']
            target = ROOT / r.get('identical_to', r.get('archive_member'))
            link = source.with_name(source.name + '.archive-link.tmp')
            assert not link.exists() and not link.is_symlink()
            link.symlink_to(os.path.relpath(target, source.parent))
            link.replace(source)
            journal.write(json.dumps(dict(path=r['path'], target=str(target.relative_to(ROOT))))+'\n')
        journal.flush();os.fsync(journal.fileno())
    for r in m['entries'] + m['deduplications']:
        source = ROOT / r['path']
        assert source.is_symlink() and os.readlink(source) == os.path.relpath(ROOT/r.get('identical_to',r.get('archive_member')), source.parent)
    for r in m['guards'] + m['retained_models']:unchanged(r)
    assert fingerprints(protected_groups()) == m['protected_data']
    m.update(status='externalized', completed_at=now(), external_sha256=v['sha256'], external_size=v['size'])
    write(ART / 'manifest.json', m)
    write(STAGE / 'manifest.json', m)
    result = dict(status='complete', completed_at=now(), external_archive=str(TARGET),
        external_size=v['size'], external_sha256=v['sha256'], externalized=len(m['entries']),
        deduplicated=len(m['deduplications']), retained_models=len(m['retained_models']),
        removed_original_allocated_bytes=m['allocated_bytes_to_release'],
        allocated_symlink_bytes=sum((ROOT/r['path']).lstat().st_blocks*512 for r in m['entries']+m['deduplications']),
        observed_local_free_delta=shutil.disk_usage(ROOT).free-before,
        protected_data=m['protected_data'], online_resume_pool_and_latest_checkpoint_verified=True)
    result['net_payload_space_released']=result['removed_original_allocated_bytes']-result['allocated_symlink_bytes']
    write(ART / 'completion.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['plan','export','release'])
    args=parser.parse_args()
    # Use the same locks as the user entry points; don't archive live writers.
    with ExitStack() as stack:
        for p in [ONLINE/'.iteration.lock', OFFLINE/'.lock', ROOT/'runtime/entity_set_mlp_cross64_role005_offline_80k_20261022/chain.lock']:
            f=stack.enter_context(p.open('a'));fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        {'plan':plan,'export':export,'release':release}[args.phase]()


if __name__=='__main__':
    main()
