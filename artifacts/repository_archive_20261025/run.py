"""Archive the four completed auxiliary runs and the original80k games to T7."""
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
import subprocess
import tarfile
import time
from zoneinfo import ZoneInfo

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
STAGE = ROOT / 'archive/20261025'
RUNTIME = ROOT / 'runtime/repository_archive_20261025'
TARGET = Path('/media/lc/T7/splendor/archive-20261025.tar.zst')
PARTIAL = TARGET.with_name(TARGET.name + '.partial')
CURRENT = ROOT / 'runs/entity_attention_policy140221_r31_r37_p60k_iteration_20261025'
OLD_ONLINE = ROOT / 'runs/entity_attention_r31_10k_p60k_iteration_20261023'
ORIGINAL = ROOT / 'runs/entity_attention_37423_random_budget_p6k_v24k_20261019'
PARENT = ROOT / 'runs/entity_attention_r20_s256n4_noble12_iteration_20261022'
OFFLINE = tuple(ROOT / 'runs' / name for name in (
    'entity_attention_128x2_shortterm_value_r21_r37_20261023',
    'entity_attention_128x2_opponent_policy_r21_r37_20261024',
    'entity_attention_128x2_continuation_r21_r37_20261024',
    'entity_attention_128x2_shortterm_value_r21_r31_vs_r37_20261024'))
COUNTS = {'offline_ordinary_checkpoint': 672, 'offline_training_metrics': 4,
          'original_80000_npz': 80000}


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n'); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path, rows):
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')


def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        return [json.loads(line) for line in handle]


def record(path, category):
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected original regular file: {path}')
    st = path.stat()
    result = dict(path=str(path.relative_to(ROOT)), category=category, size=st.st_size,
        allocated_bytes=st.st_blocks * 512, mtime_ns=st.st_mtime_ns,
        inode=[st.st_dev, st.st_ino], sha256=sha(path))
    after = path.stat()
    if (st.st_size, st.st_mtime_ns, st.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed during hashing: {path}')
    return result


def unchanged(row, *, content=True):
    path = ROOT / row['path']
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Original is no longer a regular file: {path}')
    st = path.stat()
    if (st.st_size, st.st_mtime_ns, [st.st_dev, st.st_ino]) != (row['size'], row['mtime_ns'], row['inode']):
        raise ValueError(f'Original metadata changed: {path}')
    if content and sha(path) != row['sha256']:
        raise ValueError(f'Original SHA changed: {path}')


def approved():
    selected = []
    for root in OFFLINE:
        if read(root / 'progress.json')['status'] != 'complete':
            raise ValueError(f'Offline experiment is not complete: {root}')
        keep = {Path(read(p)['checkpoint']).resolve() for p in (root / 'full_validation').glob('*.json')}
        keep.update((root / 'checkpoints' / name).resolve() for name in ('initial.pt', 'final.pt'))
        keep.add(Path(read(root / 'selection.json')['checkpoint']).resolve())
        for path in sorted((root / 'checkpoints').glob('*.pt')):
            if path.resolve() not in keep:
                st = path.lstat()
                selected.append(dict(path=str(path.relative_to(ROOT)), candidate='offline_ordinary_checkpoint',
                                     size=st.st_size, mtime_ns=st.st_mtime_ns))
        path = root / 'training_metrics.jsonl'; st = path.lstat()
        selected.append(dict(path=str(path.relative_to(ROOT)), candidate='offline_training_metrics',
                             size=st.st_size, mtime_ns=st.st_mtime_ns))
    for path in sorted(ORIGINAL.glob('rounds/*/selfplay/games/*.npz')):
        st = path.lstat()
        selected.append(dict(path=str(path.relative_to(ROOT)), candidate='original_80000_npz',
                             size=st.st_size, mtime_ns=st.st_mtime_ns))
    if dict(Counter(r['candidate'] for r in selected)) != COUNTS:
        raise ValueError('Approved checkpoint/log/original80k census differs')
    if len({r['path'] for r in selected}) != sum(COUNTS.values()):
        raise ValueError('Duplicate approved path')
    config = read(ROOT / 'configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json')
    history = {json.loads(line)['path'] for line in Path(config['history_manifest']['path']).read_text().splitlines()}
    if len(history) != 64000 or history & {str(ROOT / r['path']) for r in selected}:
        raise ValueError('Selected paths overlap the current original64k input or its count changed')
    return sorted(selected, key=lambda r: r['path'])


def protected_groups():
    return {
        'parent_46142_npz': sorted(PARENT.glob('rounds/*/selfplay/games/*.npz')),
        'old_online_60000_npz': sorted(OLD_ONLINE.glob('rounds/*/selfplay/games/*.npz')),
        'new_online_58320_npz': sorted(CURRENT.glob('rounds/*/selfplay/games/*.npz')),
    }


def fingerprints():
    expected = {'parent_46142_npz': 46142, 'old_online_60000_npz': 60000,
                'new_online_58320_npz': 58320}
    result = {}
    for name, paths in protected_groups().items():
        if len(paths) != expected[name]:
            raise ValueError(f'Protected data population changed: {name}')
        digest = hashlib.sha256(); size = 0
        for i, path in enumerate(paths, 1):
            row = record(path, name); size += row['size']
            digest.update((json.dumps(row, sort_keys=True) + '\n').encode())
            if i % 20000 == 0:
                print(f'Protected {name}: {i}/{len(paths)}', flush=True)
        result[name] = dict(count=len(paths), bytes=size, sha256_of_ordered_file_identities=digest.hexdigest())
    return result


def guard_paths():
    paths = set()
    for run in (ROOT / 'runs').iterdir():
        if not run.is_dir():
            continue
        for name in ('config.json', 'state.json', 'hybrid_state.json', 'progress.json',
                     'selection.json', 'dataset.json', 'fixed_validation.json', 'plan.json'):
            p = run / name
            if p.is_file() and not p.is_symlink():
                paths.add(p)
    for run in (*OFFLINE, ORIGINAL, OLD_ONLINE, CURRENT):
        paths.update(run.glob('rounds/*/collection_index.json'))
        paths.update(run.glob('rounds/*/replay_snapshot.json'))
        paths.update(run.glob('full_validation/*.json'))
    # Guard every non-model file in the current branch, including completed arenas
    # and the interrupted sixth-round failure records. NPZ/model guards are separate.
    for p in CURRENT.rglob('*'):
        if p.is_file() and not p.is_symlink() and p.suffix not in ('.npz', '.pt', '.pth', '.lock', '.tmp') and not p.name.endswith('lock'):
            paths.add(p)
    paths.add(ROOT / 'configs/entity_attention_v1/random_budget_policy140221_r31_r37_p60k_20261025.json')
    paths.add(ROOT / 'scripts/run_policy140221_r31_r37_p60k_iteration_20261025.sh')
    source_record = read(ROOT / 'artifacts/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/source.json')
    paths.update(ROOT / 'runtime/entity_attention_policy140221_r31_r37_p60k_iteration_20261025/source' / p
                 for p in source_record['snapshot_files'])
    for run in OFFLINE:
        evidence = ROOT / 'artifacts' / run.name
        paths.update(evidence.glob('panels/*.json'))
        paths.add(evidence / 'selected_candidate.json')
    return sorted(paths)


def plan():
    if (ART / 'manifest.json').exists() or STAGE.exists():
        raise ValueError('Batch already planned; inspect its manifest instead of overwriting it')
    selected = approved(); selected_paths = {r['path'] for r in selected}
    protected = {str(p.relative_to(ROOT)) for paths in protected_groups().values() for p in paths}
    if selected_paths & protected:
        raise ValueError('Approved paths overlap current/protected game data')
    entries = []
    for i, source in enumerate(selected, 1):
        path = ROOT / source['path']; st = path.lstat()
        if path.is_symlink() or (st.st_size, st.st_mtime_ns) != (int(source['size']), int(source['mtime_ns'])):
            raise ValueError(f'Candidate differs from approved audit: {path}')
        row = record(path, source['candidate'])
        row['archive_member'] = 'archive/20261025/' + row['path']
        entries.append(row)
        if i % 5000 == 0:
            print(f'Frozen candidates {i}/{len(selected)}', flush=True)
    if sum(r['allocated_bytes'] for r in entries) <= 0:
        raise ValueError('Candidate allocated population differs from approved audit')
    retained = []
    for top in ('runs', 'runtime', 'play_checkpoints'):
        for path in (ROOT / top).rglob('*'):
            if path.suffix in ('.pt', '.pth') and path.is_file() and not path.is_symlink() and str(path.relative_to(ROOT)) not in selected_paths:
                retained.append(record(path, 'retained_checkpoint'))
    guards = [record(p, 'resume_and_evidence_guard') for p in guard_paths()]
    if selected_paths & {r['path'] for r in guards}:
        raise ValueError('Approved path overlaps retained metadata')
    write_rows(ART / 'files.jsonl.gz', entries)
    write_rows(ART / 'retained_models.jsonl.gz', retained)
    write_rows(ART / 'guards.jsonl.gz', guards)
    manifest = dict(status='planned', created_at=now(), repository=str(ROOT),
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        runner_sha256=sha(Path(__file__)), external_archive=str(TARGET), archive_root='archive/20261025',
        scope='User-approved2026-10-25 only:672 ordinary checkpoints and4 main training_metrics.jsonl from four completed auxiliary offline experiments; original80000 historical game NPZ. Keep all selected/full-validation/initial/final checkpoints, old-online324 ordinary checkpoints, parent46142 games, old-online60000 games and new-online58320 games. No other cleanup or deduplication.',
        categories=COUNTS, total_files=len(entries), retained_models=len(retained), guard_files=len(guards),
        allocated_bytes_to_release=sum(r['allocated_bytes'] for r in entries),
        payload_bytes=sum(r['size'] for r in entries), protected_data=fingerprints(),
        records={name: dict(path=name, sha256=sha(ART / name))
                 for name in ('files.jsonl.gz', 'retained_models.jsonl.gz', 'guards.jsonl.gz')},
        metadata_format='gzip JSONL full per-file identities; compressed80676-file audit')
    write(ART / 'manifest.json', manifest)
    STAGE.mkdir(parents=True)
    for name in ('manifest.json', 'files.jsonl.gz'):
        shutil.copyfile(ART / name, STAGE / name)
    (STAGE / 'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in entries))
    (STAGE / 'README.md').write_text(
        '# 2026-10-25 冷产物归档\n\n原件流式压缩至T7，本机保留清单和原路径相对链接。'
        'files.jsonl.gz为逐路径SHA、大小、纳秒mtime、inode与归档路径；保留原字节，不去重。\n\n'
        f'从仓库根目录恢复：\n\n```bash\ntar --zstd -xf "{TARGET}" -C "{ROOT}"\n'
        'cd archive/20261025\nsha256sum -c SHA256SUMS\n```\n\n'
        '旧实验复训前，将需写入的模型／账本按files.jsonl.gz复制回原位真实文件并恢复mtime，'
        '避免通过链接修改冻结归档。本批原80k局已外置；其离线复训／完整复核须恢复。本机保留R21–37及新在线对局、全部完整留出模型、最优模型、initial/final及当前恢复依赖。\n')
    package = [dict(path=r['archive_member'], size=r['size'], mtime_ns=r['mtime_ns'], sha256=r['sha256']) for r in entries]
    package.extend(dict(path=str(p.relative_to(ROOT)), size=p.stat().st_size,
        mtime_ns=p.stat().st_mtime_ns, sha256=sha(p)) for p in sorted(STAGE.iterdir()))
    write_rows(ART / 'package_inventory.jsonl.gz', package)
    print(json.dumps({k: manifest[k] for k in ('status', 'total_files', 'categories', 'allocated_bytes_to_release', 'payload_bytes', 'retained_models')}), flush=True)


def validated_manifest():
    m = read(ART / 'manifest.json')
    for row in m['records'].values():
        if sha(ART / row['path']) != row['sha256']:
            raise ValueError('Frozen per-file manifest changed')
    return m


def validate_guards(m):
    for name in ('retained_models.jsonl.gz', 'guards.jsonl.gz'):
        for row in rows(ART / name):
            unchanged(row)
    if fingerprints() != m['protected_data']:
        raise ValueError('Protected raw data changed')


def verify_tar(path):
    expected = {r['path']: r for r in rows(ART / 'package_inventory.jsonl.gz')}; seen = set()
    process = subprocess.Popen(['zstd', '-q', '-d', '-c', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                if not member.isfile() or member.name not in expected or member.name in seen:
                    raise ValueError(f'Unexpected archive member: {member.name}')
                row = expected[member.name]
                if member.size != row['size'] or int(Decimal(member.pax_headers.get('mtime', str(member.mtime))) * 10**9) != row['mtime_ns']:
                    raise ValueError(f'Archive size/mtime mismatch: {member.name}')
                digest = hashlib.sha256(); handle = archive.extractfile(member)
                for block in iter(lambda: handle.read(4 * 1024**2), b''):
                    digest.update(block)
                if digest.hexdigest() != row['sha256']:
                    raise ValueError(f'Archive SHA mismatch: {member.name}')
                seen.add(member.name)
                if len(seen) % 10000 == 0:
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
    if m['status'] != 'planned' or TARGET.exists() or PARTIAL.exists():
        raise ValueError('Refusing to overwrite an archive or partial export')
    if not Path('/media/lc/T7').is_mount() or TARGET.parent.stat().st_dev == ROOT.stat().st_dev:
        raise ValueError('T7 must be mounted on a different filesystem')
    if shutil.disk_usage(TARGET.parent).free <= m['payload_bytes']:
        raise ValueError('Insufficient T7 space')
    entries = rows(ART / 'files.jsonl.gz')
    for row in entries:
        unchanged(row, content=False)
    validate_guards(m)
    listing = RUNTIME / 'files.nul'
    listing.write_bytes(b''.join(p.encode() + b'\0' for p in
        [r['path'] for r in entries] + [str(p.relative_to(ROOT)) for p in sorted(STAGE.iterdir())]))
    os.nice(10); started = time.monotonic()
    write(ART / 'export_state.json', dict(status='compressing', started_at=now(), partial=str(PARTIAL)))
    with PARTIAL.open('xb') as output:
        packer = subprocess.Popen(['tar', '--format=pax', '--pax-option=delete=atime,delete=ctime',
            '--transform=s,^runs/,archive/20261025/runs/,', '--transform=s,^runtime/,archive/20261025/runtime/,',
            '-C', str(ROOT), '--null', '--verbatim-files-from', '--no-recursion', '-T', str(listing), '-cf', '-'], stdout=subprocess.PIPE)
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
    with TARGET.with_name(TARGET.name + '.sha256').open('x') as handle:
        handle.write(f'{digest}  {TARGET.name}\n'); handle.flush(); os.fsync(handle.fileno())
    result.update(status='verified_external_local_originals_retained', verified_at=now(),
        target=str(TARGET), size=TARGET.stat().st_size, sha256=digest, seconds=time.monotonic()-started)
    write(ART / 'external_verification.json', result); write(ART / 'export_state.json', result)
    print(json.dumps(result), flush=True)


def release():
    m = validated_manifest(); v = read(ART / 'external_verification.json')
    if m['status'] != 'planned' or v['status'] != 'verified_external_local_originals_retained':
        raise ValueError('No verified external archive awaiting release')
    if TARGET.stat().st_size != v['size'] or sha(TARGET) != v['sha256']:
        raise ValueError('External archive identity changed')
    validate_guards(m)
    entries = rows(ART / 'files.jsonl.gz')
    for i, row in enumerate(entries, 1):
        unchanged(row)
        if i % 10000 == 0:
            print(f'Rechecked release originals {i}/{len(entries)}', flush=True)
    free_before = shutil.disk_usage(ROOT).free
    journal_path = RUNTIME / 'moves.jsonl'
    with journal_path.open('x') as journal:
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
            if i % 10000 == 0:
                print(f'Released {i}/{len(entries)}', flush=True)
        journal.flush(); os.fsync(journal.fileno())
    for row in entries:
        source = ROOT / row['path']
        if not source.is_symlink() or os.readlink(source) != os.path.relpath(ROOT / row['archive_member'], source.parent):
            raise ValueError(f'Archive link mismatch: {source}')
    validate_guards(m)
    shutil.copyfile(journal_path, STAGE / 'moves.jsonl')
    m.update(status='externalized', completed_at=now(), external_sha256=v['sha256'], external_size=v['size'])
    write(ART / 'manifest.json', m); write(STAGE / 'manifest.json', m)
    links = sum((ROOT / r['path']).lstat().st_blocks * 512 for r in entries)
    result = dict(status='complete', completed_at=now(), external_archive=str(TARGET), external_size=v['size'],
        external_sha256=v['sha256'], externalized=len(entries), categories=m['categories'],
        removed_original_allocated_bytes=m['allocated_bytes_to_release'], allocated_symlink_bytes=links,
        net_payload_space_released=m['allocated_bytes_to_release']-links,
        observed_local_free_delta=shutil.disk_usage(ROOT).free-free_before,
        protected_data=m['protected_data'], retained_models_verified=m['retained_models'],
        guard_files_verified=m['guard_files'], online_resume_inputs_unchanged=True)
    write(ART / 'completion.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'export', 'release', 'finish', 'all'))
    args = parser.parse_args(); RUNTIME.mkdir(parents=True, exist_ok=True)
    selected = approved() if args.phase in ('plan', 'all') else rows(ART / 'files.jsonl.gz')
    groups = {ROOT / '/'.join(r['path'].split('/')[:2]) for r in selected}
    groups.update((PARENT, OLD_ONLINE, CURRENT))
    with ExitStack() as stack:
        own_lock = stack.enter_context((RUNTIME / '.lock').open('a'))
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for group in sorted(groups):
            fd = os.open(group, os.O_RDONLY); stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks = set(group.glob('*.lock')) | set(group.glob('.*lock'))
            for path in sorted(locks):
                handle = stack.enter_context(path.open('a')); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        phases = ('plan', 'export', 'release') if args.phase == 'all' else (('export', 'release') if args.phase == 'finish' else (args.phase,))
        for phase in phases:
            {'plan': plan, 'export': export, 'release': release}[phase]()


if __name__ == '__main__':
    main()
