"""Read-only verification of the 2026-10-27 Git tracking cleanup."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def verify_projection(raw, projected):
    if isinstance(projected, dict) and set(projected) == {'_git_omitted_raw'}:
        marker = projected['_git_omitted_raw']
        assert type(raw).__name__ == marker['kind']
        assert len(raw) == marker['count']
        assert hashlib.sha256(encoded(raw)).hexdigest() == marker['value_sha256']
    elif isinstance(raw, dict):
        assert isinstance(projected, dict) and raw.keys() == projected.keys()
        for key in raw:
            verify_projection(raw[key], projected[key])
    elif isinstance(raw, list):
        assert isinstance(projected, list) and len(raw) == len(projected)
        for original, preserved in zip(raw, projected):
            verify_projection(original, preserved)
    else:
        assert encoded(raw) == encoded(projected)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    manifest = json.loads((here / 'manifest.json').read_bytes())
    indexed = set(subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().strip('\0').split('\0'))
    errors, changed_live, checked, projections = [], [], 0, 0
    for entry in manifest['entries']:
        path = root / entry['path']
        try:
            assert entry['path'] not in indexed, 'Raw file remains in the Git index'
            raw_bytes = path.read_bytes()
            unchanged = hashlib.sha256(raw_bytes).hexdigest() == entry['sha256']
            live = entry['path'].startswith('artifacts/offline5_vs_r37_10k_20261027/panels/') and not entry['previously_tracked']
            if not unchanged and live:
                changed_live.append(entry['path'])
                continue
            assert unchanged and len(raw_bytes) == entry['bytes'], 'Original bytes differ'
            if 'compact_summary' in entry:
                summary = json.loads((here / entry['compact_summary']).read_bytes())
                assert summary['raw'] == {k: entry[k] for k in ('path', 'bytes', 'sha256')}
                raw = ([json.loads(line) for line in raw_bytes.splitlines() if line.strip()]
                       if path.suffix == '.jsonl' else json.loads(raw_bytes))
                verify_projection(raw, summary['preserved'])
                projections += 1
            checked += 1
        except (AssertionError, OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({'path': entry['path'], 'error': str(exc)})
    paths = [e['path'] for e in manifest['entries']]
    ignore = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin', '-z'], cwd=root,
                            input=b'\0'.join(p.encode() for p in paths) + b'\0', stdout=subprocess.PIPE, check=False)
    ignored = set(ignore.stdout.decode().strip('\0').split('\0'))
    for path in sorted(set(paths) - ignored):
        errors.append({'path': path, 'error': 'Raw path is not ignored'})
    for path in (here / 'added_core_files.txt').read_text().splitlines():
        if path not in indexed:
            errors.append({'path': path, 'error': 'Core file is not in the Git index'})
        if path.endswith('.py'):
            ast.parse((root / path).read_bytes(), filename=path)
    # Negative rule must allow all explicitly versioned compact projections.
    summary_paths = [p.relative_to(root).as_posix() for p in (here / 'summaries').rglob('*.json')]
    ignore_summaries = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin', '-z'], cwd=root,
                                     input=b'\0'.join(p.encode() for p in summary_paths) + b'\0',
                                     stdout=subprocess.PIPE, check=False)
    if ignore_summaries.stdout:
        errors.append({'error': 'Compact summaries unexpectedly ignored', 'paths': ignore_summaries.stdout.decode().split('\0')})
    result = {'passed': not errors, 'scope': 'Git tracking and original preservation; no training or strength evaluation',
              'originals_verified': checked, 'compact_projections_verified': projections,
              'live_snapshots_changed': changed_live, 'errors': errors}
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
