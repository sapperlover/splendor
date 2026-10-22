"""Deduplicate the approved census, stage cold models, verify T7 tar and release payload."""
from __future__ import annotations

import argparse
from collections import Counter
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
from zoneinfo import ZoneInfo

ART = Path(__file__).resolve().parent
ROOT = ART.parents[1]
AUDIT = ROOT / "artifacts/checkpoint_storage_audit_20261021"
STAGE = ROOT / "archive/20261021"
TARGET = Path("/media/lc/T7/splendor/archive-20261021.tar.zst")
PARTIAL = TARGET.with_name(TARGET.name + ".partial")
ACTIVE = "runs/entity_attention_128x2_offline_80k_20261021/"


def now():
    return datetime.now(ZoneInfo("Asia/Singapore")).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(name):
    return [json.loads(line) for line in (AUDIT / f"{name}.jsonl").read_text().splitlines()]


def validate_original(row):
    path = ROOT / row["path"]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Original model is not an approved regular cold file: {path}")
    stat = path.stat()
    if (stat.st_size, stat.st_mtime_ns) != (row["size"], row["mtime_ns"]):
        raise ValueError(f"Original model changed since census: {path}")
    if sha256(path) != row["sha256"]:
        raise ValueError(f"Original model hash differs: {path}")


def stage():
    if (ART / "manifest.json").exists() or STAGE.exists():
        raise ValueError("Batch already initialized; inspect its manifest/journal instead of replaying stage")
    candidates = rows("ordinary_step_candidates") + rows("externalize_candidates")
    duplicates, retained = rows("deduplicate_candidates"), rows("keep_local")
    if len(candidates) != 1885 or len(duplicates) != 161:
        raise ValueError("Approved census population differs")
    selected = {r["path"] for r in candidates}
    kept = {r["path"] for r in retained}
    aliases = {r["path"] for r in duplicates}
    if len(selected) != 1885 or selected & (kept | aliases):
        raise ValueError("Storage tiers overlap")
    if any(path.startswith(ACTIVE) for path in selected | aliases):
        raise ValueError("Active training run cannot be moved or deduplicated")
    for number, row in enumerate(candidates + duplicates + retained, 1):
        validate_original(row)
        if number % 500 == 0:
            print(f"Validated {number} census checkpoints", flush=True)
    for row in duplicates:
        original = ROOT / row["identical_to"]
        if row["identical_to"] in aliases or sha256(original) != row["sha256"]:
            raise ValueError("Duplicate canonical is missing, chained or different")
    entries = []
    for row in candidates:
        path = ROOT / row["path"]
        stat = path.stat()
        entries.append({k: row[k] for k in ("path", "size", "allocated_bytes", "mtime_ns", "sha256", "category")} |
            {"archive_member": "archive/20261021/" + row["path"],
             "original_device": stat.st_dev, "original_inode": stat.st_ino, "mode": stat.st_mode & 0o777})
    plan = {"protocol": "checkpoint-cold-archive-v1", "created_at": now(),
        "repository": str(ROOT), "archive_root": "archive/20261021", "external_archive": str(TARGET),
        "status": "planned", "approval_scope": "161 audited duplicate aliases; 809 ordinary and1076 historical/EMA models; keep active attention and keep-local models.",
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "audit_inputs": {name: sha256(AUDIT / name) for name in
            ("checkpoints.jsonl", "keep_local.jsonl", "deduplicate_candidates.jsonl", "ordinary_step_candidates.jsonl", "externalize_candidates.jsonl")},
        "entries": entries, "deduplications": [{k: row[k] for k in
            ("path", "identical_to", "sha256", "size", "allocated_bytes", "mtime_ns")} for row in duplicates],
        "retained_models": retained,
        "allocated_bytes_to_externalize": sum(e["allocated_bytes"] for e in entries),
        "allocated_duplicate_bytes": sum(r["allocated_bytes"] for r in duplicates),
        "scope": "Checkpoint bytes only. No NPZ, sampling ledgers, active-run files or old archive aliases moved; existing historical manifests remain unchanged."}
    write(ART / "manifest.json", plan)
    STAGE.mkdir(parents=True)
    journal = (ART / "moves.jsonl").open("a")
    try:
        for row in duplicates:
            source, canonical = ROOT / row["path"], ROOT / row["identical_to"]
            temporary = source.with_name(source.name + ".dedup-link.tmp")
            if temporary.exists() or temporary.is_symlink():
                raise ValueError(f"Temporary link exists: {temporary}")
            temporary.symlink_to(os.path.relpath(canonical, source.parent))
            temporary.replace(source)
            journal.write(json.dumps({"action": "deduplicate", "path": row["path"], "canonical": row["identical_to"]}) + "\n")
        for number, entry in enumerate(entries, 1):
            source, destination = ROOT / entry["path"], ROOT / entry["archive_member"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() or destination.is_symlink():
                raise ValueError(f"Stage destination exists: {destination}")
            source.rename(destination)
            source.symlink_to(os.path.relpath(destination, source.parent))
            journal.write(json.dumps({"action": "stage", "path": entry["path"], "destination": entry["archive_member"]}) + "\n")
            if number % 200 == 0:
                journal.flush()
                os.fsync(journal.fileno())
                print(f"Staged {number}/{len(entries)} models", flush=True)
        journal.flush()
        os.fsync(journal.fileno())
    finally:
        journal.close()
    for entry in entries:
        source, destination = ROOT / entry["path"], ROOT / entry["archive_member"]
        stat = destination.stat()
        if not source.is_symlink() or source.resolve() != destination or stat.st_ino != entry["original_inode"]:
            raise ValueError("Model relocation/link/inode check failed")
        if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
            raise ValueError("Staged bytes/time metadata changed")
    for row in duplicates:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Duplicate link content check failed")
    plan.update(status="staged", staged_at=now())
    write(ART / "manifest.json", plan)
    shutil.copyfile(ART / "manifest.json", STAGE / "manifest.json")
    metadata = STAGE / "metadata"
    metadata.mkdir()
    for filename in ("inventory_summary.json", "keep_local.jsonl", "deduplicate_candidates.jsonl",
                     "ordinary_step_candidates.jsonl", "externalize_candidates.jsonl"):
        shutil.copyfile(AUDIT / filename, metadata / filename)
    shutil.copyfile(ROOT / "artifacts/entity_attention_ema_17_19_20261020/plan.json", metadata / "ema_plan.json")
    (STAGE / "SHA256SUMS").write_text("".join(f"{entry['sha256']}  {entry['path']}\n" for entry in entries))
    (STAGE / "README.md").write_text(
        "# 2026-10-21 checkpoint冷归档\n\n1885个模型来自已核对的809个旧恢复点及1076个历史/EMA模型。"
        "161个字节相同副本已链接到指定原件，不重复存入本包。模型字节和mtime保留，原路径留相对链接。"
        "不包含训练NPZ、账本或当前attention模型。\n\n"
        f"外盘位置：`{TARGET}`。从仓库根目录恢复：\n\n"
        f"```bash\ntar --zstd -xf '{TARGET}' -C '{ROOT}'\n"
        "cd archive/20261021\nsha256sum -c SHA256SUMS\n```\n\n"
        "只读分析可使用恢复后的原路径链接。恢复旧训练前，应将对应模型按manifest恢复为原位真实文件，"
        "不要通过归档链接写入冻结包。未解压时冷模型链接不可读；本机锚点、MLP保留模型及当前训练不依赖本包。\n")
    package = []
    by_member = {entry["archive_member"]: entry for entry in entries}
    for path in sorted(STAGE.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        stat = path.stat()
        package.append({"path": rel, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "sha256": by_member[rel]["sha256"] if rel in by_member else sha256(path),
            "checkpoint": rel in by_member})
    write(ART / "package_inventory.json", package)
    print(json.dumps({"phase": "staged", "models": len(entries), "deduplicated": len(duplicates),
        "bytes_to_externalize": plan["allocated_bytes_to_externalize"]}), flush=True)


def verify_tar(path):
    expected = {r["path"]: r for r in read(ART / "package_inventory.json")}
    observed, checkpoints = set(), 0
    decompressor = subprocess.Popen([shutil.which("zstd"), "-q", "-d", "-c", str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=decompressor.stdout, mode="r|") as archive:
            for member in archive:
                if member.isdir():
                    continue
                if not member.isfile() or member.name not in expected or member.name in observed:
                    raise ValueError(f"Unexpected/linked/repeated tar member: {member.name}")
                row = expected[member.name]
                if member.size != row["size"]:
                    raise ValueError("Archive member size differs")
                h = hashlib.sha256()
                handle = archive.extractfile(member)
                for chunk in iter(lambda: handle.read(1024**2), b""):
                    h.update(chunk)
                if h.hexdigest() != row["sha256"]:
                    raise ValueError(f"Archive member SHA differs: {member.name}")
                if row["checkpoint"]:
                    mtime = Decimal(member.pax_headers.get("mtime", str(member.mtime)))
                    if int(mtime * 10**9) != row["mtime_ns"]:
                        raise ValueError(f"Nanosecond mtime not preserved: {member.name}")
                    checkpoints += 1
                    if checkpoints % 300 == 0:
                        print(f"Verified T7 stream {checkpoints}/1885 models", flush=True)
                observed.add(member.name)
        # Drain stream padding/trailer before checking the decompressor's exit.
        for _ in iter(lambda: decompressor.stdout.read(1024**2), b""):
            pass
        if decompressor.wait() != 0 or observed != set(expected) or checkpoints != 1885:
            raise ValueError("Incomplete archive stream or decompression failure")
    finally:
        decompressor.stdout.close()
        if decompressor.poll() is None:
            decompressor.kill()
            decompressor.wait()
    return {"checkpoint_count": checkpoints, "regular_file_count": len(observed),
        "all_member_sha256_match": True, "checkpoint_nanosecond_mtime_preserved": True,
        "contains_real_models_not_symlinks": True}


def export():
    plan = read(ART / "manifest.json")
    if plan["status"] != "staged" or TARGET.exists() or PARTIAL.exists():
        raise ValueError("Batch not staged or external target/partial already exists")
    if not TARGET.parent.is_dir() or not os.access(TARGET.parent, os.W_OK):
        raise ValueError("External target is not writable")
    if shutil.disk_usage(TARGET.parent).free < plan["allocated_bytes_to_externalize"]:
        raise ValueError("External target lacks conservative uncompressed-size free space")
    os.nice(10)
    started = time.monotonic()
    write(ART / "export_state.json", {"status": "compressing", "started_at": now(), "partial": str(PARTIAL)})
    with PARTIAL.open("xb") as output:
        packer = subprocess.Popen(["tar", "--format=pax", "--pax-option=delete=atime,delete=ctime",
            "-C", str(ROOT), "-cf", "-", "archive/20261021"], stdout=subprocess.PIPE)
        compressor = subprocess.Popen([shutil.which("zstd"), "-q", "-T2", "-3", "-c"],
            stdin=packer.stdout, stdout=output)
        packer.stdout.close()
        compress_code, tar_code = compressor.wait(), packer.wait()
        if compress_code or tar_code:
            raise RuntimeError(f"tar/zstd failed: {tar_code}/{compress_code}; local payload retained")
        output.flush()
        os.fsync(output.fileno())
    print(f"Compressed directly to T7: {PARTIAL.stat().st_size} bytes; verifying full stream", flush=True)
    write(ART / "export_state.json", {"status": "verifying", "partial": str(PARTIAL), "updated_at": now()})
    verification = verify_tar(PARTIAL)
    compressed_sha = sha256(PARTIAL)
    PARTIAL.rename(TARGET)
    sidecar = TARGET.with_name(TARGET.name + ".sha256")
    with sidecar.open("x") as handle:
        handle.write(f"{compressed_sha}  {TARGET.name}\n")
        handle.flush()
        os.fsync(handle.fileno())
    result = {"status": "verified_external_local_payload_retained", "verified_at": now(),
        "target": str(TARGET), "size": TARGET.stat().st_size, "mtime_ns": TARGET.stat().st_mtime_ns,
        "sha256": compressed_sha, "sidecar": str(sidecar), "seconds": time.monotonic() - started,
        "compression": "GNU tar POSIX pax + zstd -T2 -3; direct external stream", **verification}
    write(ART / "external_verification.json", result)
    write(ART / "export_state.json", result)
    print(json.dumps(result), flush=True)


def release():
    verified = read(ART / "external_verification.json")
    plan = read(ART / "manifest.json")
    if verified["status"] != "verified_external_local_payload_retained" or plan["status"] != "staged":
        raise ValueError("Full external archive verification required before releasing models")
    if TARGET.stat().st_size != verified["size"] or sha256(TARGET) != verified["sha256"]:
        raise ValueError("Verified external archive changed; local payload retained")
    for row in plan["retained_models"]:
        if not (ROOT / row["path"]).is_file() or sha256(ROOT / row["path"]) != row["sha256"]:
            raise ValueError("Keep-local model changed before release")
    for entry in plan["entries"]:
        source, staged = ROOT / entry["path"], ROOT / entry["archive_member"]
        if not source.is_symlink() or source.resolve() != staged:
            raise ValueError("Original compatibility link changed")
        stat = staged.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (entry["size"], entry["mtime_ns"], entry["original_inode"]):
            raise ValueError("Local staged model changed before release")
    for entry in plan["entries"]:
        (ROOT / entry["archive_member"]).unlink()
    directories = sorted((p for p in STAGE.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True)
    for path in directories:
        if path.name == "metadata" or "metadata" in path.parts:
            continue
        if not any(path.iterdir()):
            path.rmdir()
    plan.update(status="externalized", completed_at=now(), external_sha256=verified["sha256"],
        external_size=verified["size"], local_payload_files_remaining=0)
    write(ART / "manifest.json", plan)
    write(STAGE / "manifest.json", plan)
    write(ART / "completion.json", {"status": "complete", "completed_at": now(),
        "deduplicated": len(plan["deduplications"]), "externalized": len(plan["entries"]),
        "removed_local_allocated_bytes": plan["allocated_bytes_to_externalize"] + plan["allocated_duplicate_bytes"],
        "external_archive": str(TARGET), "external_size": verified["size"], "external_sha256": verified["sha256"],
        "keep_local_models_verified": len(plan["retained_models"]), "active_run_excluded": ACTIVE,
        "local_metadata_retained": str(STAGE), "compatibility_links": "Cold model links require restoring the archive; deduplicated paths reference preserved canonical original paths."})
    print(json.dumps(read(ART / "completion.json")), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "export", "release"))
    args = parser.parse_args()
    runtime = ROOT / "runtime/repository_archive_20261021"
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / "operation.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"stage": stage, "export": export, "release": release}[args.phase]()


if __name__ == "__main__":
    main()
