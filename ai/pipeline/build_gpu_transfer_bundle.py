#!/usr/bin/env python3
"""Build a deterministic, clone-safe GPU execution bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    queue = json.loads((root / "queue" / "dual_5090_queue.v1.json").read_text(encoding="utf-8"))
    files: set[Path] = set()
    for relative in ("contracts", "pipeline", "queue", "tests"):
        files.update(path for path in (root / relative).rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    for relative in (
        "data/ledgers/source_ledger.v1.json",
        "data/ledgers/license_ledger.v1.json",
        "data/ledgers/truth_ledger.v1.json",
        "data/ledgers/split_ledger.v1.json",
        "data/ledgers/leakage_audit.v1.json",
        "data/ledgers/task_source_bindings.v1.json",
        "data/ledgers/ccby_multidomain_corpus.v1.json",
        "data/ledgers/ccby_rejections.v1.json",
        "data/corpora/ccby_multidomain_v1.jsonl",
        "data/staged/staging_manifest.v1.json",
        "data/staged/rag_staging_manifest.v1.json",
        "evidence/pre_gpu_disposition_244.v1.json",
        "evidence/task_contract_review.v1.json",
    ):
        files.add(root / relative)
    for job in queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]:
        if job["admission_state"] == "ADMITTED":
            files.add(root / job["staged_dataset"])
            files.add(root / job["staged_metadata"])
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"bundle inputs missing: {missing}")
    records = [
        {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(files)
    ]
    content_root = hashlib.sha256(canonical_bytes(records)).hexdigest()
    manifest = {
        "schema": "cimc.forge200.gpu-transfer-bundle.v1",
        "status": "GPU_READY_INPUTS_ONLY_NO_CHECKPOINTS",
        "records": records,
        "files": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "content_root_sha256": content_root,
        "admitted_jobs": queue["admitted_jobs"],
        "admitted_by_shard": queue["admitted_by_shard"],
        "authority_nonzero": 0,
        "contains_raw_article_xml": False,
        "contains_installer_or_wheel_cache": False,
    }
    release_dir = root / "releases"
    release_dir.mkdir(parents=True, exist_ok=True)
    archive = release_dir / f"forge200-gpu-ready-{content_root[:16]}.zip"
    manifest_name = "GPU_TRANSFER_MANIFEST.json"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for record in records:
            info = zipfile.ZipInfo(record["path"], date_time=(2026, 8, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            handle.writestr(info, (root / record["path"]).read_bytes())
        info = zipfile.ZipInfo(manifest_name, date_time=(2026, 8, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        info.external_attr = 0o644 << 16
        handle.writestr(info, json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    outer = {**manifest, "archive": str(archive.relative_to(root)).replace("\\", "/"), "archive_bytes": archive.stat().st_size, "archive_sha256": sha256_file(archive)}
    write_json(release_dir / "forge200_gpu_transfer_bundle.v1.json", outer)
    print(json.dumps({"status": outer["status"], "archive": outer["archive"], "archive_bytes": outer["archive_bytes"], "archive_sha256": outer["archive_sha256"], "files": outer["files"], "content_root_sha256": outer["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
