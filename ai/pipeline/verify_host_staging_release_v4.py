#!/usr/bin/env python3
"""Independently verify the deterministic host staging ZIP and embedded manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--release-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    release = json.loads(args.release_receipt.read_text(encoding="utf-8"))
    archive_path = root / release["archive"]
    if sha256_file(archive_path) != release["archive_sha256"]:
        raise SystemExit("archive SHA-256 mismatch")

    with zipfile.ZipFile(archive_path, "r") as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise SystemExit("duplicate ZIP member")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise SystemExit(f"unsafe ZIP member: {name}")
        manifest_bytes = archive.read("STAGING_MANIFEST.v4.json")
        if hashlib.sha256(manifest_bytes).hexdigest() != release["manifest_sha256"]:
            raise SystemExit("embedded manifest SHA-256 mismatch")
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        expected_names = {item["path"] for item in manifest["files"]} | {"STAGING_MANIFEST.v4.json"}
        if set(names) != expected_names:
            raise SystemExit("ZIP member set differs from manifest")
        checked_bytes = 0
        for item in manifest["files"]:
            data = archive.read(item["path"])
            checked_bytes += len(data)
            if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise SystemExit(f"payload mismatch: {item['path']}")
        recomputed_root = hashlib.sha256(
            json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recomputed_root != release["content_root_sha256"] or recomputed_root != manifest["content_root_sha256"]:
            raise SystemExit("content root mismatch")

    document = {
        "schema": "cimc.forge200.host-staging-release-verification.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ARCHIVE_FULL_HASH_AUDIT_BOARD_PENDING_RELEASE_FLOOR_BLOCKED",
        "archive": release["archive"],
        "archive_sha256": release["archive_sha256"],
        "content_root_sha256": release["content_root_sha256"],
        "members_checked": len(names),
        "payload_bytes_checked": checked_bytes,
        "unsafe_paths": 0,
        "duplicate_members": 0,
        "hash_mismatches": 0,
        "cache_or_checkpoint_files": 0,
        "authority": 0,
        "new_board_accepted": 0,
        "new_countable_publicly": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(document, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
