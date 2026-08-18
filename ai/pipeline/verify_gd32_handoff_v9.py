#!/usr/bin/env python3
"""Read-only reverse verification of the immutable GD32 v9 handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    handoff = args.handoff.resolve()
    manifest_path = handoff / "MANIFEST.v9.json"
    files_path = handoff / "FILES.v9.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with files_path.open("r", encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    errors = []
    for row in records:
        path = handoff / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"]:
            errors.append(row["path"])
    actual_files = {path.relative_to(handoff).as_posix() for path in handoff.rglob("*") if path.is_file()}
    expected_files = {row["path"] for row in records} | {"MANIFEST.v9.json", "FILES.v9.csv"}
    extras = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    content_root = hashlib.sha256(json.dumps([
        {"path": row["path"], "bytes": int(row["bytes"]), "sha256": row["sha256"]}
        for row in records
    ], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    embedded_ready = json.loads((handoff / "EVIDENCE/gd32_local_complete.v9.json").read_text(encoding="utf-8"))
    accepted = (
        not errors and not extras and not missing
        and len(records) == manifest["payload_files"]
        and sum(int(row["bytes"]) for row in records) == manifest["payload_bytes"]
        and sha256(files_path) == manifest["files_csv_sha256"]
        and content_root == manifest["content_root_sha256"]
        and embedded_ready["status"] == manifest["status"]
        and manifest["authority_nonzero"] == 0
        and manifest["board_accepted"] is False
    )
    result = {
        "schema": "cimc.forge200.gd32-handoff-selfverify.v9",
        "status": "PASS" if accepted else "REJECTED",
        "accepted": accepted,
        "handoff": str(handoff),
        "files": len(records),
        "bytes": sum(int(row["bytes"]) for row in records),
        "content_root_sha256": content_root,
        "manifest_sha256": sha256(manifest_path),
        "files_csv_sha256": sha256(files_path),
        "hash_mismatches": errors,
        "extra": extras,
        "missing": missing,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "files": result["files"], "bytes": result["bytes"], "mismatches": len(errors) + len(extras) + len(missing), "content_root_sha256": content_root}, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
