#!/usr/bin/env python3
"""Verify manually downloaded node bundles against frozen manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--full", action="store_true", help="also hash every archived payload member")
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts" / "cloud_transfer_20260803"
    metadata_dir, archive_dir = base / "metadata", base / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for node in ("a", "b"):
        stem = f"cimc-forge200-node-{node}-transfer-20260803"
        archive = archive_dir / f"{stem}.tar.gz"
        receipt = json.loads((metadata_dir / f"{stem}.receipt.json").read_text(encoding="utf-8"))
        manifest = json.loads((metadata_dir / f"{stem}.manifest.json").read_text(encoding="utf-8"))
        if not archive.exists():
            reports.append({"node": node.upper(), "status": "DOWNLOAD_PENDING", "expected_path": str(archive), "expected_bytes": receipt["archive_bytes"], "expected_sha256": receipt["archive_sha256"]})
            continue
        actual_sha = sha256_file(archive)
        if archive.stat().st_size != receipt["archive_bytes"] or actual_sha != receipt["archive_sha256"]:
            reports.append({"node": node.upper(), "status": "FAIL_ARCHIVE_IDENTITY", "bytes": archive.stat().st_size, "sha256": actual_sha})
            continue
        expected = {item["archive_path"]: item for item in manifest["files"]}
        errors = []
        verified = 0
        with tarfile.open(archive, mode="r:gz") as handle:
            members = {item.name: item for item in handle.getmembers() if item.isfile()}
            for path, item in expected.items():
                member = members.get(path)
                if member is None:
                    errors.append(f"MISSING:{path}")
                    continue
                if member.size != item["bytes"]:
                    errors.append(f"SIZE:{path}")
                    continue
                if args.full:
                    stream = handle.extractfile(member)
                    if stream is None:
                        errors.append(f"UNREADABLE:{path}")
                        continue
                    digest = hashlib.sha256()
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                    if digest.hexdigest() != item["sha256"]:
                        errors.append(f"SHA:{path}")
                        continue
                verified += 1
            expected_sidecars = {f"bundle/{stem}.manifest.json", f"bundle/{stem}.RESTORE.md"}
            if not expected_sidecars.issubset(members):
                errors.append("BUNDLE_SIDECARS")
        reports.append({"node": node.upper(), "status": "PASS" if not errors else "FAIL", "archive_bytes": archive.stat().st_size, "archive_sha256": actual_sha, "tar_member_files": len(members), "manifest_files_verified": verified, "full_payload_hashing": args.full, "errors": errors[:50]})
    status = "PASS" if reports and all(item["status"] == "PASS" for item in reports) else ("DOWNLOAD_PENDING" if all(item["status"] == "DOWNLOAD_PENDING" for item in reports) else "FAIL_OR_PARTIAL")
    result = {"schema": "cimc.forge200.cloud-download-verification.v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "reports": reports, "authority": 0, "board_accepted": False, "countable_model": False}
    output = root / "evidence" / "cloud5090_download_verification.v2.json"
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0 if status in {"PASS", "DOWNLOAD_PENDING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
