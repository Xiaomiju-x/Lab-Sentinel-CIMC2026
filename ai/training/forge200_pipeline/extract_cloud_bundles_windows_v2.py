#!/usr/bin/env python3
"""Safely extract verified Linux tar paths onto Windows with reversible path mapping."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path


INVALID = re.compile(r'[<>:"/\\|?*]')


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize(path: str) -> str:
    output = []
    for index, part in enumerate(path.split("/")):
        cleaned = INVALID.sub("_", part).rstrip(" .")
        if not cleaned:
            cleaned = f"__WINDOWS_EMPTY_SEGMENT_{index}__"
        output.append(cleaned)
    return "/".join(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "artifacts" / "cloud_transfer_20260803"
    reports = []
    for node in ("a", "b"):
        stem = f"cimc-forge200-node-{node}-transfer-20260803"
        archive = base / "archives" / f"{stem}.tar.gz"
        manifest = json.loads((base / "metadata" / f"{stem}.manifest.json").read_text(encoding="utf-8"))
        destination = (base / "extracted" / f"node_{node}").resolve()
        destination.mkdir(parents=True, exist_ok=True)
        mapping, errors, verified = [], [], 0
        with tarfile.open(archive, mode="r:gz") as handle:
            members = {member.name: member for member in handle.getmembers() if member.isfile()}
            for item in manifest["files"]:
                original = item["archive_path"]
                mapped = sanitize(original)
                target = (destination / Path(mapped)).resolve()
                if destination not in target.parents:
                    errors.append(f"PATH_ESCAPE:{original}")
                    continue
                member = members.get(original)
                if member is None:
                    errors.append(f"MISSING_MEMBER:{original}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
                    stream = handle.extractfile(member)
                    if stream is None:
                        errors.append(f"UNREADABLE:{original}")
                        continue
                    with target.open("wb") as output:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            output.write(block)
                if target.stat().st_size != item["bytes"] or sha256_file(target) != item["sha256"]:
                    errors.append(f"IDENTITY:{original}")
                    continue
                verified += 1
                if mapped != original:
                    mapping.append({"archive_path": original, "windows_path": mapped, "sha256": item["sha256"]})
        reports.append({"node": node.upper(), "status": "PASS" if not errors else "FAIL", "manifest_files": len(manifest["files"]), "verified_files": verified, "path_mappings": mapping, "errors": errors})
    result = {"schema": "cimc.forge200.windows-extraction.v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL", "reports": reports, "authority": 0, "board_accepted": False}
    output = root / "evidence" / "cloud5090_windows_extraction.v2.json"
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
