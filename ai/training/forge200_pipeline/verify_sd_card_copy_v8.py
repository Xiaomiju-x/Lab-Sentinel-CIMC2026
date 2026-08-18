#!/usr/bin/env python3
"""Read-only verification of a copied F200 tree on the physical microSD."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--sd-root",
        type=Path,
        required=True,
        help="Physical drive root containing F200, for example E:\\\\",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    sd_root = args.sd_root.resolve()
    target = sd_root / "F200"
    expected = inventory(source / "F200")
    actual = inventory(target) if target.is_dir() else []
    expected_by_path = {str(item["path"]): item for item in expected}
    actual_by_path = {str(item["path"]): item for item in actual}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path))
    mismatched = sorted(
        path
        for path in set(expected_by_path) & set(actual_by_path)
        if expected_by_path[path] != actual_by_path[path]
    )
    accepted = not missing and not extra and not mismatched
    result = {
        "schema": "cimc.forge200.physical-sd-copy.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHYSICAL_SD_COPY_PASS" if accepted else "PHYSICAL_SD_COPY_REJECTED",
        "accepted": accepted,
        "source": str(source),
        "sd_root": str(sd_root),
        "files": len(actual),
        "bytes": sum(int(item["bytes"]) for item in actual),
        "expected_files": len(expected),
        "expected_bytes": sum(int(item["bytes"]) for item in expected),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "files": len(actual),
                "bytes": result["bytes"],
                "mismatches": len(missing) + len(extra) + len(mismatched),
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
