#!/usr/bin/env python3
"""Build or check the deterministic SHA-256 manifest for public Git files."""

from __future__ import annotations

import argparse
import json

from _common import ROOT, iter_public_files, sha256_file


MANIFEST = ROOT / "PUBLIC_RELEASE_MANIFEST.json"


def build() -> dict:
    files = []
    for path in iter_public_files():
        files.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return {
        "schema": "lab-sentinel.public-release-manifest.v1",
        "version": "1.0.2",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = build()
    if args.check:
        if not MANIFEST.is_file():
            raise SystemExit("PUBLIC_RELEASE_MANIFEST.json is missing")
        expected = json.loads(MANIFEST.read_text(encoding="utf-8"))
        if expected != current:
            raise SystemExit("Public manifest is stale; run tools/build_public_manifest.py")
        print(f"PASS public manifest: {current['file_count']} files, {current['total_bytes']} bytes")
        return 0
    MANIFEST.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"WROTE {MANIFEST}: {current['file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
