#!/usr/bin/env python3
"""Build or verify SHA-256 checksums for public media assets."""

from __future__ import annotations

import argparse

from _common import ROOT, sha256_file


ASSET_ROOT = ROOT / "assets"
OUTPUT = ASSET_ROOT / "SHA256SUMS.txt"
EXCLUDED = {OUTPUT, ASSET_ROOT / "README.md"}


def render() -> str:
    paths = sorted(
        (path for path in ASSET_ROOT.rglob("*") if path.is_file() and path not in EXCLUDED),
        key=lambda path: path.relative_to(ASSET_ROOT).as_posix(),
    )
    return "".join(
        f"{sha256_file(path)}  {path.relative_to(ASSET_ROOT).as_posix()}\n"
        for path in paths
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != expected:
            raise SystemExit("assets/SHA256SUMS.txt is missing or stale")
        print("PASS public asset checksums")
        return 0
    OUTPUT.write_text(expected, encoding="utf-8", newline="\n")
    print(f"WROTE {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
