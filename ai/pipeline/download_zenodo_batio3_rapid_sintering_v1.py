#!/usr/bin/env python3
"""Download and verify the CC BY BaTiO3 rapid-sintering dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


RECORD_ID = 18233071
API_URL = f"https://zenodo.org/api/records/{RECORD_ID}"


def hash_file(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "CIMC-Forge200-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def download_stream(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "CIMC-Forge200-source-audit/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
        while block := response.read(4 * 1024 * 1024):
            output.write(block)
    os.replace(partial, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = root / "data/sources/zenodo_batio3_rapid_sintering_v1"
    raw_dir = root / "data/raw/zenodo_batio3_rapid_sintering_v1"
    source_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    metadata = fetch_json(API_URL)
    metadata_path = source_dir / f"zenodo_record_{RECORD_ID}.json"
    metadata_payload = (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    metadata_path.write_bytes(metadata_payload)
    if metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("unexpected or missing reusable license")
    if len(metadata["files"]) != 1:
        raise ValueError("unexpected Zenodo file inventory")

    item = metadata["files"][0]
    archive_path = raw_dir / item["key"]
    if not archive_path.exists():
        download_stream(item["links"]["self"], archive_path)
    expected_md5 = item["checksum"].split(":", 1)[1]
    actual_md5 = hash_file(archive_path, "md5")
    actual_sha256 = hash_file(archive_path, "sha256")
    if archive_path.stat().st_size != int(item["size"]) or actual_md5 != expected_md5:
        raise ValueError("Zenodo archive size or MD5 mismatch")

    receipt = {
        "schema": "cimc.forge200.zenodo-batio3-rapid-sintering-download.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_id": RECORD_ID,
        "doi": metadata["metadata"]["doi"],
        "title": metadata["metadata"]["title"],
        "license": metadata["metadata"]["license"]["id"],
        "metadata": {
            "path": metadata_path.relative_to(root).as_posix(),
            "bytes": len(metadata_payload),
            "sha256": hashlib.sha256(metadata_payload).hexdigest(),
        },
        "archive": {
            "path": archive_path.relative_to(root).as_posix(),
            "bytes": archive_path.stat().st_size,
            "md5": actual_md5,
            "sha256": actual_sha256,
        },
        "authority": 0,
        "training_actions": 0,
        "board_actions": 0,
    }
    receipt_path = source_dir / "download_receipt.v1.json"
    receipt_payload = (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_payload)
    print(json.dumps({
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": actual_sha256,
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
