#!/usr/bin/env python3
"""Download and hash-verify the CC BY Mendeley liquid-phase sintering data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "w4n4jdcgcv"
VERSION = 1
LANDING_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
FOLDERS_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/folders/{VERSION}"
ACCEPT = "application/vnd.mendeley-public-dataset.1+json"


def fetch_bytes(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "CIMC-Forge200-source-audit/1.0"}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(payload)
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_dir = root / "data/sources/mendeley_sintering_w4_v1"
    raw_dir = root / "data/raw/mendeley_sintering_w4_v1"

    landing = fetch_bytes(LANDING_URL)
    atomic_write(source_dir / "landing_page.html", landing)
    folders = json.loads(fetch_bytes(FOLDERS_URL, ACCEPT).decode("utf-8"))
    atomic_write(
        source_dir / "folders.json",
        (json.dumps(folders, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )

    all_files: list[dict] = []
    for folder in folders:
        files_url = (
            f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
            f"?folder_id={folder['id']}&version={VERSION}"
        )
        files = json.loads(fetch_bytes(files_url, ACCEPT).decode("utf-8"))
        for item in files:
            item = dict(item)
            item["source_folder_name"] = folder["name"]
            all_files.append(item)

    atomic_write(
        source_dir / "files.json",
        (json.dumps(all_files, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    if len(all_files) != 10:
        raise ValueError(f"expected 10 official files, got {len(all_files)}")

    verified = []
    for item in all_files:
        destination = raw_dir / item["filename"]
        expected_sha = item["content_details"]["sha256_hash"]
        expected_bytes = int(item["content_details"]["size"])
        if destination.exists():
            payload = destination.read_bytes()
        else:
            payload = fetch_bytes(item["content_details"]["download_url"])
            atomic_write(destination, payload)
        actual_sha = sha256_bytes(payload)
        if len(payload) != expected_bytes or actual_sha != expected_sha:
            raise ValueError(f"download verification failed: {item['filename']}")
        verified.append({
            "filename": item["filename"],
            "path": destination.relative_to(root).as_posix(),
            "bytes": len(payload),
            "sha256": actual_sha,
            "source_folder_name": item["source_folder_name"],
        })

    receipt = {
        "schema": "cimc.forge200.mendeley-sintering-w4-download.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "doi": "10.17632/w4n4jdcgcv.1",
        "landing_url": LANDING_URL,
        "license": "CC BY 4.0",
        "landing_page": {
            "path": (source_dir / "landing_page.html").relative_to(root).as_posix(),
            "bytes": len(landing),
            "sha256": sha256_bytes(landing),
        },
        "folders": {
            "path": (source_dir / "folders.json").relative_to(root).as_posix(),
            "count": len(folders),
            "sha256": sha256_bytes((source_dir / "folders.json").read_bytes()),
        },
        "files_manifest": {
            "path": (source_dir / "files.json").relative_to(root).as_posix(),
            "count": len(all_files),
            "sha256": sha256_bytes((source_dir / "files.json").read_bytes()),
        },
        "verified_files": verified,
        "total_bytes": sum(item["bytes"] for item in verified),
        "authority": 0,
        "training_actions": 0,
        "board_actions": 0,
    }
    receipt_path = source_dir / "download_receipt.v1.json"
    atomic_write(receipt_path, (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({
        "files": len(verified),
        "bytes": receipt["total_bytes"],
        "receipt": receipt_path.relative_to(root).as_posix(),
        "receipt_sha256": sha256_bytes(receipt_path.read_bytes()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
