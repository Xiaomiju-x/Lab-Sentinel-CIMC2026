#!/usr/bin/env python3
"""Download and hash-verify the CC BY SiC SPS plunger dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET_ID = "nknvz6gy6k"
VERSION = 1
DOI = "10.17632/nknvz6gy6k.1"
LANDING_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
FOLDERS_URL = f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/folders/{VERSION}"
ACCEPT = "application/vnd.mendeley-public-dataset.1+json"


def fetch(url: str, accept: str | None = None) -> bytes:
    headers = {"User-Agent": "CIMC-Forge200-source-audit/1.0"}
    if accept:
        headers["Accept"] = accept
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def sha256(payload: bytes) -> str:
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
    source = root / "data/sources/mendeley_sic_plunger_v1"
    raw = root / "data/raw/mendeley_sic_plunger_v1"
    landing = fetch(LANDING_URL)
    atomic_write(source / "landing_page.html", landing)
    folder_payload = json.loads(fetch(FOLDERS_URL, ACCEPT).decode("utf-8"))
    folders = folder_payload if isinstance(folder_payload, list) else [folder_payload]
    atomic_write(source / "folders.json", (json.dumps(folders, ensure_ascii=False, indent=2) + "\n").encode())
    files = []
    for folder in folders:
        url = (
            f"https://data.mendeley.com/public-api/datasets/{DATASET_ID}/files"
            f"?folder_id={folder['id']}&version={VERSION}"
        )
        for item in json.loads(fetch(url, ACCEPT).decode("utf-8")):
            record = dict(item)
            record["source_folder_name"] = folder["name"]
            files.append(record)
    if len(files) != 6:
        raise ValueError(f"official inventory changed: expected 6 files, got {len(files)}")
    atomic_write(source / "files.json", (json.dumps(files, ensure_ascii=False, indent=2) + "\n").encode())
    verified = []
    for item in files:
        details = item["content_details"]
        destination = raw / item["filename"]
        payload = destination.read_bytes() if destination.is_file() else fetch(details["download_url"])
        expected_bytes = int(details["size"])
        expected_sha = details["sha256_hash"].lower()
        if len(payload) != expected_bytes or sha256(payload) != expected_sha:
            raise ValueError(f"download verification failed: {item['filename']}")
        if not destination.is_file():
            atomic_write(destination, payload)
        verified.append({
            "filename": item["filename"],
            "path": destination.relative_to(root).as_posix(),
            "bytes": len(payload),
            "sha256": expected_sha,
            "source_folder_name": item["source_folder_name"],
        })
    receipt = {
        "schema": "cimc.forge200.mendeley-sic-plunger-download.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_LICENSE_AND_PAYLOAD_HASH_PASS_CONTRACT_BINDING_PENDING",
        "dataset_id": DATASET_ID,
        "version": VERSION,
        "doi": DOI,
        "landing_url": LANDING_URL,
        "license": "CC BY 4.0",
        "verified_files": verified,
        "total_bytes": sum(item["bytes"] for item in verified),
        "training_actions": 0,
        "task_contract_bindings": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    receipt_path = source / "download_receipt.v1.json"
    atomic_write(receipt_path, (json.dumps(receipt, ensure_ascii=False, indent=2) + "\n").encode())
    print(json.dumps({
        "status": receipt["status"],
        "files": len(verified),
        "bytes": receipt["total_bytes"],
        "receipt_sha256": sha256(receipt_path.read_bytes()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
