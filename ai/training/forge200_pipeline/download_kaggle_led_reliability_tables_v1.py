#!/usr/bin/env python3
"""Download and hash only the eight small public reliability CSV tables."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


BASE = "https://www.kaggle.com/api/v1/datasets/download/andreaszippelius/hellastudy-of-leds2/"
USER_AGENT = "CIMC-Forge200-source-audit/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data/sources/p107_solder_xray_pmc11126598_v1"
    inventory_path = source / "kaggle_file_inventory.v1.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected = [
        item for item in inventory["file_records"]
        if PurePosixPath(item["name"]).suffix.lower() == ".csv"
    ]
    if len(selected) != 8:
        raise ValueError(f"expected exactly eight CSV files, observed {len(selected)}")
    target_root = root / "data/raw/kaggle_led_reliability_tables_v1"
    target_root.mkdir(parents=True, exist_ok=True)
    records = []
    for item in selected:
        target = target_root / PurePosixPath(item["name"]).name
        expected = int(item["totalBytes"])
        if not target.is_file() or target.stat().st_size != expected:
            url = BASE + urllib.parse.quote(item["name"], safe="")
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
        if target.stat().st_size != expected:
            raise ValueError(f"size mismatch: {target.name}")
        records.append({
            "source_name": item["name"],
            "path": target.relative_to(root).as_posix(),
            "bytes": expected,
            "sha256": sha256_file(target),
        })
    receipt = {
        "schema": "cimc.forge200.kaggle-led-reliability-tables-download.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SOURCE_TABLES_ONLY_NOT_TASK_ADMISSION",
        "dataset": "andreaszippelius/hellastudy-of-leds2",
        "license": "CC BY-NC-SA 4.0",
        "inventory": {
            "path": inventory_path.relative_to(root).as_posix(),
            "sha256": sha256_file(inventory_path),
        },
        "records": records,
        "downloaded_bytes": sum(item["bytes"] for item in records),
        "authority": 0,
        "countable_model": False,
    }
    output = root / "evidence/kaggle_led_reliability_tables_download.v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "files": len(records),
        "bytes": receipt["downloaded_bytes"],
        "receipt_sha256": sha256_file(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
