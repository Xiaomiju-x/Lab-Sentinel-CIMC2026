#!/usr/bin/env python3
"""Download and hash-gate small official JARVIS sources for exact P contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


USER_AGENT = "CIMC-Forge200-data-audit/1.0"
SOURCES = {
    "jarvis_vacancydb_v8": {
        "article_id": 23000573,
        "file_id": 52845425,
        "file_name": "vacancydb.json.zip",
        "expected_bytes": 314_776,
        "expected_md5": "0cbfd1d49074232f4f0c7c14ed86c75c",
    },
    "jarvis_interfacedb_v2": {
        "article_id": 25832614,
        "file_id": 46355692,
        "file_name": "interface_db_dd.json.zip",
        "expected_bytes": 853_973,
        "expected_md5": "76aa72b362f470dbedc4c959211c2f6f",
    },
    "jarvis_surfacedb_v2": {
        "article_id": 25832614,
        "file_id": 46355689,
        "file_name": "surface_db_dd.json.zip",
        "expected_bytes": 3_076_691,
        "expected_md5": "e2d20b57cdd665a4a6262474392732ae",
    },
}


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            handle.write(block)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    records = []
    for directory_name, expected in SOURCES.items():
        article_id = int(expected["article_id"])
        metadata = get_json(f"https://api.figshare.com/v2/articles/{article_id}")
        if metadata.get("license", {}).get("name") != "CC BY 4.0":
            raise RuntimeError(f"FIGSHARE_LICENSE_GATE:{article_id}")
        matches = [item for item in metadata["files"] if int(item["id"]) == int(expected["file_id"])]
        if len(matches) != 1:
            raise RuntimeError(f"FIGSHARE_FILE_ID_GATE:{article_id}:{expected['file_id']}")
        item = matches[0]
        if (
            item["name"] != expected["file_name"]
            or int(item["size"]) != int(expected["expected_bytes"])
            or item["computed_md5"] != expected["expected_md5"]
        ):
            raise RuntimeError(f"FIGSHARE_METADATA_GATE:{article_id}:{item}")
        directory = root / "data" / "raw" / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        metadata_path = directory / f"figshare_article_{article_id}.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        target = directory / item["name"]
        if not target.is_file() or target.stat().st_size != int(expected["expected_bytes"]):
            download(item["download_url"], target)
        if target.stat().st_size != int(expected["expected_bytes"]) or digest(target, "md5") != expected["expected_md5"]:
            raise RuntimeError(f"FIGSHARE_PAYLOAD_GATE:{target}")
        records.append(
            {
                "article_id": article_id,
                "doi": metadata["doi"],
                "title": metadata["title"],
                "license": "CC BY 4.0",
                "file_id": int(item["id"]),
                "path": str(target.relative_to(root)).replace("\\", "/"),
                "bytes": target.stat().st_size,
                "md5": digest(target, "md5"),
                "sha256": digest(target, "sha256"),
                "metadata_path": str(metadata_path.relative_to(root)).replace("\\", "/"),
                "metadata_sha256": digest(metadata_path, "sha256"),
            }
        )
    receipt = {
        "schema": "cimc.forge200.jarvis-exact-source-download.v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "authority": 0,
        "records": records,
    }
    output = root / "evidence" / "jarvis_exact_source_download.v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": len(records), "receipt": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
