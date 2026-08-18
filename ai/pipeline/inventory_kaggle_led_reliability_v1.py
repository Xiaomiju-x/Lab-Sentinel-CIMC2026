#!/usr/bin/env python3
"""Inventory the public LED solder-reliability dataset without downloading 5.6 GB.

The inventory is resumable and is discovery-only.  It never authorizes P107 or
P122; exact task binding is performed by a separate fail-closed audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


API = "https://www.kaggle.com/api/v1/datasets/list/andreaszippelius/hellastudy-of-leds2"
USER_AGENT = "CIMC-Forge200-source-audit/1.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(page_token: str | None) -> dict:
    query = {"page_size": 200}
    if page_token:
        query["page_token"] = page_token
    url = API + "?" + urllib.parse.urlencode(query)
    for attempt in range(1, 7):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=90) as response:
                return json.load(response)
        except Exception:
            if attempt == 6:
                raise
            time.sleep(attempt * 2)
    raise AssertionError("unreachable")


def write_checkpoint(path: Path, pages: int, token: str | None, files: list[dict]) -> None:
    document = {
        "schema": "cimc.forge200.kaggle-led-reliability-inventory-checkpoint.v1",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pages_complete": pages,
        "next_page_token": token,
        "files": files,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data/sources/p107_solder_xray_pmc11126598_v1"
    source.mkdir(parents=True, exist_ok=True)
    checkpoint = source / "kaggle_file_inventory.checkpoint.json"
    output = source / "kaggle_file_inventory.v1.json"

    pages = 0
    token = None
    files: list[dict] = []
    if checkpoint.is_file():
        prior = json.loads(checkpoint.read_text(encoding="utf-8"))
        pages = int(prior["pages_complete"])
        token = prior.get("next_page_token")
        files = list(prior["files"])
        if token is None:
            checkpoint.unlink()
            return 0

    while True:
        response = fetch(token)
        chunk = response.get("datasetFiles", [])
        if not chunk:
            raise RuntimeError("Kaggle returned an empty page before EOF")
        files.extend(chunk)
        pages += 1
        token = response.get("nextPageToken") or None
        # Rewriting a growing 30+ MB JSON document on every page is wasteful.
        # Persist every 20 pages (or at EOF); an interrupted run re-fetches at
        # most 19 read-only metadata pages and never duplicates final records.
        if pages % 20 == 0 or token is None:
            write_checkpoint(checkpoint, pages, token, files)
            print(f"pages={pages} files={len(files)}", flush=True)
        if token is None:
            break

    names = [item.get("name", "") for item in files]
    top = Counter(PurePosixPath(name).parts[0] for name in names if name)
    extensions = Counter(PurePosixPath(name).suffix.lower() for name in names if name)
    keyword_counts = {
        word: sum(word in name.lower() for name in names)
        for word in ("xray", "mask", "label", "json", "failure", "resistance", "thermal", "cycle")
    }
    document = {
        "schema": "cimc.forge200.kaggle-led-reliability-file-inventory.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_DISCOVERY_ONLY_NOT_TASK_ADMISSION",
        "dataset": "andreaszippelius/hellastudy-of-leds2",
        "api": API,
        "pages": pages,
        "files": len(files),
        "listed_bytes": sum(int(item.get("totalBytes") or 0) for item in files),
        "top_level_counts": dict(sorted(top.items())),
        "extension_counts": dict(sorted(extensions.items())),
        "keyword_counts": keyword_counts,
        "keyword_examples": {
            word: [name for name in names if word in name.lower()][:100]
            for word in keyword_counts
        },
        "file_records": files,
        "authority": 0,
        "countable_model": False,
    }
    output.write_text(json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    checkpoint.unlink()
    print(json.dumps({
        "status": document["status"],
        "pages": pages,
        "files": len(files),
        "listed_bytes": document["listed_bytes"],
        "sha256": sha256_file(output),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
