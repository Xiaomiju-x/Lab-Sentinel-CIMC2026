#!/usr/bin/env python3
"""Resume and verify the two CC BY 4.0 datasets added after GPU readiness."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "carinthia_s_v1": "16895427",
    "pvd_apc_spc_v1": "16881338",
}
USER_AGENT = "CIMC-Forge200-data-audit/1.0"


def digest(path: Path, algorithm: str) -> str:
    value = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def get_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=90) as response:
        return json.load(response)


def download(url: str, target: Path, expected_size: int) -> None:
    for attempt in range(1, 7):
        current = target.stat().st_size if target.is_file() else 0
        if current == expected_size:
            return
        # Zenodo's CDN returned a nominal partial response with the full object
        # body during the first attempt.  Never append an unverified fragment;
        # restart the small public file and rely on the repository MD5 gate.
        if current:
            target.unlink()
            current = 0
        headers = {"User-Agent": USER_AGENT}
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with target.open("wb") as handle:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if target.stat().st_size == expected_size:
                return
        except Exception:
            if attempt == 6:
                raise
            time.sleep(attempt * 3)
    raise RuntimeError(f"download size mismatch: {target}")


def main() -> int:
    state_path = ROOT / "evidence" / "open_dataset_download_state.v1.json"
    records = []
    for directory_name, record_id in DATASETS.items():
        directory = ROOT / "data" / "raw" / directory_name
        directory.mkdir(parents=True, exist_ok=True)
        metadata = get_json(f"https://zenodo.org/api/records/{record_id}")
        license_id = metadata.get("metadata", {}).get("license", {}).get("id")
        if license_id != "cc-by-4.0":
            raise RuntimeError(f"license gate failed for {record_id}: {license_id}")
        metadata_path = directory / f"zenodo_record_{record_id}.json"
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        files = []
        for item in metadata["files"]:
            target = directory / item["key"]
            download(item["links"]["self"], target, int(item["size"]))
            expected_md5 = item["checksum"].split(":", 1)[1]
            actual_md5 = digest(target, "md5")
            if actual_md5 != expected_md5:
                raise RuntimeError(f"MD5 gate failed: {target}")
            files.append(
                {
                    "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "bytes": target.stat().st_size,
                    "md5": actual_md5,
                    "sha256": digest(target, "sha256"),
                }
            )
        records.append(
            {
                "zenodo_record_id": record_id,
                "doi": metadata["doi"],
                "title": metadata["metadata"]["title"],
                "license": license_id,
                "metadata_sha256": digest(metadata_path, "sha256"),
                "files": files,
            }
        )
        state = {
            "schema": "cimc.forge200.open-dataset-download.v1",
            "status": "RUNNING",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "records": records,
        }
        state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    state["status"] = "PASS"
    state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    state_path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
