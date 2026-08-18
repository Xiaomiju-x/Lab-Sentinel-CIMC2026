#!/usr/bin/env python3
"""Download a target-blind, hash-frozen subset of JARVIS LOPTICS raw files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from stage_matbench_experimental_v1 import sha256_file, write_json


INDEX_SHA256 = "1867e580149822df1fe190da7eee4a74363e5bd6ce52bf389a5c8bfdfb28a6e4"
JARVIS_SHA256 = "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"
NAME_RE = re.compile(r"^(JVASP-\d+)\.zip$")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # nosec: upstream file identity, not security
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"SINGLE_MEMBER_GATE:{path.name}")
        return json.loads(archive.read(names[0]))


def select_records(index: list[dict[str, Any]], jarvis: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    by_jid = {row["jid"]: row for row in jarvis}
    eligible = []
    for record in index:
        match = NAME_RE.match(record["name"])
        row = by_jid.get(match.group(1)) if match else None
        try:
            gap = float(row["optb88vdw_bandgap"]) if row else -1.0
        except (TypeError, ValueError):
            gap = -1.0
        if row and gap > 0.05 and int(record["size"]) <= 800 * 1024:
            eligible.append(
                {
                    **record,
                    "jid": match.group(1),
                    "optb88vdw_bandgap": gap,
                    "chemical_system": "-".join(sorted(set(row["atoms"]["elements"]))),
                    "selection_hash": hashlib.sha256(match.group(1).encode("ascii")).hexdigest(),
                }
            )
    eligible.sort(key=lambda item: (item["selection_hash"], item["jid"]))
    if len(eligible) < count:
        raise RuntimeError(f"SELECTION_COUNT_GATE:{len(eligible)}")
    return eligible[:count]


def fetch(record: dict[str, Any], output: Path) -> tuple[str, str]:
    destination = output / record["name"]
    # A small number of old Figshare records omit ``computed_md5`` while
    # retaining the identical upstream identity in ``supplied_md5``.
    expected_md5 = record.get("computed_md5") or record.get("supplied_md5")
    if not expected_md5:
        raise RuntimeError("UPSTREAM_MD5_ABSENT")
    if destination.exists() and destination.stat().st_size == int(record["size"]) and md5_file(destination) == expected_md5:
        return record["jid"], "CACHED"
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, 6):
        try:
            request = urllib.request.Request(record["download_url"], headers={"User-Agent": "CIMC-Forge200-source-audit/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    handle.write(block)
            if partial.stat().st_size != int(record["size"]) or md5_file(partial) != expected_md5:
                raise RuntimeError("SIZE_OR_MD5_GATE")
            os.replace(partial, destination)
            return record["jid"], "DOWNLOADED"
        except Exception:
            if partial.exists():
                partial.unlink()
            if attempt == 5:
                raise
            time.sleep(min(2**attempt, 20))
    raise AssertionError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    root = args.root.resolve()
    index_path = root / "data" / "raw" / "jarvis_raw_index_v1" / "figshare_data-10-28-2020.json.zip"
    jarvis_path = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    if sha256_file(index_path) != INDEX_SHA256 or sha256_file(jarvis_path) != JARVIS_SHA256:
        raise RuntimeError("SOURCE_HASH_GATE")
    index_payload = load_json_zip(index_path)
    selected = select_records(index_payload["OPT-LOPTICS"], load_json_zip(jarvis_path), args.count)
    output = root / "data" / "raw" / "jarvis_optics_p073_v1"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "selection_manifest.v1.json"
    manifest = {
        "schema": "cimc.forge200.jarvis-p073-download-selection.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SELECTED_TARGET_BLIND",
        "selection_rule": "JOINED_JID_AND_OPT_GAP_GT_0.05EV_AND_ZIP_LE_800KIB_THEN_ASC_SHA256_JID_FIRST_N",
        "count": len(selected),
        "bytes": sum(int(record["size"]) for record in selected),
        "chemical_systems": len({record["chemical_system"] for record in selected}),
        "source_index_sha256": INDEX_SHA256,
        "jarvis_table_sha256": JARVIS_SHA256,
        "upstream_article": "https://doi.org/10.6084/m9.figshare.13154159",
        "license": "CC-BY-4.0",
        "records": selected,
        "authority": 0,
    }
    write_json(manifest_path, manifest)
    completed = 0
    counts = {"CACHED": 0, "DOWNLOADED": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_by_record = {executor.submit(fetch, record, output): record for record in selected}
        for future in as_completed(future_by_record):
            record = future_by_record[future]
            try:
                _, state = future.result()
                counts[state] += 1
            except Exception as exc:
                failures.append({"jid": record["jid"], "error": f"{type(exc).__name__}:{exc}"})
            completed += 1
            if completed % 25 == 0 or completed == len(selected):
                print(json.dumps({"completed": completed, "total": len(selected), "cached": counts["CACHED"], "downloaded": counts["DOWNLOADED"], "failures": len(failures)}, sort_keys=True), flush=True)
    receipt = {
        "schema": "cimc.forge200.jarvis-p073-download-receipt.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not failures else "FAIL_CLOSED",
        "selection_manifest_sha256": sha256_file(manifest_path),
        "selected": len(selected),
        "verified": counts["CACHED"] + counts["DOWNLOADED"],
        "bytes": sum((output / record["name"]).stat().st_size for record in selected if (output / record["name"]).exists()),
        "states": counts,
        "failures": failures,
        "authority": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "jarvis_p073_download_receipt.v1.json", receipt)
    if failures:
        raise RuntimeError(f"DOWNLOAD_FAILURES:{len(failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
