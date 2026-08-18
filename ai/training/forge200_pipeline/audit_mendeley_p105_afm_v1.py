#!/usr/bin/env python3
"""Verify the CC BY EUROFER97 AFM source without manufacturing P105 labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    binding_path = root / "contracts/mendeley_p105_afm_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    archive_path = root / binding["archive"]["path"]
    if archive_path.stat().st_size != binding["archive"]["bytes"] or sha256_file(archive_path) != binding["archive"]["sha256"]:
        raise ValueError("archive hash gate failed")
    if binding["source"]["license"] != "CC BY 4.0" or binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("license or authority boundary changed")

    extracted = root / binding["extracted_directory"]
    records = []
    families = set()
    with zipfile.ZipFile(archive_path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        for member in members:
            name = Path(member.filename).name
            match = re.match(r"2016-11-24_sample(\d+)", name, flags=re.IGNORECASE)
            if not match:
                raise ValueError(f"unexpected member name: {name}")
            family = f"sample{int(match.group(1))}"
            families.add(family)
            payload = archive.read(member)
            target = extracted / name
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"extracted member mismatch: {name}")
            header = payload[:40960].decode("latin-1", errors="replace")
            samples = [int(value) for value in re.findall(r"\\Samps/line:\s*(\d+)", header)]
            if not samples or any(value != binding["expected"]["samples_per_line"] for value in samples):
                raise ValueError(f"scan shape changed: {name}")
            if binding["expected"]["height_channel"] not in header:
                raise ValueError(f"height channel missing: {name}")
            records.append({
                "filename": name,
                "sample_family": family,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "samples_per_line": samples[0],
                "height_channel_present": True,
            })
    expected = binding["expected"]
    if len(records) != expected["scan_files"] or sum(item["bytes"] for item in records) != expected["uncompressed_bytes"]:
        raise ValueError("scan inventory changed")
    if len(families) != expected["sample_families"]:
        raise ValueError("sample-family inventory changed")
    disposition = binding["candidate_disposition"]
    if disposition["candidate_id"] != "CAND-P-105" or not disposition["status"].startswith("EXACT_REJECTED"):
        raise ValueError("this audit cannot promote P105")

    audit = {
        "schema": "cimc.forge200.mendeley-p105-afm-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_LICENSE_AND_HASH_PASS_P105_EXACT_REJECTED_BASELINE_BOUNDARY",
        "source_id": binding["source_id"],
        "source": binding["source"],
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "archive": binding["archive"],
        "observed": {
            "scan_files": len(records),
            "uncompressed_bytes": sum(item["bytes"] for item in records),
            "sample_families": len(families),
            "curve_or_pixel_rows_treated_as_independent_samples": False,
            "independent_measured_Ra_Rq_peak_density_labels": 0,
            "input_derived_labels_may_bypass_frozen_baseline": False,
        },
        "records": sorted(records, key=lambda item: item["filename"]),
        "candidate_disposition": disposition,
        "training_actions": 0,
        "host_promotions": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    audit["content_root_sha256"] = content_root(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "scan_files": audit["observed"]["scan_files"],
        "sample_families": audit["observed"]["sample_families"],
        "content_root_sha256": audit["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
