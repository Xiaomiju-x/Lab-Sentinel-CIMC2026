#!/usr/bin/env python3
"""Hash-verify the SiC plunger source and retain exact-contract rejections."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


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
    binding_path = root / "contracts/mendeley_sic_plunger_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["source_id"] != "MENDELEY-NKNVZ6GY6K-V1":
        raise ValueError("source identity changed")
    if binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("nonzero authority forbidden")

    receipt_spec = binding["download_receipt"]
    receipt_path = root / receipt_spec["path"]
    if receipt_path.stat().st_size != receipt_spec["bytes"] or sha256_file(receipt_path) != receipt_spec["sha256"]:
        raise ValueError("download receipt changed")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["doi"] != binding["source"]["doi"] or receipt["license"] != binding["source"]["license"]:
        raise ValueError("DOI or license changed")
    if len(receipt["verified_files"]) != 6 or receipt["total_bytes"] != 1524140:
        raise ValueError("source inventory changed")
    for item in receipt["verified_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"source artifact mismatch: {item['filename']}")

    inspection = binding["read_only_inspection"]
    if inspection["independent_sample_family_count"] != len(inspection["independent_sample_families_observed"]):
        raise ValueError("sample family inventory is inconsistent")
    if inspection["independent_sample_family_count"] != 3 or inspection["curve_points_are_independent_experiments"]:
        raise ValueError("independence boundary changed")
    dispositions = binding["candidate_dispositions"]
    expected_ids = {
        "CAND-P-049", "CAND-P-050", "CAND-P-051", "CAND-P-066",
        "CAND-P-082", "CAND-P-085", "CAND-P-146",
    }
    if {item["candidate_id"] for item in dispositions} != expected_ids:
        raise ValueError("candidate disposition set changed")
    if any(not item["status"].startswith("EXACT_REJECTED") for item in dispositions):
        raise ValueError("this audit cannot promote candidates")

    audit = {
        "schema": "cimc.forge200.mendeley-sic-plunger-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": binding["source_id"],
        "status": "SOURCE_LICENSE_AND_HASH_PASS_SEVEN_EXACT_CONTRACTS_REJECTED",
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "source": binding["source"],
        "verified_payload": {
            "file_count": len(receipt["verified_files"]),
            "bytes": receipt["total_bytes"],
            "download_receipt": receipt_spec,
        },
        "observed": inspection,
        "candidate_dispositions": dispositions,
        "training_actions": 0,
        "host_promotions": 0,
        "leakage_safe_splits_materialized": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Retain the licensed projects as source evidence only; obtain exact record-level targets with enough independent families before any training or promotion.",
    }
    audit["content_root_sha256"] = content_root(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "rejected_exact_contracts": len(dispositions),
        "content_root_sha256": audit["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
