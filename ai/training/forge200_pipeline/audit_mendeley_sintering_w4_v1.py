#!/usr/bin/env python3
"""Verify Mendeley w4n4jdcgcv v1 and retain fail-closed task dispositions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
    binding_path = root / "contracts/mendeley_sintering_w4_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["source_id"] != "MENDELEY-W4N4JDCGCV-V1":
        raise ValueError("source identity changed")
    if binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("nonzero authority forbidden")

    verified_artifacts = {}
    for name, item in binding["artifacts"].items():
        path = root / item["path"]
        actual = sha256_file(path)
        if path.stat().st_size != item["bytes"] or actual != item["sha256"]:
            raise ValueError(f"artifact mismatch: {name}")
        verified_artifacts[name] = {**item, "verified": True}

    receipt = json.loads((root / binding["artifacts"]["download_receipt"]["path"]).read_text(encoding="utf-8"))
    if receipt["doi"] != binding["source"]["doi"] or receipt["license"] != "CC BY 4.0":
        raise ValueError("DOI or license changed")
    expected = binding["expected_inventory"]
    if len(receipt["verified_files"]) != expected["downloaded_workbooks"]:
        raise ValueError("workbook count changed")
    if receipt["total_bytes"] != expected["downloaded_bytes"]:
        raise ValueError("downloaded byte count changed")
    for item in receipt["verified_files"]:
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"workbook mismatch: {item['filename']}")

    names = [item["filename"] for item in receipt["verified_files"]]
    dynamic_rates = sorted(
        int(match.group(1))
        for name in names
        if (match := re.search(r"-(\d+)k-(?:450 to 1450|1450)", name, re.IGNORECASE))
    )
    isothermal_temperatures = sorted(
        int(match.group(1))
        for name in names
        if (match := re.search(r"-500-(\d+)-2h dwell", name, re.IGNORECASE))
    )
    if dynamic_rates != expected["constant_heating_rates_k_per_min"]:
        raise ValueError(f"heating-rate families changed: {dynamic_rates}")
    if isothermal_temperatures != expected["isothermal_temperatures_c"]:
        raise ValueError(f"isothermal families changed: {isothermal_temperatures}")

    dispositions = binding["candidate_dispositions"]
    expected_ids = {"CAND-P-049", "CAND-P-050", "CAND-P-051", "CAND-P-066", "CAND-P-082", "CAND-P-085"}
    if {item["candidate_id"] for item in dispositions} != expected_ids:
        raise ValueError("candidate disposition set changed")
    if any(item["status"] != "EXACT_REJECTED" for item in dispositions):
        raise ValueError("this audit cannot promote candidates")

    observed = {
        "downloaded_workbooks": len(receipt["verified_files"]),
        "downloaded_bytes": receipt["total_bytes"],
        "constant_heating_rate_runs": len(dynamic_rates),
        "constant_heating_rates_k_per_min": dynamic_rates,
        "isothermal_runs": len(isothermal_temperatures),
        "isothermal_temperatures_c": isothermal_temperatures,
        "material_families": 1,
        "curve_points_treated_as_independent_experiments": False,
        "relative_density_trajectory_labels": 0,
        "green_density_record_labels": 0,
        "final_grain_size_record_labels": 0,
        "activation_energy_record_labels": 0,
        "xrd_patterns": 0,
    }
    audit = {
        "schema": "cimc.forge200.mendeley-sintering-w4-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": binding["source_id"],
        "status": "SOURCE_AND_LICENSE_VERIFIED_SIX_EXACT_CONTRACTS_REJECTED",
        "contract_binding": {"path": binding_path.relative_to(root).as_posix(), "sha256": sha256_file(binding_path)},
        "source": binding["source"],
        "verified_artifacts": verified_artifacts,
        "observed": observed,
        "candidate_dispositions": dispositions,
        "training_actions": 0,
        "host_promotions": 0,
        "leakage_safe_splits_materialized": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Retain the licensed curves for external research only; seek sources with exact record-bound density, green-state, grain, activation-energy, transfer-risk, or XRD uncertainty labels.",
    }
    audit["content_root_sha256"] = content_root(audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": audit["status"], "rejected_exact_contracts": len(dispositions), "content_root_sha256": audit["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
