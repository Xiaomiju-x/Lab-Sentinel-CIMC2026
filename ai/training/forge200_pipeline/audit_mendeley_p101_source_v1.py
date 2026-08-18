#!/usr/bin/env python3
"""Verify the downloaded P101 source and retain an exact-contract rejection."""

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
    binding_path = root / "contracts/mendeley_p101_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["candidate_id"] != "CAND-P-101" or binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("invalid P101 identity or authority")

    verified_artifacts = {}
    for name, item in binding["artifacts"].items():
        path = root / item["path"]
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch: {actual_sha}")
        if "bytes" in item and path.stat().st_size != item["bytes"]:
            raise ValueError(f"{name} byte count mismatch")
        verified_artifacts[name] = {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "verified": True,
        }

    inventory = json.loads((root / binding["artifacts"]["workbook_inventory"]["path"]).read_text(encoding="utf-8"))
    sheets = inventory["sheets"]
    expected_names = ["0.5K-2", "0.5K-3", "1K-1", "1K-2", "2K-1", "2K-2", "5K-1", "5K-2"]
    if inventory["sheetCount"] != 8 or [item["name"] for item in sheets] != expected_names:
        raise ValueError("unexpected P101 workbook run inventory")
    curve_rows = sum(int(item["columns"][8]["nonEmpty"]) - 1 for item in sheets)
    if curve_rows != binding["observed_experiment_design"]["curve_rows_total"]:
        raise ValueError(f"curve row count mismatch: {curve_rows}")

    run_serials = []
    sample_masses = []
    for sheet in sheets:
        selected = {int(item["row"]): item["values"] for item in sheet["selectedRows"]}
        run_serials.append(int(selected[6][1]))
        sample_masses.append(float(selected[15][1]))
    if len(set(run_serials)) != 8:
        raise ValueError("run serials are not unique")

    receipt = {
        "schema": "cimc.forge200.mendeley-p101-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "CAND-P-101",
        "status": "SOURCE_AND_LICENSE_VERIFIED_EXACT_CONTRACT_REJECTED",
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "source": binding["source"],
        "verified_artifacts": verified_artifacts,
        "observed": {
            **binding["observed_experiment_design"],
            "unique_run_serials": len(set(run_serials)),
            "run_serials": run_serials,
            "sample_mass_mg_range": [min(sample_masses), max(sample_masses)],
            "curve_points_treated_as_independent_experiments": False,
        },
        "exact_rejection": binding["exact_rejection"],
        "training_actions": 0,
        "test_evaluation_actions": 0,
        "host_promoted": False,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Retain this source for external-validation research only; exact P101 requires multiple independently labeled resin/catalyst formulations and kinetic-parameter vectors fixed before grouped split.",
    }
    receipt["content_root_sha256"] = content_root(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "runs": len(run_serials),
        "curve_rows": curve_rows,
        "content_root_sha256": receipt["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
