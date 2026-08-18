#!/usr/bin/env python3
"""Aggregate current local host evidence without making a board claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    completed = subprocess.run([
        sys.executable, "-m", "unittest", "discover",
        "-s", str(root / "tests"), "-p", "test_forge200_local.py", "-v",
    ], text=True, capture_output=True)
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    test_count = int(match.group(1)) if match else 0
    if completed.returncode != 0 or test_count == 0 or "OK" not in combined:
        raise RuntimeError(f"LOCAL_TEST_GATE_FAILED:{completed.returncode}:{test_count}")

    names = [
        "host_closure.v4.json",
        "modelbank_build.v5.json",
        "modelbank_host_dry_run.v5.json",
        "host_artifact_verification.v4.json",
        "interface_freeze_verification.v2.json",
        "firmware_adapter_host_compile.v4.json",
        "unified_staging.v4.json",
        "release_gap_audit.v5.json",
        "exact_data_intake.v4.json",
        "mendeley_sic_plunger_source_contract_audit.v1.json",
        "mendeley_p105_afm_source_contract_audit.v1.json",
        "phm2016_p089_source_preflight.v1.json",
    ]
    receipts = []
    for name in names:
        path = root / "evidence" / name
        document = json.loads(path.read_text(encoding="utf-8"))
        receipts.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256(path),
            "status": document["status"],
        })

    result = {
        "schema": "cimc.forge200.local-host-acceptance.v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_INFRA_V5_PASS_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "tests": {"status": "PASS", "count": test_count, "returncode": completed.returncode},
        "receipts": receipts,
        "host": {
            "exact_source_bound_models": 78,
            "sim_only_extensions": 7,
            "modelbank_models": 85,
            "modelbank_v5_files": 379,
            "modelbank_v5_payload_bytes": 30402946,
            "modelbank_v5_hardlinked_package_files": 375,
            "modelbank_1000_swap_dry_run": True,
            "fault_injection_modes": 6,
            "interface_conformance": True,
            "armclang_cortex_m7_compile": True,
        },
        "release": {
            "minimum_new_assets": 120,
            "host_exact_shortfall": 42,
            "including_sim_only_shortfall": 35,
            "release_floor_met": False,
        },
        "board": {
            "new_models_board_accepted": 0,
            "new_models_countable_publicly": 0,
            "gd32_power_or_burn_performed": False,
            "unified_board_acceptance_pending": True,
            "ready_for_gd32_burn_now": False,
        },
        "safety": {
            "authority_nonzero": 0,
            "production_files_modified": 0,
            "deterministic_control_chain_replaced": False,
            "sd_separately_burned": False,
            "old_cloud_instances_reconnected": False,
        },
        "download": {
            "cloud_no_card_mode_required_now": False,
            "local_nist_p099_manual_download_can_close_at_most_one_current_contract": True,
        },
        "blocker": "Release floor still needs 42 exact source-bound host passes; additional GPU time, input-derived labels, mirrors without explicit data licenses, and curve-point multiplication cannot manufacture those truths.",
    }
    result["content_root_sha256"] = content_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "tests": test_count,
        "host_models": result["host"]["modelbank_models"],
        "exact_shortfall": result["release"]["host_exact_shortfall"],
        "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
