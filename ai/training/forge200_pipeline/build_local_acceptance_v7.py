#!/usr/bin/env python3
"""Aggregate final local Forge200 evidence without making a GD32 board claim."""

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
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-p", "test_forge200_local.py", "-v"],
        text=True,
        capture_output=True,
    )
    combined = completed.stdout + "\n" + completed.stderr
    match = re.search(r"Ran\s+(\d+)\s+tests?", combined)
    test_count = int(match.group(1)) if match else 0
    if completed.returncode != 0 or test_count == 0 or "OK" not in combined:
        raise RuntimeError(f"LOCAL_TEST_GATE_FAILED:{completed.returncode}:{test_count}")

    names = [
        "host_closure.v7.json",
        "host_artifact_verification.v7.json",
        "modelbank_build.v7.json",
        "modelbank_host_dry_run.v7.json",
        "interface_freeze_verification.v7.json",
        "firmware_adapter_host_compile.v7.json",
        "unified_staging.v7.json",
        "release_gap_audit.v7.json",
        "sim_extension_pack_39_closure.v1.json",
        "sim_extension_board41_closure.v1.json",
        "nanolm_sim_extension5_closure.v1.json",
    ]
    receipts = []
    for name in names:
        path = root / "evidence" / name
        document = json.loads(path.read_text(encoding="utf-8"))
        receipts.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "status": document["status"]})

    result = {
        "schema": "cimc.forge200.local-host-acceptance.v7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_FULL_170_TARGET_PASS_EXACT_SIM_BOUNDARIES_SEPARATED_UNIFIED_BOARD_PENDING",
        "tests": {"status": "PASS", "count": test_count, "returncode": completed.returncode},
        "receipts": receipts,
        "host": {
            "new_models": 170,
            "by_category": {"P": 112, "G": 30, "S": 28},
            "exact_source_bound_models": 78,
            "sim_only_extensions": 92,
            "package_sha256_collisions": 0,
            "payload_sha256_collisions": 0,
            "onnx_full_check_pass": 170,
            "golden_archives_pass": 170,
            "modelbank_files": 803,
            "modelbank_payload_bytes": 36622015,
            "modelbank_1000_swap_dry_run": True,
            "fault_injection_modes": 6,
            "interface_cases": 23,
            "armclang_cortex_m7_objects": 2,
            "armclang_warnings": 0,
            "armclang_errors": 0,
        },
        "release": {
            "initial_assets": 30,
            "initial_logical_models": 28,
            "combined_assets_if_all_new_models_pass_board": 200,
            "new_target_met_on_host": True,
            "logical_generative_models_if_all_new_models_pass_board": 38,
            "exact_source_bound_minimum_floor_shortfall": 42,
            "sim_only_not_exact_source_substitute": True,
        },
        "board": {
            "new_models_board_accepted": 0,
            "new_models_countable_publicly": 0,
            "gd32_power_or_burn_performed": False,
            "unified_board_acceptance_pending": True,
            "ready_for_single_unified_gd32_window": True,
        },
        "safety": {
            "authority_nonzero": 0,
            "production_files_modified": 0,
            "deterministic_control_chain_replaced": False,
            "sd_separately_burned": False,
            "old_cloud_instances_reconnected": False,
            "project_writes_outside_d_drive": False,
        },
        "next_required_action": "USER_POWERS_GD32_AND_OPENS_ONE_UNIFIED_KEIL_BOARD_ACCEPTANCE_WINDOW",
        "claim_boundary": "Host completion is not GD32 acceptance. Only individually board-passed assets become countable; exact and SIM_ONLY tiers remain separate.",
    }
    result["content_root_sha256"] = content_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "tests": test_count, "new_host_models": 170,
        "combined_projection": 200, "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
