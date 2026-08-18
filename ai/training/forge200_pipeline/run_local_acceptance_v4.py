#!/usr/bin/env python3
"""Run and freeze the complete host-only Forge200 v4 acceptance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def root_hash(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    expected = {
        "host_closure.v4.json": "HOST_CLOSURE_PARTIAL_RELEASE_FLOOR_NOT_MET_BOARD_PENDING",
        "modelbank_build.v4.json": "HOST_MODELBANK_BUILT_BOARD_PENDING",
        "modelbank_host_dry_run.v4.json": "PASS",
        "host_artifact_verification.v4.json": "PASS_HOST_ONLY_BOARD_PENDING",
        "interface_freeze_verification.v2.json": "PASS_HOST_CONFORMANCE_AND_MUTATION_BOARD_PENDING",
        "firmware_adapter_host_compile.v4.json": "PASS_ARMCLANG_CORTEX_M7_BOARD_PENDING_NOT_IN_PRODUCTION_TARGET",
        "unified_staging.v4.json": "HOST_UNIFIED_STAGING_PARTIAL_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "release_gap_audit.v4.json": "RELEASE_FLOOR_BLOCKED_BY_EXACT_DATA_AND_LATER_BOARD_ACCEPTANCE",
        "exact_data_intake.v4.json": "INTAKE_SPEC_READY_EXACT_DATA_NOT_PRESENT",
    }
    receipts = []
    for name, status in expected.items():
        path = root / "evidence" / name
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != status:
            raise SystemExit(f"unexpected {name} status: {value.get('status')!r}")
        receipts.append({"path": f"evidence/{name}", "sha256": sha256(path), "status": status})

    env = os.environ.copy()
    tooling = str(root / ".tooling/gpu4050")
    pipeline = str(root / "pipeline")
    env["PYTHONPATH"] = os.pathsep.join([tooling, pipeline, env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"), "-p", "test_*.py", "-v"],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    transcript = completed.stdout + completed.stderr
    match = re.search(r"Ran (\d+) tests", transcript)
    test_count = int(match.group(1)) if match else 0
    if completed.returncode != 0 or test_count < 17 or not transcript.rstrip().endswith("OK"):
        print(transcript)
        raise SystemExit("local acceptance tests failed or were incomplete")

    closure = json.loads((root / "evidence/host_closure.v4.json").read_text(encoding="utf-8"))
    gap = json.loads((root / "evidence/release_gap_audit.v4.json").read_text(encoding="utf-8"))
    document = {
        "schema": "cimc.forge200.local-host-acceptance.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_INFRA_PASS_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "tests": {"status": "PASS", "count": test_count, "returncode": completed.returncode},
        "receipts": receipts,
        "host": {
            "exact_source_bound_models": closure["exact_contract"]["unique_candidates"],
            "sim_only_extensions": closure["sim_only_extensions"]["unique_candidates"],
            "modelbank_models": closure["host_qualified_total_including_extensions"],
            "armclang_cortex_m7_compile": True,
            "modelbank_1000_swap_dry_run": True,
            "interface_conformance": True,
            "artifact_full_check_and_golden": True,
        },
        "release": {
            "minimum_new_assets": 120,
            "host_exact_shortfall": gap["host_exact_source_bound"]["minimum_release_shortfall"],
            "including_sim_only_shortfall": gap["minimum_release_shortfall_including_sim_only"],
            "release_floor_met": False,
        },
        "board": {
            "new_models_board_accepted": 0,
            "new_models_countable_publicly": 0,
            "gd32_power_or_burn_performed": False,
            "unified_board_acceptance_pending": True,
        },
        "safety": {
            "authority_nonzero": 0,
            "production_files_modified": 0,
            "deterministic_control_chain_replaced": False,
            "sd_separately_burned": False,
        },
        "cloud_no_card_download_required_now": False,
        "ready_for_gd32_burn_now": False,
        "blocker": "Release floor needs 42 more exact source-bound host passes before unified GD32 board acceptance; new truth cannot be manufactured by more GPU time.",
    }
    document["content_root_sha256"] = root_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "tests": test_count,
        "host_models": closure["host_qualified_total_including_extensions"],
        "exact_shortfall": document["release"]["host_exact_shortfall"],
        "status": document["status"],
        "content_root_sha256": document["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
