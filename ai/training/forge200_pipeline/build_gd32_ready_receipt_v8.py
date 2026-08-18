#!/usr/bin/env python3
"""Issue the fail-closed local-complete receipt immediately before physical GD32 work."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load(root: Path, relative: str) -> tuple[Path, dict]:
    path = root / relative
    return path, json.loads(path.read_text(encoding="utf-8"))


def record(root: Path, path: Path, data: dict) -> dict:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "status": data.get("status"),
        "content_root_sha256": data.get("content_root_sha256"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()

    paths: dict[str, tuple[Path, dict]] = {}
    for name, relative in {
        "host_closure": "evidence/host_closure.v7.json",
        "local_host": "evidence/local_host_acceptance.v7.json",
        "gap_audit": "evidence/release_gap_audit.v7.json",
        "mcu_export": "evidence/mcu_runtime_export.v8.json",
        "mcu_c": "evidence/mcu_runtime_c_verification.v8.json",
        "sd_staging": "evidence/sd_staging_verification.v8r2.json",
        "sd_faults": "evidence/sd_fault_c_verification.v8.json",
        "gd32_integration": "evidence/gd32_production_integration.v8r8.json",
        "handoff": "evidence/gd32_board_handoff_build.v8.json",
    }.items():
        paths[name] = load(root, relative)

    host = paths["host_closure"][1]
    local_host = paths["local_host"][1]
    gap = paths["gap_audit"][1]
    export = paths["mcu_export"][1]
    mcu_c = paths["mcu_c"][1]
    sd = paths["sd_staging"][1]
    faults = paths["sd_faults"][1]
    integration = paths["gd32_integration"][1]
    handoff = paths["handoff"][1]
    gates = {
        "host_170": host.get("host_qualified_total_including_extensions") == 170,
        "host_category_112_30_28": host.get("host_by_category")
        == {"G": 30, "P": 112, "S": 28},
        "exact_sim_separated": (
            host.get("exact_contract", {}).get("unique_candidates") == 78
            and host.get("sim_only_extensions", {}).get("unique_candidates") == 92
            and gap.get("host_exact_source_bound", {}).get("minimum_release_shortfall")
            == 42
        ),
        "local_host_pass": local_host.get("status")
        == "HOST_FULL_170_TARGET_PASS_EXACT_SIM_BOUNDARIES_SEPARATED_UNIFIED_BOARD_PENDING",
        "mcu_export_170": export.get("model_count") == 170,
        "portable_c_170": (
            mcu_c.get("status")
            == "PASS_170_PORTABLE_C_GOLDENS_CORTEX_M7_EXECUTION_PENDING"
            and mcu_c.get("model_count") == 170
        ),
        "sd_170": (
            sd.get("status") == "SD_STAGING_170_HOST_VERIFIED_UNIFIED_BOARD_PENDING"
            and sd.get("model_count") == 170
            and sd.get("files") == 353
        ),
        "fault_c_pass": faults.get("passed") == 7 and faults.get("failed") == 0,
        "keil_r21_pass": (
            integration.get("status")
            == "LOCAL_KEIL_AND_SD_STAGING_PASS_UNIFIED_PHYSICAL_BOARD_PENDING"
            and integration.get("keil", {}).get("full_rebuild_errors") == 0
            and integration.get("keil", {}).get("new_or_modified_source_warnings") == 0
            and integration.get("keil", {}).get("program_size", {}).get("rom_percent", 100)
            <= 88
        ),
        "authority_zero": all(
            data.get("authority_nonzero", 0) == 0 for _, data in paths.values()
        ),
        "handoff_verified": handoff.get("status")
        == "PASS_GD32_HANDOFF_ARCHIVE_VERIFIED_BOARD_PENDING",
        "board_still_pending": (
            not handoff.get("board_accepted", True)
            and host.get("new_models_board_accepted") == 0
            and host.get("new_models_countable_publicly") == 0
        ),
    }
    test_log = root / "evidence/local_test_run.v8r8.log"
    test_raw = test_log.read_bytes()
    test_text = test_raw.decode(
        "utf-16" if test_raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8",
        errors="replace",
    )
    test_match = re.search(r"Ran (\d+) tests", test_text)
    gates["local_tests_50"] = (
        test_match is not None
        and int(test_match.group(1)) == 50
        and re.search(r"(?:^|\r?\n)OK\r?\n?$", test_text) is not None
    )
    if not all(gates.values()):
        raise RuntimeError(f"GD32_READY_GATE:{gates}")

    parser_path = root / "pipeline/parse_board_receipt_v8.py"
    sd_verify_path = root / "pipeline/verify_sd_card_copy_v8.py"
    runbook_path = root / "docs/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK.md"
    result = {
        "schema": "cimc.forge200.gd32-ready.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "ready_for_gd32_burn_now": True,
        "local_gates": gates,
        "host_assets": {
            "new_total": 170,
            "by_category": {"predictive": 112, "generative": 30, "support": 28},
            "exact_source_label_split_bound": 78,
            "sim_only": 92,
            "exact_source_floor_shortfall": 42,
            "initial_board_assets": 30,
            "projected_total_after_individual_board_pass": 200,
            "projected_logical_generative_after_board_pass": 38,
        },
        "keil": integration["keil"],
        "sd_staging": integration["sd_staging"],
        "rollback": integration["rollback"],
        "test_run": {
            "tests": 50,
            "status": "PASS",
            "path": test_log.relative_to(root).as_posix(),
            "bytes": test_log.stat().st_size,
            "sha256": sha256(test_log),
        },
        "receipts": {
            name: record(root, path, data) for name, (path, data) in paths.items()
        },
        "handoff": {
            "directory": handoff["output_dir"],
            "zip": handoff["output_zip"],
            "zip_bytes": handoff["zip_bytes"],
            "zip_sha256": handoff["zip_sha256"],
            "manifest_sha256": handoff["manifest_sha256"],
        },
        "physical_tools": {
            "runbook": {
                "path": runbook_path.relative_to(root).as_posix(),
                "sha256": sha256(runbook_path),
            },
            "sd_copy_verifier": {
                "path": sd_verify_path.relative_to(root).as_posix(),
                "sha256": sha256(sd_verify_path),
            },
            "uart_parser": {
                "path": parser_path.relative_to(root).as_posix(),
                "sha256": sha256(parser_path),
            },
        },
        "remaining_physical_gates": [
            "copy and hash-verify F200 on the already-qualified FAT32 microSD",
            "user builds/downloads R2.1 with the existing CMSIS-DAP",
            "separate bounded SD-removal/power-recovery drill",
            "initial 30-asset board regression",
            "170 new assets package/golden/DWT on GD32H759",
            "64 MiB SD throughput and MAX31856 shared-bus coexistence",
            "1000 additional A/B swaps with at least four swaps per model",
            "24-hour uninterrupted soak with two-hour refusal injections",
            "parse the complete UART log to GD32_UNIFIED_BOARD_ACCEPTED",
        ],
        "cloud_or_gpu_required": False,
        "no_card_instance_required": False,
        "authority_nonzero": 0,
        "deterministic_control_chain_unchanged": True,
        "board_actions": 0,
        "board_accepted": False,
        "countable_new_models": 0,
        "claim_boundary": (
            "Local work is complete and the firmware is ready to burn, but no new model "
            "is board-accepted or publicly countable yet. The 78 EXACT and 92 SIM_ONLY "
            "tiers remain separate even after physical acceptance."
        ),
    }
    result["content_root_sha256"] = hashlib.sha256(
        canonical(
            {
                "local_gates": gates,
                "host_assets": result["host_assets"],
                "keil": result["keil"],
                "sd_staging": result["sd_staging"],
                "rollback": result["rollback"],
                "receipts": result["receipts"],
                "handoff": result["handoff"],
                "physical_tools": result["physical_tools"],
            }
        )
    ).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "ready_for_gd32_burn_now": True,
                "new_assets": 170,
                "tests": 50,
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
