#!/usr/bin/env python3
"""Final local acceptance: only the unified physical GD32 window may remain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--rollback", type=Path, required=True)
    parser.add_argument("--sd-staging", type=Path, required=True)
    parser.add_argument("--keil-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-root", type=Path, required=True,
                        help="All input and output paths must stay under this root.")
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    rollback = args.rollback.resolve()
    staging = args.sd_staging.resolve()
    keil_log = args.keil_log.resolve()
    output = args.output.resolve()
    d_scope = args.scope_root.resolve()
    for path in (root, production, rollback, staging, keil_log, output):
        try:
            path.relative_to(d_scope)
        except ValueError as exc:
            raise RuntimeError(f"D_SCOPE_GATE:{path}") from exc

    rollback_manifest = rollback / "DELTA_SHA256SUMS.csv"
    if sha256(rollback_manifest) != "289d79849fb3f78b4b6dbdc192966c73df9aa0c8ada25a67e72e1662425b42fa":
        raise RuntimeError("ROLLBACK_MANIFEST_SHA_GATE")
    with rollback_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rollback_rows = list(csv.DictReader(handle))
    rollback_bad = []
    for row in rollback_rows:
        path = rollback / row["path"]
        if not path.is_file() or path.stat().st_size != int(row["bytes"]) or sha256(path) != row["sha256"].lower():
            rollback_bad.append(row["path"])
    if rollback_bad or len(rollback_rows) != 25:
        raise RuntimeError(f"ROLLBACK_PAYLOAD_GATE:{rollback_bad}")

    config = (production / "firmware/ai_models_c/lab_build_config.h").read_text(encoding="utf-8")
    expected_defines = {
        "LAB_HARDWARE_BRINGUP": "0", "LAB_FORGE200_MODELBANK_ENABLE": "1",
        "LAB_FORGE200_BOARD_ACCEPTANCE": "1", "LAB_FORGE200_ACTION_AUTHORITY": "0",
    }
    for name, value in expected_defines.items():
        if re.search(rf"^#define\s+{name}\s+{value}\s*$", config, re.M) is None:
            raise RuntimeError(f"CONFIG_GATE:{name}")

    project = production / "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx"
    tree = ET.parse(project)
    targets = tree.findall(".//Target")
    if [target.findtext("TargetName") for target in targets] != ["R2.1", "R3-Q", "R3-P"]:
        raise RuntimeError("TARGET_IDENTITY_GATE")
    required_sources = {
        "forge200_modelbank.c", "forge200_shared_spi.c", "forge200_runtime_v8.c",
        "forge200_bus_guard.c", "forge200_board_port.c", "forge200_rag_v9.c",
        "forge200_rag_board_v9.c", "veriprocess_v9.c", "veriprocess_board_v9.c",
    }
    per_target = []
    for target in targets:
        names = {item.text for item in target.findall(".//FileName") if item.text in required_sources}
        if names != required_sources:
            raise RuntimeError(f"SOURCE_SET_GATE:{target.findtext('TargetName')}:{names}")
        per_target.append({"target": target.findtext("TargetName"), "sources": sorted(names)})
    uvopt = (production / "firmware/keil_proj/project/CIMC_GD32_Template.uvoptx").read_text(encoding="utf-8")
    if uvopt.count("<nTsel>3</nTsel>") != 3 or uvopt.count("<pMon>BIN\\CMSIS_AGDI.dll</pMon>") != 3:
        raise RuntimeError("DAPLINK_GATE")

    pairs = [
        ("modelbank_v4/forge200_modelbank.h", "forge200_modelbank.h"),
        ("modelbank_v4/forge200_modelbank.c", "forge200_modelbank.c"),
        ("modelbank_v4/forge200_shared_spi.h", "forge200_shared_spi.h"),
        ("modelbank_v4/forge200_shared_spi.c", "forge200_shared_spi.c"),
        ("modelbank_v8/forge200_runtime_v8.h", "forge200_runtime_v8.h"),
        ("modelbank_v8/forge200_runtime_v8.c", "forge200_runtime_v8.c"),
        ("modelbank_v8_gd32/forge200_bus_guard.h", "forge200_bus_guard.h"),
        ("modelbank_v8_gd32/forge200_bus_guard.c", "forge200_bus_guard.c"),
        ("modelbank_v8_gd32/forge200_board_port.h", "forge200_board_port.h"),
        ("modelbank_v8_gd32/forge200_board_port.c", "forge200_board_port.c"),
        ("rag_v9/forge200_rag_v9.h", "forge200_rag_v9.h"),
        ("rag_v9/forge200_rag_v9.c", "forge200_rag_v9.c"),
        ("rag_v9/forge200_rag_board_v9.h", "forge200_rag_board_v9.h"),
        ("rag_v9/forge200_rag_board_v9.c", "forge200_rag_board_v9.c"),
        ("veriprocess_v9/veriprocess_v9.h", "veriprocess_v9.h"),
        ("veriprocess_v9/veriprocess_v9.c", "veriprocess_v9.c"),
        ("veriprocess_v9/veriprocess_board_v9.h", "veriprocess_board_v9.h"),
        ("veriprocess_v9/veriprocess_board_v9.c", "veriprocess_board_v9.c"),
    ]
    source_records = []
    for source_relative, destination_name in pairs:
        source = root / "firmware_integration" / source_relative
        destination = production / "firmware/keil_proj/HardWare/Lab_Sentinel" / destination_name
        if not source.is_file() or not destination.is_file() or sha256(source) != sha256(destination):
            raise RuntimeError(f"SOURCE_HASH_GATE:{destination_name}")
        source_records.append({"name": destination_name, "bytes": destination.stat().st_size, "sha256": sha256(destination)})

    log_text = keil_log.read_text(encoding="latin-1")
    final_match = re.search(r'"\.\\Build\\R21\\Objects\\CIMC_R21\.axf" - (\d+) Error\(s\), (\d+) Warning\(s\)\.', log_text)
    size_match = re.search(r"Program Size: Code=(\d+) RO-data=(\d+) RW-data=(\d+) ZI-data=(\d+)", log_text)
    if final_match is None or int(final_match.group(1)) != 0 or size_match is None:
        raise RuntimeError("KEIL_GATE")
    new_warnings = [line for line in log_text.splitlines() if "warning:" in line.lower() and any(name in line.lower() for name in ("forge200_", "veriprocess_"))]
    if new_warnings:
        raise RuntimeError(f"NEW_WARNING_GATE:{new_warnings}")
    code, ro, rw, zi = map(int, size_match.groups())
    rom = code + ro + rw
    flash = 4 * 1024 * 1024
    if rom > int(flash * .88):
        raise RuntimeError(f"ROM_GATE:{rom}")
    build_root = production / "firmware/keil_proj/project/Build/R21"
    build_files = [build_root / "Objects/CIMC_R21.axf", build_root / "Objects/CIMC_R21.hex", build_root / "Listings/CIMC_R21.map"]
    outputs = [{"path": path.relative_to(production).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in build_files]

    sd_manifest = json.loads((staging / "MANIFEST.v9.json").read_text(encoding="utf-8"))
    sd_verify = json.loads((root / "evidence/sd_staging_selfverify.v9.json").read_text(encoding="utf-8"))
    rag = json.loads((root / "evidence/rag_runtime_host_acceptance.v9.json").read_text(encoding="utf-8"))
    veriprocess = json.loads((root / "evidence/veriprocess_host_acceptance.v9.json").read_text(encoding="utf-8"))
    faults = json.loads((root / "evidence/unified_fault_matrix.v9.json").read_text(encoding="utf-8"))
    parsers = json.loads((root / "evidence/board_parser_tests.v9.json").read_text(encoding="utf-8"))
    gates = {
        "sd_375_hash_complete": sd_manifest.get("file_count") == 375 and sd_verify.get("accepted") and sd_verify.get("actual_content_root_sha256") == sd_manifest.get("content_root_sha256"),
        "modelbank_170": sd_manifest.get("base_modelbank", {}).get("model_count") == 170,
        "rag_120": rag.get("status", "").startswith("PASS_HOST_RAG") and rag.get("workload", {}).get("safe") == 120,
        "veriprocess_69": veriprocess.get("status", "").startswith("PASS_HOST_AND_ARMCLANG") and veriprocess.get("host_cases", {}).get("passed") == 69,
        "fault_matrix": faults.get("accepted") is True,
        "parsers": parsers.get("status") == "PASS" and parsers.get("cases") == 4,
        "authority_zero": all(value.get("authority_nonzero", value.get("authority", 0)) == 0 for value in (sd_manifest, veriprocess, faults)) and rag.get("authority") == 0,
    }
    if not all(gates.values()):
        raise RuntimeError(f"LOCAL_EVIDENCE_GATE:{gates}")

    result = {
        "schema": "cimc.forge200.gd32-local-complete.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "rollback": {"path": str(rollback), "files": len(rollback_rows), "bytes": sum(int(row["bytes"]) for row in rollback_rows), "manifest_sha256": sha256(rollback_manifest), "mismatches": 0},
        "build_config": expected_defines,
        "keil": {
            "target": "R2.1", "compiler": "ARMCC 5.06 update 6 build 750",
            "errors": int(final_match.group(1)), "warnings": int(final_match.group(2)),
            "new_source_warnings": 0,
            "program_size": {"code": code, "ro_data": ro, "rw_data": rw, "zi_data": zi, "rom_bytes": rom, "rom_percent": rom * 100.0 / flash},
            "sources_per_target": per_target, "debugger": {"nTsel": 3, "pMon": "BIN\\CMSIS_AGDI.dll"},
            "log": {"path": keil_log.relative_to(root).as_posix(), "bytes": keil_log.stat().st_size, "sha256": sha256(keil_log)},
            "outputs": outputs,
        },
        "production_sources": source_records,
        "sd_staging": {"path": staging.relative_to(root).as_posix(), "files": sd_manifest["file_count"], "bytes": sd_manifest["bytes"], "content_root_sha256": sd_manifest["content_root_sha256"], "manifest_sha256": sha256(staging / "MANIFEST.v9.json")},
        "local_gates": gates,
        "model_inventory": {"initial_assets": 30, "new_assets": 170, "total_after_board_acceptance": 200, "new_exact": 78, "new_sim_only": 92, "new_by_category": {"P": 112, "G": 30, "S": 28}, "generative_logical_after_board_acceptance": 38},
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
        "countable_new_models": 0,
        "remaining_required_actions": [
            "COPY_AND_SHA_VERIFY_F200_ON_EXISTING_FAT32_MICROSD",
            "USER_KEIL_R21_DOWNLOAD_TO_GD32",
            "PHYSICAL_WAL_POWER_CUT_AND_RECOVERY",
            "PHYSICAL_SD_REMOVAL_FAIL_CLOSED",
            "UNIFIED_170_RAG120_VERIPROCESS_MAX31856_SHARED_SPI_ACCEPTANCE",
            "1000_PHYSICAL_AB_LOADS_AND_24H_SOAK",
            "PARSE_FINAL_UART_TO_GD32_UNIFIED_BOARD_ACCEPTED",
        ],
        "runbook": "docs/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md",
        "claim_boundary": "All local and host work is complete. No SD card or GD32 was touched; physical board acceptance remains mandatory and authority stays zero.",
    }
    result["content_root_sha256"] = hashlib.sha256(canonical({
        "rollback": result["rollback"], "keil": result["keil"], "sources": source_records,
        "sd": result["sd_staging"], "gates": gates, "inventory": result["model_inventory"],
    })).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "rom_bytes": rom, "rom_percent": result["keil"]["program_size"]["rom_percent"], "sources": len(source_records), "sd_files": sd_manifest["file_count"], "content_root_sha256": result["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
