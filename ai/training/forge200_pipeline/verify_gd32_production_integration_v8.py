#!/usr/bin/env python3
"""Verify the final local GD32/ModelBank integration without touching hardware."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
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
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def load_define(text: str, name: str) -> str:
    match = re.search(rf"^\s*#define\s+{re.escape(name)}\s+([^/\s]+)", text, re.M)
    if match is None:
        raise RuntimeError(f"MISSING_DEFINE:{name}")
    return match.group(1)


def verify_rollback(backup: Path, expected_manifest_sha: str) -> dict:
    manifest = backup / "DELTA_SHA256SUMS.csv"
    if sha256(manifest) != expected_manifest_sha.lower():
        raise RuntimeError(f"ROLLBACK_MANIFEST_GATE:{backup.name}")
    mismatches = []
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        path = backup / "payload" / Path(row["relative_path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(row["bytes"])
            or sha256(path) != row["sha256"].lower()
        ):
            mismatches.append(row["relative_path"])
    if mismatches:
        raise RuntimeError(f"ROLLBACK_PAYLOAD_GATE:{mismatches}")
    return {
        "path": str(backup),
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_sha256": sha256(manifest),
        "mismatches": 0,
    }


def parse_catalog(path: Path, source_root: str) -> list[dict]:
    raw = path.read_bytes()
    if (
        len(raw) < 128
        or raw[:4] != b"F2CT"
        or struct.unpack_from("<H", raw, 4)[0] != 1
        or struct.unpack_from("<H", raw, 6)[0] != 128
    ):
        raise RuntimeError(f"CATALOG_HEADER_GATE:{path}")
    generation, count, entry_bytes, body_bytes = struct.unpack_from(
        "<QIIQ", raw, 8
    )
    body = raw[128:]
    if (
        generation != 1
        or count != 170
        or entry_bytes != 160
        or body_bytes != len(body)
        or hashlib.sha256(body).digest() != raw[32:64]
        or raw[64:96].hex() != source_root
        or any(raw[96:128])
    ):
        raise RuntimeError(f"CATALOG_CONTENT_GATE:{path}")
    entries = []
    for index in range(count):
        entry = body[index * entry_bytes : (index + 1) * entry_bytes]
        model_id = entry[0:32].split(b"\0", 1)[0].decode("ascii")
        package_path = entry[32:56].split(b"\0", 1)[0].decode("ascii")
        golden_path = entry[56:80].split(b"\0", 1)[0].decode("ascii")
        category, tier, engine, opset, _flags, package_bytes = struct.unpack_from(
            "<BBHHHQ", entry, 80
        )
        entries.append(
            {
                "model_id": model_id,
                "package_path": package_path,
                "golden_path": golden_path,
                "category": chr(category),
                "tier": tier,
                "engine_id": engine,
                "opset": opset,
                "package_bytes": package_bytes,
                "package_sha256": entry[96:128].hex(),
                "golden_sha256": entry[128:160].hex(),
            }
        )
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--sd-root", type=Path, required=True)
    parser.add_argument("--keil-log", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    sd_root = args.sd_root.resolve()
    keil_log = args.keil_log.resolve()
    backup_root = args.backup_root.resolve()
    output = args.output.resolve()

    rollback = [
        verify_rollback(
            backup_root
            / "CIMC_PRE_FORGE200_PRODUCTION_INTEGRATION_20260804_012813",
            "55095c8cf2aaaae7d8a70d6abe7a226297543335436150f8f352e147f4b3fae9",
        ),
        verify_rollback(
            backup_root
            / "CIMC_PRE_FORGE200_FATFS_SEEK_SUPPLEMENT_20260804_015012",
            "b3b9ed64320feb57e14f6a9c7f7e094cbe9b4ccd54962d005faf86e46896ddd7",
        ),
        verify_rollback(
            backup_root
            / "CIMC_PRE_FORGE200_HEAP_SUPPLEMENT_20260804_020005",
            "bcf7859774698fa65181f74fc319c51a993e215a6eec9a28e274baf970acdc9d",
        ),
    ]

    config_path = production / "firmware/ai_models_c/lab_build_config.h"
    config = config_path.read_text(encoding="utf-8")
    defines = {
        name: load_define(config, name)
        for name in (
            "LAB_HARDWARE_BRINGUP",
            "LAB_FORGE200_MODELBANK_ENABLE",
            "LAB_FORGE200_BOARD_ACCEPTANCE",
            "LAB_FORGE200_ACTION_AUTHORITY",
        )
    }
    if defines != {
        "LAB_HARDWARE_BRINGUP": "0",
        "LAB_FORGE200_MODELBANK_ENABLE": "1",
        "LAB_FORGE200_BOARD_ACCEPTANCE": "1",
        "LAB_FORGE200_ACTION_AUTHORITY": "0",
    }:
        raise RuntimeError(f"BUILD_CONFIG_GATE:{defines}")
    freertos_config_path = (
        production / "firmware/keil_proj/FreeRTOS/src/FreeRTOSConfig.h"
    )
    freertos_config = freertos_config_path.read_text(encoding="utf-8")
    heap_match = re.search(
        r"#define\s+configTOTAL_HEAP_SIZE\s+"
        r"\(\s*\(\s*size_t\s*\)\s*\(\s*(\d+)\s*\*\s*1024\s*\)\s*\)",
        freertos_config,
    )
    if heap_match is None or int(heap_match.group(1)) != 96:
        raise RuntimeError("FREERTOS_HEAP_96_KIB_GATE")
    freertos_heap_kib = int(heap_match.group(1))

    project_path = (
        production / "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx"
    )
    project_tree = ET.parse(project_path)
    target_names = [
        element.text
        for element in project_tree.findall(".//TargetName")
        if element.text
    ]
    if target_names != ["R2.1", "R3-Q", "R3-P"]:
        raise RuntimeError(f"TARGET_IDENTITY_GATE:{target_names}")
    required_sources = {
        "forge200_modelbank.c",
        "forge200_shared_spi.c",
        "forge200_runtime_v8.c",
        "forge200_bus_guard.c",
        "forge200_board_port.c",
    }
    per_target_sources = []
    for target in project_tree.findall(".//Target"):
        target_name = target.findtext("TargetName")
        file_names = {
            value.text
            for value in target.findall(".//FileName")
            if value.text and value.text.startswith("forge200_")
        }
        if file_names != required_sources:
            raise RuntimeError(
                f"KEIL_SOURCE_GATE:{target_name}:{sorted(file_names)}"
            )
        per_target_sources.append(
            {"target": target_name, "forge200_sources": sorted(file_names)}
        )
    uvopt = (
        production / "firmware/keil_proj/project/CIMC_GD32_Template.uvoptx"
    ).read_text(encoding="utf-8")
    if uvopt.count("<nTsel>3</nTsel>") != 3 or uvopt.count(
        "<pMon>BIN\\CMSIS_AGDI.dll</pMon>"
    ) != 3:
        raise RuntimeError("DAPLINK_IDENTITY_GATE")

    source_pairs = [
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
    ]
    source_records = []
    for source_relative, destination_name in source_pairs:
        source = root / "firmware_integration" / source_relative
        destination = (
            production
            / "firmware/keil_proj/HardWare/Lab_Sentinel"
            / destination_name
        )
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"PRODUCTION_SOURCE_HASH_GATE:{destination_name}")
        source_records.append(
            {
                "source": source.relative_to(root).as_posix(),
                "destination": destination.relative_to(production).as_posix(),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    board_text = (
        production
        / "firmware/keil_proj/HardWare/Lab_Sentinel/forge200_board_port.c"
    ).read_text(encoding="utf-8")
    forbidden_calls = re.findall(
        r"\b(?:heater|relay|motor|actuator|fan12?)_[A-Za-z0-9_]+\s*\(",
        board_text,
    )
    if forbidden_calls:
        raise RuntimeError(f"CONTROL_CALL_GATE:{forbidden_calls}")

    sd_manifest = json.loads((sd_root / "MANIFEST.JSON").read_text(encoding="utf-8"))
    if (
        sd_manifest["model_count"] != 170
        or sd_manifest["exact_count"] != 78
        or sd_manifest["sim_only_count"] != 92
        or sd_manifest["authority_nonzero"] != 0
        or sd_manifest["board_actions"] != 0
    ):
        raise RuntimeError("SD_MANIFEST_BOUNDARY_GATE")
    catalog_a = parse_catalog(
        sd_root / "F200/CATALOGA.BIN",
        sd_manifest["source_manifest"]["content_root_sha256"],
    )
    catalog_b = parse_catalog(
        sd_root / "F200/CATALOGB.BIN",
        sd_manifest["source_manifest"]["content_root_sha256"],
    )
    if catalog_a != catalog_b:
        raise RuntimeError("CATALOG_AB_GATE")
    entry_checks = []
    for entry in catalog_a:
        package = sd_root / entry["package_path"].removeprefix("0:/")
        golden = sd_root / entry["golden_path"].removeprefix("0:/")
        if (
            not package.is_file()
            or not golden.is_file()
            or package.stat().st_size != entry["package_bytes"]
            or sha256(package) != entry["package_sha256"]
            or sha256(golden) != entry["golden_sha256"]
            or package.read_bytes()[12] != 0
        ):
            raise RuntimeError(f"SD_ENTRY_GATE:{entry['model_id']}")
        entry_checks.append(
            {
                "model_id": entry["model_id"],
                "category": entry["category"],
                "tier": entry["tier"],
                "engine_id": entry["engine_id"],
                "package_sha256": entry["package_sha256"],
                "golden_sha256": entry["golden_sha256"],
            }
        )
    if (
        len(entry_checks) != 170
        or len({item["model_id"] for item in entry_checks}) != 170
        or len({item["package_sha256"] for item in entry_checks}) != 170
        or sum(item["category"] == "P" for item in entry_checks) != 112
        or sum(item["category"] == "G" for item in entry_checks) != 30
        or sum(item["category"] == "S" for item in entry_checks) != 28
    ):
        raise RuntimeError("SD_ENTRY_UNIQUENESS_GATE")
    fault_names = {
        path.name for path in (sd_root / "F200/FAULT").glob("*") if path.is_file()
    }
    required_faults = {
        "BADMAG.ICM",
        "BADAUT.ICM",
        "BADPAY.ICM",
        "BADGEN.ICM",
        "BADENG.ICM",
        "BADGLD.ICM",
        "BADGLD.GLD",
        "README.TXT",
    }
    if fault_names != required_faults:
        raise RuntimeError(f"FAULT_FIXTURE_GATE:{sorted(fault_names)}")

    log_text = keil_log.read_text(encoding="latin-1")
    final_match = re.search(
        r'"\.\\Build\\R21\\Objects\\CIMC_R21\.axf" - '
        r"(\d+) Error\(s\), (\d+) Warning\(s\)\.",
        log_text,
    )
    size_match = re.search(
        r"Program Size: Code=(\d+) RO-data=(\d+) RW-data=(\d+) ZI-data=(\d+)",
        log_text,
    )
    if final_match is None or int(final_match.group(1)) != 0 or size_match is None:
        raise RuntimeError("KEIL_REBUILD_GATE")
    new_diagnostics = [
        line
        for line in log_text.splitlines()
        if "warning:" in line.lower()
        and (
            "forge200_" in line.lower()
            or "max31856.c(" in line.lower()
            or "sd_spi.c(" in line.lower()
            or "ffconf.h(" in line.lower()
        )
    ]
    if new_diagnostics:
        raise RuntimeError(f"NEW_SOURCE_WARNING_GATE:{new_diagnostics}")
    code, ro, rw, zi = map(int, size_match.groups())
    rom_bytes = code + ro + rw
    flash_bytes = 4 * 1024 * 1024
    if rom_bytes > int(flash_bytes * 0.88):
        raise RuntimeError(f"ROM_88_PERCENT_GATE:{rom_bytes}")
    build_root = production / "firmware/keil_proj/project/Build/R21"
    build_files = [
        build_root / "Objects/CIMC_R21.axf",
        build_root / "Objects/CIMC_R21.hex",
        build_root / "Listings/CIMC_R21.map",
    ]
    build_records = [
        {
            "path": path.relative_to(production).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in build_files
    ]

    result = {
        "schema": "cimc.forge200.gd32-production-integration.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "LOCAL_KEIL_AND_SD_STAGING_PASS_UNIFIED_PHYSICAL_BOARD_PENDING",
        "rollback": rollback,
        "build_config": {
            **defines,
            "configTOTAL_HEAP_SIZE_KiB": freertos_heap_kib,
        },
        "keil": {
            "target": "R2.1",
            "compiler": "ARMCC 5.06 update 6 build 750",
            "full_rebuild_errors": int(final_match.group(1)),
            "full_rebuild_warnings": int(final_match.group(2)),
            "new_or_modified_source_warnings": 0,
            "program_size": {
                "code": code,
                "ro_data": ro,
                "rw_data": rw,
                "zi_data": zi,
                "rom_bytes": rom_bytes,
                "flash_bytes": flash_bytes,
                "rom_percent": rom_bytes * 100.0 / flash_bytes,
                "rom_88_percent_limit": int(flash_bytes * 0.88),
            },
            "debugger": {
                "nTsel": 3,
                "pMon": "BIN\\CMSIS_AGDI.dll",
            },
            "sources_per_target": per_target_sources,
            "outputs": build_records,
            "log": {
                "path": keil_log.relative_to(root).as_posix(),
                "bytes": keil_log.stat().st_size,
                "sha256": sha256(keil_log),
            },
        },
        "production_sources": source_records,
        "sd_staging": {
            "path": sd_root.relative_to(root).as_posix(),
            "models": len(entry_checks),
            "by_category": {"P": 112, "G": 30, "S": 28},
            "exact": 78,
            "sim_only": 92,
            "catalog_a_sha256": sha256(sd_root / "F200/CATALOGA.BIN"),
            "catalog_b_sha256": sha256(sd_root / "F200/CATALOGB.BIN"),
            "manifest_sha256": sha256(sd_root / "MANIFEST.JSON"),
            "entry_content_root_sha256": hashlib.sha256(
                canonical(entry_checks)
            ).hexdigest(),
            "fault_fixture_count": 7,
        },
        "authority_nonzero": 0,
        "board_actions": 0,
        "ready_for_gd32_burn_now": False,
        "remaining_physical_gates": [
            "copy verified F200 staging to the already-qualified FAT32 microSD",
            "user compiles/downloads target R2.1 through existing CMSIS-DAP",
            "initial 30-model board regression",
            "170 new-model package hash/load/C golden and DWT",
            "64 MiB sequential plus random-page SD throughput",
            "1000 A/B swaps and six refusal fixtures",
            "MAX31856 shared-SPI coexistence and control-period degradation",
            "24-hour soak and final serial receipt",
        ],
        "claim_boundary": (
            "The R2.1 image links locally and the SD tree is hash-complete. "
            "No SD card or GD32 was touched; physical timing, throughput, shared-bus "
            "behavior and board acceptance remain pending."
        ),
    }
    result["content_root_sha256"] = hashlib.sha256(
        canonical(
            {
                "rollback": rollback,
                "build_config": result["build_config"],
                "keil": result["keil"],
                "production_sources": source_records,
                "sd_staging": result["sd_staging"],
            }
        )
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": 170,
                "rom_bytes": rom_bytes,
                "rom_percent": result["keil"]["program_size"]["rom_percent"],
                "new_warnings": 0,
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
