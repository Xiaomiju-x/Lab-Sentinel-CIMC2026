#!/usr/bin/env python3
"""Build and verify the immutable final local-to-board handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
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


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise RuntimeError(f"COPY_HASH_GATE:{source}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    output_dir = args.output_dir.resolve()
    output_zip = args.output_zip.resolve()
    receipt_path = args.receipt.resolve()
    if output_dir.exists() or output_zip.exists():
        raise FileExistsError("handoff output already exists; use a new immutable version")

    sd_release = root / "releases/forge200-sd-staging-v8r2-20260804"
    shutil.copytree(sd_release / "F200", output_dir / "SD_CARD_ROOT/F200")
    for name in ("FILES.CSV", "MANIFEST.JSON", "README.TXT"):
        copy_file(sd_release / name, output_dir / "SD_CARD_ROOT" / name)

    candidate_files = [
        "docs/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK.md",
        "pipeline/parse_board_receipt_v8.py",
        "pipeline/verify_sd_card_copy_v8.py",
        "evidence/host_closure.v7.json",
        "evidence/local_host_acceptance.v7.json",
        "evidence/release_gap_audit.v7.json",
        "evidence/modelbank_host_dry_run.v7.json",
        "evidence/host_artifact_verification.v7.json",
        "evidence/interface_freeze_verification.v7.json",
        "evidence/mcu_runtime_export.v8.json",
        "evidence/mcu_runtime_c_verification.v8.json",
        "evidence/firmware_runtime_host_compile.v8.json",
        "evidence/sd_staging_verification.v8r2.json",
        "evidence/sd_fault_c_verification.v8.json",
        "evidence/gd32_source_staging.v8r8.json",
        "evidence/gd32_production_integration.v8r8.json",
        "evidence/keil_r21_forge200_rebuild_v8r8.log",
    ]
    for relative in candidate_files:
        destination_group = "DOCS" if relative.startswith("docs/") else (
            "TOOLS" if relative.startswith("pipeline/") else "EVIDENCE"
        )
        copy_file(root / relative, output_dir / destination_group / Path(relative).name)

    production_files = [
        "firmware/ai_models_c/lab_build_config.h",
        "firmware/keil_proj/FreeRTOS/src/FreeRTOSConfig.h",
        "firmware/keil_proj/HardWare/Sensors/FatFs/ffconf.h",
        "firmware/keil_proj/HardWare/Sensors/sd_spi.c",
        "firmware/keil_proj/HardWare/Sensors/max31856.c",
        "firmware/keil_proj/HardWare/Lab_Sentinel/lab_sentinel.c",
        "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx",
    ]
    for relative in production_files:
        copy_file(production / relative, output_dir / "PRODUCTION_SNAPSHOT" / relative)
    integration_root = production / "firmware/keil_proj/HardWare/Lab_Sentinel"
    for name in (
        "forge200_modelbank.h",
        "forge200_modelbank.c",
        "forge200_shared_spi.h",
        "forge200_shared_spi.c",
        "forge200_runtime_v8.h",
        "forge200_runtime_v8.c",
        "forge200_bus_guard.h",
        "forge200_bus_guard.c",
        "forge200_board_port.h",
        "forge200_board_port.c",
    ):
        copy_file(
            integration_root / name,
            output_dir / "PRODUCTION_SNAPSHOT/firmware/keil_proj/HardWare/Lab_Sentinel" / name,
        )

    build_root = production / "firmware/keil_proj/project/Build/R21"
    for source, name in (
        (build_root / "Objects/CIMC_R21.axf", "CIMC_R21.axf"),
        (build_root / "Objects/CIMC_R21.hex", "CIMC_R21.hex"),
        (build_root / "Listings/CIMC_R21.map", "CIMC_R21.map"),
    ):
        copy_file(source, output_dir / "FIRMWARE_BUILD" / name)

    records = [
        {
            "path": path.relative_to(output_dir).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file())
    ]
    manifest = {
        "schema": "cimc.forge200.gd32-board-handoff.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "files_without_manifest": len(records),
        "bytes_without_manifest": sum(int(item["bytes"]) for item in records),
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "new_assets": {"total": 170, "exact": 78, "sim_only": 92},
        "projected_total_assets_after_board_pass": 200,
        "authority_nonzero": 0,
        "board_accepted": False,
        "countable_new_models": 0,
    }
    manifest_path = output_dir / "HANDOFF_MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    fixed_time = (2026, 8, 4, 0, 0, 0)
    with zipfile.ZipFile(
        output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
            info = zipfile.ZipInfo(path.relative_to(output_dir.parent).as_posix(), fixed_time)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    with zipfile.ZipFile(output_zip, "r") as archive:
        bad_member = archive.testzip()
        names = archive.namelist()
    if bad_member is not None or len(names) != len(records) + 1:
        raise RuntimeError(f"ZIP_VERIFICATION_GATE:{bad_member}:{len(names)}")
    receipt = {
        "schema": "cimc.forge200.gd32-board-handoff-build.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_GD32_HANDOFF_ARCHIVE_VERIFIED_BOARD_PENDING",
        "output_dir": str(output_dir),
        "output_zip": str(output_zip),
        "directory_files": len(records) + 1,
        "directory_bytes": sum(path.stat().st_size for path in output_dir.rglob("*") if path.is_file()),
        "manifest_sha256": sha256(manifest_path),
        "content_root_sha256": manifest["content_root_sha256"],
        "zip_bytes": output_zip.stat().st_size,
        "zip_sha256": sha256(output_zip),
        "zip_members": len(names),
        "zip_test_bad_member": bad_member,
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
    }
    receipt["receipt_content_root_sha256"] = hashlib.sha256(
        canonical({key: value for key, value in receipt.items() if key != "created_at_utc"})
    ).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
