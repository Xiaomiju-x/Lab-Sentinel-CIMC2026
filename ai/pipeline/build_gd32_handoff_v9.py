#!/usr/bin/env python3
"""Build a hash-complete, hardlink-efficient final GD32 v9 handoff directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def place(source: Path, destination: Path, methods: dict[str, int]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        methods["hardlink"] += 1
    except OSError:
        shutil.copy2(source, destination)
        methods["copy"] += 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--sd-staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-root", type=Path, required=True,
                        help="All input and output paths must stay under this root.")
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    sd = args.sd_staging.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    scope = args.scope_root.resolve()
    for path in (root, production, sd, output):
        try:
            path.relative_to(scope)
        except ValueError as exc:
            raise RuntimeError(f"D_SCOPE_GATE:{path}") from exc
    ready = json.loads((root / "evidence/gd32_local_complete.v9.json").read_text(encoding="utf-8"))
    if ready.get("status") != "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED":
        raise RuntimeError("LOCAL_COMPLETE_GATE")
    output.mkdir(parents=True)
    methods = {"hardlink": 0, "copy": 0}

    for source in sorted(sd.rglob("*")):
        if source.is_file():
            place(source, output / "SD_STAGING" / source.relative_to(sd), methods)
    build = production / "firmware/keil_proj/project/Build/R21"
    for relative in ("Objects/CIMC_R21.axf", "Objects/CIMC_R21.hex", "Listings/CIMC_R21.map"):
        place(build / relative, output / "FIRMWARE" / Path(relative).name, methods)
    project = production / "firmware/keil_proj/project"
    for name in ("CIMC_GD32_Template.uvprojx", "CIMC_GD32_Template.uvoptx"):
        place(project / name, output / "PROJECT" / name, methods)

    source_groups = (
        "firmware_integration/modelbank_v4", "firmware_integration/modelbank_v8",
        "firmware_integration/modelbank_v8_gd32", "firmware_integration/rag_v9",
        "firmware_integration/veriprocess_v9",
    )
    for group in source_groups:
        directory = root / group
        for source in sorted(directory.glob("*.[ch]")):
            place(source, output / "SOURCES" / directory.name / source.name, methods)

    tool_names = (
        "verify_sd_card_copy_v9.py", "parse_powercut_recovery_v9.py",
        "parse_board_receipt_v9.py", "test_board_parsers_v9.py",
        "verify_unified_fault_matrix_v9.py", "verify_gd32_local_complete_v9.py",
        "stage_gd32_integration_v9.py",
    )
    for name in tool_names:
        place(root / "pipeline" / name, output / "TOOLS" / name, methods)
    place(root / "docs/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md",
          output / "DOCS/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md", methods)

    evidence_names = (
        "gd32_local_complete.v9.json", "keil_r21_forge200_rebuild_v9r4.log",
        "gd32_source_staging.v9.json", "sd_staging_selfverify.v9.json",
        "rag_runtime_host_acceptance.v9.json", "veriprocess_host_acceptance.v9.json",
        "unified_fault_matrix.v9.json", "board_parser_tests.v9.json",
        "sd_fault_c_verification.v8.json",
    )
    for name in evidence_names:
        place(root / "evidence" / name, output / "EVIDENCE" / name, methods)

    (output / "BOARD_EVIDENCE_REQUIRED.txt").write_text(
        "LOCAL/HOST ACCEPTANCE IS COMPLETE. PHYSICAL GD32 ACCEPTANCE IS NOT.\n"
        "FOLLOW DOCS/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md.\n"
        "DO NOT SET board_accepted OR countable_model BEFORE GD32_UNIFIED_BOARD_ACCEPTED.\n",
        encoding="ascii",
    )
    rollback = ready["rollback"]
    (output / "ROLLBACK_REFERENCE.json").write_text(json.dumps({
        "path": rollback["path"], "files": rollback["files"], "bytes": rollback["bytes"],
        "manifest_sha256": rollback["manifest_sha256"],
        "restore": "Use RESTORE.md inside the referenced rollback directory; never build in the rollback tree.",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    records = []
    excluded = {"MANIFEST.v9.json", "FILES.v9.csv"}
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        relative = path.relative_to(output).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    files_path = output / "FILES.v9.csv"
    with files_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    content_root = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema": "cimc.forge200.gd32-unified-handoff.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "payload_files": len(records),
        "payload_bytes": sum(item["bytes"] for item in records),
        "files_csv_sha256": sha256(files_path),
        "content_root_sha256": content_root,
        "storage_methods": methods,
        "model_inventory": ready["model_inventory"],
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
        "runbook": "DOCS/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md",
    }
    (output / "MANIFEST.v9.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"], "files": manifest["payload_files"],
        "bytes": manifest["payload_bytes"], "storage_methods": methods,
        "content_root_sha256": content_root,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
