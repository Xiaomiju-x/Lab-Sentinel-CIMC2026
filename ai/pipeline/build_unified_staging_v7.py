#!/usr/bin/env python3
"""Build the final hash-only staging reference for the initial baseline and ModelBank v7."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reference(path: Path, base: Path) -> dict:
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)}


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cimc-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cimc = args.cimc_root.resolve()

    receipt_names = {
        "closure": "host_closure.v7.json",
        "modelbank": "modelbank_build.v7.json",
        "dry_run": "modelbank_host_dry_run.v7.json",
        "artifacts": "host_artifact_verification.v7.json",
        "interfaces": "interface_freeze_verification.v7.json",
        "adapter": "firmware_adapter_host_compile.v7.json",
    }
    paths = {key: root / "evidence" / name for key, name in receipt_names.items()}
    receipts = {key: json.loads(path.read_text(encoding="utf-8")) for key, path in paths.items()}
    closure = receipts["closure"]
    if closure["host_qualified_total_including_extensions"] != 170:
        raise RuntimeError("HOST_170_GATE")
    if closure["host_by_category"] != {"P": 112, "G": 30, "S": 28}:
        raise RuntimeError("CATEGORY_TARGET_GATE")
    if receipts["modelbank"]["model_count"] != 170 or receipts["dry_run"]["successful_swaps"] != 1000:
        raise RuntimeError("MODELBANK_GATE")
    if receipts["artifacts"]["onnx_full_check_pass"] != 170 or receipts["artifacts"]["golden_archives_pass"] != 170:
        raise RuntimeError("ARTIFACT_GATE")
    if receipts["interfaces"]["case_count"] != 23 or receipts["adapter"]["production_files_modified"] != 0:
        raise RuntimeError("INTERFACE_OR_ADAPTER_GATE")

    with (root / "contracts/model_roster_200.v1.tsv").open("r", encoding="utf-8", newline="") as stream:
        initial = list(csv.DictReader(stream, delimiter="\t"))[:30]
    if len(initial) != 30 or any(row["planned_status"] != "BOARD_ACCEPTED" for row in initial):
        raise RuntimeError("INITIAL_BASELINE_GATE")
    if any(row["authority"] != "0" or row["countable_now"] != "1" for row in initial):
        raise RuntimeError("INITIAL_AUTHORITY_GATE")

    ai_dir = cimc / "firmware/ai_models_c"
    include_tokens = ("weight", "golden", "image.bin", "vocab", "recipe_presets")
    baseline_files = sorted(
        (path for path in ai_dir.iterdir() if path.is_file() and any(token in path.name.lower() for token in include_tokens)),
        key=lambda item: item.name.lower(),
    )
    board_dir = cimc / "evidence/hardware_bringup/finals_hw_full_integration_20260730"
    board_files = sorted(path for path in board_dir.iterdir() if path.is_file())
    required_markers = ("models=30 authority=0", "[TGV] selftest=60007/60007 PASS", "[REG] selftest=15/15 PASS")
    boot_text = (board_dir / "full_boot_30_models_board.log").read_text(encoding="utf-8", errors="replace")
    if any(marker not in boot_text for marker in required_markers):
        raise RuntimeError("INITIAL_BOARD_EVIDENCE_GATE")

    document = {
        "schema": "cimc.forge200.unified-staging.v7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_FULL_200_ASSET_STAGING_PASS_UNIFIED_BOARD_PENDING",
        "scope": "HASH_ONLY_REFERENCE_NO_PRODUCTION_COPY_NO_BOARD_ACTION",
        "authority": 0,
        "production_files_modified": 0,
        "sd_or_board_actions": 0,
        "frozen_initial_baseline": {
            "assets": 30,
            "logical_models": 28,
            "board_accepted_historical": True,
            "roster": initial,
            "payload_references": [reference(path, cimc) for path in baseline_files],
            "board_evidence_references": [reference(path, cimc) for path in board_files],
            "board_log_markers": list(required_markers),
            "fallback_preserved": True,
        },
        "new_host_modelbank": {
            "host_qualified": 170,
            "by_category": closure["host_by_category"],
            "exact_source_bound": closure["exact_contract"]["unique_candidates"],
            "sim_only_extensions": closure["sim_only_extensions"]["unique_candidates"],
            "board_accepted": 0,
            "countable_publicly": 0,
            "authority_nonzero": 0,
            "receipts": {key: reference(path, root) for key, path in paths.items()},
            "modelbank_content_root_sha256": receipts["modelbank"]["content_root_sha256"],
        },
        "combined_projection": {
            "assets_if_all_new_host_models_later_pass_board": 200,
            "logical_generative_models_if_all_new_host_models_later_pass_board": 38,
            "not_a_board_acceptance_claim": True,
            "new_asset_target": 170,
            "total_asset_target": 200,
            "host_target_met": True,
            "exact_source_bound_minimum_floor_shortfall": 42,
            "sim_only_extensions_are_not_exact_source_substitutes": True,
        },
        "unified_board_sequence_pending": [
            "restore LAB_HARDWARE_BRINGUP=0 and retain Keil target R2.1/DAPLink configuration",
            "link isolated authority-zero adapter without replacing the deterministic control chain",
            "validate ModelBank loader, microSD/FatFs, and MAX31856 shared-SPI exclusion and mode restoration",
            "exercise catalog A/B, SHA, generation, golden, commit, fallback, and power-loss recovery",
            "publish only individually board-passed assets; failed candidates remain non-countable",
        ],
    }
    document["content_root_sha256"] = content_root(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": document["status"], "initial_assets": 30, "new_host_assets": 170,
        "combined_projection": 200, "content_root_sha256": document["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
