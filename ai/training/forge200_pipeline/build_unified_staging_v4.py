#!/usr/bin/env python3
"""Build a hash-only reference joining the frozen 30-asset baseline and ModelBank v4."""

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
    return {
        "path": path.relative_to(base).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


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

    closure_path = root / "evidence/host_closure.v4.json"
    bank_path = root / "evidence/modelbank_build.v4.json"
    dry_path = root / "evidence/modelbank_host_dry_run.v4.json"
    adapter_path = root / "evidence/firmware_adapter_host_compile.v4.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    dry = json.loads(dry_path.read_text(encoding="utf-8"))
    adapter = json.loads(adapter_path.read_text(encoding="utf-8"))

    if closure["host_qualified_total_including_extensions"] != 85:
        raise SystemExit("host closure is not the frozen 85-model v4 closure")
    if bank["model_count"] != 85 or dry["successful_swaps"] != 1000:
        raise SystemExit("ModelBank v4 build/dry-run is incomplete")
    if adapter["production_files_modified"] != 0:
        raise SystemExit("firmware adapter reports a production modification")

    with (root / "contracts/model_roster_200.v1.tsv").open("r", encoding="utf-8", newline="") as stream:
        roster = list(csv.DictReader(stream, delimiter="\t"))
    initial = roster[:30]
    if len(initial) != 30 or any(row["planned_status"] != "BOARD_ACCEPTED" for row in initial):
        raise SystemExit("frozen 30-asset roster prefix is invalid")
    if any(row["authority"] != "0" or row["countable_now"] != "1" for row in initial):
        raise SystemExit("frozen baseline authority/countability contract changed")

    ai_dir = cimc / "firmware/ai_models_c"
    include_tokens = ("weight", "golden", "image.bin", "vocab", "recipe_presets")
    baseline_files = [
        path for path in ai_dir.iterdir()
        if path.is_file() and any(token in path.name.lower() for token in include_tokens)
    ]
    baseline_files.sort(key=lambda item: item.name.lower())
    if not baseline_files:
        raise SystemExit("no frozen baseline payload files found")

    board_dir = cimc / "evidence/hardware_bringup/finals_hw_full_integration_20260730"
    board_files = sorted(path for path in board_dir.iterdir() if path.is_file())
    boot_log = board_dir / "full_boot_30_models_board.log"
    boot_text = boot_log.read_text(encoding="utf-8", errors="replace")
    required_markers = ("models=30 authority=0", "[TGV] selftest=60007/60007 PASS", "[REG] selftest=15/15 PASS")
    if any(marker not in boot_text for marker in required_markers):
        raise SystemExit("historical 30-model board log markers are incomplete")

    exact = closure["exact_contract"]["unique_candidates"]
    sim = closure["sim_only_extensions"]["unique_candidates"]
    document = {
        "schema": "cimc.forge200.unified-staging.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_UNIFIED_STAGING_PARTIAL_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
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
            "host_qualified": 85,
            "exact_source_bound": exact,
            "sim_only_extensions": sim,
            "board_accepted": 0,
            "countable_publicly": 0,
            "authority_nonzero": 0,
            "modelbank_receipt": reference(bank_path, root),
            "dry_run_receipt": reference(dry_path, root),
            "firmware_adapter_receipt": reference(adapter_path, root),
            "modelbank_content_root_sha256": bank["content_root_sha256"],
        },
        "combined_projection": {
            "assets_if_all_new_host_models_later_pass_board": 115,
            "not_a_board_acceptance_claim": True,
            "new_release_floor": 120,
            "total_release_floor": 150,
            "floor_met": False,
            "exact_source_bound_shortfall": 42,
            "including_sim_only_shortfall": 35,
        },
        "unified_board_sequence_pending": [
            "restore LAB_HARDWARE_BRINGUP=0 and retain Keil target R2.1/DAPLink configuration",
            "link isolated adapter without replacing deterministic authority chain",
            "validate SD/ModelBank/MAX31856 shared SPI CS exclusion and mode restoration",
            "exercise catalog A/B, SHA/generation/golden/commit/fallback and power-loss recovery",
            "publish only individually board-passed assets; failed candidates remain non-countable",
        ],
    }
    document["content_root_sha256"] = content_root(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "baseline_payload_files": len(baseline_files),
        "board_evidence_files": len(board_files),
        "host_models": 85,
        "combined_projection": 115,
        "content_root_sha256": document["content_root_sha256"],
        "status": document["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
