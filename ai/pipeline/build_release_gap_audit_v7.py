#!/usr/bin/env python3
"""Freeze all 244 dispositions after the full host target while preserving exact/SIM boundaries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_root(document: dict) -> str:
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
    closure_path = root / "evidence/host_closure.v7.json"
    disposition_path = root / "evidence/pre_gpu_disposition_244.v1.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    with (root / "contracts/candidate_pool_244.v1.tsv").open("r", encoding="utf-8", newline="") as stream:
        pool = list(csv.DictReader(stream, delimiter="\t"))
    if len(pool) != 244 or disposition["candidate_count"] != 244:
        raise RuntimeError("CANDIDATE_UNIVERSE_GATE")

    exact_ids = {row["candidate_id"] for row in closure["exact_contract"]["records"]}
    sim_ids = {row["candidate_id"] for row in closure["sim_only_extensions"]["records"]}
    if exact_ids & sim_ids or len(exact_ids) != 78 or len(sim_ids) != 92:
        raise RuntimeError("TIER_SEPARATION_GATE")
    pre_gpu = {row["candidate_id"]: row for row in disposition["records"]}
    records = []
    final_counts = Counter()
    reason_counts = Counter()
    for row in pool:
        candidate_id = row["candidate_id"]
        if candidate_id in exact_ids:
            state, reason = "HOST_EXACT_SOURCE_BOUND_PASS_BOARD_PENDING", "EXACT_CONTRACT_G0_G9_HOST_PASS"
        elif candidate_id in sim_ids:
            state, reason = "HOST_SIM_ONLY_EXTENSION_PASS_BOARD_PENDING", "SIMULATION_NOT_EXPERIMENTAL_SOURCE_GATE_SUBSTITUTE"
        elif pre_gpu[candidate_id]["disposition"] == "PRE_GPU_REJECTED_WITH_EVIDENCE":
            state, reason = "PRE_GPU_EXACT_REJECTED_WITH_EVIDENCE", pre_gpu[candidate_id]["reason_code"]
        else:
            state, reason = "EXACT_CONTRACT_NOT_PROMOTED", "NO_EXACT_HOST_PROMOTION_IN_FROZEN_CLOSURE"
        records.append({
            "candidate_id": candidate_id,
            "category": candidate_id.split("-")[1],
            "final_state": state,
            "reason_code": reason,
            "authority": 0,
            "board_accepted": False,
            "countable_new_model": False,
        })
        final_counts[state] += 1
        reason_counts[reason] += 1

    target = {"P": 112, "G": 30, "S": 28}
    exact_by = Counter(value.split("-")[1] for value in exact_ids)
    sim_by = Counter(value.split("-")[1] for value in sim_ids)
    host_by = {key: exact_by[key] + sim_by[key] for key in target}
    if host_by != target:
        raise RuntimeError(f"FULL_TARGET_GATE:{host_by}")
    document = {
        "schema": "cimc.forge200.release-gap-audit.v7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_FULL_170_TARGET_MET_EXACT_SOURCE_FLOOR_STILL_SHORT_42_UNIFIED_BOARD_PENDING",
        "candidate_universe": 244,
        "target_new_assets": {"total": 170, "by_category": target},
        "minimum_new_assets": 120,
        "host_exact_source_bound": {
            "total": 78,
            "by_category": {key: exact_by[key] for key in target},
            "target_shortfall_by_category": {key: target[key] - exact_by[key] for key in target},
            "minimum_release_shortfall": 42,
        },
        "host_sim_only_extensions": {
            "total": 92,
            "by_category": {key: sim_by[key] for key in target},
            "not_source_gate_substitutes": True,
        },
        "host_total_including_sim_only": 170,
        "host_total_by_category": host_by,
        "host_target_gap": 0,
        "initial_board_baseline": {"assets": 30, "logical_models": 28, "logical_generative_models": 8},
        "combined_projection_if_all_new_host_assets_later_board_pass": 200,
        "combined_logical_generative_if_all_new_host_assets_later_board_pass": 38,
        "new_board_accepted": 0,
        "new_countable_publicly": 0,
        "final_state_counts": dict(sorted(final_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
        "claim_boundary": [
            "The 170-host-asset target is met only when 92 explicitly labeled SIM_ONLY extensions are included.",
            "Only 78 assets pass frozen exact source/label/split contracts; the exact minimum floor remains short by 42.",
            "All 170 new assets remain authority=0, board-pending, and publicly non-countable until individual unified-board acceptance.",
        ],
        "additional_cloud_or_local_training_required_for_current_host_target": False,
        "cloud_no_card_download_required_now": False,
        "references": {
            "closure": {"path": "evidence/host_closure.v7.json", "sha256": sha256(closure_path)},
            "pre_gpu_disposition": {"path": "evidence/pre_gpu_disposition_244.v1.json", "sha256": sha256(disposition_path)},
        },
    }
    document["content_root_sha256"] = content_root(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": document["status"], "exact": 78, "sim_only": 92, "host_total": 170,
        "records": len(records), "content_root_sha256": document["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
