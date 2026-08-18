#!/usr/bin/env python3
"""Freeze the honest Forge200 release-count gap after local host closure."""

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
    closure_path = root / "evidence/host_closure.v4.json"
    disposition_path = root / "evidence/pre_gpu_disposition_244.v1.json"
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    disposition = json.loads(disposition_path.read_text(encoding="utf-8"))
    with (root / "contracts/candidate_pool_244.v1.tsv").open("r", encoding="utf-8", newline="") as stream:
        pool = list(csv.DictReader(stream, delimiter="\t"))
    if len(pool) != 244 or disposition["candidate_count"] != 244:
        raise SystemExit("candidate universe is not 244")

    exact_ids = {row["candidate_id"] for row in closure["exact_contract"]["records"]}
    sim_ids = {row["candidate_id"] for row in closure["sim_only_extensions"]["records"]}
    rejected_after_touch = set(closure["rejected_candidates"])
    pre_gpu = {row["candidate_id"]: row for row in disposition["records"]}
    records = []
    reason_counts = Counter()
    final_counts = Counter()
    for row in pool:
        candidate_id = row["candidate_id"]
        pre = pre_gpu[candidate_id]
        if candidate_id in exact_ids:
            state = "HOST_EXACT_SOURCE_BOUND_PASS_BOARD_PENDING"
            reason = "EXACT_CONTRACT_G0_G9_HOST_PASS"
        elif candidate_id in sim_ids:
            state = "HOST_SIM_ONLY_EXTENSION_PASS_BOARD_PENDING"
            reason = "PHYSICS_SIM_NOT_SOURCE_GATE_SUBSTITUTE"
        elif candidate_id in rejected_after_touch:
            state = "HOST_EVALUATED_NOT_PROMOTED"
            reason = "FROZEN_TEST_OR_DATA_GATE_NOT_PASSED"
        elif pre["disposition"] == "PRE_GPU_REJECTED_WITH_EVIDENCE":
            state = "PRE_GPU_REJECTED_WITH_EVIDENCE"
            reason = pre["reason_code"]
        else:
            state = "NO_FROZEN_HOST_PROMOTION"
            reason = "NO_HOST_PROMOTION_RECEIPT_IN_FROZEN_CLOSURE"
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

    exact_by_category = Counter(item.split("-")[1] for item in exact_ids)
    sim_by_category = Counter(item.split("-")[1] for item in sim_ids)
    target = {"P": 112, "G": 30, "S": 28}
    exact_shortfall = {key: target[key] - exact_by_category[key] for key in target}
    document = {
        "schema": "cimc.forge200.release-gap-audit.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RELEASE_FLOOR_BLOCKED_BY_EXACT_DATA_AND_LATER_BOARD_ACCEPTANCE",
        "candidate_universe": 244,
        "target_new_assets": {"total": 170, "by_category": target},
        "minimum_new_assets": 120,
        "host_exact_source_bound": {
            "total": len(exact_ids),
            "by_category": dict(sorted(exact_by_category.items())),
            "target_shortfall_by_category": exact_shortfall,
            "minimum_release_shortfall": 120 - len(exact_ids),
        },
        "host_sim_only_extensions": {
            "total": len(sim_ids),
            "by_category": dict(sorted(sim_by_category.items())),
            "not_source_gate_substitutes": True,
        },
        "host_total_including_sim_only": len(exact_ids | sim_ids),
        "minimum_release_shortfall_including_sim_only": 120 - len(exact_ids | sim_ids),
        "initial_board_baseline": {"assets": 30, "logical_models": 28},
        "combined_projection_if_all_new_host_assets_later_board_pass": 30 + len(exact_ids | sim_ids),
        "new_board_accepted": 0,
        "new_countable_publicly": 0,
        "final_state_counts": dict(sorted(final_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "records": records,
        "blockers": [
            "At least 42 additional exact source/label/split-bound host passes are needed for the 120-new-asset release floor; SIM_ONLY extensions do not close frozen source gates.",
            "All new assets remain board-pending and authority=0; host compilation and simulation do not make them publicly countable.",
            "Additional GPU time alone cannot create missing experimental truth, expert judgment labels, semantic taxonomy mappings, or an explicit reusable source license.",
        ],
        "cloud_no_card_download_required_now": False,
        "additional_gpu_training_productive_without_new_exact_data": False,
        "references": {
            "closure": {"path": "evidence/host_closure.v4.json", "sha256": sha256(closure_path)},
            "pre_gpu_disposition": {"path": "evidence/pre_gpu_disposition_244.v1.json", "sha256": sha256(disposition_path)},
        },
    }
    document["content_root_sha256"] = root_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "exact": len(exact_ids), "sim_only": len(sim_ids), "host_total": len(exact_ids | sim_ids),
        "exact_floor_shortfall": 120 - len(exact_ids),
        "including_sim_shortfall": 120 - len(exact_ids | sim_ids),
        "records": len(records), "content_root_sha256": document["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
