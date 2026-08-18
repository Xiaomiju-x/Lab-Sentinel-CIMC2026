#!/usr/bin/env python3
"""Generate the fail-closed data intake specification for unresolved Forge200 tasks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def root_hash(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def required_proof(source_gate: str, reason: str) -> list[str]:
    proof = [
        "record_id and immutable source content hash",
        "input fields exactly matching input_contract with physical units",
        "target fields exactly matching target_label with provenance",
        "group/family/run identifier fixed before split",
        "train/validation/test assignment with zero group overlap",
    ]
    if "EXPERT" in source_gate or "EXPERT" in reason:
        proof += ["independent expert annotation protocol", "adjudication record and blinded test labels"]
    elif "TEAM" in source_gate or "BOARD" in source_gate or "L0_L1" in source_gate:
        proof += ["team measurement protocol and instrument/run identifier", "raw observation link; metadata-only logs are insufficient"]
    else:
        proof += ["explicit reusable data license", "official source URL/DOI and downloaded-file SHA-256"]
    return proof


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    gap = json.loads((root / "evidence/release_gap_audit.v4.json").read_text(encoding="utf-8"))
    gap_by_id = {row["candidate_id"]: row for row in gap["records"]}
    with (root / "contracts/candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8", newline="") as stream:
        contracts = list(csv.DictReader(stream, delimiter="\t"))

    intake = []
    for contract in contracts:
        gap_row = gap_by_id[contract["candidate_id"]]
        if gap_row["final_state"] == "HOST_EXACT_SOURCE_BOUND_PASS_BOARD_PENDING":
            continue
        intake.append({
            "candidate_id": contract["candidate_id"],
            "objective_id": contract["objective_id"],
            "current_state": gap_row["final_state"],
            "reason_code": gap_row["reason_code"],
            "input_contract": contract["input_contract"],
            "target_label": contract["target_label"],
            "source_gate": contract["source_gate"],
            "frozen_baseline": contract["baseline"],
            "primary_metric": contract["primary_metric"],
            "parameter_cap": contract["parameter_cap"],
            "required_proof": required_proof(contract["source_gate"], gap_row["reason_code"]),
            "prohibited_substitutions": [
                "teacher_or_API_output_as_ground_truth",
                "fixture_or_template_text_as_experimental_truth",
                "physics_simulation_for_an_experimental_source_gate",
                "random_seed_checkpoint_quantization_or_prompt_role_as_a_new_model",
                "reusing_the_same_touched_test_set_for_tuning",
            ],
        })
    if len(intake) != 166:  # 244 minus 78 exact passes; includes seven SIM_ONLY tasks.
        raise SystemExit(f"unexpected intake task count: {len(intake)}")
    counts = Counter(item["current_state"] for item in intake)
    document = {
        "schema": "cimc.forge200.exact-data-intake.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "INTAKE_SPEC_READY_EXACT_DATA_NOT_PRESENT",
        "minimum_additional_exact_host_passes_required": 42,
        "target_additional_exact_host_passes_required": 92,
        "unresolved_or_non_exact_tasks": len(intake),
        "current_state_counts": dict(sorted(counts.items())),
        "selection_rule": "Materialize any contract-compliant tasks, freeze group split before training, then retain every independent candidate that exceeds its preregistered baseline under G0-G9.",
        "records": intake,
        "acceptance_boundary": {
            "data_arrival_alone_is_not_model_acceptance": True,
            "training_and_frozen_test_evaluation_still_required": True,
            "board_acceptance_still_required_after_host_release_floor": True,
            "authority": 0,
        },
    }
    document["content_root_sha256"] = root_hash(document)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "records": len(intake),
        "minimum_additional_exact": 42,
        "status": document["status"],
        "content_root_sha256": document["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
