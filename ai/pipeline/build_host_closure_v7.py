#!/usr/bin/env python3
"""Build the final 170-asset host closure with explicit exact/SIM separation."""

from __future__ import annotations

import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_host_closure_v4 import canonical, sha, validate_package, write


EXACT_STATUS = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING"
SIM_STATUSES = {
    "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY",
    "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING",
}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    abi = json.loads((root / "contracts" / "model_package_abi.v1.json").read_text(encoding="utf-8"))
    engines = {item["engine_id"] for item in abi["engines"]}
    choices: dict[tuple[str, str], list[tuple[Path, dict]]] = collections.defaultdict(list)
    rejected: dict[str, set[str]] = collections.defaultdict(set)
    for path in root.glob("artifacts/**/promotion_receipt.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidate_id = receipt.get("candidate_id")
        status = receipt.get("status", "")
        if not candidate_id:
            continue
        exact_ok = status == EXACT_STATUS and receipt.get("host_contract_pass") is not False
        legacy_sim_ok = status == "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" and receipt.get("host_contract_pass") is not False
        extension_ok = status == "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING" and receipt.get("host_extension_pass") is True and receipt.get("host_contract_pass") is False
        if exact_ok or legacy_sim_ok or extension_ok:
            choices[(candidate_id, status)].append((path, receipt))
        elif "REJECTED" in status:
            rejected[candidate_id].add(status)

    exact, extensions = [], []
    for (candidate_id, status), items in sorted(choices.items()):
        items.sort(key=lambda item: ("local4050" not in str(item[0]).lower(), -item[0].stat().st_mtime, str(item[0])))
        receipt_path, receipt = items[0]
        if receipt.get("authority") != 0 or receipt.get("board_accepted") is not False or receipt.get("countable_model") is not False:
            raise RuntimeError(f"AUTHORITY_BOARD_GATE:{candidate_id}")
        is_exact = status == EXACT_STATUS
        if not is_exact and status not in SIM_STATUSES:
            raise RuntimeError(f"UNKNOWN_SIM_STATUS:{candidate_id}:{status}")
        if status == "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING" and not str(receipt.get("original_task_contract_status", "")).startswith("UNCHANGED_FAIL_CLOSED"):
            raise RuntimeError(f"SIM_EXTENSION_BOUNDARY_GATE:{candidate_id}")
        directory = receipt_path.parent
        package_path = directory / receipt["package"]["path"]
        package = validate_package(package_path, receipt, engines)
        golden = directory / "golden_vectors.npz"
        output_schema = directory / "output_schema.json"
        model_card = directory / "model_card.md"
        if not golden.is_file() or sha(golden) != (receipt.get("golden_sha256") or package["golden_sha256_header"]):
            raise RuntimeError(f"GOLDEN_GATE:{candidate_id}")
        if package["golden_sha256_header"] != sha(golden):
            raise RuntimeError(f"HEADER_GOLDEN_GATE:{candidate_id}")
        selection = None
        frozen_test = None
        selection_kind = None
        if status == "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING":
            pretest = directory / "baseline_selection_frozen_before_test.json"
            validation_record = directory / "validation_selection_record.v1.json"
            frozen_test_path = directory / "frozen_test_evaluation.v1.json"
            if pretest.is_file():
                selection, selection_kind = pretest, "PRE_TEST_FREEZE"
            elif validation_record.is_file():
                selection, selection_kind = validation_record, "VALIDATION_ONLY_SELECTION_RECORDED_AFTER_TRAINER"
            else:
                raise RuntimeError(f"SELECTION_EVIDENCE_GATE:{candidate_id}")
            if not frozen_test_path.is_file():
                raise RuntimeError(f"FROZEN_TEST_GATE:{candidate_id}")
            frozen_test = frozen_test_path
        record = {
            "candidate_id": candidate_id,
            "category": candidate_id.split("-")[1],
            "status": status,
            "truth_class": receipt.get("truth_class"),
            "public_claim_scope": receipt.get("public_claim_scope"),
            "original_task_contract_status": receipt.get("original_task_contract_status"),
            "promotion_receipt": str(receipt_path.relative_to(root)).replace("\\", "/"),
            "promotion_receipt_sha256": sha(receipt_path),
            "package": {**package, "path": str(package_path.relative_to(root)).replace("\\", "/")},
            "golden": {"path": str(golden.relative_to(root)).replace("\\", "/"), "sha256": sha(golden)},
            "output_schema": {"path": str(output_schema.relative_to(root)).replace("\\", "/"), "sha256": sha(output_schema)} if output_schema.is_file() else None,
            "model_card": {"path": str(model_card.relative_to(root)).replace("\\", "/"), "sha256": sha(model_card)} if model_card.is_file() else None,
            "selection_evidence": {"path": str(selection.relative_to(root)).replace("\\", "/"), "sha256": sha(selection), "kind": selection_kind} if selection else None,
            "frozen_test": {"path": str(frozen_test.relative_to(root)).replace("\\", "/"), "sha256": sha(frozen_test)} if frozen_test else None,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        }
        (exact if is_exact else extensions).append(record)

    exact_ids = {record["candidate_id"] for record in exact}
    extensions = [record for record in extensions if record["candidate_id"] not in exact_ids]
    all_records = exact + extensions
    ids = [record["candidate_id"] for record in all_records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("DUPLICATE_CANDIDATE_GATE")
    category_counts = collections.Counter(record["category"] for record in all_records)
    target = {"P": 112, "G": 30, "S": 28}
    target_gap = {key: max(value - category_counts.get(key, 0), 0) for key, value in target.items()}
    package_hashes = [record["package"]["sha256"] for record in all_records]
    payload_hashes = [record["package"]["payload_sha256"] for record in all_records]
    integrity = {
        "package_hashes": len(package_hashes),
        "unique_package_hashes": len(set(package_hashes)),
        "package_collisions": len(package_hashes) - len(set(package_hashes)),
        "payload_hashes": len(payload_hashes),
        "unique_payload_hashes": len(set(payload_hashes)),
        "payload_collisions": len(payload_hashes) - len(set(payload_hashes)),
    }
    if any(integrity[key] for key in ("package_collisions", "payload_collisions")):
        raise RuntimeError(f"WEIGHT_COLLISION_GATE:{integrity}")
    full_target_met = len(all_records) == 170 and target_gap == {"P": 0, "G": 0, "S": 0}
    result = {
        "schema": "cimc.forge200.host-closure.v7",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_FULL_170_TARGET_MET_IF_UNIFIED_BOARD_PASSES_EXACT_AND_SIM_BOUNDARIES_SEPARATED" if full_target_met else "HOST_TARGET_OPEN_BOARD_PENDING",
        "exact_contract": {"unique_candidates": len(exact), "by_category": dict(collections.Counter(record["category"] for record in exact)), "records": exact},
        "sim_only_extensions": {"unique_candidates": len(extensions), "by_category": dict(collections.Counter(record["category"] for record in extensions)), "records": extensions, "not_substitutes_for_frozen_source_gates": True},
        "host_qualified_total_including_extensions": len(all_records),
        "host_by_category": dict(sorted(category_counts.items())),
        "initial_board_baseline": {"assets": 30, "logical_models": 28, "logical_generative_models": 8},
        "combined_assets_if_all_host_assets_later_board_pass": 30 + len(all_records),
        "combined_logical_generative_if_all_host_assets_later_board_pass": 8 + category_counts.get("G", 0),
        "release_floor": {"total_assets": 150, "new_assets_required": 120, "exact_new_shortfall": max(120 - len(exact), 0), "including_sim_extension_shortfall": max(120 - len(all_records), 0), "asset_floor_met_if_all_host_assets_board_pass": 30 + len(all_records) >= 150, "exact_source_bound_floor_met": len(exact) >= 120},
        "full_target": {"new_assets": 170, "by_category": target, "gap_by_category": target_gap, "total_gap": sum(target_gap.values()), "met": full_target_met},
        "integrity": integrity,
        "rejected_candidates": sorted({candidate_id for candidate_id in rejected if candidate_id not in exact_ids}),
        "new_models_board_accepted": 0,
        "new_models_countable_publicly": 0,
        "authority_nonzero": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(canonical({"exact": exact, "extensions": extensions, "integrity": integrity})).hexdigest()
    output = root / "evidence" / "host_closure.v7.json"
    write(output, result)
    print(json.dumps({"status": result["status"], "exact": len(exact), "extensions": len(extensions), "host_total": len(all_records), "by_category": result["host_by_category"], "combined_assets": result["combined_assets_if_all_host_assets_later_board_pass"], "combined_logical_generative": result["combined_logical_generative_if_all_host_assets_later_board_pass"], "integrity": integrity, "content_root_sha256": result["content_root_sha256"]}, sort_keys=True))
    return 0 if full_target_met else 2


if __name__ == "__main__":
    raise SystemExit(main())
