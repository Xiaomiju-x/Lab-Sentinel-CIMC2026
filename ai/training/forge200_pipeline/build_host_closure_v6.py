#!/usr/bin/env python3
"""Build v6 closure with exact, legacy SIM_ONLY, and frozen v1 SIM extensions."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from build_host_closure_v4 import canonical, sha, validate_package, write


EXACT_STATUS = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING"
LEGACY_SIM_STATUS = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY"
SIM_V1_STATUS = "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or (root / "evidence" / "host_closure.v6.json")
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
        admitted = False
        if status == EXACT_STATUS:
            admitted = receipt.get("host_contract_pass") is not False
        elif status == LEGACY_SIM_STATUS:
            admitted = receipt.get("host_contract_pass") is not False
        elif status == SIM_V1_STATUS:
            admitted = receipt.get("host_extension_pass") is True and receipt.get("host_contract_pass") is False
        if admitted:
            choices[(candidate_id, status)].append((path, receipt))
        elif "REJECTED" in status:
            rejected[candidate_id].add(status)

    exact: list[dict] = []
    extensions: list[dict] = []
    for (candidate_id, status), items in sorted(choices.items()):
        items.sort(key=lambda item: ("local4050" not in str(item[0]).lower(), -item[0].stat().st_mtime, str(item[0])))
        path, receipt = items[0]
        if receipt.get("authority") != 0 or receipt.get("board_accepted") is not False or receipt.get("countable_model") is not False:
            raise RuntimeError(f"AUTHORITY_BOARD_GATE:{candidate_id}")
        if status == SIM_V1_STATUS and receipt.get("original_task_contract_status") != "UNCHANGED_FAIL_CLOSED":
            raise RuntimeError(f"SIM_EXTENSION_BOUNDARY_GATE:{candidate_id}")
        package_path = path.parent / receipt["package"]["path"]
        package = validate_package(package_path, receipt, engines)
        golden = path.parent / "golden_vectors.npz"
        output_schema = path.parent / "output_schema.json"
        model_card = path.parent / "model_card.md"
        frozen_selection = path.parent / "baseline_selection_frozen_before_test.json"
        frozen_test = path.parent / "frozen_test_evaluation.v1.json"
        if not golden.is_file() or sha(golden) != (receipt.get("golden_sha256") or package["golden_sha256_header"]):
            raise RuntimeError(f"GOLDEN_GATE:{candidate_id}")
        if package["golden_sha256_header"] != sha(golden):
            raise RuntimeError(f"HEADER_GOLDEN_GATE:{candidate_id}")
        if status == SIM_V1_STATUS and (not frozen_selection.is_file() or not frozen_test.is_file()):
            raise RuntimeError(f"FROZEN_TEST_GATE:{candidate_id}")
        record = {
            "candidate_id": candidate_id,
            "category": candidate_id.split("-")[1],
            "status": status,
            "truth_class": receipt.get("truth_class"),
            "public_claim_scope": receipt.get("public_claim_scope"),
            "original_task_contract_status": receipt.get("original_task_contract_status"),
            "promotion_receipt": str(path.relative_to(root)).replace("\\", "/"),
            "promotion_receipt_sha256": sha(path),
            "package": {**package, "path": str(package_path.relative_to(root)).replace("\\", "/")},
            "golden": {"path": str(golden.relative_to(root)).replace("\\", "/"), "sha256": sha(golden)},
            "output_schema": {"path": str(output_schema.relative_to(root)).replace("\\", "/"), "sha256": sha(output_schema)} if output_schema.is_file() else None,
            "model_card": {"path": str(model_card.relative_to(root)).replace("\\", "/"), "sha256": sha(model_card)} if model_card.is_file() else None,
            "frozen_selection": {"path": str(frozen_selection.relative_to(root)).replace("\\", "/"), "sha256": sha(frozen_selection)} if frozen_selection.is_file() else None,
            "frozen_test": {"path": str(frozen_test.relative_to(root)).replace("\\", "/"), "sha256": sha(frozen_test)} if frozen_test.is_file() else None,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        }
        (exact if status == EXACT_STATUS else extensions).append(record)

    exact_ids = {record["candidate_id"] for record in exact}
    extensions = [record for record in extensions if record["candidate_id"] not in exact_ids]
    all_records = exact + extensions
    if len({record["candidate_id"] for record in all_records}) != len(all_records):
        raise RuntimeError("DUPLICATE_CANDIDATE_GATE")
    package_hashes = [record["package"]["sha256"] for record in all_records]
    payload_hashes = [record["package"]["payload_sha256"] for record in all_records]
    counts = lambda rows: dict(collections.Counter(record["category"] for record in rows))
    category_counts = collections.Counter(record["category"] for record in all_records)
    target = {"P": 112, "G": 30, "S": 28}
    target_gap = {key: max(value - category_counts.get(key, 0), 0) for key, value in target.items()}
    total_asset_floor_met_if_board_passes = 30 + len(all_records) >= 150
    result = {
        "schema": "cimc.forge200.host-closure.v6",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_ASSET_FLOOR_MET_IF_UNIFIED_BOARD_PASSES_FULL_TARGET_OPEN_BOARD_PENDING" if total_asset_floor_met_if_board_passes else "HOST_ASSET_FLOOR_OPEN_BOARD_PENDING",
        "exact_contract": {"unique_candidates": len(exact), "by_category": counts(exact), "records": exact},
        "sim_only_extensions": {
            "unique_candidates": len(extensions),
            "by_category": counts(extensions),
            "records": extensions,
            "not_substitutes_for_frozen_source_gates": True,
        },
        "host_qualified_total_including_extensions": len(all_records),
        "host_by_category": dict(sorted(category_counts.items())),
        "initial_board_baseline": {"assets": 30, "logical_models": 28},
        "combined_assets_if_all_host_assets_later_board_pass": 30 + len(all_records),
        "release_floor": {
            "total_assets": 150,
            "new_assets_required": 120,
            "exact_new_shortfall": max(120 - len(exact), 0),
            "including_sim_extension_shortfall": max(120 - len(all_records), 0),
            "asset_floor_met_if_all_host_assets_board_pass": total_asset_floor_met_if_board_passes,
            "exact_source_bound_floor_met": len(exact) >= 120,
        },
        "full_target": {"new_assets": 170, "by_category": target, "gap_by_category": target_gap, "total_gap": sum(target_gap.values())},
        "integrity": {
            "package_hashes": len(package_hashes),
            "unique_package_hashes": len(set(package_hashes)),
            "package_collisions": len(package_hashes) - len(set(package_hashes)),
            "payload_hashes": len(payload_hashes),
            "unique_payload_hashes": len(set(payload_hashes)),
            "payload_collisions": len(payload_hashes) - len(set(payload_hashes)),
        },
        "rejected_candidates": sorted({candidate_id for candidate_id in rejected if candidate_id not in exact_ids}),
        "new_models_board_accepted": 0,
        "new_models_countable_publicly": 0,
        "authority_nonzero": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(
        canonical({"exact": exact, "extensions": extensions, "integrity": result["integrity"]})
    ).hexdigest()
    write(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "exact": len(exact),
                "extensions": len(extensions),
                "host_total": len(all_records),
                "combined_assets": result["combined_assets_if_all_host_assets_later_board_pass"],
                "target_gap": target_gap,
                "integrity": result["integrity"],
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
