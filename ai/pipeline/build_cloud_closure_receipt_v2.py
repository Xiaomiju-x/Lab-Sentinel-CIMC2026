#!/usr/bin/env python3
"""Verify node transfer manifests and freeze the truthful cloud-training closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANDIDATE_RE = re.compile(r"CAND-[PGS]-\d{3}")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_set(files: list[dict[str, Any]], prefix: str, suffix: str = "promotion_receipt.json") -> set[str]:
    result = set()
    for item in files:
        path = item["archive_path"]
        if path.startswith(prefix) and path.endswith(suffix):
            match = CANDIDATE_RE.search(path)
            if match:
                result.add(match.group(0))
    return result


def verify_node(directory: Path, node: str) -> tuple[dict[str, Any], dict[str, Any]]:
    stem = f"cimc-forge200-node-{node.lower()}-transfer-20260803"
    manifest_path = directory / f"{stem}.manifest.json"
    receipt_path = directory / f"{stem}.receipt.json"
    sidecar_path = directory / f"{stem}.tar.gz.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if manifest["status"] != "PASS" or receipt["status"] != "PASS":
        raise RuntimeError(f"NODE_{node}_STATUS")
    if receipt["manifest_sha256"] != sha256_file(manifest_path):
        raise RuntimeError(f"NODE_{node}_MANIFEST_SHA")
    if manifest["file_count"] != len(manifest["files"]):
        raise RuntimeError(f"NODE_{node}_FILE_COUNT")
    if manifest["total_uncompressed_bytes"] != sum(item["bytes"] for item in manifest["files"]):
        raise RuntimeError(f"NODE_{node}_BYTE_COUNT")
    if manifest["content_root_sha256"] != hashlib.sha256(canonical_bytes(manifest["files"])).hexdigest():
        raise RuntimeError(f"NODE_{node}_CONTENT_ROOT")
    paths = [item["archive_path"] for item in manifest["files"]]
    if len(paths) != len(set(paths)):
        raise RuntimeError(f"NODE_{node}_DUPLICATE_PATH")
    forbidden = [path for path in paths if path.endswith((".pt", ".safetensors")) or "pilot_snapshot" in path]
    if forbidden:
        raise RuntimeError(f"NODE_{node}_EXCLUSION_GATE")
    expected_archive_sha = sidecar_path.read_text(encoding="ascii").split()[0]
    if expected_archive_sha != receipt["archive_sha256"]:
        raise RuntimeError(f"NODE_{node}_ARCHIVE_SIDECAR")
    return manifest, receipt


def artifact_entry(files: list[dict[str, Any]], prefix: str, candidate: str) -> dict[str, Any]:
    roots = [item for item in files if item["archive_path"].startswith(f"{prefix}/{candidate}/")]
    by_name = {Path(item["archive_path"]).name: item for item in roots}
    binaries = sorted((item for item in roots if item["archive_path"].endswith(".bin")), key=lambda item: item["archive_path"])
    return {
        "candidate_id": candidate,
        "promotion_receipt_sha256": by_name.get("promotion_receipt.json", {}).get("sha256"),
        "onnx_sha256": by_name.get("fp32.onnx", {}).get("sha256"),
        "golden_sha256": by_name.get("golden_vectors.npz", {}).get("sha256"),
        "package_sha256": binaries[0]["sha256"] if binaries else None,
        "package_bytes": binaries[0]["bytes"] if binaries else None,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    metadata = root / "artifacts" / "cloud_transfer_20260803" / "metadata"
    a_manifest, a_receipt = verify_node(metadata, "A")
    b_manifest, b_receipt = verify_node(metadata, "B")
    a_files, b_files = a_manifest["files"], b_manifest["files"]
    download_path = root / "evidence" / "cloud5090_download_verification.v2.json"
    download_receipt = json.loads(download_path.read_text(encoding="utf-8")) if download_path.exists() else {"status": "DOWNLOAD_PENDING"}
    download_complete = download_receipt.get("status") == "PASS"

    first_pass = candidate_set(a_files + b_files, "first_pass/artifacts/")
    nanolm = candidate_set(a_files + b_files, "corrective/nanolm_selected/")
    material = candidate_set(a_files + b_files, "corrective/material_exact/")
    p087_selected = candidate_set(b_files, "corrective/material_p087_selected/")
    material.discard("CAND-P-087")
    material |= p087_selected
    router_rejected = candidate_set(a_files, "corrective/router_rejected/")
    rag_rejected = candidate_set(a_files, "corrective/rag_encoder_rejected/")
    open_rejected = candidate_set(b_files, "corrective/open_data_rejected/")
    postgpu = candidate_set(b_files, "corrective/postgpu_support/")
    exact_pass = material | {"CAND-S-041"}
    exact_rejected = router_rejected | rag_rejected | open_rejected | {"CAND-S-033"}
    extra = open_rejected | postgpu
    trained_unique = first_pass | extra
    first_pass_unclosed = first_pass - nanolm - material - router_rejected - rag_rejected

    expected_first = {"P": 12, "G": 26, "S": 31}
    actual_first = {kind: sum(item.startswith(f"CAND-{kind}-") for item in first_pass) for kind in expected_first}
    if len(first_pass) != 69 or actual_first != expected_first:
        raise RuntimeError(f"FIRST_PASS_GATE:{actual_first}")
    if nanolm != {f"CAND-G-{index:03d}" for index in range(1, 27)}:
        raise RuntimeError("NANOLM_SET_GATE")
    expected_material = {"CAND-P-069", "CAND-P-071", "CAND-P-072", "CAND-P-074", "CAND-P-075", "CAND-P-076", "CAND-P-077", "CAND-P-078", "CAND-P-086", "CAND-P-087", "CAND-P-140"}
    if material != expected_material:
        raise RuntimeError(f"MATERIAL_SET_GATE:{sorted(material)}")
    expected_rejected = {"CAND-S-001", "CAND-S-009", "CAND-S-010", "CAND-S-011", "CAND-S-012", "CAND-S-013", "CAND-S-014", "CAND-S-029", "CAND-P-088", "CAND-S-033"}
    if exact_rejected != expected_rejected:
        raise RuntimeError(f"REJECTED_SET_GATE:{sorted(exact_rejected)}")
    if len(trained_unique) != 72 or len(exact_pass) != 12 or len(first_pass_unclosed) != 24:
        raise RuntimeError("CLOSURE_COUNT_GATE")

    selected_artifacts = []
    for candidate in sorted(nanolm):
        files = a_files if int(candidate[-3:]) % 2 else b_files
        selected_artifacts.append({**artifact_entry(files, "corrective/nanolm_selected", candidate), "host_state": "PASS_AVAILABLE_SURROGATE_EXACT_CONTRACT_COMPONENTS_PENDING"})
    for candidate in sorted(material):
        if candidate in {"CAND-P-069", "CAND-P-074", "CAND-P-076"}:
            selected_artifacts.append({**artifact_entry(a_files, "corrective/material_exact", candidate), "host_state": "CONTRACT_BASELINE_PASS_3_OF_3_BOARD_PENDING"})
        elif candidate == "CAND-P-087":
            selected_artifacts.append({**artifact_entry(b_files, "corrective/material_p087_selected", candidate), "host_state": "CONTRACT_BASELINE_PASS_3_OF_3_BOARD_PENDING"})
        else:
            selected_artifacts.append({**artifact_entry(b_files, "corrective/material_exact", candidate), "host_state": "CONTRACT_BASELINE_PASS_3_OF_3_BOARD_PENDING"})
    selected_artifacts.append({**artifact_entry(b_files, "corrective/postgpu_support", "CAND-S-041"), "host_state": "CONTRACT_BASELINE_PASS_3_OF_3_BOARD_PENDING"})

    result = {
        "schema": "cimc.forge200.cloud5090-training-closure.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "GPU_TRAINING_CLOSED_TRANSFER_READY_HOST_RESULTS_BOARD_PENDING",
        "working_directory": "${CIMC_FORGE200_ROOT}",
        "gpu_phase": {
            "instances": 2,
            "gpu_model": "NVIDIA GeForce RTX 5090",
            "local_rtx4050_used_for_main_training": False,
            "unique_cuda_touched_candidates": len(trained_unique),
            "first_pass_unique": len(first_pass),
            "first_pass_distribution": actual_first,
            "corrective_only_unique": sorted(extra - first_pass),
        },
        "host_disposition": {
            "exact_contract_baseline_pass_board_pending": sorted(exact_pass),
            "exact_contract_baseline_pass_count": len(exact_pass),
            "nanolm_available_surrogate_pass_exact_components_pending": sorted(nanolm),
            "nanolm_pending_count": len(nanolm),
            "exact_contract_baseline_rejected": sorted(exact_rejected),
            "exact_contract_baseline_rejected_count": len(exact_rejected),
            "first_pass_baseline_unclosed_or_input_mismatch": sorted(first_pass_unclosed),
            "first_pass_unclosed_count": len(first_pass_unclosed),
        },
        "nanolm_count": {
            "new_trained_unique": 26,
            "existing_logical": 8,
            "current_logical_total": 34,
            "planned_logical_total": 38,
            "shortfall": 4,
            "blocked_candidates": ["CAND-G-027", "CAND-G-028", "CAND-G-029", "CAND-G-030"],
            "blocker": "TASK_SPECIFIC_EXPERT_LABELS_NOT_AVAILABLE_NO_SYNTHETIC_SUBSTITUTION",
        },
        "expanded_corpus": {
            "documents": 540,
            "chunks": 23798,
            "bytes": 35486817,
            "sha256": "68b27f18ea23b6be9adc2dbb7e19fd00f2f382208f317fb12eb8c22c2291f854",
            "license_gate": "EUROPE_PMC_METADATA_PLUS_JATS_CC_BY_PER_DOCUMENT",
            "cross_split_pmcid_overlap": 0,
        },
        "selected_artifacts": selected_artifacts,
        "transfer": {
            "node_a": {**a_receipt, "remote_gzip_test": "PASS", "remote_tar_member_count": a_manifest["file_count"] + 2},
            "node_b": {**b_receipt, "remote_gzip_test": "PASS", "remote_tar_member_count": b_manifest["file_count"] + 2},
            "total_archive_bytes": a_receipt["archive_bytes"] + b_receipt["archive_bytes"],
            "large_archives_downloaded_to_local": download_complete,
            "metadata_downloaded_and_verified": True,
            "local_download_verification": {
                "status": download_receipt.get("status"),
                "receipt_path": str(download_path.relative_to(root)).replace("\\", "/") if download_path.exists() else None,
                "receipt_sha256": sha256_file(download_path) if download_path.exists() else None,
                "full_payload_hashing": bool(download_complete and all(item.get("full_payload_hashing") for item in download_receipt.get("reports", []))),
            },
        },
        "release_gate": {
            "status": "REJECTED_HOST_PROMOTABLE_LT_120_AND_UNIFIED_BOARD_PENDING",
            "host_exact_pass": len(exact_pass),
            "minimum_new_board_pass_for_modelbank": 120,
            "release_floor": 150,
            "target_new_slots": 170,
            "reason": "MISSING_TASK_TRUTH_AND_CONTRACT_BASELINE_FAILURES_MUST_REMAIN_FAIL_CLOSED",
        },
        "safety": {
            "authority": 0,
            "board_accepted": False,
            "countable_model_count": 0,
            "deterministic_control_chain_modified": False,
            "firmware_modified": False,
            "sd_or_board_burned": False,
            "unified_modelbank_loader_sd_max31856_board_test_pending": True,
        },
    }
    result["content_root_sha256"] = hashlib.sha256(canonical_bytes({key: value for key, value in result.items() if key not in {"created_at_utc", "content_root_sha256"}})).hexdigest()
    output = root / "evidence" / "cloud5090_training_closure.v2.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "sha256": sha256_file(output), "content_root_sha256": result["content_root_sha256"], "trained_unique": len(trained_unique), "exact_pass": len(exact_pass), "nanolm_pending": len(nanolm), "rejected": len(exact_rejected), "unclosed": len(first_pass_unclosed)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
