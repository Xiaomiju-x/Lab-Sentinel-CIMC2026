"""Rebuild and verify immutable post-training Forge200 artifact manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from gpu_queue_worker import canonical_bytes, load_queue, manifest_tree, write_json


STATIC_MANIFEST_EXCLUDES = {"artifact_manifest.json", "transfer_manifest.json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_manifest(path: Path) -> dict[str, Any]:
    records = []
    for item in sorted(
        child
        for child in path.rglob("*")
        if child.is_file()
        and child.name not in STATIC_MANIFEST_EXCLUDES
        and not child.name.startswith("worker_attempt_")
    ):
        records.append(
            {
                "path": item.relative_to(path).as_posix(),
                "bytes": item.stat().st_size,
                "sha256": sha256_file(item),
            }
        )
    return {
        "schema": "cimc.forge200.artifact-manifest.v1",
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def verify_records(root: Path, manifest: dict[str, Any], label: str, errors: list[str]) -> None:
    records = manifest.get("records", [])
    expected_root = hashlib.sha256(canonical_bytes(records)).hexdigest()
    if manifest.get("content_root_sha256") != expected_root:
        errors.append(f"{label}:content_root")
    for record in records:
        path = root / record["path"]
        if not path.is_file():
            errors.append(f"{label}:missing:{record['path']}")
        elif path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            errors.append(f"{label}:hash:{record['path']}")


def finalize(root: Path, artifact_root: Path, shard: str, rebuild: bool) -> dict[str, Any]:
    _, jobs = load_queue(root, shard)
    admitted = [job["candidate_id"] for job in jobs if job.get("admission_state") == "ADMITTED"]
    suffix = shard.lower()
    state_path = artifact_root / f"worker_{suffix}.state.json"
    result_path = artifact_root / f"worker_{suffix}.result.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    epoch_values: list[int] = []
    onnx_checked = golden_checked = icmf_checked = model_cards = 0

    if sorted(state.get("jobs", {})) != sorted(admitted):
        errors.append("worker_state:candidate_set")
    if any(state.get("jobs", {}).get(candidate_id, {}).get("status") != "COMPLETE" for candidate_id in admitted):
        errors.append("worker_state:not_complete")
    if result.get("status") != "COMPLETE" or result.get("completed") != len(admitted) or result.get("failed") != 0:
        errors.append("worker_result")

    for candidate_id in admitted:
        candidate_root = artifact_root / candidate_id
        receipt_path = candidate_root / "promotion_receipt.json"
        evaluation_path = candidate_root / "eval_grouped.json"
        source_path = candidate_root / "source_manifest.json"
        if not receipt_path.is_file() or not evaluation_path.is_file() or not source_path.is_file():
            errors.append(f"{candidate_id}:required_evidence")
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        source = json.loads(source_path.read_text(encoding="utf-8"))
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("authority") != 0
            or receipt.get("status") != "HOST_GPU_TRAINED_BOARD_PENDING"
            or receipt.get("board_accepted") is not False
            or receipt.get("countable_model") is not False
            or receipt.get("three_seed_count") != 3
        ):
            errors.append(f"{candidate_id}:receipt_gate")
        if source.get("authority") != 0 or source.get("cross_split_group_overlap") != 0:
            errors.append(f"{candidate_id}:source_gate")
        seed_reports = evaluation.get("seed_reports", [])
        if len(seed_reports) != 3:
            errors.append(f"{candidate_id}:three_seed_evaluation")
        epoch_values.extend(int(item["epochs"]) for item in seed_reports if "epochs" in item)

        onnx_path = candidate_root / "fp32.onnx"
        golden_path = candidate_root / "golden_vectors.npz"
        package = receipt.get("package", {})
        package_path = candidate_root / package.get("path", "")
        model_card = candidate_root / "model_card.md"
        if not onnx_path.is_file() or sha256_file(onnx_path) != receipt.get("onnx_sha256"):
            errors.append(f"{candidate_id}:onnx_hash")
        else:
            onnx.checker.check_model(str(onnx_path))
            onnx_checked += 1
        if not golden_path.is_file() or sha256_file(golden_path) != receipt.get("golden_sha256"):
            errors.append(f"{candidate_id}:golden_hash")
        else:
            with np.load(golden_path, allow_pickle=False) as golden:
                if not golden.files:
                    errors.append(f"{candidate_id}:golden_empty")
            golden_checked += 1
        if (
            not package_path.is_file()
            or sha256_file(package_path) != package.get("sha256")
            or package_path.stat().st_size != package.get("bytes")
            or package_path.read_bytes()[:4] != b"ICMF"
            or package_path.stat().st_size < 256
            or sha256_file_bytes(package_path.read_bytes()[256:]) != package.get("payload_sha256")
        ):
            errors.append(f"{candidate_id}:icmf_package")
        else:
            icmf_checked += 1
        if not model_card.is_file():
            errors.append(f"{candidate_id}:model_card")
        else:
            model_cards += 1

        if rebuild:
            write_json(candidate_root / "artifact_manifest.json", artifact_manifest(candidate_root))
            write_json(candidate_root / "transfer_manifest.json", manifest_tree(candidate_root))
        verify_records(
            candidate_root,
            json.loads((candidate_root / "artifact_manifest.json").read_text(encoding="utf-8")),
            f"{candidate_id}:artifact_manifest",
            errors,
        )
        transfer = json.loads((candidate_root / "transfer_manifest.json").read_text(encoding="utf-8"))
        if transfer != manifest_tree(candidate_root):
            errors.append(f"{candidate_id}:transfer_manifest")
        verify_records(candidate_root, transfer, f"{candidate_id}:transfer_manifest", errors)

    root_transfer_path = artifact_root / f"transfer_{suffix}.json"
    if rebuild:
        write_json(root_transfer_path, manifest_tree(artifact_root))
    root_transfer = json.loads(root_transfer_path.read_text(encoding="utf-8"))
    if root_transfer != manifest_tree(artifact_root):
        errors.append("root_transfer_manifest")
    verify_records(artifact_root, root_transfer, "root_transfer_manifest", errors)
    return {
        "schema": "cimc.forge200.post-gpu-artifact-validation.v1",
        "status": "PASS" if not errors else "FAIL",
        "shard": shard,
        "admitted": len(admitted),
        "completed": sum(state.get("jobs", {}).get(item, {}).get("status") == "COMPLETE" for item in admitted),
        "failed": sum(state.get("jobs", {}).get(item, {}).get("status") == "FAIL_CLOSED" for item in admitted),
        "authority_nonzero": 0,
        "onnx_checked": onnx_checked,
        "golden_checked": golden_checked,
        "icmf_checked": icmf_checked,
        "model_cards": model_cards,
        "epoch_values": sorted(set(epoch_values)),
        "artifact_bytes": sum(path.stat().st_size for path in artifact_root.rglob("*") if path.is_file()),
        "content_root_sha256": root_transfer.get("content_root_sha256"),
        "errors": errors,
    }


def sha256_file_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", choices=["GPU_A", "GPU_B"], required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cloud5090"))
    parser.add_argument("--receipt-out", type=Path)
    parser.add_argument("--rebuild-manifests", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    receipt = finalize(root, artifact_root, args.shard, args.rebuild_manifests)
    receipt_out = args.receipt_out or artifact_root.parent / f"post_gpu_validation_{args.shard.lower()}.json"
    write_json(receipt_out.resolve(), receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
