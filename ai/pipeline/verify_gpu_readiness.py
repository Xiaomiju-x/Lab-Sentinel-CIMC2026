#!/usr/bin/env python3
"""Fail-closed local acceptance for the Forge200 pre-GPU phase."""

from __future__ import annotations

import csv
import hashlib
import json
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def manifest_errors(root: Path, manifest: dict[str, Any]) -> list[str]:
    errors = []
    for record in manifest["records"]:
        path = root / record["path"]
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            errors.append(record["path"])
    if hashlib.sha256(canonical_bytes(manifest["records"])).hexdigest() != manifest["content_root_sha256"]:
        errors.append("CONTENT_ROOT")
    return errors


def verify_fixture_packages(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records, errors = [], []
    fixture_root = root / "artifacts" / "fixture_dry_run"
    receipt = load_json(fixture_root / "dry_run_receipt.v1.json")
    for task in receipt["tasks"]:
        candidate = task["candidate_id"]
        directory = fixture_root / candidate
        package_path = directory / task["package"]["path"]
        raw = package_path.read_bytes()
        magic, schema, header_bytes, engine, opset, authority = struct.unpack_from("<4sHHHHB", raw, 0)
        record = {
            "candidate_id": candidate,
            "magic": magic.decode("ascii", errors="replace"),
            "schema": schema,
            "header_bytes": header_bytes,
            "engine_id": engine,
            "opset": opset,
            "authority": authority,
            "onnx_status": task["onnx"]["status"],
            "package_sha256": sha256_file(package_path),
        }
        records.append(record)
        if (magic, schema, header_bytes, engine, opset, authority) != (b"ICMF", 1, 256, 240, 1, 0):
            errors.append(f"{candidate}: ABI header")
        if task["onnx"]["status"] != "ONNX_CHECKER_PASS":
            errors.append(f"{candidate}: ONNX")
        artifact_manifest = load_json(directory / "artifact_manifest.json")
        errors.extend(f"{candidate}:{item}" for item in manifest_errors(directory, artifact_manifest))
    return records, errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    local_tooling = root / ".tooling" / "python"
    if local_tooling.is_dir():
        sys.path.insert(0, str(local_tooling))
    errors: list[str] = []
    tasks = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    pool = read_tsv(root / "contracts" / "candidate_pool_244.v1.tsv")
    if len(tasks) != 244 or len(pool) != 244:
        errors.append("244_CONTRACT_COUNT")
    if any(row.get("authority") != "0" or any(not value for value in row.values()) for row in tasks + pool):
        errors.append("TASK_EMPTY_OR_AUTHORITY")
    task_review = load_json(root / "evidence" / "task_contract_review.v1.json")
    if task_review.get("status") != "PASS" or task_review.get("errors"):
        errors.append("TASK_REVIEW")
    source = load_json(root / "data" / "ledgers" / "source_ledger.v1.json")
    license_ledger = load_json(root / "data" / "ledgers" / "license_ledger.v1.json")
    truth = load_json(root / "data" / "ledgers" / "truth_ledger.v1.json")
    splits = load_json(root / "data" / "ledgers" / "split_ledger.v1.json")
    leakage = load_json(root / "data" / "ledgers" / "leakage_audit.v1.json")
    if source.get("status") != "PASS" or license_ledger.get("status") != "PASS" or truth.get("status") != "FROZEN":
        errors.append("SOURCE_LICENSE_TRUTH")
    if splits.get("status") != "PASS" or leakage.get("status") != "PASS" or leakage.get("cross_split_group_overlap_total") != 0:
        errors.append("SPLIT_LEAKAGE")
    queue = load_json(root / "queue" / "dual_5090_queue.v1.json")
    jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
    if len(jobs) != 244 or len(queue["jobs"]["GPU_A"]) != 148 or len(queue["jobs"]["GPU_B"]) != 96:
        errors.append("QUEUE_COUNTS")
    if any(job.get("authority") != 0 for job in jobs):
        errors.append("QUEUE_AUTHORITY")
    admitted = [job for job in jobs if job.get("admission_state") == "ADMITTED"]
    rejected = [job for job in jobs if job.get("admission_state") == "PRE_GPU_REJECTED_WITH_EVIDENCE"]
    unresolved_jobs = [job for job in jobs if job.get("admission_state") not in {"ADMITTED", "PRE_GPU_REJECTED_WITH_EVIDENCE"}]
    for job in admitted:
        dataset = root / job.get("staged_dataset", "")
        if not dataset.is_file() or sha256_file(dataset) != job.get("staged_dataset_sha256"):
            errors.append(f"{job['candidate_id']}:STAGED_HASH")
    abi = load_json(root / "contracts" / "model_package_abi.v1.json")
    if abi.get("authority") != 0 or abi.get("status") != "FROZEN_FOR_GPU_TRAINING_BOARD_PENDING":
        errors.append("ABI_FREEZE")
    expected_offset = 0
    for field in abi["fields"]:
        if field["offset"] != expected_offset:
            errors.append("ABI_FIELD_GAP")
        expected_offset += field["bytes"]
    if expected_offset != abi["header_bytes"]:
        errors.append("ABI_HEADER_LENGTH")
    interfaces = []
    for relative in (
        "contracts/schemas/evidence_card_v2.schema.json",
        "contracts/schemas/sintergraph_psp_r1.schema.json",
        "contracts/schemas/chronospec_r4.events.v1.json",
    ):
        path = root / relative
        value = load_json(path)
        serialized = json.dumps(value, sort_keys=True)
        if '"const": 0' not in serialized and value.get("authority") != 0:
            errors.append(relative + ":AUTHORITY")
        interfaces.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    fixture_packages, fixture_errors = verify_fixture_packages(root)
    errors.extend(fixture_errors)
    local_manifest_path = root / "evidence" / "local_readiness_manifest.v1.json"
    local_manifest = load_json(local_manifest_path)
    for record in local_manifest["records"]:
        artifact = root / record["path"]
        if artifact.is_file():
            record["bytes"] = artifact.stat().st_size
            record["sha256"] = sha256_file(artifact)
    local_manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    local_manifest["content_root_sha256"] = hashlib.sha256(canonical_bytes(local_manifest["records"])).hexdigest()
    local_manifest_path.write_text(json.dumps(local_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    errors.extend("LOCAL_MANIFEST:" + item for item in manifest_errors(root, local_manifest))
    staging = load_json(root / "data" / "staged" / "staging_manifest.v1.json")
    rag_staging = load_json(root / "data" / "staged" / "rag_staging_manifest.v1.json")
    disposition = load_json(root / "evidence" / "pre_gpu_disposition_244.v1.json")
    if disposition.get("status") != "PASS" or disposition.get("candidate_count") != 244 or disposition.get("unresolved") != 0:
        errors.append("PRE_GPU_DISPOSITION")
    if disposition.get("admitted") != len(admitted) or disposition.get("pre_gpu_rejected_with_evidence") != len(rejected):
        errors.append("PRE_GPU_DISPOSITION_COUNTS")
    state_counts = Counter(job["pre_gpu_rejection"]["reason_code"] for job in rejected)
    admitted_by_shard = Counter(job["gpu_shard"] for job in admitted)
    toolchain_status = "PASS" if not errors else "FAIL"
    full_data_ready = not unresolved_jobs and len(admitted) + len(rejected) == 244 and admitted_by_shard["GPU_A"] > 0 and admitted_by_shard["GPU_B"] > 0
    full_status = "GPU_READY" if toolchain_status == "PASS" and full_data_ready else "GPU_READY_REJECTED_PRE_GPU_GATES"
    manifest_files: set[Path] = set()
    for relative in ("contracts", "docs", "pipeline", "tests", "data/ledgers", "data/splits", "data/staged", "data/corpora", "data/raw", "data/metadata", "queue", "artifacts/fixture_dry_run", "evidence/rag_fixture_dry_run_v1", "releases"):
        base = root / relative
        if base.is_dir():
            manifest_files.update(item for item in base.rglob("*") if item.is_file() and "__pycache__" not in item.parts)
    manifest_files.add(root / "AGENTS.md")
    manifest_records = [
        {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(manifest_files)
    ]
    readiness_manifest = {
        "schema": "cimc.forge200.gpu-readiness-manifest.v1",
        "records": manifest_records,
        "files": len(manifest_records),
        "bytes": sum(item["bytes"] for item in manifest_records),
        "content_root_sha256": hashlib.sha256(canonical_bytes(manifest_records)).hexdigest(),
    }
    manifest_output = root / "evidence" / "gpu_readiness_manifest.v1.json"
    manifest_output.write_text(json.dumps(readiness_manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    try:
        import numpy
        import onnx
        local_versions = {"numpy": numpy.__version__, "onnx": onnx.__version__}
    except ImportError as exc:
        local_versions = {"error": str(exc)}
        errors.append("LOCAL_EXPORT_DEPENDENCY_IMPORT")
        toolchain_status = "FAIL"
        full_status = "GPU_READY_REJECTED_PRE_GPU_DATA_GATES"
    receipt = {
        "schema": "cimc.forge200.gpu-readiness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CIMC",
        "status": full_status,
        "toolchain_status": toolchain_status,
        "pilot_infrastructure_status": "BOTH_SHARDS_REAL_TASK_PILOT_READY" if toolchain_status == "PASS" and admitted_by_shard["GPU_A"] and admitted_by_shard["GPU_B"] else "NOT_READY",
        "gpu_actions": 0,
        "remote_connections": 0,
        "board_actions": 0,
        "production_modifications": 0,
        "authority_nonzero": 0,
        "contracts": {"candidate_tasks": len(tasks), "candidate_pool": len(pool), "review_status": task_review["status"]},
        "data": {
            "source_status": source["status"],
            "license_status": license_ledger["status"],
            "truth_status": truth["status"],
            "split_status": splits["status"],
            "leakage_status": leakage["status"],
            "tabular_staged_pass": staging["staged_pass"],
            "rag_staged_pass": rag_staging["staged_pass"],
            "ccby_documents": load_json(root / "data" / "ledgers" / "ccby_multidomain_corpus.v1.json")["document_count"],
            "ccby_chunks": load_json(root / "data" / "ledgers" / "ccby_multidomain_corpus.v1.json")["chunk_count"],
        },
        "queue": {
            "jobs": len(jobs),
            "gpu_a_jobs": len(queue["jobs"]["GPU_A"]),
            "gpu_b_jobs": len(queue["jobs"]["GPU_B"]),
            "admitted": len(admitted),
            "pre_gpu_rejected_with_evidence": len(rejected),
            "unresolved": len(unresolved_jobs),
            "admitted_by_shard": dict(admitted_by_shard),
            "rejected_by_reason": dict(sorted(state_counts.items())),
            "admitted_estimated_gpu_minutes": {
                shard: sum(job["estimated_gpu_minutes"] for job in admitted if job["gpu_shard"] == shard) for shard in ("GPU_A", "GPU_B")
            },
            "admitted_static_wall_hours": max(sum(job["estimated_gpu_minutes"] for job in admitted if job["gpu_shard"] == shard) for shard in ("GPU_A", "GPU_B")) / 60.0,
            "vram_gib_max": {
                shard: max((job["estimated_vram_gib"] for job in queue["jobs"][shard]), default=0) for shard in ("GPU_A", "GPU_B")
            },
            "cache_budget_bytes": queue["cache_budget_bytes"],
            "checkpoint_budget_bytes": queue["checkpoint_budget_bytes"],
        },
        "abi": {"path": "contracts/model_package_abi.v1.json", "sha256": sha256_file(root / "contracts" / "model_package_abi.v1.json"), "fixture_packages": fixture_packages},
        "interfaces": interfaces,
        "tooling": {
            "local_versions": local_versions,
            "gpu_requirements_sha256": sha256_file(root / "pipeline" / "requirements-gpu-cu128.lock.txt"),
            "readiness_manifest": "evidence/gpu_readiness_manifest.v1.json",
            "readiness_manifest_sha256": sha256_file(manifest_output),
            "content_root_sha256": readiness_manifest["content_root_sha256"],
            "files": readiness_manifest["files"],
            "bytes": readiness_manifest["bytes"],
        },
        "commands": {
            "audit_a": "python pipeline/gpu_queue_worker.py --shard GPU_A --mode audit",
            "audit_b": "python pipeline/gpu_queue_worker.py --shard GPU_B --mode audit",
            "pilot_a": "python pipeline/gpu_queue_worker.py --shard GPU_A --mode pilot --artifact-root artifacts/cloud5090 --pilot-jobs 12 --pilot-epochs 40 --max-minutes 60 --resume",
            "pilot_b": "python pipeline/gpu_queue_worker.py --shard GPU_B --mode pilot --artifact-root artifacts/cloud5090 --pilot-jobs 8 --pilot-epochs 40 --max-minutes 60 --resume",
            "eta_at_2h": "python pipeline/reestimate_gpu_eta.py --artifact-root artifacts/cloud5090 --elapsed-hours 2",
            "full_a": "python pipeline/gpu_queue_worker.py --shard GPU_A --mode full --artifact-root artifacts/cloud5090 --resume",
            "full_b": "python pipeline/gpu_queue_worker.py --shard GPU_B --mode full --artifact-root artifacts/cloud5090 --resume",
        },
        "blocking_conditions": [] if full_data_ready else ["One or more candidates lack a final admitted or evidenced pre-GPU rejection disposition."],
        "scope_limitations": [
            f"{len(rejected)} candidates are pre-GPU rejected with evidence and will not consume CUDA.",
            "The 150-new-model release floor cannot be claimed from this admitted pool alone; only post-training winners with complete board evidence may be counted.",
            "GPU_READY authorizes only the admitted queue and does not authorize board deployment, deterministic control, or promotion of literature/API output to ground truth.",
        ],
        "transfer_bundle": load_json(root / "releases" / "forge200_gpu_transfer_bundle.v1.json"),
        "errors": errors,
    }
    output = root / "evidence" / "gpu_readiness_receipt.v1.json"
    write_json = lambda path, value: path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    write_json(output, receipt)
    print(json.dumps({"status": full_status, "toolchain": toolchain_status, "admitted": len(admitted), "pre_gpu_rejected": len(rejected), "unresolved": len(unresolved_jobs), "errors": len(errors)}, sort_keys=True))
    return 0 if toolchain_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
