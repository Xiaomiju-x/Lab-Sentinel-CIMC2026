#!/usr/bin/env python3
"""Recoverable single-GPU shard worker for the Forge200 queue.

The two workers are launched independently on separate RTX5090 instances; no
cross-public-network DDP is used.  This process performs pre-GPU admission,
timeouts, retries, heartbeat monitoring, resume, and continuous artifact hash
manifests.  It never opens SSH or provisions cloud resources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
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


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def load_queue(root: Path, shard: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    queue = json.loads((root / "queue" / "dual_5090_queue.v1.json").read_text(encoding="utf-8"))
    if queue.get("project") != "CIMC" or queue.get("no_cross_public_network_ddp") is not True:
        raise RuntimeError("QUEUE_PROJECT_OR_DDP_GATE")
    jobs = queue["jobs"][shard]
    if any(job.get("authority") != 0 or job.get("gpu_shard") != shard for job in jobs):
        raise RuntimeError("QUEUE_AUTHORITY_OR_SHARD_GATE")
    return queue, jobs


def manifest_tree(path: Path) -> dict[str, Any]:
    records = []
    if path.is_dir():
        for item in sorted(
            child
            for child in path.rglob("*")
            if child.is_file() and not child.name.startswith("transfer_")
        ):
            records.append({"path": str(item.relative_to(path)).replace("\\", "/"), "bytes": item.stat().st_size, "sha256": sha256_file(item)})
    return {"schema": "cimc.forge200.transfer-manifest.v1", "records": records, "bytes": sum(item["bytes"] for item in records), "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def audit(root: Path, shard: str) -> dict[str, Any]:
    queue, jobs = load_queue(root, shard)
    errors = []
    admitted = []
    for job in jobs:
        if job.get("admission_state") == "ADMITTED":
            dataset = root / job.get("staged_dataset", "")
            if not dataset.is_file() or sha256_file(dataset) != job.get("staged_dataset_sha256"):
                errors.append(f"{job['candidate_id']}: staged dataset hash")
            else:
                admitted.append(job["candidate_id"])
    return {
        "schema": "cimc.forge200.shard-audit.v1",
        "status": "PASS" if not errors else "FAIL",
        "shard": shard,
        "queue_status": queue["status"],
        "jobs": len(jobs),
        "admitted": len(admitted),
        "blocked": len(jobs) - len(admitted),
        "admitted_candidates": admitted,
        "errors": errors,
        "authority_nonzero": 0,
    }


def run_child(command: list[str], timeout_seconds: float, heartbeat_path: Path, heartbeat_stale_seconds: float) -> tuple[int, str]:
    # A completed pilot leaves a heartbeat behind.  On a later full/resume
    # invocation its old mtime must not be interpreted as a hang before the
    # newly launched trainer has had a chance to publish its first heartbeat.
    if heartbeat_path.is_file():
        heartbeat_path.touch()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    output: list[str] = []
    started = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - started > timeout_seconds:
            process.kill()
            output.append("WORKER_TIMEOUT")
            break
        if heartbeat_path.is_file() and time.time() - heartbeat_path.stat().st_mtime > heartbeat_stale_seconds:
            process.kill()
            output.append("HEARTBEAT_STALE")
            break
        time.sleep(2)
    stdout, _ = process.communicate()
    if stdout:
        output.append(stdout)
    return process.returncode if process.returncode is not None else 124, "".join(output)


def execute(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    queue, jobs = load_queue(root, args.shard)
    admitted = [job for job in jobs if job.get("admission_state") == "ADMITTED"]
    if args.mode == "pilot":
        admitted = admitted[: args.pilot_jobs]
    if not admitted:
        raise RuntimeError(f"NO_ADMITTED_JOBS_{args.shard}")
    worker_state_path = artifact_root / f"worker_{args.shard.lower()}.state.json"
    if worker_state_path.is_file() and args.resume:
        state = json.loads(worker_state_path.read_text(encoding="utf-8"))
    else:
        state = {"schema": "cimc.forge200.worker-state.v1", "shard": args.shard, "started_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    wall_started = time.monotonic()
    for job in admitted:
        candidate_id = job["candidate_id"]
        prior = state["jobs"].get(candidate_id, {})
        if prior.get("status") == "COMPLETE" and args.resume:
            continue
        if args.mode == "pilot" and time.monotonic() - wall_started >= args.max_minutes * 60:
            break
        attempts = int(prior.get("attempts", 0))
        status = "FAIL_CLOSED"
        while attempts <= int(job["max_retries"]):
            attempts += 1
            state["jobs"][candidate_id] = {"status": "RUNNING", "attempts": attempts, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
            write_json(worker_state_path, state)
            metadata = json.loads((root / job["staged_metadata"]).read_text(encoding="utf-8"))
            trainer = "gpu_train_rag_job.py" if metadata.get("task_kind") in {"token_lm", "contrastive_embedding"} else "gpu_train_job.py"
            command = [
                sys.executable,
                str(root / "pipeline" / trainer),
                "--candidate-id",
                candidate_id,
                "--root",
                str(root),
                "--artifact-root",
                str(artifact_root),
                "--device",
                args.device,
                "--resume",
            ]
            if args.mode == "pilot":
                command.extend(["--max-epochs", str(args.pilot_epochs)])
            exit_code, output = run_child(
                command,
                timeout_seconds=float(job["timeout_minutes"]) * 60,
                heartbeat_path=artifact_root / candidate_id / "heartbeat.json",
                heartbeat_stale_seconds=max(float(job["heartbeat_seconds"]) * 4, 120.0),
            )
            (artifact_root / candidate_id / f"worker_attempt_{attempts}.log").write_text(output, encoding="utf-8")
            receipt_path = artifact_root / candidate_id / "promotion_receipt.json"
            if exit_code == 0 and receipt_path.is_file():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                if receipt.get("authority") == 0 and receipt.get("status") == "HOST_GPU_TRAINED_BOARD_PENDING":
                    status = "COMPLETE"
                    break
            failure = json.loads((artifact_root / candidate_id / "failure.json").read_text(encoding="utf-8")) if (artifact_root / candidate_id / "failure.json").is_file() else {}
            if failure.get("error", "").startswith(("BLOCKED_PRE_GPU", "STAGED_", "SPLIT_", "AUTHORITY_", "DATASET_")):
                break
        state["jobs"][candidate_id] = {"status": status, "attempts": attempts, "updated_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(worker_state_path, state)
        write_json(artifact_root / candidate_id / "transfer_manifest.json", manifest_tree(artifact_root / candidate_id))
        write_json(artifact_root / f"transfer_{args.shard.lower()}.json", manifest_tree(artifact_root))
    complete = sum(item.get("status") == "COMPLETE" for item in state["jobs"].values())
    failed = sum(item.get("status") == "FAIL_CLOSED" for item in state["jobs"].values())
    result = {
        "schema": "cimc.forge200.worker-result.v1",
        "status": "COMPLETE" if failed == 0 else "COMPLETE_WITH_FAILURES",
        "shard": args.shard,
        "mode": args.mode,
        "admitted_in_shard": len(admitted),
        "completed": complete,
        "failed": failed,
        "runtime_seconds": time.monotonic() - wall_started,
        "authority_nonzero": 0,
        "no_cross_public_network_ddp": True,
    }
    write_json(artifact_root / f"worker_{args.shard.lower()}.result.json", result)
    write_json(artifact_root / f"transfer_{args.shard.lower()}.json", manifest_tree(artifact_root))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", choices=["GPU_A", "GPU_B"], required=True)
    parser.add_argument("--mode", choices=["audit", "pilot", "full"], default="audit")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cloud5090"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pilot-jobs", type=int, default=4)
    parser.add_argument("--pilot-epochs", type=int, default=12)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "audit":
        result = audit(root, args.shard)
    else:
        result = execute(args)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"PASS", "COMPLETE"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
