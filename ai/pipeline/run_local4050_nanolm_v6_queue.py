#!/usr/bin/env python3
"""Resumable two-process local RTX4050 queue for NanoLM exact-v6 assets."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


IDS = ["CAND-G-001", "CAND-G-003", *[f"CAND-G-{number:03d}" for number in range(4, 27)]]


def write_state(path: Path, value: dict[str, Any]) -> None:
    value["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-parallel", type=int, default=2)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = root / "artifacts" / "local4050_nanolm_contract_v6"
    log_root = artifact_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state_path = artifact_root / "queue_state.json"
    state: dict[str, Any] = {
        "schema": "cimc.forge200.local4050-nanolm-queue.v6",
        "status": "RUNNING",
        "candidate_ids": IDS,
        "max_parallel": args.max_parallel,
        "authority": 0,
        "completed": [],
        "failed": [],
        "running": {},
        "pending": [],
    }
    pending = []
    for candidate_id in IDS:
        receipt = artifact_root / candidate_id / "promotion_receipt.json"
        failure = artifact_root / candidate_id / "failure.json"
        if receipt.is_file() and not failure.is_file():
            state["completed"].append(candidate_id)
        else:
            pending.append(candidate_id)
    running: dict[str, tuple[subprocess.Popen[bytes], Any, Any]] = {}
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root / ".tooling" / "gpu4050"), str(root / ".tooling" / "python"), str(root / "pipeline")]
    )
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["TEMP"] = str(root / ".tooling" / "tmp_gpu4050")
    environment["TMP"] = environment["TEMP"]
    while pending or running:
        while pending and len(running) < args.max_parallel:
            candidate_id = pending.pop(0)
            number = int(candidate_id[-3:])
            foundation = number <= 6
            command = [
                sys.executable,
                str(root / "pipeline" / "gpu_train_nanolm_v2_job.py"),
                "--candidate-id", candidate_id,
                "--root", str(root),
                "--artifact-root", str(artifact_root),
                "--staged-subdir", "staged_nanolm_contract_exact_v6",
                "--supervised-only",
                "--grounded-claim-selection",
                "--resume",
                "--device", "cuda:0",
                "--batch-size", "32",
                "--max-epochs", "24" if foundation else "18",
                "--min-epochs", "12" if foundation else "10",
                "--early-stop-patience", "4",
                "--qat-epochs", "3",
            ]
            stdout_handle = (log_root / f"{candidate_id}.out.log").open("ab")
            stderr_handle = (log_root / f"{candidate_id}.err.log").open("ab")
            process = subprocess.Popen(
                command,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=environment,
                creationflags=creation_flags,
            )
            running[candidate_id] = (process, stdout_handle, stderr_handle)
        finished = []
        for candidate_id, (process, stdout_handle, stderr_handle) in running.items():
            code = process.poll()
            if code is None:
                continue
            stdout_handle.close()
            stderr_handle.close()
            if code == 0 and (artifact_root / candidate_id / "promotion_receipt.json").is_file():
                state["completed"].append(candidate_id)
            else:
                state["failed"].append({"candidate_id": candidate_id, "exit_code": code})
            finished.append(candidate_id)
        for candidate_id in finished:
            running.pop(candidate_id)
        state["running"] = {candidate_id: process.pid for candidate_id, (process, _, _) in running.items()}
        state["pending"] = pending
        state["status"] = "RUNNING" if pending or running else ("COMPLETE" if not state["failed"] else "COMPLETE_WITH_FAILURES")
        write_state(state_path, state)
        if pending or running:
            time.sleep(2)
    return 0 if not state["failed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
