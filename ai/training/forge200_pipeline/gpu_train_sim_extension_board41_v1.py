#!/usr/bin/env python3
"""Run the frozen SIM extension trainer for board-facing P001-P041."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from gpu_train_job import write_json
from gpu_train_sim_extension_pack_39_v1 import train_one


def prepare_immutable_training_view(root: Path) -> Path:
    """Create D-drive hardlinks matching the frozen trainer's input layout."""
    view = root / ".runtime" / "board41_train_view_v1"
    dataset_view = view / "data" / "staged_sim_extension_pack_39_v1"
    contract_view = view / "contracts" / "sim_extension_pack_39.v1.json"
    dataset_view.mkdir(parents=True, exist_ok=True)
    contract_view.parent.mkdir(parents=True, exist_ok=True)
    source_contract = root / "contracts" / "sim_extension_board41.v1.json"
    shutil.copy2(source_contract, contract_view)
    source_data = root / "data" / "staged_sim_extension_board41_v1"
    for source in source_data.iterdir():
        if not source.is_file():
            continue
        target = dataset_view / source.name
        if target.exists():
            if target.stat().st_size != source.stat().st_size:
                raise RuntimeError(f"TRAINING_VIEW_SIZE_GATE:{target}")
            continue
        os.link(source, target)
    return view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-id")
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "contracts" / "sim_extension_board41.v1.json").read_text(encoding="utf-8"))
    tasks = config["tasks"]
    if args.candidate_id:
        tasks = [task for task in tasks if task["candidate_id"] == args.candidate_id]
        if not tasks:
            raise RuntimeError("UNKNOWN_CANDIDATE_ID")
    training_view = prepare_immutable_training_view(root)
    results = []
    for index, task in enumerate(tasks, 1):
        task_for_trainer = {**task, "outputs": [f"normalized_output_{i}" for i in range(task["output_dimensions"])]}
        print(json.dumps({"event": "TASK_START", "index": index, "total": len(tasks), "candidate_id": task["candidate_id"]}), flush=True)
        receipt = train_one(training_view, args.artifact_root.resolve(), task_for_trainer, config, args.device)
        results.append(receipt)
        print(json.dumps({"event": "TASK_DONE", "candidate_id": task["candidate_id"], "status": receipt["status"], "runtime_seconds": receipt.get("runtime_seconds")}, sort_keys=True), flush=True)
    passed = sum(bool(item.get("host_extension_pass")) for item in results)
    closure = {
        "schema": "cimc.forge200.sim-extension-board41-closure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed == len(results) else "PARTIAL",
        "requested_tasks": len(results),
        "host_extension_passes": passed,
        "rejections": len(results) - passed,
        "records": results,
        "original_exact_contract_promotions": 0,
        "training_view": str(training_view.relative_to(root)).replace("\\", "/"),
        "training_view_payloads_are_ntfs_hardlinks": True,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "sim_extension_board41_closure.v1.json", closure)
    print(json.dumps({"status": closure["status"], "passed": passed, "rejected": len(results) - passed}, sort_keys=True))
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
