#!/usr/bin/env python3
"""Re-estimate independent shard ETA from completed promotion receipts."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/cloud5090"))
    parser.add_argument("--elapsed-hours", type=float, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    queue = json.loads((root / "queue" / "dual_5090_queue.v1.json").read_text(encoding="utf-8"))
    shards = {}
    for shard in ("GPU_A", "GPU_B"):
        admitted = [job for job in queue["jobs"][shard] if job["admission_state"] == "ADMITTED"]
        completed = []
        for job in admitted:
            receipt_path = artifact_root / job["candidate_id"] / "promotion_receipt.json"
            if not receipt_path.is_file():
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") == "HOST_GPU_TRAINED_BOARD_PENDING" and receipt.get("authority") == 0:
                completed.append((job, float(receipt["runtime_seconds"])))
        completed_estimate = sum(job["estimated_gpu_minutes"] for job, _ in completed)
        completed_actual_minutes = sum(seconds for _, seconds in completed) / 60.0
        scale = completed_actual_minutes / completed_estimate if completed_estimate > 0 else None
        remaining_estimate = sum(job["estimated_gpu_minutes"] for job in admitted if job["candidate_id"] not in {item[0]["candidate_id"] for item in completed})
        remaining_hours = remaining_estimate * scale / 60.0 if scale is not None else None
        shards[shard] = {
            "admitted": len(admitted),
            "completed": len(completed),
            "completed_estimated_gpu_minutes": completed_estimate,
            "completed_actual_minutes": completed_actual_minutes,
            "actual_to_static_scale": scale,
            "remaining_static_gpu_minutes": remaining_estimate,
            "projected_remaining_hours": remaining_hours,
        }
    projections = [value["projected_remaining_hours"] for value in shards.values() if value["projected_remaining_hours"] is not None]
    projected_total_wall = args.elapsed_hours + max(projections) if len(projections) == 2 else None
    status = "PAUSE_EXCEEDS_10H_EXPLAIN_BEFORE_CONTINUE" if projected_total_wall is not None and projected_total_wall > 10.0 else "CONTINUE_WITHIN_10H_WINDOW" if projected_total_wall is not None else "INSUFFICIENT_COMPLETIONS_FOR_ETA"
    report = {
        "schema": "cimc.forge200.gpu-eta-reestimate.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_hours": args.elapsed_hours,
        "status": status,
        "projected_total_wall_hours": projected_total_wall,
        "shards": shards,
        "rule": "At 2 elapsed hours pause before continuing when projected total dual-card wall time exceeds 10 hours.",
    }
    output = artifact_root / f"eta_at_{args.elapsed_hours:g}h.json"
    write_json(output, report)
    print(json.dumps(report, sort_keys=True))
    return 2 if status.startswith("PAUSE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
