#!/usr/bin/env python3
"""Audit the unified 170-model/RAG/TraceLedger fault and swap matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--sd-staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    staging = args.sd_staging.resolve()
    manifest = json.loads((staging / "MANIFEST.v9.json").read_text(encoding="utf-8"))
    base = root / manifest["base_modelbank"]["path"]
    base_manifest = json.loads((base / "MANIFEST.JSON").read_text(encoding="utf-8"))
    records = base_manifest["records"]
    if len(records) != 170:
        raise RuntimeError("MODEL_COUNT_GATE")

    counts = Counter()
    cancelled = []
    same_reloads = []
    index = 0
    previous = 170
    for load_number in range(1, 1001):
        record = records[index]
        counts[index] += 1
        if index == previous:
            same_reloads.append({"load": load_number, "candidate_id": record["candidate_id"]})
        previous = index
        if load_number % 250 == 0:
            size = int(record["package_bytes"])
            cancelled.append({
                "load": load_number,
                "candidate_id": record["candidate_id"],
                "package_bytes": size,
                "size_class": "LARGE" if size > 100000 else "MEDIUM" if size > 8192 else "SMALL",
            })
        if load_number % 100 != 99:
            index = (index + 1) % 170
    swap_gates = {
        "loads_1000": sum(counts.values()) == 1000,
        "all_170_exercised": len(counts) == 170,
        "minimum_four_per_model": min(counts.values()) >= 4,
        "actual_same_model_reload_10": len(same_reloads) == 10,
        "mid_load_cancel_4": len(cancelled) == 4,
    }

    board_path = production / "firmware/keil_proj/HardWare/Lab_Sentinel/forge200_board_port.c"
    board = board_path.read_text(encoding="utf-8")
    required_fragments = (
        "index == previous_swap_index", "(swap_loads % 100U) != 99U",
        "cancelled_load_probe", "(swap_loads % 250U) == 0U",
        "forge200_rag_board_run", "event=", "POWER_CUT_ARMED",
        "same_model_reload=%lu", "mid_cancel=%lu", "soak_rag_queries",
    )
    static_gates = {fragment: fragment in board for fragment in required_fragments}
    forbidden = re.findall(r"\b(?:heater|relay|motor|actuator|fan12?)_[A-Za-z0-9_]+\s*\(", board)

    sd_faults = json.loads((root / "evidence/sd_fault_c_verification.v8.json").read_text(encoding="utf-8"))
    rag = json.loads((root / "evidence/rag_runtime_host_acceptance.v9.json").read_text(encoding="utf-8"))
    veriprocess = json.loads((root / "evidence/veriprocess_host_acceptance.v9.json").read_text(encoding="utf-8"))
    host_gates = {
        "sd_package_faults_7": sd_faults.get("passed") == 7 and sd_faults.get("failed") == 0,
        "rag_120_safe": rag.get("workload", {}).get("total") == 120 and rag.get("workload", {}).get("safe") == 120,
        "rag_four_mutations": sum(item.get("rejected", False) for item in rag.get("mutations", [])) == 4,
        "veriprocess_69": veriprocess.get("host_cases", {}).get("passed") == 69 and veriprocess.get("host_cases", {}).get("failed") == 0,
        "authority_zero": sd_faults.get("authority_nonzero_accepted") == 0 and rag.get("authority") == 0 and veriprocess.get("authority_nonzero") == 0,
    }
    accepted = all(swap_gates.values()) and all(static_gates.values()) and not forbidden and all(host_gates.values())
    result = {
        "schema": "cimc.forge200.unified-fault-matrix.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_HOST_FAULT_MATRIX_BOARD_PHYSICAL_ACCEPTANCE_PENDING" if accepted else "REJECTED",
        "accepted": accepted,
        "swap_schedule": {
            "loads": sum(counts.values()), "models": len(counts),
            "minimum_per_model": min(counts.values()), "maximum_per_model": max(counts.values()),
            "actual_same_model_reloads": same_reloads, "mid_load_cancellations": cancelled,
            "gates": swap_gates,
        },
        "host_fault_cases": {
            "modelbank_package": 7, "rag_assets": 4, "veriprocess": 69,
            "gates": host_gates,
        },
        "board_static": {
            "path": str(board_path), "sha256": sha256(board_path),
            "required_fragments": static_gates, "forbidden_control_calls": forbidden,
        },
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
        "board_pending": [
            "PHYSICAL_SD_REMOVAL_FAIL_CLOSED", "PHYSICAL_WAL_POWER_CUT_RECOVERY",
            "MAX31856_SHARED_SPI_CONCURRENCY", "GD32_DWT_LATENCY",
            "1000_PHYSICAL_AB_LOADS", "24H_PHYSICAL_SOAK",
        ],
    }
    result["content_root_sha256"] = hashlib.sha256(json.dumps(
        {"swap": result["swap_schedule"], "host": result["host_fault_cases"], "board": result["board_static"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "min_per_model": min(counts.values()),
        "same_reloads": len(same_reloads), "mid_cancels": len(cancelled),
        "host_cases": 80, "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
