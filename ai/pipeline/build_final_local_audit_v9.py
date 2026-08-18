#!/usr/bin/env python3
"""Last fail-closed audit before yielding for physical GD32 power-on."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
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
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-root", type=Path, required=True,
                        help="All input and output paths must stay under this root.")
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    handoff = args.handoff.resolve()
    output = args.output.resolve()
    scope = args.scope_root.resolve()
    for path in (root, production, handoff, output):
        try:
            path.relative_to(scope)
        except ValueError as exc:
            raise RuntimeError(f"D_SCOPE_GATE:{path}") from exc

    unit = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(root / "tests"),
         "-p", "test_forge200_local.py", "-v"],
        cwd=production, text=True, encoding="utf-8", errors="replace",
        capture_output=True, timeout=180, check=False,
    )
    combined = unit.stdout + unit.stderr
    match = re.search(r"Ran (\d+) tests", combined)
    unit_count = int(match.group(1)) if match else -1
    if unit.returncode != 0 or unit_count != 50 or "OK" not in combined:
        raise RuntimeError(f"UNIT_GATE:{unit.returncode}:{combined[-2000:]}")

    receipts = {
        "local_complete": root / "evidence/gd32_local_complete.v9.json",
        "rag": root / "evidence/rag_runtime_host_acceptance.v9.json",
        "veriprocess": root / "evidence/veriprocess_host_acceptance.v9.json",
        "fault_matrix": root / "evidence/unified_fault_matrix.v9.json",
        "parser_tests": root / "evidence/board_parser_tests.v9.json",
        "sd_selfverify": root / "evidence/sd_staging_selfverify.v9.json",
        "handoff_selfverify": root / "evidence/gd32_handoff_selfverify.v9.json",
    }
    loaded = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in receipts.items()}
    gates = {
        "unit_50": unit_count == 50 and unit.returncode == 0,
        "local_complete": loaded["local_complete"].get("status") == "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "rag": loaded["rag"].get("workload", {}).get("safe") == 120,
        "veriprocess": loaded["veriprocess"].get("host_cases", {}).get("passed") == 69,
        "fault_matrix": loaded["fault_matrix"].get("accepted") is True,
        "parser_tests": loaded["parser_tests"].get("status") == "PASS",
        "sd_selfverify": loaded["sd_selfverify"].get("accepted") is True,
        "handoff_selfverify": loaded["handoff_selfverify"].get("accepted") is True,
        "handoff_manifest": json.loads((handoff / "MANIFEST.v9.json").read_text(encoding="utf-8")).get("status") == "GD32_READY_LOCAL_COMPLETE_UNIFIED_PHYSICAL_BOARD_REQUIRED",
        "runbook": (root / "docs/FORGE200_UNIFIED_GD32_BOARD_ACCEPTANCE_RUNBOOK_V9.md").is_file(),
        "full_mode": "#define LAB_HARDWARE_BRINGUP 0" in (production / "firmware/ai_models_c/lab_build_config.h").read_text(encoding="utf-8"),
    }
    if not all(gates.values()):
        raise RuntimeError(f"FINAL_GATE:{gates}")
    receipt_records = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for name, path in receipts.items()
    }
    result = {
        "schema": "cimc.forge200.final-local-audit.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_LOCAL_ONLY_GD32_POWER_AND_PHYSICAL_ACCEPTANCE_REQUIRED",
        "gates": gates,
        "unit_tests": {"passed": 50, "failed": 0},
        "receipts": receipt_records,
        "handoff": {
            "path": str(handoff),
            "manifest_sha256": sha256(handoff / "MANIFEST.v9.json"),
            "content_root_sha256": json.loads((handoff / "MANIFEST.v9.json").read_text(encoding="utf-8"))["content_root_sha256"],
        },
        "remaining_scope": "PHYSICAL_GD32_ONLY",
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
        "cloud_connections": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(json.dumps(
        {"gates": gates, "receipts": receipt_records, "handoff": result["handoff"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "unit_tests": 50,
        "local_gates": len(gates), "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
