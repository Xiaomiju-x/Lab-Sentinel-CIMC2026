#!/usr/bin/env python3
"""Run the exact SD refusal fixtures through the portable C ModelBank."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--sd-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sd_root = args.sd_root.resolve()
    runner = args.runner.resolve()
    good_golden = sd_root / "F200/P001.GLD"
    cases = [
        ("BAD_MAGIC", "F200/FAULT/BADMAG.ICM", good_golden, "CAND-P-001", 1, 3),
        ("AUTHORITY_NONZERO", "F200/FAULT/BADAUT.ICM", good_golden, "CAND-P-001", 1, 4),
        ("PAYLOAD_CORRUPT", "F200/FAULT/BADPAY.ICM", good_golden, "CAND-P-001", 1, 8),
        ("ROLLBACK_GENERATION", "F200/FAULT/BADGEN.ICM", good_golden, "CAND-P-001", 1, 9),
        ("UNKNOWN_ENGINE", "F200/FAULT/BADENG.ICM", good_golden, "CAND-P-001", 1, 5),
        ("GOLDEN_CORRUPT", "F200/FAULT/BADGLD.ICM", sd_root / "F200/FAULT/BADGLD.GLD", "CAND-P-001", 1, 10),
        ("MODEL_ID_MISMATCH", "F200/P001.ICM", good_golden, "CAND-P-999", 1, 7),
    ]
    records = []
    for name, package_rel, golden, model, generation, expected in cases:
        package = sd_root / package_rel
        completed = subprocess.run(
            [
                str(runner),
                str(package),
                str(golden),
                model,
                str(generation),
                str(expected),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"FAULT_C_GATE:{name}:{completed.returncode}:"
                f"{completed.stdout}:{completed.stderr}"
            )
        report = json.loads(completed.stdout)
        if not report["pass"] or report["status"] != expected:
            raise RuntimeError(f"FAULT_STATUS_GATE:{name}:{report}")
        records.append(
            {
                "case": name,
                "package": package.relative_to(root).as_posix(),
                "package_sha256": sha256(package),
                "golden": golden.relative_to(root).as_posix(),
                "golden_sha256": sha256(golden),
                "expected_status": expected,
                "actual_status": report["status"],
                "pass": True,
            }
        )
    result = {
        "schema": "cimc.forge200.sd-fault-c-verification.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_SEVEN_C_LOADER_REFUSAL_CASES_BOARD_PENDING",
        "runner": {
            "path": runner.relative_to(root).as_posix(),
            "bytes": runner.stat().st_size,
            "sha256": sha256(runner),
        },
        "cases": records,
        "passed": len(records),
        "failed": 0,
        "authority_nonzero_accepted": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical(records)).hexdigest(),
        "claim_boundary": (
            "Portable C loader refusal behavior passed on x86. Physical card "
            "removal, power loss and bus electrical behavior remain board pending."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": result["passed"],
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
