#!/usr/bin/env python3
"""Export compact, hash-bound validation history without model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_optional(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected: dict[str, dict[str, Any]] = {}
    roots = []
    for precedence, artifact_root in enumerate(args.artifact_root):
        artifact_root = artifact_root.resolve()
        roots.append({"path": str(artifact_root), "precedence": precedence})
        for receipt_path in sorted(artifact_root.glob("CAND-*/promotion_receipt.json")):
            candidate_root = receipt_path.parent
            candidate_id = candidate_root.name
            package_path = candidate_root / "w8_or_w8a8.bin"
            selected[candidate_id] = {
                "candidate_id": candidate_id,
                "precedence": precedence,
                "artifact_root": str(artifact_root),
                "promotion_receipt": read_optional(receipt_path),
                "promotion_receipt_sha256": sha256_file(receipt_path),
                "evaluation": read_optional(candidate_root / "eval_grouped.json"),
                "evaluation_sha256": sha256_file(candidate_root / "eval_grouped.json"),
                "calibration": read_optional(candidate_root / "calibration_ood.json"),
                "quantization": read_optional(candidate_root / "quantization_parity.json"),
                "source_manifest": read_optional(candidate_root / "source_manifest.json"),
                "baseline_report": read_optional(candidate_root / "baseline_report.json"),
                "contract_baseline_evaluation": read_optional(candidate_root / "contract_baseline_evaluation.json"),
                "package_payload_sha256": hashlib.sha256(package_path.read_bytes()[256:]).hexdigest(),
            }
    records = [selected[key] for key in sorted(selected)]
    report = {
        "schema": "cimc.forge200.frozen-validation-history.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selection_rule": "candidate_id_last_artifact_root_wins",
        "roots": roots,
        "candidate_count": len(records),
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "authority": 0,
        "board_accepted": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "candidate_count": len(records), "content_root_sha256": report["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
