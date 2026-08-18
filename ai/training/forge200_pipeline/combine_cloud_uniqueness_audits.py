#!/usr/bin/env python3
"""Combine shard audits with explicit last-report-wins corrective precedence."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    selected: dict[str, dict[str, Any]] = {}
    sources = []
    for precedence, path in enumerate(args.input):
        report = json.loads(path.read_text(encoding="utf-8"))
        sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "precedence": precedence,
                "candidate_count": report["candidate_count"],
                "source_status": report["status"],
            }
        )
        for record in report["records"]:
            selected[record["candidate_id"]] = {
                **record,
                "selected_from": str(path),
                "precedence": precedence,
            }
    records = [selected[key] for key in sorted(selected)]
    by_payload: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_payload[record["payload_sha256"]].append(record["candidate_id"])
    collisions = [
        {"payload_sha256": digest, "candidate_ids": ids}
        for digest, ids in sorted(by_payload.items())
        if len(ids) > 1
    ]
    errors = []
    if collisions:
        errors.append("duplicate_selected_weight_payloads")
    if any(record.get("authority") != 0 for record in records):
        errors.append("authority_nonzero")
    if any(record.get("board_accepted") is not False for record in records):
        errors.append("invalid_board_state")
    content = {"sources": sources, "records": records, "payload_collisions": collisions}
    report = {
        "schema": "cimc.forge200.combined-cloud-model-uniqueness.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "selection_rule": "candidate_id_last_report_wins_corrective_precedence",
        "candidate_count": len(records),
        "unique_weight_payloads": len(by_payload),
        "generative_candidates": sum(item["candidate_id"].startswith("CAND-G-") for item in records),
        "predictive_candidates": sum(item["candidate_id"].startswith("CAND-P-") for item in records),
        "support_candidates": sum(item["candidate_id"].startswith("CAND-S-") for item in records),
        "countable_models": sum(item.get("countable_model") is True for item in records),
        "board_accepted": sum(item.get("board_accepted") is True for item in records),
        "authority_nonzero": sum(item.get("authority") != 0 for item in records),
        **content,
        "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
        "errors": errors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: report[key] for key in (
                "status", "candidate_count", "unique_weight_payloads",
                "predictive_candidates", "generative_candidates", "support_candidates",
                "countable_models", "board_accepted", "authority_nonzero", "errors",
                "content_root_sha256",
            )},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
