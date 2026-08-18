#!/usr/bin/env python3
"""Refresh the release gap with post-v4 source audits and no count gaming."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    prior_path = root / "evidence/release_gap_audit.v4.json"
    prior = json.loads(prior_path.read_text(encoding="utf-8"))
    audit_names = [
        "mendeley_sic_plunger_source_contract_audit.v1.json",
        "mendeley_p105_afm_source_contract_audit.v1.json",
        "phm2016_p089_source_preflight.v1.json",
        "nist_p099_source_contract_audit.v1.json",
        "p073_residual_exploration_quarantine.v1.json",
    ]
    audits = []
    for name in audit_names:
        path = root / "evidence" / name
        document = json.loads(path.read_text(encoding="utf-8"))
        audits.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "status": document["status"],
            "training_actions": document.get("training_actions", document.get("actions", {}).get("training_promotion_actions", 0)),
            "host_promotions": document.get("host_promotions", document.get("actions", {}).get("host_promotions", 0)),
        })
    if any(item["training_actions"] or item["host_promotions"] for item in audits):
        raise RuntimeError("POST_V4_AUDIT_PROMOTION_FORBIDDEN")

    result = {
        "schema": "cimc.forge200.release-gap-audit.v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RELEASE_FLOOR_BLOCKED_BY_EXACT_DATA_AND_LATER_BOARD_ACCEPTANCE",
        "prior": {
            "path": prior_path.relative_to(root).as_posix(),
            "sha256": sha256(prior_path),
            "content_root_sha256": prior["content_root_sha256"],
        },
        "host_state": {
            "exact_source_bound": 78,
            "sim_only_extensions": 7,
            "host_total": 85,
            "new_assets_floor": 120,
            "exact_source_bound_shortfall": 42,
            "including_sim_only_shortfall": 35,
            "new_promotions_after_v4": 0,
        },
        "post_v4_source_audits": audits,
        "manual_acquisition_queue": [
            {
                "candidate_id": "CAND-P-099",
                "source": "NIST MDS2-3838",
                "files": [
                    {
                        "name": "intensity_sets.zip",
                        "bytes": 419600000,
                        "sha256": "5cd9f4caff80e9afab83515032347a17e9974554ea148f01280090504807e078"
                    },
                    {
                        "name": "mask_sets.zip",
                        "bytes": 276900,
                        "sha256": "5925dc95478e2cfc3c9ec54bfef888c7596db35fdf41bb929d6b96b8562ab562"
                    }
                ],
                "promotion_before_hash_verification": False
            }
        ],
        "non_download_blockers": [
            {
                "candidate_id": "CAND-P-089",
                "reason": "PHM2016 semantic match exists, but official payload and explicit reusable dataset license are not both materialized."
            },
            {
                "candidate_id": "CAND-P-105",
                "reason": "AFM source is licensed and hashed, but input-derived roughness targets cannot bypass or beat the frozen deterministic baseline."
            },
            {
                "candidate_id": "CAND-P-073",
                "reason": "Validation gate failed; the mistakenly evaluated test result is quarantined and cannot be reused for selection."
            }
        ],
        "authority_nonzero": 0,
        "board_accepted": 0,
        "countable_models": 0,
        "production_files_modified": 0,
        "gd32_actions": 0,
    }
    result["content_root_sha256"] = content_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "exact_source_bound": result["host_state"]["exact_source_bound"],
        "exact_shortfall": result["host_state"]["exact_source_bound_shortfall"],
        "audits": len(audits),
        "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
