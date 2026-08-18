#!/usr/bin/env python3
"""Audit potentially reusable data inside the CIMC workspace only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    candidate_root = Path(__file__).resolve().parents[1]
    cimc_root = candidate_root.parents[1] / "CIMC"
    specs = [
        (
            "FROZEN_NANOLM_TEMPLATE_CORPUS",
            list((cimc_root / "model" / "nanolm").glob("*.jsonl")),
            "CONTROLLED_FIXTURE",
            "FROZEN_BASELINE_ONLY",
            "Hand-written/template-generated furnace text; contains prescriptive wording and is not licensed multidomain QA ground truth.",
        ),
        (
            "FROZEN_ROOT_EGUARD_SIM_LABELS",
            list((cimc_root / "model" / "root_eguard_r2" / "artifacts" / "dataset").glob("*.jsonl")),
            "PHYSICS_SIM",
            "FROZEN_BASELINE_ONLY",
            "SIM_PROCESS labels belong to the existing Root-eGuard evaluation protocol and do not satisfy new independent task labels.",
        ),
        (
            "FROZEN_BASELINE_NUMPY_ARTIFACTS",
            list((cimc_root / "model").glob("ai*/*.npy")) + list((cimc_root / "model").glob("ai*/*.npz")),
            "MIXED_FROZEN_BASELINE",
            "FROZEN_BASELINE_ONLY",
            "Arrays are inputs/checkpoints of already counted baseline tasks; no record-level license/truth manifest permits relabelling them for new objectives.",
        ),
        (
            "HARDWARE_BRINGUP_LOGS",
            list((cimc_root / "evidence" / "hardware_bringup").rglob("*")),
            "METADATA_ONLY",
            "NO_TRAINING_LABELS",
            "Board logs and build receipts prove hardware behavior but are not run/session-bound supervised records for Forge200 tasks.",
        ),
        (
            "CHRONOSPEC_REPLAY_RECORDS",
            list((cimc_root / "artifacts" / "chronospec_r3_host_diff_smoke").glob("*.jsonl"))
            + list((cimc_root / "evidence" / "chronospec_r3_host_100k_20260716").glob("*.jsonl")),
            "CONTROLLED_FIXTURE",
            "FROZEN_BASELINE_ONLY",
            "Deterministic conformance replay is valid interface evidence, not new materials/process task truth.",
        ),
    ]
    groups = []
    for group_id, paths, truth_class, decision, reason in specs:
        files = sorted({path for path in paths if path.is_file()})
        records = [
            {
                "path": str(path.relative_to(cimc_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ]
        groups.append(
            {
                "group_id": group_id,
                "truth_class": truth_class,
                "decision": decision,
                "new_task_training_allowed": False,
                "reason": reason,
                "files": len(records),
                "bytes": sum(item["bytes"] for item in records),
                "records": records,
            }
        )
    referenced_real = cimc_root / "exp_ground_truth" / "observed_pl.csv"
    receipt = {
        "schema": "cimc.forge200.existing-cimc-asset-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_NO_ADDITIONAL_ADMISSIBLE_NEW_TASK_DATA",
        "scope": str(cimc_root),
        "groups": groups,
        "referenced_real_source": {
            "path": "exp_ground_truth/observed_pl.csv",
            "exists": referenced_real.is_file(),
            "decision": "NOT_ADMITTED_MISSING_ARTIFACT" if not referenced_real.is_file() else "REQUIRES_RECORD_LEVEL_REVIEW",
        },
        "additional_admitted_new_tasks": 0,
        "authority": 0,
    }
    output = candidate_root / "data" / "ledgers" / "cimc_existing_asset_audit.v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "groups": len(groups), "files": sum(item["files"] for item in groups), "additional_admitted": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
