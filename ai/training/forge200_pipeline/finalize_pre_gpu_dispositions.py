#!/usr/bin/env python3
"""Resolve every Forge200 candidate to admitted or evidenced pre-GPU rejection."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def contracts(root: Path) -> dict[str, dict[str, str]]:
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def rejection_reason(candidate_id: str, source_gate: str, data_state: str) -> tuple[str, str]:
    if candidate_id.startswith("CAND-P-"):
        if any(token in source_gate for token in ("TEAM_MEASURED", "BOARD_MEASURED")) or "RECORD_LEVEL_MANIFEST_REQUIRED" in data_state:
            return "TEAM_OR_BOARD_RECORD_LABELS_ABSENT", "Available CIMC board evidence is metadata-only and contains no task target rows."
        if "L2" in source_gate or "L2" in data_state:
            return "L2_EXPERIMENTAL_TARGETS_ABSENT", "No record-level L2 experimental labels matching this exact target are materialized."
        if "SEMANTICS" in data_state:
            return "SOURCE_TARGET_SEMANTICS_MISMATCH", "A source artifact exists, but its field semantics do not equal the frozen target contract."
        if "RECORD_LEVEL_LABEL_GATE" in data_state:
            return "RECORD_LEVEL_TARGET_BINDING_ABSENT", "The source family is present but exact per-record target binding and grouped split are absent."
        return "TASK_SPECIFIC_OPEN_DATA_NOT_MATERIALIZED", "No licensed artifact with this exact input, target, and split unit is materialized."
    if candidate_id in {"CAND-G-027", "CAND-G-028", "CAND-G-029", "CAND-G-030"}:
        return "EXPERT_HYPOTHESIS_OR_PLAN_TARGETS_ABSENT", "Licensed evidence text cannot mechanically supply ranked hypotheses, falsifiers, measurement plans, or final adjudication labels."
    if candidate_id.startswith("CAND-G-"):
        return "EXPERT_REVIEW_LABELS_ABSENT", "The frozen gate requires expert-reviewed labels; API or corpus text is explicitly forbidden as ground truth."
    support_reasons = {
        "CAND-S-008": ("EXPECTED_INFORMATION_GAIN_LABELS_ABSENT", "Evidence-gain targets require observed uncertainty reduction or expert measurement judgments."),
        "CAND-S-031": ("EXPERT_NLI_ADJUDICATION_ABSENT", "Shared cross-domain NLI requires expert adjudication beyond controlled exact-span mutations."),
        "CAND-S-032": ("SESSION_QUERY_REWRITE_PAIRS_ABSENT", "No source-bound operator session/rewrite relevance pairs are available."),
        "CAND-S-033": ("POST_GPU_ROUTER_OUTPUT_DEPENDENCY", "Calibration inputs require frozen router/retriever logits produced after the first GPU pass."),
        "CAND-S-034": ("EXPERT_CITATION_SPANS_ABSENT", "The spare contract requires expert answer-to-source span annotations."),
        "CAND-S-035": ("DOCUMENT_TABLE_CELL_PAIRS_ABSENT", "Numeric claim labels with document-and-table split and supporting cells are not curated."),
        "CAND-S-037": ("TEMPORAL_RELEVANCE_JUDGMENTS_ABSENT", "Batch-split temporal relevance judgments are unavailable."),
        "CAND-S-038": ("CURATED_SOURCE_AUDIT_LABELS_ABSENT", "Publisher-family source trust audit labels are unavailable."),
        "CAND-S-039": ("EXPERT_DUPLICATE_GROUPS_ABSENT", "Exact text identity is insufficient for the required independent-evidence expert groups."),
        "CAND-S-040": ("EXPERT_CONTRADICTION_EDGE_LABELS_ABSENT", "Direct, conditional, and temporal contradiction edge types require expert labels."),
        "CAND-S-041": ("FROZEN_MODEL_VALIDATION_HISTORY_ABSENT", "The competence target depends on validation history that does not exist before GPU training."),
        "CAND-S-042": ("LICENSED_REVISION_CASES_ABSENT", "The required source-split revision and supersession cases are not materialized."),
        "CAND-S-043": ("EXPERT_MATERIAL_ENTITY_LINKS_ABSENT", "Formula normalization alone cannot replace expert entity links and unresolved labels."),
        "CAND-S-045": ("EXPERT_RELATION_LABELS_ABSENT", "Material-process-property relations and argument spans require expert annotations."),
        "CAND-S-046": ("EXPERT_TABLE_CELL_LINKS_ABSENT", "Open tables exist but expert claim-to-cell support links do not."),
        "CAND-S-047": ("EXPERT_FIGURE_PANEL_LINKS_ABSENT", "Open captions exist but expert claim-to-panel support links do not."),
        "CAND-S-048": ("FROZEN_COORDINATION_DATA_ABSENT", "Joint coordination targets depend on frozen outputs from the first-stage models."),
    }
    return support_reasons.get(candidate_id, ("EXACT_TARGET_LABELS_ABSENT", "No exact target labels satisfying the frozen source gate are available."))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    task_contracts = contracts(root)
    queue_path = root / "queue" / "dual_5090_queue.v1.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    evidence_paths = {
        "asset_audit": "data/ledgers/cimc_existing_asset_audit.v1.json",
        "team_evidence": "data/ledgers/team_hardware_evidence_records.v1.json",
        "source_bindings": "data/ledgers/task_source_bindings.v1.json",
        "ccby_corpus": "data/ledgers/ccby_multidomain_corpus.v1.json",
        "rag_staging": "data/staged/rag_staging_manifest.v1.json",
        "tabular_staging": "data/staged/staging_manifest.v1.json",
    }
    evidence_hashes = {name: {"path": path, "sha256": sha256_file(root / path)} for name, path in evidence_paths.items()}
    records = []
    for shard in ("GPU_A", "GPU_B"):
        for job in queue["jobs"][shard]:
            candidate_id = job["candidate_id"]
            contract = task_contracts[candidate_id]
            contract_sha = hashlib.sha256(canonical_bytes(contract)).hexdigest()
            if job.get("admission_state") == "ADMITTED":
                metadata_path = root / job["staged_metadata"]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                record = {
                    "candidate_id": candidate_id,
                    "category": job["category"],
                    "gpu_shard": shard,
                    "disposition": "ADMITTED",
                    "reason_code": "EXACT_SOURCE_LABEL_SPLIT_BINDING_PASS",
                    "source_gate": contract["source_gate"],
                    "task_contract_sha256": contract_sha,
                    "staged_dataset": job["staged_dataset"],
                    "staged_dataset_sha256": job["staged_dataset_sha256"],
                    "staged_metadata_sha256": sha256_file(metadata_path),
                    "split_sha256": metadata["split_sha256"],
                    "truth_class": metadata["truth_class"],
                    "authority": 0,
                }
            else:
                data_state = job.get("data_binding", {}).get("full_data_state", "UNKNOWN")
                reason_code, explanation = rejection_reason(candidate_id, contract["source_gate"], data_state)
                job["admission_state"] = "PRE_GPU_REJECTED_WITH_EVIDENCE"
                job["pre_gpu_rejection"] = {"reason_code": reason_code, "evidence": list(evidence_paths.values())}
                job.pop("staged_dataset", None)
                job.pop("staged_dataset_sha256", None)
                job.pop("staged_metadata", None)
                record = {
                    "candidate_id": candidate_id,
                    "category": job["category"],
                    "gpu_shard": shard,
                    "disposition": "PRE_GPU_REJECTED_WITH_EVIDENCE",
                    "reason_code": reason_code,
                    "explanation": explanation,
                    "source_gate": contract["source_gate"],
                    "data_state": data_state,
                    "task_contract_sha256": contract_sha,
                    "evidence": evidence_hashes,
                    "proxy_substitution_forbidden": True,
                    "authority": 0,
                }
            records.append(record)
    counts = Counter(record["disposition"] for record in records)
    category_admitted = Counter(record["category"] for record in records if record["disposition"] == "ADMITTED")
    shard_admitted = Counter(record["gpu_shard"] for record in records if record["disposition"] == "ADMITTED")
    if len(records) != 244 or counts["ADMITTED"] == 0 or counts["ADMITTED"] + counts["PRE_GPU_REJECTED_WITH_EVIDENCE"] != 244:
        raise RuntimeError("DISPOSITION_COVERAGE_GATE")
    queue["admitted_jobs"] = counts["ADMITTED"]
    queue["rejected_jobs"] = counts["PRE_GPU_REJECTED_WITH_EVIDENCE"]
    queue["blocked_jobs"] = 0
    queue["admitted_by_shard"] = dict(shard_admitted)
    queue["status"] = "GPU_READY_ADMITTED_AND_PRE_GPU_REJECTED_RESOLVED"
    write_json(queue_path, queue)
    for shard, filename in (("GPU_A", "gpu_a.queue.json"), ("GPU_B", "gpu_b.queue.json")):
        write_json(root / "queue" / filename, {"schema": queue["schema"], "shard": shard, "jobs": queue["jobs"][shard]})
    receipt = {
        "schema": "cimc.forge200.pre-gpu-disposition-244.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_count": len(records),
        "admitted": counts["ADMITTED"],
        "pre_gpu_rejected_with_evidence": counts["PRE_GPU_REJECTED_WITH_EVIDENCE"],
        "unresolved": 0,
        "admitted_by_category": dict(category_admitted),
        "admitted_by_shard": dict(shard_admitted),
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "authority_nonzero": 0,
        "teacher_promoted_to_ground_truth": 0,
    }
    write_json(root / "evidence" / "pre_gpu_disposition_244.v1.json", receipt)
    print(json.dumps({key: receipt[key] for key in ("status", "candidate_count", "admitted", "pre_gpu_rejected_with_evidence", "unresolved", "admitted_by_category", "admitted_by_shard")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
