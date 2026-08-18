#!/usr/bin/env python3
"""Build frozen local ledgers, group splits, candidate review and GPU queues.

This program is deliberately CPU-only.  It never opens a camera, serial port,
GPIO, network socket, Keil project, or accelerator.  It consumes only files
already present in the approved Forge200 candidate directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TRUTH_CLASSES = [
    "TEAM_MEASURED",
    "OPEN_EXPERIMENT",
    "LITERATURE_CURATED_EXPERIMENT",
    "ANON_PRODUCTION",
    "OPEN_COMPUTED_DFT",
    "PHYSICS_SIM",
    "STRUCTURE_DERIVED",
    "CONTROLLED_FIXTURE",
    "SYNTHETIC_AUGMENTATION",
    "TEACHER_CANDIDATE",
    "METADATA_ONLY",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if not rows:
        raise RuntimeError(f"empty TSV: {path}")
    return rows


def split_for_group(namespace: str, group: str) -> str:
    bucket = int(hashlib.sha256(f"{namespace}:{group}".encode()).hexdigest()[:8], 16) % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def parse_elements(formula: str) -> str:
    elements = sorted(set(re.findall(r"[A-Z][a-z]?", formula or "")))
    return "-".join(elements) if elements else "UNKNOWN"


def review_tasks(root: Path) -> dict[str, Any]:
    pool_path = root / "contracts" / "candidate_pool_244.v1.tsv"
    task_path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    pool = read_tsv(pool_path)
    tasks = read_tsv(task_path)
    errors: list[str] = []
    required = [
        "candidate_id",
        "objective_id",
        "input_contract",
        "target_label",
        "source_gate",
        "baseline",
        "primary_metric",
        "parameter_cap",
        "consumer",
        "replacement_for",
        "status",
        "authority",
    ]
    if len(pool) != 244 or len(tasks) != 244:
        errors.append(f"expected 244 pool/tasks, got {len(pool)}/{len(tasks)}")
    pool_by_id = {row["candidate_id"]: row for row in pool}
    if len(pool_by_id) != len(pool):
        errors.append("duplicate candidate_id in candidate pool")
    if len({row["objective_id"] for row in tasks}) != len(tasks):
        errors.append("duplicate objective_id in task contracts")
    row_receipts: list[dict[str, str]] = []
    for row in tasks:
        cid = row.get("candidate_id", "")
        missing = [field for field in required if not row.get(field, "").strip()]
        if missing:
            errors.append(f"{cid}: blank fields {','.join(missing)}")
        if row.get("authority") != "0":
            errors.append(f"{cid}: authority is not zero")
        if cid not in pool_by_id:
            errors.append(f"{cid}: absent from candidate pool")
        else:
            pool_row = pool_by_id[cid]
            if row["replacement_for"] != pool_row["replacement_for"]:
                errors.append(f"{cid}: replacement whitelist mismatch")
            if pool_row["authority"] != "0":
                errors.append(f"{cid}: pool authority is not zero")
        row_receipts.append(
            {
                "candidate_id": cid,
                "objective_id": row.get("objective_id", ""),
                "row_sha256": sha256_bytes(canonical_bytes(row)),
            }
        )
    category_counts = Counter(row["category"] for row in pool)
    primary_count = sum(row["replacement_for"] == "PRIMARY_TARGET" for row in pool)
    spare_count = len(pool) - primary_count
    if category_counts != Counter({"PREDICTIVE": 148, "GENERATIVE": 48, "SUPPORT": 48}):
        errors.append(f"category counts mismatch: {dict(category_counts)}")
    if (primary_count, spare_count) != (170, 74):
        errors.append(f"target/spare mismatch: {primary_count}/{spare_count}")
    receipt = {
        "schema": "cimc.forge200.task-contract-review.v1",
        "status": "PASS" if not errors else "FAIL",
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_pool_sha256": sha256_file(pool_path),
        "candidate_task_contracts_sha256": sha256_file(task_path),
        "candidate_count": len(pool),
        "task_contract_count": len(tasks),
        "primary_targets": primary_count,
        "spares": spare_count,
        "categories": dict(sorted(category_counts.items())),
        "unique_candidate_ids": len(pool_by_id),
        "unique_objective_ids": len({row["objective_id"] for row in tasks}),
        "authority_nonzero": sum(row["authority"] != "0" for row in tasks),
        "blank_field_rows": sum(any(not value.strip() for value in row.values()) for row in tasks),
        "errors": errors,
        "row_receipts": row_receipts,
    }
    write_json(root / "evidence" / "task_contract_review.v1.json", receipt)
    if errors:
        raise RuntimeError("task contract review failed: " + "; ".join(errors[:8]))
    return receipt


def build_source_ledgers(root: Path) -> dict[str, Any]:
    raw_specs = [
        {
            "source_id": "jarvis_dft_v11",
            "path": "data/raw/jarvis_dft_v11/jdft_3d-9-24-2025.json.zip",
            "expected_bytes": 48_447_610,
            "expected_sha256": "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5",
            "pid": "10.6084/m9.figshare.6815699.v11",
            "canonical_url": "https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699",
            "version": "11:file-64391379",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "truth_class": "OPEN_COMPUTED_DFT",
            "training_allowed": True,
            "rag_allowed": True,
            "scope": "DFT computational properties only; never experimental truth",
            "metadata_snapshot": "data/metadata/figshare_6815699_v11.json",
        },
        {
            "source_id": "ipop_v3",
            "path": "data/raw/ipop_v3/Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv",
            "expected_bytes": 987_261,
            "expected_sha256": "9ebbee222e7b21faac0919761fbc8ee76c304c8ae3c8c89f0ab384a3d53b2924",
            "pid": "10.6084/m9.figshare.24771186.v1",
            "canonical_url": "https://figshare.com/articles/dataset/24771186",
            "version": "1:file-43535559",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "truth_class": "LITERATURE_CURATED_EXPERIMENT",
            "training_allowed": True,
            "rag_allowed": True,
            "scope": "Literature-curated optical properties; preserve upstream DOI and conditions",
            "metadata_snapshot": "data/metadata/figshare_24771186_v1.json",
        },
        {
            "source_id": "uci_secom",
            "path": "data/raw/uci_secom/secom.zip",
            "expected_bytes": 1_964_989,
            "expected_sha256": "eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e",
            "pid": "10.24432/C54305",
            "canonical_url": "https://archive.ics.uci.edu/dataset/179/secom",
            "version": "official-record-fetched-2026-08-01",
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "truth_class": "ANON_PRODUCTION",
            "training_allowed": True,
            "rag_allowed": True,
            "scope": "One anonymous pass/fail target; no claim of live fab integration",
            "metadata_snapshot": "data/metadata/uci_secom_record_20260801.html",
        },
    ]
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for spec in raw_specs:
        path = root / spec["path"]
        metadata = root / spec["metadata_snapshot"]
        actual_bytes = path.stat().st_size if path.exists() else -1
        actual_sha = sha256_file(path) if path.exists() else ""
        if actual_bytes != spec["expected_bytes"] or actual_sha != spec["expected_sha256"]:
            errors.append(f"{spec['source_id']}: raw artifact mismatch")
        if not metadata.exists():
            errors.append(f"{spec['source_id']}: metadata snapshot missing")
        records.append(
            {
                **{key: value for key, value in spec.items() if not key.startswith("expected_")},
                "artifact_bytes": actual_bytes,
                "artifact_sha256": actual_sha,
                "metadata_snapshot_sha256": sha256_file(metadata) if metadata.exists() else "",
                "decision": "TRAIN_RAG_GO" if not errors else "FAIL_CLOSED",
            }
        )
    team_evidence = root.parents[1] / "CIMC" / "evidence" / "hardware_bringup" / "finals_hw_full_integration_20260730"
    team_records = []
    if team_evidence.is_dir():
        for path in sorted(item for item in team_evidence.rglob("*") if item.is_file()):
            team_records.append(
                {
                    "path": str(path.relative_to(team_evidence)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "truth_class": "METADATA_ONLY",
                    "training_label_present": False,
                }
            )
    else:
        errors.append("cimc_team_hardware_evidence_20260730: path missing")
    team_manifest_path = root / "data" / "ledgers" / "team_hardware_evidence_records.v1.json"
    team_manifest = {
        "schema": "cimc.forge200.team-evidence-records.v1",
        "source_id": "cimc_team_hardware_evidence_20260730",
        "status": "METADATA_ONLY_NO_TRAINING_LABELS" if team_records else "FAIL",
        "records": team_records,
        "record_count": len(team_records),
        "bytes": sum(item["bytes"] for item in team_records),
        "content_root_sha256": sha256_bytes(canonical_bytes(team_records)),
    }
    write_json(team_manifest_path, team_manifest)
    records.extend(
        [
            {
                "source_id": "cimc_team_hardware_evidence_20260730",
                "path": "CIMC/evidence/hardware_bringup/finals_hw_full_integration_20260730",
                "pid": "TEAM_LEDGER:FINALS_HW_20260730",
                "version": "immutable-baseline-manifest:C811AF...C4ABB",
                "license": "TEAM_OWNED_PRIVATE",
                "license_url": "internal://team-ownership-ledger",
                "truth_class": "METADATA_ONLY",
                "training_allowed": False,
                "rag_allowed": True,
                "scope": "Only records with run/session binding; build logs alone are METADATA_ONLY",
                "decision": "NO_LABELED_TRAINING_RECORDS",
                "record_manifest": "data/ledgers/team_hardware_evidence_records.v1.json",
                "record_manifest_sha256": sha256_file(team_manifest_path),
            },
            {
                "source_id": "forge200_controlled_fixture_v1",
                "path": "data/fixtures",
                "pid": "CIMC:FORGE200:FIXTURE:V1",
                "version": "generated-by-hashed-tooling",
                "license": "TEAM_OWNED_PRIVATE",
                "license_url": "internal://fixture-contract",
                "truth_class": "CONTROLLED_FIXTURE",
                "training_allowed": False,
                "rag_allowed": False,
                "scope": "Toolchain dry-run and throughput pilot only; never model quality",
                "decision": "PIPELINE_ONLY",
            },
            {
                "source_id": "teacher_candidate_pool",
                "path": "not-materialized",
                "pid": "CIMC:TEACHER:CANDIDATES",
                "version": "none",
                "license": "SOURCE_BOUND_PER_RECORD_REQUIRED",
                "license_url": "internal://teacher-governance",
                "truth_class": "TEACHER_CANDIDATE",
                "training_allowed": False,
                "rag_allowed": False,
                "scope": "Never ground truth; private environment variables only; no API call in E0-E2",
                "decision": "NO_TEACHER_GENERATION_PERFORMED",
            },
            {
                "source_id": "standards_metadata_registry",
                "path": "metadata-only",
                "pid": "ISO/ASTM/IEC/SEMI/JEDEC/IPC/IEEE identifiers",
                "version": "public-title-number-scope-only",
                "license": "PROPRIETARY_METADATA_ONLY",
                "license_url": "publisher-specific-public-scope-pages",
                "truth_class": "METADATA_ONLY",
                "training_allowed": False,
                "rag_allowed": False,
                "scope": "No standards full text, tables, figures, embeddings, prompts or summaries",
                "decision": "METADATA_ONLY",
            },
        ]
    )
    source_ledger = {
        "schema": "cimc.forge200.source-ledger.v1",
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CIMC",
        "runtime_dependency_projects": [],
        "records": records,
        "errors": errors,
    }
    write_json(root / "data" / "ledgers" / "source_ledger.v1.json", source_ledger)
    license_ledger = {
        "schema": "cimc.forge200.license-ledger.v1",
        "status": source_ledger["status"],
        "policy": {
            "publicly_visible_is_not_a_license": True,
            "teacher_output_is_not_ground_truth": True,
            "standards_fulltext_allowed": False,
            "unknown_license_action": "QUARANTINE",
            "license_change_action": "QUARANTINE_AND_REHASH",
        },
        "records": [
            {
                "source_id": item["source_id"],
                "license": item["license"],
                "license_url": item["license_url"],
                "training_allowed": item["training_allowed"],
                "rag_allowed": item["rag_allowed"],
                "decision": item["decision"],
                "metadata_snapshot_sha256": item.get("metadata_snapshot_sha256"),
            }
            for item in records
        ],
    }
    write_json(root / "data" / "ledgers" / "license_ledger.v1.json", license_ledger)
    truth_ledger = {
        "schema": "cimc.forge200.truth-ledger.v1",
        "status": "FROZEN",
        "classes": TRUTH_CLASSES,
        "report_metrics_separately": True,
        "forbidden_promotions": [
            "TEACHER_CANDIDATE_TO_GROUND_TRUTH",
            "OPEN_COMPUTED_DFT_TO_EXPERIMENT",
            "CONTROLLED_FIXTURE_TO_TEAM_MEASURED",
            "SYNTHETIC_AUGMENTATION_TO_ANON_PRODUCTION",
            "METADATA_ONLY_TO_TRAINING_LABEL",
        ],
    }
    write_json(root / "data" / "ledgers" / "truth_ledger.v1.json", truth_ledger)
    if errors:
        raise RuntimeError("source ledger failed: " + "; ".join(errors))
    return source_ledger


def build_jarvis_split(root: Path) -> dict[str, Any]:
    archive = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    output = root / "data" / "splits" / "jarvis_dft_v11.assignments.tsv"
    with zipfile.ZipFile(archive) as zf:
        member = zf.namelist()[0]
        records = json.loads(zf.read(member))
    assignments: list[tuple[str, str, str, str]] = []
    ids: set[str] = set()
    duplicate_ids = 0
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    for row in records:
        jid = str(row.get("jid", ""))
        formula = str(row.get("formula", ""))
        group = parse_elements(formula)
        split = split_for_group("jarvis_chemical_system", group)
        if jid in ids:
            duplicate_ids += 1
        ids.add(jid)
        split_groups[split].add(group)
        split_counts[split] += 1
        assignments.append((jid, formula, group, split))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["jid", "formula", "chemical_system_group", "split"])
        writer.writerows(assignments)
    overlap = (
        split_groups["train"] & split_groups["validation"]
        | split_groups["train"] & split_groups["test"]
        | split_groups["validation"] & split_groups["test"]
    )
    return {
        "source_id": "jarvis_dft_v11",
        "records": len(records),
        "groups": len(set(group for _, _, group, _ in assignments)),
        "counts": dict(split_counts),
        "duplicate_ids": duplicate_ids,
        "cross_split_group_overlap": len(overlap),
        "assignment_path": str(output.relative_to(root)).replace("\\", "/"),
        "assignment_sha256": sha256_file(output),
        "truth_class": "OPEN_COMPUTED_DFT",
        "group_key": "strict_chemical_system",
    }


def build_ipop_split(root: Path) -> dict[str, Any]:
    source = root / "data" / "raw" / "ipop_v3" / "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    output = root / "data" / "splits" / "ipop_v3.assignments.tsv"
    duplicate_output = root / "data" / "splits" / "ipop_v3.exact_duplicates.tsv"
    assignments: list[tuple[str, str, str, str, str]] = []
    duplicates: list[tuple[str, str, str]] = []
    split_groups: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    first_tag_by_hash: dict[str, str] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            tag = str(row.get("Tag", "")).strip()
            row_hash = sha256_bytes(canonical_bytes(row))
            if row_hash in first_tag_by_hash:
                duplicates.append((tag, first_tag_by_hash[row_hash], row_hash))
                continue
            first_tag_by_hash[row_hash] = tag
            host = str(row.get("Host", "UNKNOWN")).strip()
            dopant = str(row.get("1st dopant", "UNKNOWN")).strip()
            doi = str(row.get("Reference", "UNKNOWN")).strip().lower()
            group = f"{doi}|{host}|{dopant}"
            split = split_for_group("ipop_doi_host_dopant", group)
            split_groups[split].add(group)
            split_counts[split] += 1
            assignments.append((tag, doi, host, dopant, split))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["tag", "doi_family", "host_family", "dopant_family", "split"])
        writer.writerows(assignments)
    with duplicate_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["duplicate_tag", "canonical_tag", "normalized_row_sha256"])
        writer.writerows(duplicates)
    overlap = (
        split_groups["train"] & split_groups["validation"]
        | split_groups["train"] & split_groups["test"]
        | split_groups["validation"] & split_groups["test"]
    )
    return {
        "source_id": "ipop_v3",
        "records": len(assignments),
        "groups": len(set(f"{doi}|{host}|{dopant}" for _, doi, host, dopant, _ in assignments)),
        "counts": dict(split_counts),
        "exact_duplicate_rows_quarantined": len(duplicates),
        "cross_split_group_overlap": len(overlap),
        "assignment_path": str(output.relative_to(root)).replace("\\", "/"),
        "assignment_sha256": sha256_file(output),
        "duplicate_ledger_path": str(duplicate_output.relative_to(root)).replace("\\", "/"),
        "duplicate_ledger_sha256": sha256_file(duplicate_output),
        "truth_class": "LITERATURE_CURATED_EXPERIMENT",
        "group_key": "doi_family+host_family+dopant_family",
    }


def build_secom_split(root: Path) -> dict[str, Any]:
    archive = root / "data" / "raw" / "uci_secom" / "secom.zip"
    output = root / "data" / "splits" / "uci_secom.assignments.tsv"
    with zipfile.ZipFile(archive) as zf:
        lines = zf.read("secom_labels.data").decode("utf-8").splitlines()
    parsed: list[tuple[int, int, str, datetime]] = []
    for index, line in enumerate(lines):
        label_text, timestamp_text = line.split(maxsplit=1)
        timestamp_text = timestamp_text.strip().strip('"')
        stamp = datetime.strptime(timestamp_text, "%d/%m/%Y %H:%M:%S")
        parsed.append((index, int(label_text), timestamp_text, stamp))
    parsed.sort(key=lambda item: item[3])
    n = len(parsed)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)
    assignments: list[tuple[int, int, str, str, str]] = []
    counts: Counter[str] = Counter()
    label_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for rank, (index, label, timestamp_text, stamp) in enumerate(parsed):
        split = "train" if rank < train_end else "validation" if rank < validation_end else "test"
        time_group = stamp.strftime("%Y-%m-%d")
        counts[split] += 1
        label_counts[split][label] += 1
        assignments.append((index, label, timestamp_text, time_group, split))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_index", "pass_fail_label", "timestamp", "time_group", "split"])
        writer.writerows(assignments)
    return {
        "source_id": "uci_secom",
        "records": len(assignments),
        "counts": dict(counts),
        "label_counts": {key: dict(value) for key, value in label_counts.items()},
        "cross_split_row_overlap": 0,
        "chronological_order": True,
        "group_id_limit": "anonymous dataset has no public lot/tool/wafer ID; chronological test-point split is the strongest available",
        "assignment_path": str(output.relative_to(root)).replace("\\", "/"),
        "assignment_sha256": sha256_file(output),
        "truth_class": "ANON_PRODUCTION",
        "group_key": "chronological_test_point_block",
    }


def build_splits(root: Path) -> dict[str, Any]:
    sources = [build_jarvis_split(root), build_ipop_split(root), build_secom_split(root)]
    errors = []
    for source in sources:
        if source.get("cross_split_group_overlap", source.get("cross_split_row_overlap", 0)) != 0:
            errors.append(f"{source['source_id']}: split leakage")
        if min(source["counts"].values()) <= 0:
            errors.append(f"{source['source_id']}: empty split")
    manifest = {
        "schema": "cimc.forge200.split-ledger.v1",
        "status": "PASS" if not errors else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_before_augmentation": True,
        "fit_train_only": ["imputer", "scaler", "feature_selector", "pca", "calibrator", "augmentation_parameters"],
        "same_run_post_sinter_leakage_forbidden": True,
        "sources": sources,
        "errors": errors,
    }
    write_json(root / "data" / "ledgers" / "split_ledger.v1.json", manifest)
    leakage = {
        "schema": "cimc.forge200.leakage-audit.v1",
        "status": "PASS" if not errors else "FAIL",
        "global_entity_namespaces_are_source_scoped": True,
        "cross_split_group_overlap_total": sum(
            item.get("cross_split_group_overlap", item.get("cross_split_row_overlap", 0)) for item in sources
        ),
        "ipop_exact_duplicates_quarantined": next(
            item["exact_duplicate_rows_quarantined"] for item in sources if item["source_id"] == "ipop_v3"
        ),
        "secom_group_id_limit": "GROUP_ID_LIMITED_CHRONOLOGICAL_SPLIT",
        "same_run_future_metrology_forbidden": ["XRD", "PL", "SEM", "EDS", "POST_RUN_QUALITY"],
        "pair_patch_chunk_augmentation_after_split_only": True,
        "teacher_may_view_test": False,
        "fit_train_only": manifest["fit_train_only"],
    }
    write_json(root / "data" / "ledgers" / "leakage_audit.v1.json", leakage)
    if errors:
        raise RuntimeError("split build failed: " + "; ".join(errors))
    return manifest


def select_data_binding(candidate_id: str) -> dict[str, str]:
    category = candidate_id.split("-")[1]
    number = int(candidate_id.split("-")[2])
    if category == "P":
        # Only targets with a direct, semantics-matching field in the pinned
        # JARVIS artifact are admitted.  Computed properties may not satisfy
        # contracts that explicitly require experimental truth, and proxy
        # fields (for example max_dij for d33) are not silently relabelled.
        if number in {69, 71, 72, 74, 75, 76, 77, 78, 86, 142}:
            return {"source_family": "jarvis_dft_v11", "truth_class": "OPEN_COMPUTED_DFT", "full_data_state": "MATERIALIZED"}
        if number in {67, 68, 70, 73, 84, 140, 141, 143, 144, 145}:
            return {"source_family": "jarvis_dft_v11", "truth_class": "OPEN_COMPUTED_DFT", "full_data_state": "FAIL_CLOSED_TARGET_SEMANTICS_NOT_MATERIALIZED"}
        if 42 <= number <= 66 or 79 <= number <= 85 or number in {138, 139}:
            return {"source_family": "ipop_v3+team_l2", "truth_class": "LITERATURE_CURATED_EXPERIMENT", "full_data_state": "MATERIALIZED_WITH_RECORD_LEVEL_LABEL_GATE"}
        if number == 87:
            return {"source_family": "uci_secom", "truth_class": "ANON_PRODUCTION", "full_data_state": "MATERIALIZED"}
        if number <= 41 or 113 <= number <= 137:
            return {"source_family": "cimc_team_hardware_evidence_20260730", "truth_class": "TEAM_MEASURED", "full_data_state": "RECORD_LEVEL_MANIFEST_REQUIRED"}
        return {"source_family": "task_specific_open_or_team_source", "truth_class": "METADATA_ONLY", "full_data_state": "FAIL_CLOSED_UNTIL_ARTIFACT_HASH"}
    if category == "G":
        return {"source_family": "licensed_multidomain_corpus", "truth_class": "METADATA_ONLY", "full_data_state": "CORPUS_BUILD_REQUIRED_TEACHER_NOT_GROUND_TRUTH"}
    return {"source_family": "licensed_multidomain_judgments", "truth_class": "METADATA_ONLY", "full_data_state": "JUDGMENT_BUILD_REQUIRED"}


def build_bindings_and_queue(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    pool = read_tsv(root / "contracts" / "candidate_pool_244.v1.tsv")
    tasks = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    task_by_id = {row["candidate_id"]: row for row in tasks}
    bindings = []
    queue_a = []
    queue_b = []
    estimate_total_minutes = {"GPU_A": 0.0, "GPU_B": 0.0}
    for index, candidate in enumerate(pool):
        cid = candidate["candidate_id"]
        task = task_by_id[cid]
        binding = select_data_binding(cid)
        bindings.append({"candidate_id": cid, "objective_id": task["objective_id"], **binding})
        if candidate["category"] == "PREDICTIVE":
            gpu = "GPU_A"
            minutes = 2.4 if candidate["replacement_for"] == "PRIMARY_TARGET" else 1.8
            batch_key = f"PRED_BATCH_{(index // 12):02d}"
        else:
            gpu = "GPU_B"
            if candidate["category"] == "GENERATIVE":
                minutes = 9.0 if candidate["replacement_for"] == "PRIMARY_TARGET" else 6.0
                batch_key = f"GEN_BATCH_{(index // 4):02d}"
            else:
                minutes = 2.6 if candidate["replacement_for"] == "PRIMARY_TARGET" else 2.0
                batch_key = f"SUP_BATCH_{(index // 8):02d}"
        item = {
            "job_id": f"FORGE200-{cid}",
            "candidate_id": cid,
            "objective_id": task["objective_id"],
            "category": candidate["category"],
            "target_slot": candidate["target_slot"],
            "replacement_for": candidate["replacement_for"],
            "authority": 0,
            "gpu_shard": gpu,
            "batch_key": batch_key,
            "estimated_gpu_minutes": minutes,
            "estimated_vram_gib": 18 if candidate["category"] == "GENERATIVE" else 8 if candidate["category"] == "PREDICTIVE" else 10,
            "seeds": [20260801, 20260802, 20260803],
            "checkpoint_interval_steps": 200,
            "heartbeat_seconds": 30,
            "timeout_minutes": 90 if candidate["category"] == "GENERATIVE" else 30,
            "max_retries": 2,
            "retain_all_completed_artifacts": True,
            "data_binding": binding,
            "admission_state": "ADMITTED" if binding["full_data_state"] == "MATERIALIZED" else "BLOCKED_PRE_GPU",
            "admission_checks": ["PROJECT_CIMC", "SOURCE_LICENSE", "SPLIT_HASH", "NO_TEST_LEAKAGE", "ENGINE_ABI", "AUTHORITY_ZERO"],
        }
        (queue_a if gpu == "GPU_A" else queue_b).append(item)
        estimate_total_minutes[gpu] += minutes
    binding_ledger = {
        "schema": "cimc.forge200.task-source-bindings.v1",
        "status": "FROZEN_FAIL_CLOSED",
        "records": bindings,
        "materialized_direct_count": sum(item["full_data_state"] == "MATERIALIZED" for item in bindings),
        "gpu_admitted_count": sum(item["full_data_state"] == "MATERIALIZED" for item in bindings),
        "gpu_blocked_count": sum(item["full_data_state"] != "MATERIALIZED" for item in bindings),
        "record_or_corpus_build_required_count": sum("REQUIRED" in item["full_data_state"] for item in bindings),
        "fail_closed_count": sum("FAIL_CLOSED" in item["full_data_state"] for item in bindings),
    }
    write_json(root / "data" / "ledgers" / "task_source_bindings.v1.json", binding_ledger)
    queue = {
        "schema": "cimc.forge200.dual-5090-queue.v1",
        "status": "PREPARED_RECOVERABLE_PILOT_FIRST",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "CIMC",
        "candidate_jobs": len(pool),
        "admitted_jobs": sum(item["admission_state"] == "ADMITTED" for item in queue_a + queue_b),
        "blocked_jobs": sum(item["admission_state"] != "ADMITTED" for item in queue_a + queue_b),
        "gpu_a_jobs": len(queue_a),
        "gpu_b_jobs": len(queue_b),
        "estimate_gpu_minutes": estimate_total_minutes,
        "estimated_wall_hours_after_batching": round(max(estimate_total_minutes.values()) / 60.0, 2),
        "pilot": {"duration_minutes_min": 30, "duration_minutes_max": 60, "eta_reestimate_after_hours": 2, "pause_if_wall_hours_exceed": 10},
        "cache_budget_bytes": 64 * 1024 * 1024 * 1024,
        "checkpoint_budget_bytes": 110 * 1024 * 1024 * 1024,
        "artifact_return_policy": "HASH_MANIFEST_AND_CONTINUOUS_DOWNLOAD",
        "no_cross_public_network_ddp": True,
        "jobs": {"GPU_A": queue_a, "GPU_B": queue_b},
    }
    write_json(root / "queue" / "dual_5090_queue.v1.json", queue)
    write_json(root / "queue" / "gpu_a.queue.json", {"schema": queue["schema"], "shard": "GPU_A", "jobs": queue_a})
    write_json(root / "queue" / "gpu_b.queue.json", {"schema": queue["schema"], "shard": "GPU_B", "jobs": queue_b})
    return binding_ledger, queue


def build_release_manifest(root: Path, outputs: Iterable[Path]) -> dict[str, Any]:
    records = []
    for path in sorted(set(outputs)):
        records.append(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "schema": "cimc.forge200.local-readiness-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": records,
    }
    manifest["content_root_sha256"] = sha256_bytes(canonical_bytes(records))
    write_json(root / "evidence" / "local_readiness_manifest.v1.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    review = review_tasks(root)
    source = build_source_ledgers(root)
    splits = build_splits(root)
    bindings, queue = build_bindings_and_queue(root)
    outputs = [
        root / "evidence" / "task_contract_review.v1.json",
        root / "data" / "ledgers" / "source_ledger.v1.json",
        root / "data" / "ledgers" / "license_ledger.v1.json",
        root / "data" / "ledgers" / "truth_ledger.v1.json",
        root / "data" / "ledgers" / "team_hardware_evidence_records.v1.json",
        root / "data" / "ledgers" / "split_ledger.v1.json",
        root / "data" / "ledgers" / "leakage_audit.v1.json",
        root / "data" / "ledgers" / "task_source_bindings.v1.json",
        root / "data" / "splits" / "jarvis_dft_v11.assignments.tsv",
        root / "data" / "splits" / "ipop_v3.assignments.tsv",
        root / "data" / "splits" / "ipop_v3.exact_duplicates.tsv",
        root / "data" / "splits" / "uci_secom.assignments.tsv",
        root / "queue" / "dual_5090_queue.v1.json",
        root / "queue" / "gpu_a.queue.json",
        root / "queue" / "gpu_b.queue.json",
    ]
    manifest = build_release_manifest(root, outputs)
    summary = {
        "status": "PASS",
        "task_contracts": review["task_contract_count"],
        "sources": len(source["records"]),
        "split_sources": len(splits["sources"]),
        "bindings": len(bindings["records"]),
        "queue_jobs": queue["candidate_jobs"],
        "estimated_wall_hours": queue["estimated_wall_hours_after_batching"],
        "content_root_sha256": manifest["content_root_sha256"],
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
