#!/usr/bin/env python3
"""Freeze S041 from pre-existing validation history with model-family holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    output: dict[str, float] = {}
    if isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)) and math.isfinite(float(value)):
        output[prefix] = float(value)
    elif isinstance(value, dict):
        for key, child in value.items():
            output.update(flatten_numeric(child, f"{prefix}.{key}" if prefix else str(key)))
    return output


def choose_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        records,
        key=lambda item: (
            item.get("contract_baseline_evaluation") is not None,
            int(item.get("promotion_receipt", {}).get("parameter_count", 0)),
            int(item.get("precedence", 0)),
        ),
    )


def split_for(candidate_id: str) -> int:
    bucket = int(hashlib.sha256(candidate_id.encode()).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def seed_label(record: dict[str, Any], seed_report: dict[str, Any], seed_index: int) -> int:
    contract_eval = record.get("contract_baseline_evaluation") or {}
    candidate_scores = contract_eval.get("candidate_three_seed_primary_composite") or []
    baseline_metrics = contract_eval.get("baseline_metrics") or {}
    if seed_index < len(candidate_scores) and "primary_composite" in baseline_metrics:
        return int(float(candidate_scores[seed_index]) > float(baseline_metrics["primary_composite"]))
    evaluation = record["evaluation"]
    baseline = evaluation.get("baseline") or {}
    test = seed_report.get("test") or {}
    task_kind = str(record.get("source_manifest", {}).get("task_kind", ""))
    if task_kind == "regression" and "mae" in test and "mae" in baseline:
        return int(float(test["mae"]) < float(baseline["mae"]))
    if task_kind in {"classification", "contrastive_embedding"} and "balanced_accuracy" in test and "balanced_accuracy" in baseline:
        return int(float(test["balanced_accuracy"]) > float(baseline["balanced_accuracy"]))
    if "token_nll" in test and "token_nll" in baseline:
        return int(float(test["token_nll"]) < float(baseline["token_nll"]))
    if "answer_token_nll" in test and "unigram_answer_token_nll" in baseline:
        return int(float(test["answer_token_nll"]) < float(baseline["unigram_answer_token_nll"]))
    return int(bool(evaluation.get("baseline_proxy_pass", False)))


def validation_quality(record: dict[str, Any], seed_report: dict[str, Any]) -> float:
    generation = seed_report.get("generation") or {}
    validation = seed_report.get("validation") or {}
    if "primary_composite" in generation:
        return float(generation["primary_composite"])
    if "balanced_accuracy" in validation:
        return float(validation["balanced_accuracy"])
    if "mae" in validation:
        return -float(validation["mae"])
    if "answer_token_nll" in validation:
        return -float(validation["answer_token_nll"])
    if "token_nll" in validation:
        return -float(validation["token_nll"])
    values = list(flatten_numeric(validation).values())
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    history_paths = [
        root / "artifacts" / "cloud5090" / "corrective_transfers" / "frozen_validation_history_A.json",
        root / "artifacts" / "cloud5090" / "corrective_transfers" / "frozen_validation_history_B.json",
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in history_paths:
        history = json.loads(path.read_text(encoding="utf-8"))
        if history["status"] != "PASS" or history["authority"] != 0:
            raise RuntimeError("HISTORY_GATE")
        for record in history["records"]:
            grouped.setdefault(record["candidate_id"], []).append(record)
    selected = {candidate_id: choose_record(records) for candidate_id, records in grouped.items()}
    feature_rows: list[dict[str, float]] = []
    labels, groups, splits, qualities, seeds = [], [], [], [], []
    for candidate_id, record in sorted(selected.items()):
        evaluation = record["evaluation"]
        receipt = record["promotion_receipt"]
        source = record["source_manifest"]
        category = candidate_id[5]
        for seed_index, seed_report in enumerate(evaluation.get("seed_reports", [])):
            values: dict[str, float] = {
                "category_P": float(category == "P"),
                "category_G": float(category == "G"),
                "category_S": float(category == "S"),
                "log_parameters": math.log1p(float(receipt.get("parameter_count", 0))),
                "log_package_bytes": math.log1p(float(receipt.get("package", {}).get("bytes", 0))),
                "runtime_log_seconds": math.log1p(float(receipt.get("runtime_seconds", 0))),
                "source_records_log": math.log1p(float(source.get("records", 0))),
                "source_features_log": math.log1p(float(source.get("features", 0))),
                "authority": float(receipt.get("authority", 0)),
                "board_accepted": float(bool(receipt.get("board_accepted", False))),
            }
            for prefix in ("validation", "pre_qat_validation", "generation"):
                values.update({f"seed.{key}": value for key, value in flatten_numeric(seed_report.get(prefix, {}), prefix).items()})
            values.update({f"baseline.{key}": value for key, value in flatten_numeric(evaluation.get("baseline", {})).items()})
            values.update({f"calibration.{key}": value for key, value in flatten_numeric(record.get("calibration") or {}).items()})
            feature_rows.append(values)
            labels.append(seed_label(record, seed_report, seed_index))
            groups.append(candidate_id)
            splits.append(split_for(candidate_id))
            qualities.append(validation_quality(record, seed_report))
            seeds.append(int(seed_report.get("seed", 20260801 + seed_index)))
    feature_names = sorted({key for row in feature_rows for key in row})
    x = np.asarray([[row.get(key, 0.0) for key in feature_names] for row in feature_rows], dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    split = np.asarray(splits, dtype=np.int8)
    quality = np.asarray(qualities, dtype=np.float32)
    train_quality = np.sort(quality[split == 0])
    baseline_probability = np.searchsorted(train_quality, quality, side="right") / max(len(train_quality), 1)
    baseline_probability = np.clip(baseline_probability, 0.02, 0.98).astype(np.float32)
    if min(np.sum(split == code) for code in (0, 1, 2)) < 12:
        raise RuntimeError("HOLDOUT_TOO_SMALL")
    output = root / "data" / "staged_postgpu" / "CAND-S-041.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        groups=np.asarray(groups),
        split=split,
        seed=np.asarray(seeds, dtype=np.int64),
        validation_quality=quality,
        baseline_probability=baseline_probability,
        candidate_id=np.asarray("CAND-S-041"),
        task_kind=np.asarray("binary_probability"),
        authority=np.asarray(0, dtype=np.int8),
    )
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == "CAND-S-041")
    counts = {name: int(np.sum(split == code)) for code, name in enumerate(("train", "validation", "test"))}
    metadata = {
        "schema": "cimc.forge200.postgpu-competence-history.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-S-041",
        "records": len(y),
        "source_model_families": len(selected),
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "counts": counts,
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "split_unit": "candidate_id_model_family",
        "cross_split_group_overlap": 0,
        "label_rule": "frozen_validation_features_predict_heldout_test_baseline_gate",
        "baseline_contract": contract["baseline"],
        "primary_metric_contract": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "history_sources": [{"path": str(path.relative_to(root)).replace("\\", "/"), "sha256": sha256_file(path)} for path in history_paths],
        "truth_class": "POST_GPU_FROZEN_VALIDATION_TO_HELDOUT_TEST_QUALITY_GATE",
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    write_json(root / "evidence" / "s041_competence_history_staging.v1.json", metadata)
    print(json.dumps({"status": "PASS", "records": len(y), "models": len(selected), "features": len(feature_names), "positive": int(y.sum()), "negative": int(len(y)-y.sum()), "counts": counts, "sha256": metadata["sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
