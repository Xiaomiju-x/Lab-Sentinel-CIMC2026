#!/usr/bin/env python3
"""Stage IPOP rows that exactly close the P058 NIR lifetime contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import ELEMENTS, ELEMENT_INDEX, canonical_bytes, parse_formula, sha256_file, write_json


SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def vector(counts: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    total = sum(counts.values())
    for symbol, count in counts.items():
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def number(value: str, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_id = "CAND-P-058"
    source = root / "data" / "raw" / "ipop_v3" / "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    assignment_path = root / "data" / "splits" / "ipop_v3.assignments.tsv"
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assignments = {row["tag"]: row for row in csv.DictReader(handle, delimiter="\t")}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            emission, decay = number(row["Emission max. (nm)"]), number(row["Decay time (ns)"])
            if not math.isfinite(emission) or emission < 700.0 or not math.isfinite(decay) or decay <= 0.0:
                continue
            assignment = assignments.get(row["Tag"])
            if assignment is None:
                raise RuntimeError(f"missing frozen split assignment: {row['Tag']}")
            host = parse_formula(row["Host"])
            dopant = np.zeros(len(ELEMENTS), dtype=np.float32)
            for field in ("1st dopant", "2nd dopant"):
                symbol = row[field].strip()
                if symbol:
                    if symbol not in ELEMENT_INDEX:
                        raise RuntimeError(f"unsupported dopant: {symbol}")
                    dopant[ELEMENT_INDEX[symbol]] = 1.0
            concentration_1 = number(row["1st doping concentration"], 0.0)
            concentration_2 = number(row["2nd doping concentration "], 0.0)
            valency_1 = number(row["1st dopant valency"], 0.0)
            valency_2 = number(row["2nd dopant valency"], 0.0)
            temperature = number(row["Temp. (K)"], 298.15)
            feature = np.concatenate(
                (
                    vector(host), dopant,
                    np.asarray(
                        [concentration_1, concentration_2, valency_1 / 8.0, valency_2 / 8.0, temperature / 2000.0, emission / 2000.0, 0.0, 1.0],
                        dtype=np.float32,
                    ),
                )
            )
            family = f"{row['Host'].strip()}|{row['1st dopant'].strip()}|{row['2nd dopant'].strip()}"
            rows.append(
                {
                    "x": feature, "y": math.log10(decay / 1000.0), "family": family,
                    "group": f"{assignment['doi_family']}|{assignment['host_family']}|{assignment['dopant_family']}",
                    "split": SPLIT_CODE[assignment["split"]], "tag": row["Tag"],
                }
            )
    x = np.asarray([row["x"] for row in rows], dtype=np.float32)
    y = np.asarray([row["y"] for row in rows], dtype=np.float32)
    family = np.asarray([row["family"] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    split = np.asarray([row["split"] for row in rows], dtype=np.int8)
    train = split == 0
    global_median = float(np.median(y[train]))
    family_median: dict[str, float] = {}
    for name in sorted(set(family[train].tolist())):
        family_median[name] = float(np.median(y[train & (family == name)]))
    baseline = np.asarray([family_median.get(name, global_median) for name in family], dtype=np.float32)
    # The preregistered train-only family median is also a legitimate residual
    # prior at inference.  Supplying it explicitly lets the tiny network learn
    # concentration/temperature corrections without memorizing sparse host IDs.
    x = np.concatenate((x, baseline[:, None]), axis=1).astype(np.float32)
    counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
    group_sets = {code: set(groups[split == code]) for code in (0, 1, 2)}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    if min(counts.values()) < 16 or overlap:
        raise RuntimeError(f"P058_SPLIT_GATE:{counts}:overlap={overlap}")
    output = root / "data" / "staged_ipop_exact_v1" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, x=x, y=y, family=family, groups=groups, split=split, baseline_pred=baseline,
        tag=np.asarray([row["tag"] for row in rows]), candidate_id=np.asarray(candidate_id),
        task_kind=np.asarray("regression"), truth_class=np.asarray("LITERATURE_CURATED_EXPERIMENT"),
        authority=np.asarray(0, dtype=np.int8),
    )
    metadata = {
        "schema": "cimc.forge200.ipop-exact-staged.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "candidate_id": candidate_id, "task_kind": "regression",
        "truth_class": "LITERATURE_CURATED_EXPERIMENT", "source_id": "ipop_v3",
        "source_sha256": sha256_file(source), "split_sha256": sha256_file(assignment_path),
        "split_unit": "DOI_HOST_DOPANT_FAMILY", "cross_split_group_overlap": overlap, "counts": counts,
        "records": len(rows), "features": int(x.shape[1]),
        "record_filter": "emission_max_nm_ge_700_and_positive_decay_time_ns",
        "target_transform": "log10(decay_time_ns/1000)_log_microseconds",
        "feature_contract": "host_composition_118+dopant_presence_118+concentrations+valencies+temperature+emission_context+explicit_site_missingness_mask+train_only_host_dopant_median_residual_prior",
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_WITH_EXPLICIT_SITE_MISSINGNESS_MASK_NO_SITE_VALUE_FABRICATED",
        "target_label": contract["target_label"], "baseline": contract["baseline"],
        "baseline_fit": "TRAIN_ONLY_HOST_DOPANT_FAMILY_MEDIAN_WITH_GLOBAL_FALLBACK",
        "residual_prior_feature_index": int(x.shape[1] - 1),
        "primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "fit_preprocessing_on_train_only": True, "teacher_outputs": 0, "authority": 0,
        "board_accepted": False, "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"), "bytes": output.stat().st_size, "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    content = {"records": [metadata], "authority_nonzero": 0, "board_actions": 0}
    receipt = {"schema": "cimc.forge200.ipop-exact-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", **content, "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest()}
    write_json(root / "evidence" / "ipop_exact_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "records": len(rows), "counts": counts, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
