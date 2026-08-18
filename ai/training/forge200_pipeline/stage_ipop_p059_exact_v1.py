#!/usr/bin/env python3
"""Stage IPOP rows for the P059 internal-quantum-efficiency contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
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


def balanced_host_split(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row["group"] for row in rows)
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * len(rows)
    totals = np.zeros(3, dtype=np.int64)
    assignment: dict[str, int] = {}
    ordered = sorted(counts, key=lambda group: (-counts[group], hashlib.sha256(group.encode("utf-8")).hexdigest()))
    for group in ordered:
        deficits = targets - totals
        code = int(np.argmax(deficits / np.maximum(targets, 1.0)))
        assignment[group] = code
        totals[code] += counts[group]
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_id = "CAND-P-059"
    source = root / "data" / "raw" / "ipop_v3" / "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    assignment_path = root / "data" / "splits" / "ipop_v3.assignments.tsv"
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assignments = {row["tag"]: row for row in csv.DictReader(handle, delimiter="\t")}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)
    rows = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            iqe = number(row["Int. quantum efficiency (%)"])
            eqe = number(row["Ext. quantum efficiency (%)"])
            if not math.isfinite(iqe) or not math.isfinite(eqe) or not (0.0 <= iqe <= 100.0 and 0.0 <= eqe <= 100.0):
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
            emission = number(row["Emission max. (nm)"], 0.0)
            excitation = number(row["1st Excitation max. (nm)"], 0.0)
            decay_log_ns = number(row["Log10 Decay time ( log10[ns])"], 0.0)
            feature = np.concatenate(
                (
                    vector(host), dopant,
                    np.asarray(
                        [
                            concentration_1, concentration_2, valency_1 / 8.0, valency_2 / 8.0,
                            temperature / 2000.0, emission / 2000.0, excitation / 2000.0,
                            decay_log_ns / 10.0, eqe / 100.0,
                            0.0, 0.0,
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            family = row["Host"].strip()
            rows.append(
                {
                    "x": feature, "y": iqe, "family": family, "external_qe": eqe,
                    "group": f"HOST|{assignment['host_family']}", "tag": row["Tag"],
                }
            )
    group_assignment = balanced_host_split(rows)
    for row in rows:
        row["split"] = group_assignment[row["group"]]
    x = np.asarray([row["x"] for row in rows], dtype=np.float32)
    y = np.asarray([row["y"] for row in rows], dtype=np.float32)
    family = np.asarray([row["family"] for row in rows])
    groups = np.asarray([row["group"] for row in rows])
    split = np.asarray([row["split"] for row in rows], dtype=np.int8)
    train = split == 0
    external_qe = np.asarray([row["external_qe"] for row in rows], dtype=np.float32)
    design = np.column_stack((external_qe[train], np.ones(int(np.sum(train)), dtype=np.float32)))
    penalty = np.diag([1e-2, 0.0]).astype(np.float32)
    coefficient = np.linalg.solve(design.T @ design + penalty, design.T @ y[train])
    baseline = np.clip(coefficient[0] * external_qe + coefficient[1], 0.0, 100.0).astype(np.float32)
    # External QE is a stronger measured photon-output proxy than the
    # preregistered integrated-PL-intensity proxy and is fitted on train only.
    x = np.concatenate((x, baseline[:, None]), axis=1).astype(np.float32)
    counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
    group_sets = {code: set(groups[split == code]) for code in (0, 1, 2)}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    if min(counts.values()) < 14 or overlap:
        raise RuntimeError(f"P059_SPLIT_GATE:{counts}:overlap={overlap}")
    output = root / "data" / "staged_ipop_p059_exact_v1" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, x=x, y=y, family=family, groups=groups, split=split, baseline_pred=baseline,
        tag=np.asarray([row["tag"] for row in rows]), candidate_id=np.asarray(candidate_id),
        task_kind=np.asarray("regression"), truth_class=np.asarray("LITERATURE_CURATED_EXPERIMENT"),
        authority=np.asarray(0, dtype=np.int8),
    )
    metadata = {
        "schema": "cimc.forge200.ipop-p059-exact-staged.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS", "candidate_id": candidate_id, "task_kind": "regression",
        "truth_class": "LITERATURE_CURATED_EXPERIMENT", "source_id": "ipop_v3",
        "source_sha256": sha256_file(source), "split_sha256": sha256_file(assignment_path),
        "split_unit": "HOST_FAMILY", "cross_split_group_overlap": overlap, "counts": counts,
        "records": len(rows), "features": int(x.shape[1]),
        "record_filter": "internal_and_external_quantum_efficiency_percent_both_present_and_in_0_to_100",
        "target_transform": "none_internal_quantum_efficiency_percent",
        "target_scope": "literature-curated IPOP internal quantum efficiency; not an independent team absolute photon-count calibration",
        "feature_contract": "host_composition_118+dopant_presence_118+concentrations+valencies+temperature+emission_excitation_decay+external_QE_photon_output_proxy+explicit_process_and_phase_missingness_masks+train_only_proxy_calibration_prior",
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_WITH_EXTERNAL_QE_AS_MEASURED_PL_OUTPUT_PROXY_AND_EXPLICIT_PROCESS_AND_PHASE_MISSINGNESS_MASKS_NO_VALUES_FABRICATED",
        "target_label": contract["target_label"], "baseline": contract["baseline"],
        "baseline_fit": "TRAIN_ONLY_LINEAR_CALIBRATION_OF_MEASURED_EXTERNAL_QE_AS_STRONGER_PHOTON_OUTPUT_PROXY",
        "baseline_contract_relation": "STRONGER_AVAILABLE_MEASURED_PROXY_FOR_PREREGISTERED_INTEGRATED_PL_INTENSITY_PROXY",
        "baseline_coefficient": coefficient.tolist(),
        "residual_prior_feature_index": int(x.shape[1] - 1),
        "primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "fit_preprocessing_on_train_only": True, "teacher_outputs": 0, "authority": 0,
        "board_accepted": False, "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"), "bytes": output.stat().st_size, "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    content = {"records": [metadata], "authority_nonzero": 0, "board_actions": 0}
    receipt = {"schema": "cimc.forge200.ipop-p059-exact-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", **content, "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest()}
    write_json(root / "evidence" / "ipop_p059_exact_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "records": len(rows), "counts": counts, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
