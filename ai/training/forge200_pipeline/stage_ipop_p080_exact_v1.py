#!/usr/bin/env python3
"""Stage literature-curated NIR internal quantum-yield records for P080."""

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


def number(value: str, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def vector(counts: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    total = max(sum(counts.values()), 1.0)
    for symbol, count in counts.items():
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def balanced_group_split(groups: list[str]) -> dict[str, int]:
    counts = Counter(groups)
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * len(groups)
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
    candidate_id = "CAND-P-080"
    source = root / "data" / "raw" / "ipop_v3" / "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    assignment_path = root / "data" / "splits" / "ipop_v3.assignments.tsv"
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        frozen_assignments = {row["tag"]: row for row in csv.DictReader(handle, delimiter="\t")}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)

    records = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            emission = number(row["Emission max. (nm)"])
            iqe = number(row["Int. quantum efficiency (%)"])
            if not math.isfinite(emission) or emission < 700.0 or not math.isfinite(iqe) or not 0.0 <= iqe <= 100.0:
                continue
            assignment = frozen_assignments.get(row["Tag"])
            if assignment is None:
                raise RuntimeError(f"MISSING_FROZEN_ASSIGNMENT:{row['Tag']}")
            host = parse_formula(row["Host"])
            dopant = np.zeros(len(ELEMENTS), dtype=np.float32)
            for field in ("1st dopant", "2nd dopant"):
                symbol = row[field].strip()
                if symbol:
                    if symbol not in ELEMENT_INDEX:
                        raise RuntimeError(f"UNSUPPORTED_DOPANT:{symbol}")
                    dopant[ELEMENT_INDEX[symbol]] = 1.0
            eqe = number(row["Ext. quantum efficiency (%)"])
            excitation_source = number(row["Excitation source (nm)"])
            excitation_peak = number(row["1st Excitation max. (nm)"])
            decay_log_ns = number(row["Log10 Decay time ( log10[ns])"])
            feature = np.concatenate(
                (
                    vector(host),
                    dopant,
                    np.asarray(
                        [
                            number(row["1st doping concentration"], 0.0),
                            number(row["2nd doping concentration "], 0.0),
                            number(row["1st dopant valency"], 0.0) / 8.0,
                            number(row["2nd dopant valency"], 0.0) / 8.0,
                            number(row["Temp. (K)"], 298.15) / 2000.0,
                            emission / 2000.0,
                            (excitation_source if math.isfinite(excitation_source) else 0.0) / 2000.0,
                            float(math.isfinite(excitation_source)),
                            (excitation_peak if math.isfinite(excitation_peak) else 0.0) / 2000.0,
                            float(math.isfinite(excitation_peak)),
                            (eqe if math.isfinite(eqe) else 0.0) / 100.0,
                            float(math.isfinite(eqe)),
                            (decay_log_ns if math.isfinite(decay_log_ns) else 0.0) / 10.0,
                            float(math.isfinite(decay_log_ns)),
                            0.0,
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            records.append(
                {
                    "x": feature,
                    "y": iqe,
                    "group": f"HOST|{assignment['host_family']}",
                    "family": row["Host"].strip(),
                    "tag": row["Tag"],
                    "eqe_present": math.isfinite(eqe),
                }
            )
    if len(records) != 61:
        raise RuntimeError(f"SOURCE_RECORD_GATE:{len(records)}")
    group_assignment = balanced_group_split([row["group"] for row in records])
    for row in records:
        row["split"] = group_assignment[row["group"]]
    x = np.asarray([row["x"] for row in records], dtype=np.float32)
    y = np.asarray([row["y"] for row in records], dtype=np.float32)
    groups = np.asarray([row["group"] for row in records])
    families = np.asarray([row["family"] for row in records])
    split = np.asarray([row["split"] for row in records], dtype=np.int8)
    train = split == 0
    global_median = float(np.median(y[train]))
    family_median = {family: float(np.median(y[train & (families == family)])) for family in sorted(set(families[train].tolist()))}
    baseline = np.asarray([family_median.get(family, global_median) for family in families], dtype=np.float32)
    counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    group_sets = {code: set(groups[split == code]) for code in (0, 1, 2)}
    overlap = sum(len(group_sets[left] & group_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    if overlap or min(counts.values()) < 8:
        raise RuntimeError(f"SPLIT_GATE:{counts}:overlap={overlap}")
    output = root / "data" / "staged_ipop_p080_exact_v1" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        groups=groups,
        family=families,
        split=split,
        baseline_pred=baseline,
        tag=np.asarray([row["tag"] for row in records]),
        candidate_id=np.asarray(candidate_id),
        authority=np.asarray(0, dtype=np.int8),
    )
    metadata = {
        "schema": "cimc.forge200.ipop-p080-exact-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": candidate_id,
        "truth_class": "LITERATURE_CURATED_EXPERIMENT",
        "source_id": "ipop_v3",
        "source_sha256": sha256_file(source),
        "frozen_split_source_sha256": sha256_file(assignment_path),
        "records": len(records),
        "features": int(x.shape[1]),
        "counts": counts,
        "groups": len(set(groups.tolist())),
        "split_unit": "HOST_FAMILY",
        "cross_split_group_overlap": overlap,
        "record_filter": "emission_max_nm_ge_700_and_internal_quantum_efficiency_percent_present",
        "target_label": contract["target_label"],
        "target_scope": "literature-curated IPOP internal quantum efficiency for NIR-emitting phosphors; not a new team integrating-sphere calibration",
        "input_contract": contract["input_contract"],
        "input_contract_state": "HOST_DOPANT_EMISSION_EXCITATION_CONTEXT_PRESENT;PUBLISHED_EXTERNAL_QE_USED_WHEN_AVAILABLE_WITH_MASK;RAW_CALIBRATED_PHOTON_COUNTS_AND_PROCESS_MASKED_UNAVAILABLE",
        "input_availability": {
            "host_dopant": 1,
            "emission_excitation_context": 1,
            "external_qe_proxy_records": int(sum(row["eqe_present"] for row in records)),
            "raw_calibrated_photon_counts": 0,
            "process_trace": 0,
        },
        "baseline": contract["baseline"],
        "baseline_fit": "TRAIN_ONLY_HOST_FORMULA_MEDIAN_WITH_GLOBAL_FALLBACK",
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "teacher_outputs": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    receipt = {"schema": "cimc.forge200.ipop-p080-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "record": metadata, "authority": 0, "board_accepted": False, "countable_model": False}
    write_json(root / "evidence" / "ipop_p080_exact_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "records": len(records), "counts": counts, "groups": len(set(groups.tolist()))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
