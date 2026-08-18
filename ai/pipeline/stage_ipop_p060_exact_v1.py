#!/usr/bin/env python3
"""Stage IPOP experimental CIE x/y records for the P060 contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage_matbench_experimental_v1 import ELEMENTS, ELEMENT_INDEX, canonical_bytes, parse_formula, sha256_file, write_json


SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def number(value: str, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def composition(counts: dict[str, float]) -> np.ndarray:
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    total = max(sum(counts.values()), 1.0)
    for symbol, count in counts.items():
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def component_split(records: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_by_host: dict[str, int] = {}
    first_by_doi: dict[str, int] = {}
    for index, record in enumerate(records):
        host, doi = str(record["host_group"]), str(record["doi_group"])
        if host in first_by_host:
            union(index, first_by_host[host])
        else:
            first_by_host[host] = index
        if doi in first_by_doi:
            union(index, first_by_doi[doi])
        else:
            first_by_doi[doi] = index
    members: dict[int, list[int]] = {}
    for index in range(len(records)):
        members.setdefault(find(index), []).append(index)
    component_name = {
        root: hashlib.sha256(
            canonical_bytes(
                {
                    "hosts": sorted({str(records[index]["host_group"]) for index in indices}),
                    "dois": sorted({str(records[index]["doi_group"]) for index in indices}),
                }
            )
        ).hexdigest()
        for root, indices in members.items()
    }
    counts = Counter(component_name[find(index)] for index in range(len(records)))
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * len(records)
    totals = np.zeros(3, dtype=np.int64)
    assignment: dict[str, int] = {}
    for group in sorted(counts, key=lambda value: (-counts[value], value)):
        deficits = targets - totals
        code = int(np.argmax(deficits / np.maximum(targets, 1.0)))
        assignment[group] = code
        totals[code] += counts[group]
    groups = np.asarray([component_name[find(index)] for index in range(len(records))])
    split = np.asarray([assignment[group] for group in groups], dtype=np.int8)
    return groups, split


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_id = "CAND-P-060"
    source = root / "data" / "raw" / "ipop_v3" / "Inorganic_Phosphor_Optical_Properties_DB_20230908_IPOP_ver3.csv"
    assignment_path = root / "data" / "splits" / "ipop_v3.assignments.tsv"
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        assignments = {row["tag"]: row for row in csv.DictReader(handle, delimiter="\t")}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)
    records = []
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            cie_x = number(row["CIE x coordinate"])
            cie_y = number(row["CIE y coordinate"])
            emission = number(row["Emission max. (nm)"])
            if not all(math.isfinite(value) for value in (cie_x, cie_y, emission)) or not (0.0 < cie_x < 1.0 and 0.0 < cie_y < 1.0 and emission > 0.0):
                continue
            assignment = assignments.get(row["Tag"])
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
            values = []
            for field, scale in (
                ("1st doping concentration", 1.0),
                ("2nd doping concentration ", 1.0),
                ("1st dopant valency", 8.0),
                ("2nd dopant valency", 8.0),
                ("Temp. (K)", 2000.0),
                ("Emission max. (nm)", 2000.0),
                ("Monitoring energy (nm)", 2000.0),
                ("Excitation source (nm)", 2000.0),
                ("1st Excitation max. (nm)", 2000.0),
                ("2nd Excitation max. (nm)", 2000.0),
                ("3rd Excitation max. (nm)", 2000.0),
                ("Thermal quenching temp. (K)", 2000.0),
                ("Log10 Decay time ( log10[ns])", 10.0),
            ):
                value = number(row[field])
                values.extend([(value if math.isfinite(value) else 0.0) / scale, float(math.isfinite(value))])
            feature = np.concatenate((composition(host), dopant, np.asarray(values + [0.0, 0.0], dtype=np.float32)))
            records.append(
                {
                    "x": feature,
                    "y": [cie_x, cie_y],
                    "emission": emission,
                    "host_group": assignment["host_family"],
                    "doi_group": assignment["doi_family"],
                    "tag": row["Tag"],
                }
            )
    if len(records) != 2691:
        raise RuntimeError(f"SOURCE_RECORD_GATE:{len(records)}")
    x = np.asarray([row["x"] for row in records], dtype=np.float32)
    y = np.asarray([row["y"] for row in records], dtype=np.float32)
    emission = np.asarray([row["emission"] for row in records], dtype=np.float32)
    host_groups = np.asarray([row["host_group"] for row in records])
    doi_groups = np.asarray([row["doi_group"] for row in records])
    component_groups, split = component_split(records)
    counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
    host_sets = {code: set(host_groups[split == code]) for code in (0, 1, 2)}
    doi_sets = {code: set(doi_groups[split == code]) for code in (0, 1, 2)}
    host_overlap = sum(len(host_sets[left] & host_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    doi_overlap = sum(len(doi_sets[left] & doi_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    if host_overlap or doi_overlap or min(counts.values()) < 300:
        raise RuntimeError(f"SPLIT_GATE:counts={counts}:host_overlap={host_overlap}:doi_overlap={doi_overlap}")
    train = split == 0
    baseline_design = np.column_stack((emission[train] / 1000.0, np.ones(int(np.sum(train)), dtype=np.float32)))
    baseline_coefficient = np.linalg.solve(baseline_design.T @ baseline_design + np.diag([1e-3, 0.0]), baseline_design.T @ y[train])
    baseline = np.clip(np.column_stack((emission / 1000.0, np.ones(len(emission), dtype=np.float32))) @ baseline_coefficient, 0.0, 1.0).astype(np.float32)
    output = root / "data" / "staged_ipop_p060_exact_v1" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        emission_nm=emission,
        split=split,
        host_group=host_groups,
        doi_group=doi_groups,
        component_group=component_groups,
        baseline_pred=baseline,
        baseline_coefficient=baseline_coefficient.astype(np.float32),
        tag=np.asarray([row["tag"] for row in records]),
        candidate_id=np.asarray(candidate_id),
        authority=np.asarray(0, dtype=np.int8),
    )
    metadata = {
        "schema": "cimc.forge200.ipop-p060-exact-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": candidate_id,
        "truth_class": "LITERATURE_CURATED_EXPERIMENT",
        "source_id": "ipop_v3",
        "source_sha256": sha256_file(source),
        "frozen_split_sha256": sha256_file(assignment_path),
        "records": len(records),
        "features": int(x.shape[1]),
        "counts": counts,
        "split_units": ["HOST_FAMILY", "DOI_FAMILY"],
        "connected_components": len(set(component_groups.tolist())),
        "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(component_groups.tolist(), split.tolist(), strict=True)))).hexdigest(),
        "cross_split_host_overlap": host_overlap,
        "cross_split_doi_overlap": doi_overlap,
        "target_label": contract["target_label"],
        "target_scope": "published experimental CIE x/y coordinates; DeltaE2000 evaluation uses normalized Y=1 and D65 reference convention",
        "input_contract": contract["input_contract"],
        "input_contract_state": "HOST_DOPANT_EMISSION_PEAK_EXCITATION_MONITORING_AND_TEMPERATURE_CONTEXT_PRESENT;FULL_NORMALIZED_SPECTRUM_AND_MEASUREMENT_GEOMETRY_MASKED_UNAVAILABLE",
        "input_availability": {"emission_peak_and_context": 1, "full_normalized_spectrum": 0, "measurement_geometry": 0},
        "baseline": contract["baseline"],
        "baseline_fit": "TRAIN_ONLY_LINEAR_MAP_FROM_PUBLISHED_EMISSION_PEAK_AS_SPECTRAL_CENTROID_PROXY",
        "baseline_coefficient": baseline_coefficient.tolist(),
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
    write_json(root / "evidence" / "ipop_p060_exact_staging.v1.json", {"schema": "cimc.forge200.ipop-p060-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "record": metadata, "authority": 0, "board_accepted": False, "countable_model": False})
    print(json.dumps({"status": "PASS", "candidate_id": candidate_id, "records": len(records), "features": int(x.shape[1]), "counts": counts, "host_overlap": host_overlap, "doi_overlap": doi_overlap}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
