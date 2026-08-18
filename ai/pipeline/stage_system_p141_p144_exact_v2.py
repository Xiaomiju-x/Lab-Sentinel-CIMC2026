#!/usr/bin/env python3
"""Stage the source-bound sysTEm experimental contracts for P141 and P144.

The split is frozen without looking at target values.  Rows sharing either a
source paper or an element system are placed in one connected component so a
paper or compound family cannot cross train/validation/test boundaries.
"""

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
import pandas as pd

from stage_matbench_experimental_v1 import (
    canonical_bytes,
    features,
    parse_formula,
    sha256_file,
    write_json,
)


COMMIT = "172aaea637d4222076bf227b797f75c38d2d741c"
DATA_SHA256 = "32bd60388a685fd770dde1dc01627d38d7c9107d65de7645761f6b23b647aefe"
LICENSE_SHA256 = "7ad065eb67b96c0ccf62e5086ff2778160390cd67c6eadd61bfd869aed31a20f"
README_SHA256 = "6f3bad2696fa0327c1f81c68f17ee9d3d42196d165698f5e98ed4e125c7da55c"
DOI = "10.1007/s40192-026-00446-5"
PREPRINT_DOI = "10.26434/chemrxiv-2025-4gxmc"
SPLIT_NAMES = ("train", "validation", "test")


def finite(value: Any) -> float:
    result = float(value)
    return result if math.isfinite(result) else math.nan


def normalize_source(value: Any) -> str:
    return str(value).strip().lower().replace("https://doi.org/", "").rstrip("/")


def element_system(formula: str) -> str:
    return "-".join(sorted(parse_formula(formula)))


def connected_components(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    parent = list(range(len(frame)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    sources = [normalize_source(value) for value in frame["Source Paper"]]
    systems = [element_system(str(value)) for value in frame["Pymatgen Composition"]]
    for values in (sources, systems):
        first: dict[str, int] = {}
        for index, value in enumerate(values):
            if value in first:
                union(index, first[value])
            else:
                first[value] = index
    members: dict[int, list[int]] = {}
    for index in range(len(frame)):
        members.setdefault(find(index), []).append(index)
    names = {
        root: hashlib.sha256(
            canonical_bytes(
                {
                    "row_ids": [int(frame.iloc[index]["#"]) for index in indices],
                    "source_papers": sorted({sources[index] for index in indices}),
                    "element_systems": sorted({systems[index] for index in indices}),
                }
            )
        ).hexdigest()
        for root, indices in members.items()
    }
    components = np.asarray([names[find(index)] for index in range(len(frame))])
    return components, np.asarray(systems)


def balanced_assignment(components: np.ndarray) -> dict[str, int]:
    weights = Counter(components.tolist())
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * len(components)
    totals = np.zeros(3, dtype=np.int64)
    assignment: dict[str, int] = {}
    for component in sorted(weights, key=lambda value: (-weights[value], value)):
        deficits = targets - totals
        split = int(np.argmax(deficits / np.maximum(targets, 1.0)))
        assignment[component] = split
        totals[split] += weights[component]
    return assignment


def source_features(formula: str, temperature: float, year: float, mixed: bool) -> np.ndarray:
    composition = features(parse_formula(formula), "transport")
    scaled_temperature = temperature / 1500.0
    return np.concatenate(
        (
            composition,
            np.asarray(
                [
                    scaled_temperature,
                    scaled_temperature**2,
                    math.log1p(temperature) / 8.0,
                    year / 2030.0,
                    float(mixed),
                    0.0,  # structure descriptor unavailable
                    0.0,  # carrier descriptor unavailable
                ],
                dtype=np.float32,
            ),
        )
    )


def overlap_count(values: np.ndarray, split: np.ndarray) -> int:
    sets = {code: set(values[split == code].tolist()) for code in (0, 1, 2)}
    return sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw" / "system_thermoelectric_v1"
    dataset = raw / "sysTEm_dataset.xlsx"
    license_path = raw / "LICENSE"
    readme = raw / "README.md"
    if sha256_file(dataset) != DATA_SHA256 or sha256_file(license_path) != LICENSE_SHA256 or sha256_file(readme) != README_SHA256:
        raise RuntimeError("SOURCE_HASH_GATE")
    license_text = license_path.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    if "MIT License" not in license_text or DOI not in readme_text or "experimental thermoelectric" not in readme_text.lower():
        raise RuntimeError("LICENSE_PID_SCOPE_GATE")

    frame = pd.read_excel(dataset)
    required = {
        "#", "Source Paper", "Pymatgen Composition", "Type of Formula", "Year", "Temperature (K)",
        "Electrical Conductivity (S/cm)", "Seebeck Coefficient (µV/K)", "Power Factor (µW/cmK²)",
    }
    if len(frame) != 8650 or not required.issubset(frame.columns) or frame["#"].nunique() != len(frame):
        raise RuntimeError("UPSTREAM_SCHEMA_GATE")
    components, systems = connected_components(frame)
    assignment = balanced_assignment(components)
    row_split = np.asarray([assignment[value] for value in components], dtype=np.int8)
    sources = np.asarray([normalize_source(value) for value in frame["Source Paper"]])
    formulas = frame["Pymatgen Composition"].astype(str).to_numpy()
    if overlap_count(components, row_split) or overlap_count(systems, row_split) or overlap_count(sources, row_split):
        raise RuntimeError("SPLIT_LEAKAGE_GATE")

    split_path = root / "data" / "splits" / "system_v1.connected.assignments.tsv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["row_id", "component_group", "split", "source_paper", "element_system", "composition"])
        for index, row in frame.iterrows():
            writer.writerow([int(row["#"]), components[index], SPLIT_NAMES[row_split[index]], row["Source Paper"], systems[index], formulas[index]])

    common_x: list[np.ndarray] = []
    for _, row in frame.iterrows():
        common_x.append(
            source_features(
                str(row["Pymatgen Composition"]), finite(row["Temperature (K)"]), finite(row["Year"]),
                str(row["Type of Formula"]).strip().lower().startswith("mixed"),
            )
        )
    common_x_array = np.asarray(common_x, dtype=np.float32)

    task_rows: dict[str, list[int]] = {"CAND-P-141": [], "CAND-P-144": []}
    p141_targets: list[float] = []
    p144_targets: list[float] = []
    p144_baseline: list[float] = []
    for index, row in frame.iterrows():
        temperature = finite(row["Temperature (K)"])
        seebeck = finite(row["Seebeck Coefficient (µV/K)"])
        conductivity = finite(row["Electrical Conductivity (S/cm)"])
        power_factor = finite(row["Power Factor (µW/cmK²)"])
        if 0.0 < temperature <= 2000.0 and math.isfinite(seebeck) and abs(seebeck) <= 5000.0:
            task_rows["CAND-P-141"].append(index)
            p141_targets.append(seebeck)
        if (
            0.0 < temperature <= 2000.0 and math.isfinite(seebeck) and math.isfinite(conductivity)
            and conductivity > 0.0 and math.isfinite(power_factor) and 0.0 <= power_factor <= 100000.0
        ):
            task_rows["CAND-P-144"].append(index)
            p144_targets.append(power_factor)
            p144_baseline.append(seebeck**2 * conductivity * 1e-6)
    if len(task_rows["CAND-P-141"]) != 8582 or len(task_rows["CAND-P-144"]) != 8575:
        raise RuntimeError(f"ELIGIBLE_RECORD_GATE:{len(task_rows['CAND-P-141'])}:{len(task_rows['CAND-P-144'])}")

    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    stage_root = root / "data" / "staged_system_p141_p144_exact_v2"
    stage_root.mkdir(parents=True, exist_ok=True)
    definitions = {
        "CAND-P-141": {
            "targets": np.asarray(p141_targets, dtype=np.float32),
            "truth": "OPEN_EXPERIMENTAL_DIGITIZED_SYSTEMATICALLY_VERIFIED_SEEBECK",
            "scope": "direct experimental Seebeck coefficient in uV/K with sign; source structure and carrier descriptors are explicitly unavailable",
            "baseline": "TRAIN_ONLY_DESCRIPTOR_RIDGE",
        },
        "CAND-P-144": {
            "targets": np.asarray(p144_targets, dtype=np.float32),
            "truth": "OPEN_EXPERIMENTAL_DIGITIZED_SYSTEMATICALLY_VERIFIED_POWER_FACTOR",
            "scope": "direct experimental power factor in uW/cmK2; measured Seebeck and conductivity are model inputs",
            "baseline": "MEASURED_SEEBECK_SQUARED_TIMES_MEASURED_CONDUCTIVITY",
        },
    }
    evidence_records = []
    for candidate_id, definition in definitions.items():
        selected = np.asarray(task_rows[candidate_id], dtype=np.int64)
        x = common_x_array[selected]
        if candidate_id == "CAND-P-144":
            seebeck = frame.iloc[selected]["Seebeck Coefficient (µV/K)"].to_numpy(dtype=np.float32)
            conductivity = frame.iloc[selected]["Electrical Conductivity (S/cm)"].to_numpy(dtype=np.float32)
            x = np.column_stack(
                (
                    x,
                    np.arcsinh(seebeck / 200.0),
                    np.log1p(conductivity),
                    (seebeck >= 0.0).astype(np.float32),
                )
            ).astype(np.float32)
        arrays: dict[str, np.ndarray] = {
            "x": x,
            "y": definition["targets"],
            "row_id": frame.iloc[selected]["#"].to_numpy(dtype=np.int32),
            "family": systems[selected],
            "source_paper": sources[selected],
            "component_group": components[selected],
            "split": row_split[selected],
        }
        if candidate_id == "CAND-P-144":
            arrays["baseline_pred"] = np.asarray(p144_baseline, dtype=np.float32)
        output = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(output, **arrays)
        split_counts = {SPLIT_NAMES[code]: int(np.sum(arrays["split"] == code)) for code in (0, 1, 2)}
        contract_sha = hashlib.sha256(canonical_bytes(contracts[candidate_id])).hexdigest()
        metadata = {
            "schema": "cimc.forge200.system-contract-exact-dataset.v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "candidate_id": candidate_id,
            "dataset_id": "system_thermoelectric_v1",
            "records": len(selected),
            "features": int(x.shape[1]),
            "split_counts": split_counts,
            "split_method": "TARGET_BLIND_CONNECTED_COMPONENT_BY_SOURCE_PAPER_OR_ELEMENT_SYSTEM_THEN_BALANCED_70_15_15",
            "cross_split_component_overlap": overlap_count(arrays["component_group"], arrays["split"]),
            "cross_split_family_overlap": overlap_count(arrays["family"], arrays["split"]),
            "cross_split_source_paper_overlap": overlap_count(arrays["source_paper"], arrays["split"]),
            "source": {
                "repository": "https://github.com/tankylz/sysTEm_dataset",
                "commit": COMMIT,
                "publication_doi": DOI,
                "preprint_doi": PREPRINT_DOI,
                "repository_license": "MIT",
                "publication_license": "CC-BY-4.0",
                "dataset_sha256": DATA_SHA256,
                "license_sha256": LICENSE_SHA256,
                "readme_sha256": README_SHA256,
            },
            "truth_class": definition["truth"],
            "target_scope": definition["scope"],
            "baseline_execution": definition["baseline"],
            "task_contract_sha256": contract_sha,
            "split_assignments_sha256": sha256_file(split_path),
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
            "sha256": sha256_file(output),
        }
        if any(metadata[key] != 0 for key in ("cross_split_component_overlap", "cross_split_family_overlap", "cross_split_source_paper_overlap")):
            raise RuntimeError(f"{candidate_id}:POST_STAGE_LEAKAGE_GATE")
        write_json(output.with_suffix(".metadata.json"), metadata)
        evidence_records.append(metadata)
    write_json(
        root / "evidence" / "system_p141_p144_exact_staging.v2.json",
        {
            "schema": "cimc.forge200.system-exact-staging.v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "records": evidence_records,
            "authority_nonzero": 0,
            "board_actions": 0,
        },
    )
    print(json.dumps({"status": "PASS", "tasks": {record["candidate_id"]: record["split_counts"] for record in evidence_records}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
