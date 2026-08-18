#!/usr/bin/env python3
"""Stage source-bound experimental thermoelectric contracts P141 and P144."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import ELEMENTS, ELEMENT_INDEX, canonical_bytes, sha256_file, write_json


ZIP_SHA256 = "edf55673db9fdcae2d5702e9f1d68680cbe239c64dcca68a0dc7e8a9aae19c54"
RECORD_SHA256 = "c9ff76a962ddb87c98beccbf97f212544304b253044b271de99f7547e95c59c3"
SAMPLE_SHA256 = "f65d0717864bee5702ad1e7815c658105d0e6e0e8397e35db19ae1d9a5195503"
RAW_SHA256 = "ed907cd16ec9c7be1cc36e8e242f152b41485247fd581009db96b9b0b95d690c"
COLLOCATED_SHA256 = "cc0cc7d363106a3849c266a8230721e2657534c97a56ffbe9c8a3eb1aaaf6fd9"
ELEMENT_RE = re.compile(r"[A-Z][a-z]?")


def number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def element_features(sample: dict[str, str]) -> np.ndarray:
    tokens = [token for token in ELEMENT_RE.findall(f"{sample['Composition_by_element']} {sample['Composition_detailed']}") if token in ELEMENT_INDEX]
    if not tokens:
        raise ValueError(f"NO_ELEMENTS:{sample['sample_id']}")
    counts = Counter(tokens)
    presence = np.zeros(len(ELEMENTS), dtype=np.float32)
    frequency = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count in counts.items():
        presence[ELEMENT_INDEX[symbol]] = 1.0
        frequency[ELEMENT_INDEX[symbol]] = count / len(tokens)
    return np.concatenate((presence, frequency))


def hashed_onehot(value: str, width: int = 16) -> np.ndarray:
    result = np.zeros(width, dtype=np.float32)
    result[int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % width] = 1.0
    return result


def sample_features(sample: dict[str, str]) -> np.ndarray:
    dimension = str(sample.get("mat_dimension(bulk, film, 1D, 2D)", "unknown")).strip().lower()
    dimension_values = [float(dimension == value) for value in ("bulk", "film", "1d", "2d")]
    return np.concatenate(
        (
            element_features(sample),
            np.asarray(
                [
                    len(set(ELEMENT_RE.findall(sample["Composition_detailed"]))) / 16.0,
                    number(sample.get("YEAR"), 2000.0) / 2030.0,
                    *dimension_values,
                ],
                dtype=np.float32,
            ),
            hashed_onehot(str(sample.get("SINTERING", "unknown"))),
        )
    )


def connected_components(samples: dict[int, dict[str, str]]) -> dict[int, str]:
    ids = sorted(samples)
    index_of = {sample_id: index for index, sample_id in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for field in ("GROUP", "BASEMAT", "DOI"):
        first: dict[str, int] = {}
        for sample_id in ids:
            value = str(samples[sample_id].get(field, "")).strip().lower()
            if not value or value == "na":
                continue
            index = index_of[sample_id]
            if value in first:
                union(index, first[value])
            else:
                first[value] = index
    members: dict[int, list[int]] = {}
    for sample_id in ids:
        members.setdefault(find(index_of[sample_id]), []).append(sample_id)
    names = {
        root: hashlib.sha256(
            canonical_bytes(
                {
                    "sample_ids": member_ids,
                    "families": sorted({samples[sample_id]["GROUP"] for sample_id in member_ids}),
                    "base_materials": sorted({samples[sample_id]["BASEMAT"] for sample_id in member_ids}),
                    "dois": sorted({samples[sample_id]["DOI"] for sample_id in member_ids}),
                }
            )
        ).hexdigest()
        for root, member_ids in members.items()
    }
    return {sample_id: names[find(index_of[sample_id])] for sample_id in ids}


def balanced_split(component_by_sample: dict[int, str], weights: Counter[str]) -> dict[str, int]:
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * sum(weights.values())
    totals = np.zeros(3, dtype=np.int64)
    assignment: dict[str, int] = {}
    for component in sorted(weights, key=lambda value: (-weights[value], value)):
        deficits = targets - totals
        code = int(np.argmax(deficits / np.maximum(targets, 1.0)))
        assignment[component] = code
        totals[code] += weights[component]
    return assignment


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw_root = root / "data" / "raw" / "tematdb_zenodo_15518036"
    publication = raw_root / "byungkiryu-teMatDb-68b6ec6" / "teMatDb_publication" / "teMatDb272_dataset_20250515"
    source_zip = raw_root / "teMatDb-v1.1.6a.zip"
    source_record = raw_root / "zenodo_record_15518036.json"
    sample_path = publication / "teMatDb_samples.csv"
    raw_path = publication / "teMatDb_rawTEPs.csv"
    collocated_path = publication / "teMatDb_collocatedTEPs.csv"
    expected = {
        source_zip: ZIP_SHA256,
        source_record: RECORD_SHA256,
        sample_path: SAMPLE_SHA256,
        raw_path: RAW_SHA256,
        collocated_path: COLLOCATED_SHA256,
    }
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise RuntimeError(f"SOURCE_HASH_GATE:{path.name}")
    record = json.loads(source_record.read_text(encoding="utf-8"))
    if record["metadata"]["license"]["id"] != "cc-by-4.0" or record["metadata"]["doi"] != "10.5281/zenodo.15518036":
        raise RuntimeError("LICENSE_PID_GATE")
    with sample_path.open("r", encoding="utf-8-sig", newline="") as handle:
        samples = {int(row["sample_id"]): row for row in csv.DictReader(handle)}
    if len(samples) != 272:
        raise RuntimeError(f"SAMPLE_GATE:{len(samples)}")
    component_by_sample = connected_components(samples)
    sample_feature = {sample_id: sample_features(sample) for sample_id, sample in samples.items()}

    alpha_rows: list[tuple[int, float, float]] = []
    with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(row["sample_id"])
            if sample_id not in samples or row["tepname"] != "alpha":
                continue
            temperature, alpha = number(row["Temperature"]), number(row["tepvalue"])
            if math.isfinite(temperature) and math.isfinite(alpha) and 0.0 < temperature <= 2000.0 and abs(alpha) <= 0.01:
                alpha_rows.append((sample_id, temperature, alpha * 1e6))
    # One upstream alpha row is the explicit (T=0, alpha=0) origin placeholder
    # and is not a physical measurement point.
    if len(alpha_rows) != 3852:
        raise RuntimeError(f"P141_RECORD_GATE:{len(alpha_rows)}")
    weights = Counter(component_by_sample[sample_id] for sample_id, _, _ in alpha_rows)
    component_assignment = balanced_split(component_by_sample, weights)
    split_assignment_path = root / "data" / "splits" / "tematdb_v116.assignments.tsv"
    split_assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with split_assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_id", "component_group", "split", "doi", "family", "base_material"])
        for sample_id in sorted(samples):
            component = component_by_sample[sample_id]
            writer.writerow([sample_id, component, ("train", "validation", "test")[component_assignment[component]], samples[sample_id]["DOI"], samples[sample_id]["GROUP"], samples[sample_id]["BASEMAT"]])

    def p141_feature(sample_id: int, temperature: float) -> np.ndarray:
        temp = temperature / 1500.0
        return np.concatenate((sample_feature[sample_id], np.asarray([temp, temp**2, math.log1p(temperature) / 8.0, 0.0, 0.0], dtype=np.float32)))

    p141_x = np.asarray([p141_feature(sample_id, temperature) for sample_id, temperature, _ in alpha_rows], dtype=np.float32)
    p141_y = np.asarray([target for _, _, target in alpha_rows], dtype=np.float32)
    p141_sample = np.asarray([sample_id for sample_id, _, _ in alpha_rows], dtype=np.int32)
    p141_family = np.asarray([samples[sample_id]["GROUP"] for sample_id, _, _ in alpha_rows])
    p141_component = np.asarray([component_by_sample[sample_id] for sample_id, _, _ in alpha_rows])
    p141_split = np.asarray([component_assignment[value] for value in p141_component], dtype=np.int8)

    p144_rows: list[tuple[int, float, float, float, float]] = []
    with collocated_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = int(row["sample_id"])
            if sample_id not in samples:
                continue
            temperature, alpha, rho, kappa, zt = (number(row[field]) for field in ("Temperature", "alpha", "rho", "kappa", "ZT_author_declared"))
            if not all(math.isfinite(value) for value in (temperature, alpha, rho, kappa, zt)) or temperature <= 0.0 or rho <= 0.0 or kappa <= 0.0 or zt < 0.0:
                continue
            target = zt * kappa / temperature * 10000.0
            baseline = alpha**2 / rho * 10000.0
            if not all(math.isfinite(value) and 0.0 <= value <= 1e7 for value in (target, baseline)):
                continue
            p144_rows.append((sample_id, temperature, alpha * 1e6, 1.0 / rho, target))
    # Two upstream interpolated rows carry a small negative author-declared ZT
    # and are excluded because a negative power-factor target is non-physical.
    if len(p144_rows) != 56639:
        raise RuntimeError(f"P144_RECORD_GATE:{len(p144_rows)}")

    def p144_feature(sample_id: int, temperature: float, alpha_uv: float, conductivity: float) -> np.ndarray:
        return np.concatenate(
            (
                sample_feature[sample_id],
                np.asarray(
                    [
                        temperature / 1500.0,
                        math.log1p(temperature) / 8.0,
                        math.asinh(alpha_uv / 200.0),
                        math.log1p(conductivity) / 20.0,
                        float(alpha_uv >= 0.0),
                        0.0,
                        0.0,
                    ],
                    dtype=np.float32,
                ),
            )
        )

    p144_x = np.asarray([p144_feature(sample_id, temperature, alpha_uv, conductivity) for sample_id, temperature, alpha_uv, conductivity, _ in p144_rows], dtype=np.float32)
    p144_y = np.asarray([target for *_, target in p144_rows], dtype=np.float32)
    p144_baseline = np.asarray([(alpha_uv * 1e-6) ** 2 * conductivity * 10000.0 for _, _, alpha_uv, conductivity, _ in p144_rows], dtype=np.float32)
    p144_sample = np.asarray([sample_id for sample_id, *_ in p144_rows], dtype=np.int32)
    p144_family = np.asarray([samples[sample_id]["GROUP"] for sample_id, *_ in p144_rows])
    p144_component = np.asarray([component_by_sample[sample_id] for sample_id, *_ in p144_rows])
    p144_split = np.asarray([component_assignment[value] for value in p144_component], dtype=np.int8)

    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    stage_root = root / "data" / "staged_tematdb_p141_p144_exact_v1"
    stage_root.mkdir(parents=True, exist_ok=True)
    task_arrays = {
        "CAND-P-141": {"x": p141_x, "y": p141_y, "sample_id": p141_sample, "family": p141_family, "component_group": p141_component, "split": p141_split},
        "CAND-P-144": {"x": p144_x, "y": p144_y, "baseline_pred": p144_baseline, "sample_id": p144_sample, "family": p144_family, "component_group": p144_component, "split": p144_split},
    }
    truth = {
        "CAND-P-141": "OPEN_EXPERIMENTAL_DIGITIZED_SEEBECK",
        "CAND-P-144": "OPEN_EXPERIMENTAL_INTERPOLATED_DERIVED_POWER_FACTOR",
    }
    target_scope = {
        "CAND-P-141": "raw digitized experimental Seebeck coefficient converted from V/K to uV/K; sign is derived without teacher labels",
        "CAND-P-144": "power factor derived independently from author-declared ZT times experimental kappa divided by temperature; baseline uses experimental alpha squared over rho",
    }
    input_state = {
        "CAND-P-141": "COMPOSITION_TEMPERATURE_DIMENSION_SYNTHESIS_PRESENT;CRYSTAL_STRUCTURE_AND_CARRIER_FEATURES_MASKED_UNAVAILABLE",
        "CAND-P-144": "EXPERIMENTAL_SEEBECK_CONDUCTIVITY_TEMPERATURE_COMPOSITION_DIMENSION_SYNTHESIS_PRESENT;CRYSTAL_STRUCTURE_AND_CARRIER_DETAIL_MASKED_UNAVAILABLE",
    }
    records = []
    for candidate_id, arrays in task_arrays.items():
        output = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(output, **arrays, candidate_id=np.asarray(candidate_id), authority=np.asarray(0, dtype=np.int8))
        split = arrays["split"]
        component = arrays["component_group"]
        family = arrays["family"]
        counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
        component_sets = {code: set(component[split == code]) for code in (0, 1, 2)}
        family_sets = {code: set(family[split == code]) for code in (0, 1, 2)}
        component_overlap = sum(len(component_sets[left] & component_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        family_overlap = sum(len(family_sets[left] & family_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        if component_overlap or family_overlap or min(counts.values()) < 400:
            raise RuntimeError(f"SPLIT_GATE:{candidate_id}:{counts}:{component_overlap}:{family_overlap}")
        contract = contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.tematdb-exact-staged.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "candidate_id": candidate_id,
            "truth_class": truth[candidate_id],
            "source_id": "tematdb_zenodo_15518036_v1.1.6",
            "source_url": "https://zenodo.org/records/15518036",
            "source_pid": "10.5281/zenodo.15518036",
            "source_archive_sha256": ZIP_SHA256,
            "source_table_sha256": RAW_SHA256 if candidate_id == "CAND-P-141" else COLLOCATED_SHA256,
            "sample_table_sha256": SAMPLE_SHA256,
            "license": "CC BY 4.0",
            "records": len(arrays["y"]),
            "features": int(arrays["x"].shape[1]),
            "samples": int(len(set(arrays["sample_id"].tolist()))),
            "counts": counts,
            "split_units": ["COMPOUND_FAMILY", "BASE_MATERIAL", "DOI", "SAMPLE_CURVE"],
            "connected_components": len(set(component.tolist())),
            "cross_split_component_overlap": component_overlap,
            "cross_split_family_overlap": family_overlap,
            "split_assignment_sha256": sha256_file(split_assignment_path),
            "input_contract": contract["input_contract"],
            "input_contract_state": input_state[candidate_id],
            "target_label": contract["target_label"],
            "target_scope": target_scope[candidate_id],
            "baseline": contract["baseline"],
            "baseline_execution": "TRAIN_ONLY_DESCRIPTOR_RIDGE" if candidate_id == "CAND-P-141" else "EXPERIMENTAL_SEEBECK_SQUARED_TIMES_CONDUCTIVITY",
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
        records.append(metadata)
    receipt = {"schema": "cimc.forge200.tematdb-p141-p144-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "records": records, "authority": 0, "board_accepted": False, "countable_model": False, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}
    write_json(root / "evidence" / "tematdb_p141_p144_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "tasks": [{"candidate_id": record["candidate_id"], "records": record["records"], "counts": record["counts"], "features": record["features"]} for record in records]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
