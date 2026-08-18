#!/usr/bin/env python3
"""Stage source-bound JARVIS-DFT maximum IR-mode intensity records for P143."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import ELEMENTS, ELEMENT_INDEX


SOURCE_SHA256 = "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"
SOURCE_URL = "https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699"
SOURCE_PID = "10.6084/m9.figshare.6815699.v11"
ACTIVE_THRESHOLD = 0.1


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def composition(elements: list[str]) -> np.ndarray:
    counts = Counter(elements)
    total = max(sum(counts.values()), 1)
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count in counts.items():
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def structure_features(atoms: dict[str, Any]) -> np.ndarray:
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    lengths = np.linalg.norm(lattice, axis=1)
    angles = np.asarray(atoms.get("angles", [90.0, 90.0, 90.0]), dtype=np.float64)
    volume = abs(float(np.linalg.det(lattice)))
    nat = max(len(atoms["elements"]), 1)
    return np.asarray(
        [
            math.log1p(nat) / 7.0,
            *(np.log1p(lengths) / 6.0).tolist(),
            *(angles / 180.0).tolist(),
            math.log1p(volume / nat) / 6.0,
            float(np.min(lengths) / max(np.max(lengths), 1e-9)),
            float(np.std(lengths) / max(np.mean(lengths), 1e-9)),
        ],
        dtype=np.float32,
    )


def mode_features(values: list[Any]) -> tuple[np.ndarray, float]:
    modes = np.asarray([numeric(value, float("nan")) for value in values], dtype=np.float64)
    modes = modes[np.isfinite(modes)]
    if not len(modes):
        raise ValueError("empty_phonon_modes")
    positive = modes[modes > 0.0]
    if not len(positive):
        positive = np.asarray([0.0], dtype=np.float64)
    quantiles = np.quantile(positive, [0.10, 0.25, 0.50, 0.75, 0.90])
    clipped = np.clip(positive, 0.0, 2000.0)
    histogram, _ = np.histogram(clipped, bins=np.linspace(0.0, 2000.0, 17))
    histogram = histogram.astype(np.float64) / max(len(positive), 1)
    vector = np.asarray(
        [
            math.log1p(len(modes)) / 6.0,
            float(np.mean(modes < 0.0)),
            numeric(np.min(modes)) / 2000.0,
            numeric(np.max(positive)) / 2000.0,
            numeric(np.mean(positive)) / 1000.0,
            numeric(np.std(positive)) / 1000.0,
            *(quantiles / 2000.0).tolist(),
            *histogram.tolist(),
        ],
        dtype=np.float32,
    )
    return vector, float(np.max(positive))


def chemical_system(row: dict[str, Any]) -> str:
    return "-".join(sorted(set(row["atoms"]["elements"])))


def split_for_group(group: str) -> int:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    if sha256_file(source) != SOURCE_SHA256:
        raise RuntimeError("SOURCE_HASH_GATE")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != 1 or ".." in Path(names[0]).parts:
            raise RuntimeError("ARCHIVE_LAYOUT_GATE")
        rows = json.loads(archive.read(names[0]))
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    contract = contracts["CAND-P-143"]

    features, targets, groups, jids, frequency_proxy = [], [], [], [], []
    for row in rows:
        target = numeric(row.get("max_ir_mode"), float("nan"))
        modes = row.get("modes")
        if not math.isfinite(target) or target < 0.0 or not isinstance(modes, list) or not modes:
            continue
        mode_vector, maximum_frequency = mode_features(modes)
        atoms = row["atoms"]
        scalar = np.asarray(
            [
                numeric(row.get("formation_energy_peratom")),
                numeric(row.get("optb88vdw_bandgap")),
                numeric(row.get("mbj_bandgap")),
                float(row.get("mbj_bandgap") not in (None, "", "na")),
                numeric(row.get("density")) / 25.0,
                numeric(row.get("spg_number")) / 230.0,
                numeric(row.get("dfpt_piezo_max_dielectric")) / 100.0,
                numeric(row.get("dfpt_piezo_max_dielectric_electronic")) / 100.0,
                numeric(row.get("dfpt_piezo_max_dielectric_ionic")) / 100.0,
                numeric(row.get("bulk_modulus_kv")) / 400.0,
                numeric(row.get("shear_modulus_gv")) / 250.0,
                0.0,
                0.0,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate((composition(atoms["elements"]), structure_features(atoms), mode_vector, scalar)))
        targets.append(target)
        groups.append(chemical_system(row))
        jids.append(str(row["jid"]))
        frequency_proxy.append(maximum_frequency)
    if len(features) != 3307:
        raise RuntimeError(f"EXACT_RECORD_GATE:{len(features)}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    group_array = np.asarray(groups)
    split = np.asarray([split_for_group(group) for group in groups], dtype=np.int8)
    sets = {code: set(group_array[split == code]) for code in (0, 1, 2)}
    overlap = sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    active_counts = {
        name: {"active": int(np.sum(y[split == code] > ACTIVE_THRESHOLD)), "inactive": int(np.sum(y[split == code] <= ACTIVE_THRESHOLD))}
        for name, code in (("train", 0), ("validation", 1), ("test", 2))
    }
    if overlap or min(counts.values()) < 250 or min(item["inactive"] for item in active_counts.values()) < 10:
        raise RuntimeError(f"SPLIT_GATE:overlap={overlap}:counts={counts}:active={active_counts}")

    stage_root = root / "data" / "staged_jarvis_p143_exact_v1"
    stage_root.mkdir(parents=True, exist_ok=True)
    output = stage_root / "CAND-P-143.npz"
    np.savez_compressed(
        output,
        x=x,
        y=y,
        groups=group_array,
        jid=np.asarray(jids),
        split=split,
        frequency_proxy=np.asarray(frequency_proxy, dtype=np.float32),
        active_threshold=np.asarray(ACTIVE_THRESHOLD, dtype=np.float32),
        authority=np.asarray(0, dtype=np.int8),
        candidate_id=np.asarray("CAND-P-143"),
    )
    metadata = {
        "schema": "cimc.forge200.jarvis-ir-intensity-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-P-143",
        "truth_class": "OPEN_COMPUTED_DFT_IR_INTENSITY",
        "source_id": "jarvis_dft_v11",
        "source_url": SOURCE_URL,
        "source_pid": SOURCE_PID,
        "source_sha256": SOURCE_SHA256,
        "source_property": "max_ir_mode",
        "source_property_semantics": "published per-material maximum infrared mode intensity",
        "license": "CC BY 4.0",
        "records": len(x),
        "features": int(x.shape[1]),
        "groups": len(set(groups)),
        "counts": counts,
        "active_threshold": ACTIVE_THRESHOLD,
        "active_counts": active_counts,
        "split_unit": "CHEMICAL_SYSTEM_MATERIAL_FAMILY",
        "cross_split_group_overlap": overlap,
        "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(groups, split.tolist(), strict=True)))).hexdigest(),
        "input_contract": contract["input_contract"],
        "input_contract_state": "COMPOSITION_STRUCTURE_AND_FULL_PUBLISHED_PHONON_FREQUENCY_LIST_PRESENT;MODE_EIGENVECTOR_AND_BORN_CHARGE_MASKED_UNAVAILABLE",
        "input_availability": {"phonon_mode_frequencies": 1, "mode_eigenvectors": 0, "born_charge_features": 0},
        "target_label": contract["target_label"],
        "target_scope": "MAXIMUM_IR_MODE_INTENSITY_PER_MATERIAL_AND_ACTIVE_CLASS_AT_FIXED_0P1_THRESHOLD",
        "baseline": contract["baseline"],
        "baseline_execution": "TRAIN_ONLY_MAX_POSITIVE_PHONON_FREQUENCY_QUANTILE_BIN_MEAN_LOG_INTENSITY",
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "teacher_outputs": 0,
        "experimental_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema": "cimc.forge200.jarvis-p143-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "record": metadata,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    evidence = root / "evidence" / "jarvis_p143_staging.v1.json"
    evidence.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": len(x), "features": int(x.shape[1]), "counts": counts, "active_counts": active_counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
