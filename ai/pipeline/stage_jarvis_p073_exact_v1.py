#!/usr/bin/env python3
"""Stage JARVIS LOPTICS spectra for the P073 absorption-onset contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from download_jarvis_p073_subset_v1 import JARVIS_SHA256, INDEX_SHA256, md5_file
from stage_matbench_experimental_v1 import canonical_bytes, features, parse_formula, sha256_file, write_json


HC_EV_CM = 1.239841984e-4
ABSORPTION_THRESHOLD_CM_INV = 1.0e4
SUSTAINED_POINTS = 5
SPLIT_NAMES = ("train", "validation", "test")


def load_json_zip(path: Path) -> Any:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError(f"SINGLE_MEMBER_GATE:{path.name}")
        return json.loads(archive.read(names[0]))


def dielectric_rows(node: ET.Element, kind: str) -> np.ndarray:
    rows = []
    for item in node.findall(f"./{kind}/array/set/r"):
        values = [float(value) for value in (item.text or "").split()]
        if len(values) != 7:
            raise ValueError(f"DIELECTRIC_ROW_WIDTH:{kind}:{len(values)}")
        rows.append(values)
    result = np.asarray(rows, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] < 100:
        raise ValueError(f"DIELECTRIC_GRID:{kind}:{result.shape}")
    return result


def absorption_onset(path: Path) -> tuple[float, dict[str, float | int]]:
    with zipfile.ZipFile(path) as archive:
        if "vasprun.xml" not in archive.namelist():
            raise ValueError("VASPRUN_ABSENT")
        root = ET.fromstring(archive.read("vasprun.xml"))
    nodes = root.findall(".//dielectricfunction")
    if len(nodes) != 1:
        raise ValueError(f"DIELECTRIC_FUNCTION_COUNT:{len(nodes)}")
    imaginary = dielectric_rows(nodes[0], "imag")
    real = dielectric_rows(nodes[0], "real")
    if imaginary.shape != real.shape or not np.allclose(imaginary[:, 0], real[:, 0], atol=1e-7):
        raise ValueError("DIELECTRIC_GRID_MISMATCH")
    energy = real[:, 0]
    epsilon_real = np.mean(real[:, 1:4], axis=1)
    epsilon_imag = np.mean(imaginary[:, 1:4], axis=1)
    magnitude = np.sqrt(epsilon_real**2 + epsilon_imag**2)
    extinction = np.sqrt(np.maximum((magnitude - epsilon_real) / 2.0, 0.0))
    absorption = 4.0 * math.pi * np.maximum(energy, 0.0) * extinction / HC_EV_CM
    active = (energy >= 0.05) & np.isfinite(absorption) & (absorption >= ABSORPTION_THRESHOLD_CM_INV)
    sustained = np.convolve(active.astype(np.int8), np.ones(SUSTAINED_POINTS, dtype=np.int8), mode="valid")
    hits = np.flatnonzero(sustained == SUSTAINED_POINTS)
    if not len(hits):
        raise ValueError("NO_SUSTAINED_ABSORPTION_ONSET")
    onset = float(energy[int(hits[0])])
    if not 0.05 <= onset <= 20.0:
        raise ValueError(f"ONSET_RANGE:{onset}")
    return onset, {
        "grid_points": int(len(energy)),
        "energy_step_median_eV": float(np.median(np.diff(energy))),
        "energy_max_eV": float(np.max(energy)),
    }


def balanced_assignment(systems: list[str]) -> dict[str, int]:
    weights = Counter(systems)
    targets = np.asarray([0.70, 0.15, 0.15], dtype=np.float64) * len(systems)
    totals = np.zeros(3, dtype=np.int64)
    result: dict[str, int] = {}
    for system in sorted(weights, key=lambda value: (-weights[value], value)):
        split = int(np.argmax((targets - totals) / np.maximum(targets, 1.0)))
        result[system] = split
        totals[split] += weights[system]
    return result


def overlap_count(groups: np.ndarray, split: np.ndarray) -> int:
    sets = {code: set(groups[split == code].tolist()) for code in (0, 1, 2)}
    return sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))


def model_features(row: dict[str, Any], spectrum_meta: dict[str, float | int]) -> np.ndarray:
    composition = features(parse_formula(str(row["formula"])), "structure")
    values = [
        float(row.get("density") or 0.0) / 25.0,
        float(row.get("nat") or len(row["atoms"]["elements"])) / 100.0,
        float(row.get("spg_number") or 0.0) / 230.0,
        float(row.get("formation_energy_peratom") or 0.0) / 10.0,
        math.log1p(int(spectrum_meta["grid_points"])) / 10.0,
        float(spectrum_meta["energy_step_median_eV"]),
        float(spectrum_meta["energy_max_eV"]) / 50.0,
    ]
    return np.concatenate((composition, np.asarray(values, dtype=np.float32)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source_root = root / "data" / "raw" / "jarvis_optics_p073_v1"
    selection_path = source_root / "selection_manifest.v1.json"
    receipt_path = root / "evidence" / "jarvis_p073_download_receipt.v1.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if selection["count"] != 500 or receipt["status"] != "PASS" or receipt["verified"] != 500 or receipt["selection_manifest_sha256"] != sha256_file(selection_path):
        raise RuntimeError("DOWNLOAD_RECEIPT_GATE")
    jarvis_path = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    index_path = root / "data" / "raw" / "jarvis_raw_index_v1" / "figshare_data-10-28-2020.json.zip"
    if sha256_file(jarvis_path) != JARVIS_SHA256 or sha256_file(index_path) != INDEX_SHA256:
        raise RuntimeError("SOURCE_HASH_GATE")
    jarvis = {row["jid"]: row for row in load_json_zip(jarvis_path)}
    systems = [record["chemical_system"] for record in selection["records"]]
    assignment = balanced_assignment(systems)

    x, y, baseline, groups, jids, splits = [], [], [], [], [], []
    source_records, rejection_counts = [], Counter()
    for record in selection["records"]:
        path = source_root / record["name"]
        expected_md5 = record.get("computed_md5") or record.get("supplied_md5")
        if not path.exists() or path.stat().st_size != int(record["size"]) or md5_file(path) != expected_md5:
            raise RuntimeError(f"RAW_FILE_IDENTITY_GATE:{record['jid']}")
        try:
            target, spectrum_meta = absorption_onset(path)
            row = jarvis[record["jid"]]
            gap = float(row["optb88vdw_bandgap"])
            if not math.isfinite(gap) or gap <= 0.05:
                raise ValueError("BASELINE_GAP_GATE")
            x.append(model_features(row, spectrum_meta))
            y.append(target)
            baseline.append(gap)
            groups.append(record["chemical_system"])
            jids.append(record["jid"])
            splits.append(assignment[record["chemical_system"]])
            source_records.append({"jid": record["jid"], "name": record["name"], "bytes": record["size"], "md5": expected_md5, "onset_definition_version": 1})
        except (ValueError, KeyError, ET.ParseError, zipfile.BadZipFile) as exc:
            rejection_counts[f"{type(exc).__name__}:{exc}"] += 1
    if len(x) < 400:
        raise RuntimeError(f"ELIGIBLE_SPECTRUM_GATE:{len(x)}")
    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    baseline_array = np.asarray(baseline, dtype=np.float32)
    group_array = np.asarray(groups)
    split_array = np.asarray(splits, dtype=np.int8)
    if overlap_count(group_array, split_array):
        raise RuntimeError("CHEMICAL_SYSTEM_SPLIT_LEAKAGE_GATE")
    split_counts = {SPLIT_NAMES[code]: int(np.sum(split_array == code)) for code in (0, 1, 2)}
    if min(split_counts.values()) < 40:
        raise RuntimeError(f"SPLIT_SIZE_GATE:{split_counts}")

    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    candidate_id = "CAND-P-073"
    stage_root = root / "data" / "staged_jarvis_p073_exact_v1"
    stage_root.mkdir(parents=True, exist_ok=True)
    output = stage_root / f"{candidate_id}.npz"
    np.savez_compressed(output, x=x_array, y=y_array, baseline_pred=baseline_array, family=group_array, component_group=group_array, jid=np.asarray(jids), split=split_array)
    source_root_hash = hashlib.sha256(canonical_bytes(sorted(source_records, key=lambda item: item["jid"]))).hexdigest()
    metadata = {
        "schema": "cimc.forge200.jarvis-p073-contract-exact-dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": candidate_id,
        "records": len(y_array),
        "features": int(x_array.shape[1]),
        "split_counts": split_counts,
        "split_method": "TARGET_BLIND_ELEMENT_SYSTEM_GROUP_BALANCED_70_15_15",
        "cross_split_component_overlap": overlap_count(group_array, split_array),
        "cross_split_family_overlap": overlap_count(group_array, split_array),
        "source_selected": 500,
        "source_rejections": dict(sorted(rejection_counts.items())),
        "source_content_root_sha256": source_root_hash,
        "source_selection_manifest_sha256": sha256_file(selection_path),
        "source_download_receipt_sha256": sha256_file(receipt_path),
        "source_index_sha256": INDEX_SHA256,
        "source_jarvis_table_sha256": JARVIS_SHA256,
        "source_pid": "10.6084/m9.figshare.13154159",
        "source_license": "CC-BY-4.0",
        "truth_class": "OPEN_COMPUTED_DFT_DIELECTRIC_SPECTRUM_DERIVED",
        "target_scope": "isotropic optical absorption onset from the raw LOPTICS complex dielectric function",
        "target_definition": {"absorption_threshold_cm_inverse": ABSORPTION_THRESHOLD_CM_INV, "minimum_energy_eV": 0.05, "sustained_points": SUSTAINED_POINTS, "formula": "alpha_cm_inverse=4*pi*energy_eV*extinction_coefficient/(hc_eV_cm)"},
        "baseline_execution": "PUBLISHED_OPTB88VDW_BANDGAP_AS_ABSORPTION_ONSET",
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contracts[candidate_id])).hexdigest(),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    write_json(root / "evidence" / "jarvis_p073_exact_staging.v1.json", {"schema": "cimc.forge200.jarvis-p073-exact-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "record": metadata, "authority_nonzero": 0, "board_actions": 0})
    print(json.dumps({"status": "PASS", "records": len(y_array), "split_counts": split_counts, "rejections": dict(rejection_counts)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
