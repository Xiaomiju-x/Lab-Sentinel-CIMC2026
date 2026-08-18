#!/usr/bin/env python3
"""Restage the five material/process tasks whose first GPU inputs were incomplete.

This is a corrective, fail-closed staging pass.  It keeps the licensed JARVIS
and UCI labels and the already frozen group assignments, but binds every model
to the input fields named by its task contract.  It never changes authority or
promotes a host artifact to a board-accepted model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENTS)}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
MEV_PER_A2_TO_J_PER_M2 = 0.01602176634


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric(value: Any) -> float:
    if isinstance(value, bool):
        return math.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def composition(formula: str) -> np.ndarray | None:
    vector = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count_text in re.findall(
        r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula
    ):
        if symbol not in ELEMENT_INDEX:
            return None
        vector[ELEMENT_INDEX[symbol]] += float(count_text) if count_text else 1.0
    total = float(vector.sum())
    if total <= 0:
        return None
    return vector / total


def dim_code(value: Any) -> float:
    text = str(value or "").lower()
    if text.startswith("2d"):
        return 1.0
    if text.startswith("1d"):
        return 0.5
    if text.startswith("0d"):
        return 0.25
    return 0.0


def common_features(row: dict[str, Any]) -> np.ndarray | None:
    comp = composition(str(row.get("formula", "")))
    if comp is None:
        return None
    return np.concatenate(
        (
            comp,
            np.asarray(
                [
                    numeric(row.get("density")),
                    numeric(row.get("nat")),
                    numeric(row.get("spg_number") or row.get("spg")),
                    dim_code(row.get("dimensionality")),
                ],
                dtype=np.float32,
            ),
        )
    )


def gap_features(row: dict[str, Any]) -> np.ndarray:
    values = np.asarray(
        [
            numeric(row.get("optb88vdw_bandgap")),
            numeric(row.get("mbj_bandgap")),
            numeric(row.get("hse_gap")),
        ],
        dtype=np.float32,
    )
    return np.concatenate((values, np.isfinite(values).astype(np.float32)))


def structure_features(row: dict[str, Any]) -> np.ndarray:
    atoms = row.get("atoms") if isinstance(row.get("atoms"), dict) else {}
    abc = np.asarray(atoms.get("abc", [math.nan] * 3), dtype=np.float32)
    angles = np.asarray(atoms.get("angles", [math.nan] * 3), dtype=np.float32)
    lattice = np.asarray(atoms.get("lattice_mat", []), dtype=np.float32)
    coords = np.asarray(atoms.get("coords", []), dtype=np.float32)
    spans = np.full(3, math.nan, dtype=np.float32)
    if coords.ndim == 2 and coords.shape[1] == 3 and len(coords):
        spans = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
    volume = math.nan
    if lattice.shape == (3, 3):
        volume = abs(float(np.linalg.det(lattice)))
    nat = max(numeric(row.get("nat")), 1.0)
    gap_proxy = float(np.nanmax(abc - spans)) if np.all(np.isfinite(abc - spans)) else math.nan
    return np.concatenate(
        (
            abc,
            angles,
            spans,
            np.asarray([volume / nat, gap_proxy], dtype=np.float32),
        )
    )


def elastic_features(row: dict[str, Any]) -> np.ndarray:
    tensor = np.asarray(row.get("elastic_tensor", []), dtype=np.float32)
    upper = np.full(21, math.nan, dtype=np.float32)
    if tensor.shape == (6, 6):
        upper = tensor[np.triu_indices(6)].astype(np.float32)
    scalar = np.asarray(
        [
            numeric(row.get("bulk_modulus_kv")),
            numeric(row.get("shear_modulus_gv")),
            numeric(row.get("poisson")),
            numeric(row.get("epsx")),
            numeric(row.get("epsy")),
            numeric(row.get("epsz")),
            numeric(row.get("dfpt_piezo_max_dielectric_electronic")),
            numeric(row.get("dfpt_piezo_max_dielectric_ionic")),
        ],
        dtype=np.float32,
    )
    return np.concatenate((scalar, upper))


def contract_hashes(root: Path) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    rows = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    selected = {row["candidate_id"]: row for row in rows}
    return (
        {key: hashlib.sha256(canonical_bytes(row)).hexdigest() for key, row in selected.items()},
        selected,
    )


def save_dataset(
    root: Path,
    candidate_id: str,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    output = root / "data" / "staged_contract_v2" / f"{candidate_id}.npz"
    split = arrays["split"].astype(np.int8)
    groups = arrays["groups"].astype(str)
    counts = {
        name: int(np.sum(split == code))
        for name, code in SPLIT_CODE.items()
    }
    group_sets = {code: set(groups[split == code]) for code in SPLIT_CODE.values()}
    overlap = sum(
        len(group_sets[a] & group_sets[b])
        for a, b in ((0, 1), (0, 2), (1, 2))
    )
    errors = []
    if min(counts.values()) < 16:
        errors.append("INSUFFICIENT_SPLIT_ROWS")
    if overlap:
        errors.append("GROUP_SPLIT_LEAKAGE")
    if arrays["x"].ndim != 2 or not len(arrays["x"]):
        errors.append("INVALID_FEATURE_MATRIX")
    status = "PASS" if not errors else "FAIL_CLOSED"
    if status == "PASS":
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            **arrays,
            candidate_id=np.asarray(candidate_id),
            task_kind=np.asarray(metadata["task_kind"]),
            truth_class=np.asarray(metadata["truth_class"]),
            authority=np.asarray(0, dtype=np.int8),
        )
    record = {
        "schema": "cimc.forge200.contract-exact-staged-dataset.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "counts": counts,
        "records": int(len(arrays["x"])),
        "features": int(arrays["x"].shape[1]),
        "cross_split_group_overlap": overlap,
        "fit_preprocessing_on_train_only": True,
        **metadata,
        "errors": errors,
    }
    if status == "PASS":
        record.update(
            {
                "path": str(output.relative_to(root)).replace("\\", "/"),
                "bytes": output.stat().st_size,
                "sha256": sha256_file(output),
            }
        )
    write_json(output.with_suffix(".metadata.json"), record)
    return record


def stage_jarvis(root: Path, hashes: dict[str, str], contracts: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    artifact = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    assignment_path = root / "data" / "splits" / "jarvis_dft_v11.assignments.tsv"
    assignments = {row["jid"]: row for row in read_tsv(assignment_path)}
    with zipfile.ZipFile(artifact) as archive:
        rows = json.load(archive.open(archive.namelist()[0]))
    rows_by_id = {str(row.get("jid")): row for row in rows}
    results: list[dict[str, Any]] = []

    definitions = {
        "CAND-P-069": {
            "target": "formation_energy_peratom",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row),
                    np.asarray([numeric(row.get("encut")), numeric(row.get("kpoint_length_unit"))], dtype=np.float32),
                )
            ),
            "feature_contract": "composition+crystal_lattice_geometry+bandgap_and_DFT_provenance_metadata",
            "target_units": "eV_per_atom",
            "baseline_fields": "crystal_dimensionality_chemistry_family_train_mean",
        },
        "CAND-P-071": {
            "target": "mbj_bandgap",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row)[[0, 2, 3, 5]],
                    np.asarray([numeric(row.get("encut")), numeric(row.get("kpoint_length_unit"))], dtype=np.float32),
                )
            ),
            "feature_contract": "composition+structure+PBE_gap+HSE_gap+gap_masks+TBmBJ_computation_metadata",
            "target_units": "eV",
            "baseline_fields": "optb88vdw_PBE_gap_train_linear_correction",
        },
        "CAND-P-072": {
            "target": "dfpt_piezo_max_dielectric",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate((common_features(row), gap_features(row))),
            "feature_contract": "composition_118+density+nat+space_group+dimensionality+three_bandgaps+bandgap_presence_masks",
            "target_units": "relative_static_dielectric_constant",
            "baseline_fields": "optb88vdw_bandgap_train_fitted_inverse_linear",
        },
        "CAND-P-078": {
            "target": "exfoliation_energy",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row),
                    structure_features(row),
                    gap_features(row),
                    np.asarray(
                        [
                            numeric(row.get("formation_energy_peratom")),
                            numeric(row.get("encut")),
                            numeric(row.get("kpoint_length_unit")),
                        ],
                        dtype=np.float32,
                    ),
                )
            ),
            "feature_contract": "layered_composition+lattice_abc_angles+coordinate_span+interlayer_gap_proxy+volume_per_atom+DFT_metadata",
            "target_units": "J_per_m2_converted_from_JARVIS_meV_per_A2",
            "baseline_fields": "crystal_system_and_dimensionality_family_train_median",
        },
        "CAND-P-086": {
            "target": "min_ir_mode",
            "task_kind": "classification",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row),
                    gap_features(row),
                    np.asarray(
                        [
                            numeric(row.get("formation_energy_peratom")),
                            numeric(row.get("bulk_modulus_kv")),
                            numeric(row.get("shear_modulus_gv")),
                            numeric(row.get("poisson")),
                            numeric(row.get("avg_elec_mass")),
                            numeric(row.get("avg_hole_mass")),
                            numeric(row.get("max_ir_mode")),
                            numeric(row.get("encut")),
                            numeric(row.get("kpoint_length_unit")),
                        ],
                        dtype=np.float32,
                    ),
                )
            ),
            "feature_contract": "composition+structure+formation_energy+elastic_and_IR_force_constant_proxies+DFT_metadata",
            "target_units": "stable_1_if_min_IR_mode_nonnegative_else_unstable_0",
            "baseline_fields": "formation_energy_per_atom_train_fitted_threshold",
        },
        "CAND-P-074": {
            "target": "avg_elec_mass",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row),
                    np.asarray([numeric(row.get("formation_energy_peratom")), numeric(row.get("encut")), numeric(row.get("kpoint_length_unit"))], dtype=np.float32),
                )
            ),
            "feature_contract": "composition+structure+band_edge_gap_features+DFT_metadata",
            "target_units": "electron_mass_m0",
            "baseline_fields": "crystal_dimensionality_material_family_train_median",
        },
        "CAND-P-075": {
            "target": "avg_hole_mass",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row),
                    np.asarray([numeric(row.get("formation_energy_peratom")), numeric(row.get("encut")), numeric(row.get("kpoint_length_unit"))], dtype=np.float32),
                )
            ),
            "feature_contract": "composition+structure+valence_band_gap_features+DFT_metadata",
            "target_units": "hole_mass_m0",
            "baseline_fields": "crystal_dimensionality_material_family_train_median",
        },
        "CAND-P-076": {
            "target": "bulk_modulus_kv",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row), elastic_features(row),
                )
            ),
            "feature_contract": "composition+structure+density+elastic_tensor_and_dielectric_metadata",
            "target_units": "GPa",
            "baseline_fields": "density_train_linear_regression",
        },
        "CAND-P-077": {
            "target": "shear_modulus_gv",
            "task_kind": "regression",
            "feature_builder": lambda row: np.concatenate(
                (
                    common_features(row), structure_features(row), gap_features(row), elastic_features(row),
                )
            ),
            "feature_contract": "composition+structure+density+bulk_poisson_elastic_tensor_metadata",
            "target_units": "GPa",
            "baseline_fields": "fixed_Poisson_0p25_relation_from_bulk_modulus",
        },
    }
    for candidate_id, spec in definitions.items():
        x, y, groups, splits, baseline_group = [], [], [], [], []
        for jid, assignment in assignments.items():
            row = rows_by_id.get(jid)
            if row is None:
                continue
            target = numeric(row.get(spec["target"]))
            base = common_features(row)
            if not math.isfinite(target) or base is None:
                continue
            vector = spec["feature_builder"](row)
            x.append(vector)
            if candidate_id == "CAND-P-078":
                y.append(target * MEV_PER_A2_TO_J_PER_M2)
            elif spec["task_kind"] == "classification":
                y.append(int(target >= 0.0))
            else:
                y.append(target)
            groups.append(assignment["chemical_system_group"])
            splits.append(SPLIT_CODE[assignment["split"]])
            baseline_group.append(
                f"{row.get('crys', 'UNKNOWN')}|{row.get('dimensionality', 'UNKNOWN')}"
            )
        arrays = {
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64 if spec["task_kind"] == "classification" else np.float32),
            "groups": np.asarray(groups),
            "split": np.asarray(splits, dtype=np.int8),
            "baseline_group": np.asarray(baseline_group),
        }
        results.append(
            save_dataset(
                root,
                candidate_id,
                arrays,
                {
                    "task_kind": spec["task_kind"],
                    "truth_class": "OPEN_COMPUTED_DFT",
                    "source_id": "jarvis_dft_v11",
                    "source_sha256": sha256_file(artifact),
                    "split_sha256": sha256_file(assignment_path),
                    "task_contract_sha256": hashes[candidate_id],
                    "input_contract": contracts[candidate_id]["input_contract"],
                    "target_label": contracts[candidate_id]["target_label"],
                    "baseline": contracts[candidate_id]["baseline"],
                    "primary_metric": contracts[candidate_id]["primary_metric"],
                    "parameter_cap": contracts[candidate_id]["parameter_cap"],
                    "feature_contract": spec["feature_contract"],
                    "target_field": spec["target"],
                    "target_units": spec["target_units"],
                    "baseline_fields": spec["baseline_fields"],
                },
            )
        )

    candidate_id = "CAND-P-140"
    x, y, groups, splits = [], [], [], []
    for row in rows:
        target = numeric(row.get("dfpt_piezo_max_dij"))
        base = common_features(row)
        if not math.isfinite(target) or base is None:
            continue
        prototype = str(
            row.get("spg_symbol") or row.get("spg_number") or row.get("spg") or "UNKNOWN"
        )
        bucket = int(hashlib.sha256(prototype.encode("utf-8")).hexdigest()[:8], 16) % 100
        split = 0 if bucket < 70 else 1 if bucket < 85 else 2
        x.append(np.concatenate((base, gap_features(row), elastic_features(row))))
        y.append(target)
        groups.append(prototype)
        splits.append(split)
    results.append(
        save_dataset(
            root,
            candidate_id,
            {
                "x": np.asarray(x, dtype=np.float32),
                "y": np.asarray(y, dtype=np.float32),
                "groups": np.asarray(groups),
                "split": np.asarray(splits, dtype=np.int8),
            },
            {
                "task_kind": "regression",
                "truth_class": "OPEN_COMPUTED_DFT",
                "source_id": "jarvis_dft_v11",
                "source_sha256": sha256_file(artifact),
                "split_sha256": hashlib.sha256(
                    canonical_bytes(sorted(zip(groups, splits, strict=True)))
                ).hexdigest(),
                "task_contract_sha256": hashes[candidate_id],
                "input_contract": contracts[candidate_id]["input_contract"],
                "target_label": contracts[candidate_id]["target_label"],
                "baseline": contracts[candidate_id]["baseline"],
                "primary_metric": contracts[candidate_id]["primary_metric"],
                "parameter_cap": contracts[candidate_id]["parameter_cap"],
                "feature_contract": "composition+structure+symmetry+bandgap+elastic_tensor_upper_triangle+dielectric_descriptors",
                "target_field": "dfpt_piezo_max_dij",
                "target_units": "JARVIS_DFPT_max_abs_dij_computed",
                "baseline_fields": "composition_descriptors_gradient_boosting_train_only",
            },
        )
    )
    return results


def stage_secom(root: Path, hashes: dict[str, str], contracts: dict[str, dict[str, str]]) -> dict[str, Any]:
    candidate_id = "CAND-P-087"
    artifact = root / "data" / "raw" / "uci_secom" / "secom.zip"
    legacy_assignment_path = root / "data" / "splits" / "uci_secom.assignments.tsv"
    legacy_rows = read_tsv(legacy_assignment_path)
    # The first split cut two calendar-day groups at row-count boundaries.  A
    # time-group contract must keep every day wholly in one split, so freeze a
    # corrected chronological day-family assignment here.
    ordered_groups = sorted({row["time_group"] for row in legacy_rows})
    train_end = max(1, int(len(ordered_groups) * 0.70))
    validation_end = max(train_end + 1, int(len(ordered_groups) * 0.85))
    group_split = {
        group: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, group in enumerate(ordered_groups)
    }
    corrected_rows = [
        {**row, "split": group_split[row["time_group"]]} for row in legacy_rows
    ]
    assignments = {int(row["row_index"]): row for row in corrected_rows}
    assignment_path = (
        root / "data" / "splits" / "uci_secom.time_group_v2.assignments.tsv"
    )
    assignment_path.parent.mkdir(parents=True, exist_ok=True)
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("row_index", "pass_fail_label", "timestamp", "time_group", "split"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(corrected_rows)
    with zipfile.ZipFile(artifact) as archive:
        data_lines = archive.read("secom.data").decode("utf-8").splitlines()
        label_lines = archive.read("secom_labels.data").decode("utf-8").splitlines()
    x, y, groups, splits = [], [], [], []
    for index, (data_line, label_line) in enumerate(
        zip(data_lines, label_lines, strict=True)
    ):
        values = np.asarray(
            [float(value) if value != "NaN" else math.nan for value in data_line.split()],
            dtype=np.float32,
        )
        missing = (~np.isfinite(values)).astype(np.float32)
        x.append(np.concatenate((values, missing)))
        y.append(1 if int(label_line.split(maxsplit=1)[0]) == 1 else 0)
        groups.append(assignments[index]["time_group"])
        splits.append(SPLIT_CODE[assignments[index]["split"]])
    return save_dataset(
        root,
        candidate_id,
        {
            "x": np.asarray(x, dtype=np.float32),
            "y": np.asarray(y, dtype=np.int64),
            "groups": np.asarray(groups),
            "split": np.asarray(splits, dtype=np.int8),
        },
        {
            "task_kind": "classification",
            "truth_class": "ANONYMIZED_PRODUCTION_SECOM",
            "source_id": "uci_secom",
            "source_sha256": sha256_file(artifact),
            "split_sha256": sha256_file(assignment_path),
            "task_contract_sha256": hashes[candidate_id],
            "input_contract": contracts[candidate_id]["input_contract"],
            "target_label": contracts[candidate_id]["target_label"],
            "baseline": contracts[candidate_id]["baseline"],
            "primary_metric": contracts[candidate_id]["primary_metric"],
            "parameter_cap": contracts[candidate_id]["parameter_cap"],
            "feature_contract": "SECOM_590_sensor_values+590_explicit_missingness_mask;time_group_split",
            "split_correction": "chronological_calendar_day_groups_70_15_15_no_day_crosses_splits",
            "target_field": "pass_fail_yield_label",
            "target_units": "binary_fail_1_pass_0",
            "baseline_fields": "same_train_imputation_and_scaling_regularized_logistic_regression",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    hashes, contracts = contract_hashes(root)
    records = stage_jarvis(root, hashes, contracts)
    records.append(stage_secom(root, hashes, contracts))
    errors = [f"{item['candidate_id']}:{error}" for item in records for error in item["errors"]]
    content = {"records": records, "errors": errors}
    manifest = {
        "schema": "cimc.forge200.material-contract-correction.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "candidate_count": len(records),
        "passed": sum(item["status"] == "PASS" for item in records),
        "authority_nonzero": 0,
        "board_accepted": 0,
        "countable_models": 0,
        "correction_reason": "FIRST_GPU_STAGING_OMITTED_CONTRACT_INPUT_FIELDS",
        **content,
        "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }
    write_json(root / "evidence" / "material_contract_data_correction.v2.json", manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "candidate_count": manifest["candidate_count"],
                "passed": manifest["passed"],
                "content_root_sha256": manifest["content_root_sha256"],
                "errors": errors,
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
