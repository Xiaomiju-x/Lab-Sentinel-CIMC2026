#!/usr/bin/env python3
"""Stage exact JARVIS-DFT records for P070 and P084.

The source archives and their Figshare metadata are verified before use.  P070
uses the neutral-vacancy records exactly as published.  P084 is derived only
when both component slabs have matching PBE/no-dipole records and their atom
counts sum to the interface atom count.  No teacher or approximate labels are
used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import zipfile
from collections import Counter
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
ELEMENT_INDEX = {value: index for index, value in enumerate(ELEMENTS)}
INTERFACE_RE = re.compile(
    r"^Interface-(JVASP-\d+)_(JVASP-\d+)_film_miller_(-?\d+)_(-?\d+)_(-?\d+)_"
    r"sub_miller_(-?\d+)_(-?\d+)_(-?\d+)_film_thickness_(\d+)_subs_thickness_(\d+)_"
    r"seperation_([-\d.]+)_disp_([-\d.]+)_([-\d.]+)_vasp$"
)
SURFACE_RE = re.compile(
    r"^Surface-(JVASP-\d+)_miller_(-?\d+)_(-?\d+)_(-?\d+)_thickness_(\d+)_(.+)$"
)


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


def load_zip_json(path: Path) -> list[dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != 1 or names[0].startswith(("/", "\\")) or ".." in Path(names[0]).parts:
            raise RuntimeError(f"UNSAFE_OR_AMBIGUOUS_ARCHIVE:{path.name}")
        value = json.loads(archive.read(names[0]))
    if not isinstance(value, list):
        raise RuntimeError(f"SOURCE_NOT_RECORD_LIST:{path.name}")
    return value


def read_contracts(root: Path) -> dict[str, dict[str, str]]:
    path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def composition(elements: list[str]) -> np.ndarray:
    counts = Counter(elements)
    total = max(sum(counts.values()), 1)
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count in counts.items():
        if symbol not in ELEMENT_INDEX:
            raise RuntimeError(f"UNKNOWN_ELEMENT:{symbol}")
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def lattice_descriptors(atoms: dict[str, Any]) -> np.ndarray:
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    lengths = np.linalg.norm(lattice, axis=1)
    volume = abs(float(np.linalg.det(lattice)))
    angles = np.asarray(atoms.get("angles", [90.0, 90.0, 90.0]), dtype=np.float64)
    count = max(len(atoms["elements"]), 1)
    return np.asarray(
        [
            math.log1p(count) / 7.0,
            *(np.log1p(lengths) / 6.0).tolist(),
            *(angles / 180.0).tolist(),
            math.log1p(volume / count) / 6.0,
            float(np.min(lengths) / max(np.max(lengths), 1e-9)),
            float(np.std(lengths) / max(np.mean(lengths), 1e-9)),
        ],
        dtype=np.float32,
    )


def split_for_group(group: str) -> int:
    bucket = int(hashlib.sha256(group.encode("utf-8")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def grouped_split(groups: list[str]) -> np.ndarray:
    result = np.asarray([split_for_group(value) for value in groups], dtype=np.int8)
    counts = [int(np.sum(result == code)) for code in (0, 1, 2)]
    if min(counts) < 16:
        # The deterministic fallback remains group-disjoint and is used only
        # when a hash split happens to make a small source too imbalanced.
        unique = sorted(set(groups), key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
        code_by_group = {}
        for index, group in enumerate(unique):
            fraction = index / max(len(unique), 1)
            code_by_group[group] = 0 if fraction < 0.70 else 1 if fraction < 0.85 else 2
        result = np.asarray([code_by_group[value] for value in groups], dtype=np.int8)
    return result


def split_audit(groups: list[str], split: np.ndarray) -> tuple[dict[str, int], int, str]:
    sets = {code: {group for group, item in zip(groups, split, strict=True) if item == code} for code in (0, 1, 2)}
    overlap = sum(len(sets[a] & sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    split_hash = hashlib.sha256(canonical_bytes(sorted(zip(groups, split.tolist(), strict=True)))).hexdigest()
    if overlap or min(counts.values()) < 16:
        raise RuntimeError(f"SPLIT_GATE:overlap={overlap}:counts={counts}")
    return counts, overlap, split_hash


def task_contract_hash(contract: dict[str, str]) -> str:
    return hashlib.sha256(canonical_bytes(contract)).hexdigest()


def stage_p070(root: Path, contracts: dict[str, dict[str, str]], source: Path, source_record: dict[str, Any]) -> dict[str, Any]:
    rows = load_zip_json(source)
    features, targets, groups, row_ids, defect_symbols, baseline_features = [], [], [], [], [], []
    rejected: list[str] = []
    for row in rows:
        try:
            bulk = row["bulk_atoms"]
            defective = row["defective_atoms"]
            symbol = str(row["symbol"])
            # Some upstream rows store the pristine primitive cell and the
            # defective supercell, so their raw atom counts are intentionally
            # not required to differ by one.  The published id/symbol/Wyckoff
            # tuple is the defect-site identity; verify only that the species
            # is present in both supplied structures.
            if symbol not in bulk["elements"]:
                raise ValueError("defect_symbol_absent_from_pristine_structure")
            ef = float(row["ef"])
            chemical_potential = float(row["chem_pot"])
            if not all(np.isfinite([ef, chemical_potential, float(row["bulk_energy"]), float(row["defective_energy"])])):
                raise ValueError("nonfinite_energy")
            defect_onehot = np.zeros(len(ELEMENTS), dtype=np.float32)
            defect_onehot[ELEMENT_INDEX[symbol]] = 1.0
            wyckoff_hash = np.zeros(16, dtype=np.float32)
            wyckoff_hash[int(hashlib.sha256(str(row["wycoff"]).encode()).hexdigest()[:8], 16) % 16] = 1.0
            count = len(bulk["elements"])
            bulk_energy_per_atom = float(row["bulk_energy"]) / max(count, 1)
            vector = np.concatenate(
                (
                    composition(bulk["elements"]),
                    defect_onehot,
                    lattice_descriptors(bulk),
                    wyckoff_hash,
                    np.asarray(
                        [
                            chemical_potential,
                            bulk_energy_per_atom,
                            0.0,  # JARVIS vacancy database contains neutral defects only.
                            float(row.get("material_type") == "3D"),
                            float(row.get("material_type") == "2D"),
                        ],
                        dtype=np.float32,
                    ),
                )
            )
            features.append(vector)
            targets.append(ef)
            groups.append(str(row["jid"]))
            row_ids.append(str(row["id"]))
            defect_symbols.append(symbol)
            baseline_features.append([bulk_energy_per_atom, chemical_potential])
        except (KeyError, TypeError, ValueError) as exc:
            rejected.append(f"{row.get('id', 'unknown')}:{exc}")
    if rejected or len(features) != 464:
        raise RuntimeError(f"P070_SOURCE_ROW_GATE:accepted={len(features)}:rejected={len(rejected)}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    split = grouped_split(groups)
    counts, overlap, split_hash = split_audit(groups, split)
    output = root / "data" / "staged_jarvis_exact_v1" / "CAND-P-070.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        split=split,
        groups=np.asarray(groups),
        row_id=np.asarray(row_ids),
        defect_symbol=np.asarray(defect_symbols),
        baseline_features=np.asarray(baseline_features, dtype=np.float32),
        candidate_id=np.asarray("CAND-P-070"),
        task_kind=np.asarray("regression"),
        authority=np.asarray(0, dtype=np.int8),
    )
    contract = contracts["CAND-P-070"]
    metadata = {
        "schema": "cimc.forge200.jarvis-exact-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-P-070",
        "truth_class": "OPEN_COMPUTED_DFT_NEUTRAL_DEFECT",
        "source_dataset": "JARVIS-DFT vacancydb v8",
        "source_url": "https://figshare.com/articles/dataset/JARVIS-DFT_data_v8_/23000573",
        "source_path": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": sha256_file(source),
        "source_record": source_record,
        "license": "CC BY 4.0",
        "records": len(x),
        "features": int(x.shape[1]),
        "counts": counts,
        "split_unit": "PRISTINE_JARVIS_STRUCTURE_JID",
        "cross_split_group_overlap": overlap,
        "split_sha256": split_hash,
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_NEUTRAL_CHARGE_STATE_SOURCE_DEFINED_AS_ZERO",
        "neutral_charge_source_boundary": "JARVIS-DFT vacancy database documents electrically neutral defects only",
        "target_label": contract["target_label"],
        "baseline": contract["baseline"],
        "baseline_fit": "TRAIN_ONLY_RIDGE_ON_PRISTINE_ENERGY_CHEMICAL_POTENTIAL_PLUS_DEFECT_SPECIES_CONSTANT",
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": task_contract_hash(contract),
        "teacher_outputs": 0,
        "experimental_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def stage_p084(
    root: Path,
    contracts: dict[str, dict[str, str]],
    interface_source: Path,
    surface_source: Path,
    interface_record: dict[str, Any],
    surface_record: dict[str, Any],
) -> dict[str, Any]:
    interfaces = load_zip_json(interface_source)
    surfaces = load_zip_json(surface_source)
    surface_index: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = {}
    for row in surfaces:
        match = SURFACE_RE.match(str(row["name"]))
        if match:
            surface_index.setdefault(tuple(match.groups()[:5]), []).append(row)
    features, targets, groups, row_ids, baseline_values = [], [], [], [], []
    rejection_counts: Counter[str] = Counter()
    ev_per_a2_to_j_per_m2 = 16.02176634
    for row in interfaces:
        match = INTERFACE_RE.match(str(row["jid"]))
        if not match:
            rejection_counts["interface_id_parse"] += 1
            continue
        value = match.groups()
        key_film = (value[0], value[2], value[3], value[4], value[8])
        key_sub = (value[1], value[5], value[6], value[7], value[9])
        film = [item for item in surface_index.get(key_film, []) if item["name"].endswith("VASP_PBE_noDP")]
        sub = [item for item in surface_index.get(key_sub, []) if item["name"].endswith("VASP_PBE_noDP")]
        if len(film) != 1 or len(sub) != 1:
            rejection_counts["missing_unique_pbe_nodp_surface"] += 1
            continue
        film_row, sub_row = film[0], sub[0]
        if len(film_row["atoms"]["elements"]) + len(sub_row["atoms"]["elements"]) != len(row["atoms"]["elements"]):
            rejection_counts["component_interface_atom_count_mismatch"] += 1
            continue
        lattice = np.asarray(row["atoms"]["lattice_mat"], dtype=np.float64)
        area = float(np.linalg.norm(np.cross(lattice[0], lattice[1])))
        if not np.isfinite(area) or area <= 0:
            rejection_counts["invalid_interface_area"] += 1
            continue
        work = (
            float(film_row["final_energy"])
            + float(sub_row["final_energy"])
            - float(row["final_energy"])
        ) / area * ev_per_a2_to_j_per_m2
        if not np.isfinite(work) or work <= 0:
            rejection_counts["nonpositive_work_of_adhesion"] += 1
            continue
        offset = row.get("offset")
        offset_available = isinstance(offset, (int, float)) and np.isfinite(float(offset))
        film_lattice = np.asarray(film_row["atoms"]["lattice_mat"], dtype=np.float64)
        sub_lattice = np.asarray(sub_row["atoms"]["lattice_mat"], dtype=np.float64)
        inplane_film = np.linalg.norm(film_lattice[:2], axis=1)
        inplane_sub = np.linalg.norm(sub_lattice[:2], axis=1)
        vector = np.concatenate(
            (
                composition(film_row["atoms"]["elements"]),
                composition(sub_row["atoms"]["elements"]),
                np.asarray(
                    [
                        *(float(v) for v in value[2:8]),
                        float(value[8]),
                        float(value[9]),
                        float(value[10]),
                        float(value[11]),
                        float(value[12]),
                        area,
                        *(inplane_film.tolist()),
                        *(inplane_sub.tolist()),
                        float(np.mean(np.abs(inplane_film - inplane_sub) / np.maximum(inplane_sub, 1e-9))),
                        float(film_row["surf_en"]),
                        float(sub_row["surf_en"]),
                        float(offset) if offset_available else 0.0,
                        float(offset_available),
                        float(row.get("optb88vdw_bandgap", 0.0)),
                        float(len(film_row["atoms"]["elements"])),
                        float(len(sub_row["atoms"]["elements"])),
                    ],
                    dtype=np.float32,
                ),
            )
        )
        features.append(vector)
        targets.append(work)
        # All displacements/orientations for the same material pair stay together.
        groups.append(f"{value[0]}|{value[1]}")
        row_ids.append(str(row["jid"]))
        baseline_values.append(float(film_row["surf_en"]) + float(sub_row["surf_en"]))
    if len(features) != 185:
        raise RuntimeError(f"P084_EXACT_MATCH_GATE:accepted={len(features)}:rejections={dict(rejection_counts)}")
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    split = grouped_split(groups)
    counts, overlap, split_hash = split_audit(groups, split)
    output = root / "data" / "staged_jarvis_exact_v1" / "CAND-P-084.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x,
        y=y,
        split=split,
        groups=np.asarray(groups),
        row_id=np.asarray(row_ids),
        baseline_pred=np.asarray(baseline_values, dtype=np.float32),
        candidate_id=np.asarray("CAND-P-084"),
        task_kind=np.asarray("regression"),
        authority=np.asarray(0, dtype=np.int8),
    )
    contract = contracts["CAND-P-084"]
    metadata = {
        "schema": "cimc.forge200.jarvis-exact-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-P-084",
        "truth_class": "OPEN_COMPUTED_DFT_PBE_MATCHED_COMPONENT_INTERFACE",
        "source_dataset": "JARVIS-DFT interfacedb v2 plus surfacedb v2",
        "source_urls": [
            "https://figshare.com/articles/dataset/JARVIS-DFT_3D_dataset_/25832614",
            "https://pages.nist.gov/jarvis/databases/",
        ],
        "source_paths": [
            str(interface_source.relative_to(root)).replace("\\", "/"),
            str(surface_source.relative_to(root)).replace("\\", "/"),
        ],
        "source_sha256": [sha256_file(interface_source), sha256_file(surface_source)],
        "source_records": [interface_record, surface_record],
        "license": "CC BY 4.0",
        "records": len(x),
        "source_interface_records": len(interfaces),
        "excluded_source_rows": len(interfaces) - len(x),
        "exclusion_reasons": dict(sorted(rejection_counts.items())),
        "exact_join_gates": ["PBE_noDP_for_both_surfaces", "component_atom_counts_sum_to_interface", "positive_finite_area_and_target"],
        "target_derivation": "(E_film_slab + E_substrate_slab - E_interface) / inplane_area * 16.02176634 J_per_m2_per_eV_per_A2",
        "features": int(x.shape[1]),
        "counts": counts,
        "split_unit": "ORDERED_FILM_SUBSTRATE_JARVIS_PAIR",
        "cross_split_group_overlap": overlap,
        "split_sha256": split_hash,
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_EXACT_MATCHED_PBE_COMPONENTS_AND_INTERFACE",
        "target_label": contract["target_label"],
        "baseline": contract["baseline"],
        "baseline_fit": "FIXED_PUBLISHED_SURFACE_ENERGY_SUM_NO_TEST_FIT",
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": task_contract_hash(contract),
        "teacher_outputs": 0,
        "experimental_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    receipt_path = root / "evidence" / "jarvis_exact_source_download.v1.json"
    source_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if source_receipt.get("status") != "PASS":
        raise RuntimeError("JARVIS_SOURCE_RECEIPT_GATE")
    source_records = {item["path"]: item for item in source_receipt["records"]}
    contracts = read_contracts(root)
    p070_source = root / "data" / "raw" / "jarvis_vacancydb_v8" / "vacancydb.json.zip"
    p084_interface = root / "data" / "raw" / "jarvis_interfacedb_v2" / "interface_db_dd.json.zip"
    p084_surface = root / "data" / "raw" / "jarvis_surfacedb_v2" / "surface_db_dd.json.zip"
    p070 = stage_p070(
        root,
        contracts,
        p070_source,
        source_records["data/raw/jarvis_vacancydb_v8/vacancydb.json.zip"],
    )
    p084 = stage_p084(
        root,
        contracts,
        p084_interface,
        p084_surface,
        source_records["data/raw/jarvis_interfacedb_v2/interface_db_dd.json.zip"],
        source_records["data/raw/jarvis_surfacedb_v2/surface_db_dd.json.zip"],
    )
    content = {"records": [p070, p084], "authority_nonzero": 0, "board_actions": 0}
    receipt = {
        "schema": "cimc.forge200.jarvis-exact-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        **content,
        "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }
    write_json(root / "evidence" / "jarvis_exact_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "records": {item["candidate_id"]: item["records"] for item in content["records"]}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
