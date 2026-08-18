#!/usr/bin/env python3
"""Stage source-bound JARVIS host-stability rankings for P138."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import ELEMENTS, ELEMENT_INDEX


HOST_ANIONS = {"O", "N", "F", "Cl", "Br", "I", "S", "Se"}


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


def composition(elements: list[str]) -> np.ndarray:
    counts = Counter(elements)
    total = sum(counts.values())
    result = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count in counts.items():
        result[ELEMENT_INDEX[symbol]] = count / total
    return result


def structure_features(atoms: dict[str, Any]) -> np.ndarray:
    lattice = np.asarray(atoms["lattice_mat"], dtype=np.float64)
    lengths = np.linalg.norm(lattice, axis=1)
    volume = abs(float(np.linalg.det(lattice)))
    angles = np.asarray(atoms.get("angles", [90.0, 90.0, 90.0]), dtype=np.float64)
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


def split_for_group(group: str) -> int:
    bucket = int(hashlib.sha256(group.encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    expected_sha = "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"
    if sha256_file(source) != expected_sha:
        raise RuntimeError("JARVIS_SOURCE_HASH_GATE")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != 1:
            raise RuntimeError("JARVIS_ARCHIVE_GATE")
        source_rows = json.loads(archive.read(names[0]))

    filtered = []
    for row in source_rows:
        elements = set(row["atoms"]["elements"])
        if not (2 <= len(elements) <= 5 and elements & HOST_ANIONS):
            continue
        if float(row["optb88vdw_bandgap"]) < 1.0:
            continue
        if not np.isfinite(float(row["formation_energy_peratom"])) or not np.isfinite(float(row["ehull"])):
            continue
        system = "-".join(sorted(elements))
        filtered.append((system, row))
    counts = Counter(system for system, _ in filtered)
    filtered = [(system, row) for system, row in filtered if counts[system] >= 3]
    by_system: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for system, row in filtered:
        by_system[system].append(row)
    if len(filtered) != 8174 or len(by_system) != 1383:
        raise RuntimeError(f"HOST_FILTER_GATE:records={len(filtered)}:groups={len(by_system)}")

    x, y, groups, jids, baseline = [], [], [], [], []
    for system in sorted(by_system):
        rows = by_system[system]
        formation = np.asarray([float(row["formation_energy_peratom"]) for row in rows], dtype=np.float64)
        group_stats = np.asarray(
            [
                len(rows) / 128.0,
                float(np.min(formation)),
                float(np.mean(formation)),
                float(np.std(formation)),
                float(np.quantile(formation, 0.25)),
                float(np.median(formation)),
                float(np.quantile(formation, 0.75)),
            ],
            dtype=np.float32,
        )
        for row in rows:
            energy = float(row["formation_energy_peratom"])
            material = np.asarray(
                [
                    energy,
                    energy - float(np.min(formation)),
                    float(np.mean(formation <= energy)),
                    float(row["optb88vdw_bandgap"]),
                    float(row["density"]),
                    float(row["spg_number"]) / 230.0,
                    float(len(set(row["atoms"]["elements"]))) / 5.0,
                    float(row.get("dimensionality") == "3D"),
                ],
                dtype=np.float32,
            )
            x.append(np.concatenate((composition(row["atoms"]["elements"]), structure_features(row["atoms"]), group_stats, material)))
            y.append(float(row["ehull"]))
            groups.append(system)
            jids.append(str(row["jid"]))
            baseline.append(energy)
    x_array = np.asarray(x, dtype=np.float32)
    y_array = np.asarray(y, dtype=np.float32)
    split = np.asarray([split_for_group(group) for group in groups], dtype=np.int8)
    sets = {code: {group for group, item in zip(groups, split, strict=True) if item == code} for code in (0, 1, 2)}
    overlap = sum(len(sets[a] & sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    split_counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    if overlap or min(split_counts.values()) < 500:
        raise RuntimeError(f"SPLIT_GATE:overlap={overlap}:counts={split_counts}")

    contract_path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with contract_path.open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == "CAND-P-138")
    output = root / "data" / "staged_jarvis_p138_exact_v1" / "CAND-P-138.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=x_array,
        y=y_array,
        groups=np.asarray(groups),
        jid=np.asarray(jids),
        split=split,
        baseline_score=np.asarray(baseline, dtype=np.float32),
        authority=np.asarray(0, dtype=np.int8),
        candidate_id=np.asarray("CAND-P-138"),
    )
    metadata = {
        "schema": "cimc.forge200.jarvis-p138-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-P-138",
        "truth_class": "OPEN_COMPUTED_DFT_HOST_STABILITY_SCREENING",
        "source_id": "jarvis_dft_v11",
        "source_url": "https://figshare.com/articles/dataset/jdft_3d-7-7-2018_json/6815699",
        "source_pid": "10.6084/m9.figshare.6815699.v11",
        "source_sha256": expected_sha,
        "license": "CC BY 4.0",
        "records": len(x_array),
        "features": int(x_array.shape[1]),
        "chemical_system_groups": len(set(groups)),
        "counts": split_counts,
        "split_unit": "CHEMICAL_SYSTEM",
        "cross_split_group_overlap": overlap,
        "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(groups, split.tolist(), strict=True)))).hexdigest(),
        "host_candidate_filter": "2_TO_5_ELEMENTS_AND_CONTAINS_O_N_F_Cl_Br_I_S_Se_AND_OPT_B88VDW_GAP_GE_1EV_AND_AT_LEAST_3_POLYMORPHS_PER_CHEMICAL_SYSTEM",
        "scope_boundary": "computed thermodynamic host stability screening only; not an experimental phosphor performance label",
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_COMPOSITION_STRUCTURE_FORMATION_ENERGY_AND_SOURCE_DERIVED_COMPETING_PHASE_FEATURES",
        "target_label": contract["target_label"],
        "target_field": "ehull",
        "target_semantics": "published JARVIS energy above convex hull; lower is more stable",
        "baseline": contract["baseline"],
        "baseline_execution": "published formation_energy_peratom ascending after fixed host filter",
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
    receipt = {
        "schema": "cimc.forge200.jarvis-p138-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "record": metadata,
        "supersedes_pre_gpu_rejection_reason": "RECORD_LEVEL_TARGET_BINDING_ABSENT",
        "superseding_evidence": "published per-record ehull plus chemical-system split and exact competing-phase feature materialization",
        "authority_nonzero": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical_bytes(metadata)).hexdigest(),
    }
    write_json(root / "evidence" / "jarvis_p138_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "records": len(x_array), "groups": len(set(groups)), "counts": split_counts, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
