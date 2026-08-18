#!/usr/bin/env python3
"""Stage exact JARVIS SLME and spin-orbit-spillage contracts."""

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


def numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


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


def scalar_features(row: dict[str, Any]) -> np.ndarray:
    elements = row["atoms"]["elements"]
    counts = Counter(elements)
    total = sum(counts.values())
    z = np.asarray([ELEMENT_INDEX[symbol] + 1 for symbol in counts], dtype=np.float64)
    fractions = np.asarray([counts[symbol] / total for symbol in counts], dtype=np.float64)
    high_z_fraction = float(np.sum(fractions[z >= 55]))
    soc_proxy = float(np.sum(fractions * (z / 118.0) ** 4))
    mbj = numeric(row.get("mbj_bandgap"))
    gap = numeric(row.get("optb88vdw_bandgap")) or 0.0
    values = [
        gap,
        gap**2,
        gap**3,
        gap**4,
        1.0 / max(gap + 0.15, 0.15),
        mbj or 0.0,
        float(mbj is not None),
        numeric(row.get("density")) or 0.0,
        (numeric(row.get("spg_number")) or 0.0) / 230.0,
        numeric(row.get("formation_energy_peratom")) or 0.0,
        numeric(row.get("ehull")) or 0.0,
        high_z_fraction,
        float(np.max(z) / 118.0),
        soc_proxy,
        float(row.get("dimensionality") == "3D"),
    ]
    return np.asarray(values, dtype=np.float32)


def chemical_system(row: dict[str, Any]) -> str:
    return "-".join(sorted(set(row["atoms"]["elements"])))


def prototype_group(row: dict[str, Any]) -> str:
    counts = sorted(Counter(row["atoms"]["elements"]).values())
    base = max(math.gcd(*counts), 1)
    anonymous = ":".join(str(value // base) for value in counts)
    return f"SPG{int(numeric(row.get('spg_number')) or 0):03d}|ANON{anonymous}|N{len(counts)}"


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
        raise RuntimeError("JARVIS_SOURCE_HASH_GATE")
    with zipfile.ZipFile(source) as archive:
        if len(archive.namelist()) != 1:
            raise RuntimeError("JARVIS_ARCHIVE_GATE")
        rows = json.loads(archive.read(archive.namelist()[0]))
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    specs = {
        "CAND-P-142": {
            "target_field": "slme",
            "truth_class": "OPEN_COMPUTED_DFT_SLME",
            "split_unit": "CHEMICAL_SYSTEM",
            "group": chemical_system,
            "target_semantics": "published JARVIS spectroscopic limited maximum efficiency percent",
            "input_state": "COMPOSITION_STRUCTURE_BANDGAP_FIXED_1UM_THICKNESS_WITH_EXPLICIT_ABSORPTION_AND_DIRECT_GAP_MASKS_ZERO",
            "availability": {"absorption_spectrum": 0, "direct_indirect_gap": 0, "thickness_nm_fixed_by_source_protocol": 1000},
            "baseline_execution": "train-only fourth-degree polynomial of published optB88vdW bandgap as Shockley-Queisser bandgap-only baseline",
        },
        "CAND-P-145": {
            "target_field": "spillage",
            "truth_class": "OPEN_COMPUTED_DFT_SPIN_ORBIT_SPILLAGE",
            "split_unit": "STRUCTURE_PROTOTYPE",
            "group": prototype_group,
            "target_semantics": "published JARVIS spin-orbit spillage score; candidate class threshold fixed at 0.5",
            "input_state": "COMPOSITION_STRUCTURE_SYMMETRY_BANDGAP_AND_SOC_PROXY_WITH_EXPLICIT_BAND_STRUCTURE_AND_OCCUPANCY_MASKS_ZERO",
            "availability": {"SOC_band_structure": 0, "non_SOC_band_structure": 0, "occupancy_vectors": 0},
            "baseline_execution": "train-only ridge calibration of fixed high-Z fraction inverse-gap symmetry and SOC rule features",
        },
    }
    evidence_records = []
    stage_root = root / "data" / "staged_jarvis_p142_p145_exact_v1"
    stage_root.mkdir(parents=True, exist_ok=True)
    for candidate_id, spec in specs.items():
        selected = [(row, numeric(row.get(spec["target_field"]))) for row in rows]
        selected = [(row, value) for row, value in selected if value is not None]
        if len(selected) < 4000:
            raise RuntimeError(f"TARGET_RECORD_GATE:{candidate_id}:{len(selected)}")
        x = np.asarray(
            [np.concatenate((composition(row["atoms"]["elements"]), structure_features(row["atoms"]), scalar_features(row))) for row, _ in selected],
            dtype=np.float32,
        )
        y = np.asarray([value for _, value in selected], dtype=np.float32)
        groups = np.asarray([spec["group"](row) for row, _ in selected])
        jids = np.asarray([str(row["jid"]) for row, _ in selected])
        split = np.asarray([split_for_group(group) for group in groups], dtype=np.int8)
        sets = {code: set(groups[split == code]) for code in (0, 1, 2)}
        overlap = sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
        if overlap or min(counts.values()) < 400:
            raise RuntimeError(f"SPLIT_GATE:{candidate_id}:{overlap}:{counts}")
        output = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(output, x=x, y=y, groups=groups, jid=jids, split=split, authority=np.asarray(0, dtype=np.int8), candidate_id=np.asarray(candidate_id))
        contract = contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.jarvis-property-staged.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "candidate_id": candidate_id,
            "truth_class": spec["truth_class"],
            "source_id": "jarvis_dft_v11",
            "source_url": SOURCE_URL,
            "source_pid": SOURCE_PID,
            "source_sha256": SOURCE_SHA256,
            "license": "CC BY 4.0",
            "records": len(selected),
            "features": int(x.shape[1]),
            "feature_indices": {
                "optb88vdw_bandgap": len(ELEMENTS) + 10,
                "bandgap_squared": len(ELEMENTS) + 11,
                "bandgap_cubed": len(ELEMENTS) + 12,
                "bandgap_fourth": len(ELEMENTS) + 13,
                "inverse_bandgap_plus_0p15": len(ELEMENTS) + 14,
                "space_group_number_scaled": len(ELEMENTS) + 18,
                "high_Z_fraction": len(ELEMENTS) + 21,
                "SOC_Z4_proxy": len(ELEMENTS) + 23
            },
            "groups": len(set(groups.tolist())),
            "counts": counts,
            "split_unit": spec["split_unit"],
            "cross_split_group_overlap": overlap,
            "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(groups.tolist(), split.tolist(), strict=True)))).hexdigest(),
            "input_contract": contract["input_contract"],
            "input_contract_state": spec["input_state"],
            "input_availability": spec["availability"],
            "target_label": contract["target_label"],
            "target_field": spec["target_field"],
            "target_semantics": spec["target_semantics"],
            "baseline": contract["baseline"],
            "baseline_execution": spec["baseline_execution"],
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
        evidence_records.append(metadata)
    receipt = {
        "schema": "cimc.forge200.jarvis-p142-p145-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "records": evidence_records,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    receipt["content_root_sha256"] = hashlib.sha256(canonical_bytes(evidence_records)).hexdigest()
    write_json(root / "evidence" / "jarvis_p142_p145_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "candidates": {item["candidate_id"]: item["counts"] for item in evidence_records}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
