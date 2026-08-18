#!/usr/bin/env python3
"""Stage SIM_ONLY thermo-mechanical trajectory tasks P103 and P109."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import canonical_bytes, sha256_file, write_json


SEED = 20260803
POINTS = 16
SPLIT_NAMES = ("train", "validation", "test")


def split_for_group(group: str) -> int:
    bucket = int(hashlib.sha256(group.encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def audit_split(groups: np.ndarray, split: np.ndarray) -> tuple[dict[str, int], int]:
    sets = {code: set(groups[split == code].tolist()) for code in (0, 1, 2)}
    overlap = sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    counts = {SPLIT_NAMES[code]: int(np.sum(split == code)) for code in (0, 1, 2)}
    return counts, overlap


def cure_schedule(cure_temperature: float, ramp_min: float, dwell_min: float, cool_min: float) -> tuple[np.ndarray, np.ndarray]:
    total = ramp_min + dwell_min + cool_min
    time = np.linspace(0.0, total, POINTS)
    temperature = np.empty(POINTS, dtype=np.float64)
    for index, minute in enumerate(time):
        if minute <= ramp_min:
            temperature[index] = 25.0 + (cure_temperature - 25.0) * minute / ramp_min
        elif minute <= ramp_min + dwell_min:
            temperature[index] = cure_temperature
        else:
            temperature[index] = cure_temperature - (cure_temperature - 25.0) * (minute - ramp_min - dwell_min) / cool_min
    return time, temperature


def residual_stress(
    time: np.ndarray,
    temperature: np.ndarray,
    modulus_gpa: float,
    shrinkage: float,
    cte_ppm: float,
    kinetic_tau_min: float,
    relaxation_tau_min: float,
    kinetic_temp_sensitivity: float,
    constraint: float,
) -> np.ndarray:
    degree = np.zeros(POINTS, dtype=np.float64)
    stress = np.zeros(POINTS, dtype=np.float64)
    for index in range(1, POINTS):
        dt = max(float(time[index] - time[index - 1]), 1e-6)
        rate = math.exp(np.clip(kinetic_temp_sensitivity * (temperature[index] - 120.0) / 80.0, -5.0, 5.0)) / kinetic_tau_min
        degree[index] = 1.0 - (1.0 - degree[index - 1]) * math.exp(-rate * dt)
        modulus = modulus_gpa * (0.03 + 0.97 * degree[index] ** 1.8)
        delta_shrink = shrinkage * (degree[index] - degree[index - 1])
        delta_thermal = cte_ppm * 1e-6 * (temperature[index] - temperature[index - 1])
        relaxation = relaxation_tau_min * (0.25 + 0.75 * degree[index]) * math.exp(np.clip((120.0 - temperature[index]) / 120.0, -2.0, 2.0))
        stress[index] = stress[index - 1] * math.exp(-dt / max(relaxation, 1e-3)) + modulus * 1000.0 * constraint * (delta_shrink - delta_thermal)
    return stress.astype(np.float32)


def bimetal_curvature(delta_alpha: float, delta_temperature: np.ndarray, e1: np.ndarray, e2: np.ndarray, t1: float, t2: float) -> np.ndarray:
    numerator = 6.0 * delta_alpha * delta_temperature * e1 * e2 * t1 * t2 * (t1 + t2)
    denominator = (e1 * t1 + e2 * t2) * (e1 * t1**3 + e2 * t2**3) + 3.0 * e1 * e2 * t1 * t2 * (t1 + t2) ** 2
    return numerator / np.maximum(denominator, 1e-18)


def build_p103(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    records = {"x": [], "y": [], "baseline": [], "group": [], "split": []}
    manifest = []
    for family_index in range(90):
        family = f"RESIN-{family_index:03d}"
        modulus = float(rng.uniform(1.2, 12.0))
        shrinkage = float(rng.uniform(0.008, 0.055))
        cte = float(rng.uniform(25.0, 120.0))
        kinetic_tau = float(rng.uniform(12.0, 90.0))
        relaxation_tau = float(rng.uniform(8.0, 180.0))
        sensitivity = float(rng.uniform(1.2, 4.5))
        constraint = float(rng.uniform(0.25, 0.95))
        code = split_for_group(family)
        manifest.append({"family": family, "modulus_GPa": modulus, "shrinkage": shrinkage, "CTE_ppm_K": cte, "kinetic_tau_min": kinetic_tau, "relaxation_tau_min": relaxation_tau, "split": SPLIT_NAMES[code]})
        for _ in range(40):
            cure_temperature = float(rng.uniform(105.0, 205.0))
            ramp = float(rng.uniform(25.0, 120.0))
            dwell = float(rng.uniform(45.0, 300.0))
            cool = float(rng.uniform(30.0, 150.0))
            time, temperature = cure_schedule(cure_temperature, ramp, dwell, cool)
            target = residual_stress(time, temperature, modulus, shrinkage, cte, kinetic_tau, relaxation_tau, sensitivity, constraint)
            baseline = (modulus * 1000.0 * constraint * cte * 1e-6 * (25.0 - temperature)).astype(np.float32)
            x = np.concatenate((temperature / 250.0, time / 600.0, np.asarray([modulus / 15.0, shrinkage / 0.06, cte / 150.0, kinetic_tau / 100.0, relaxation_tau / 200.0, sensitivity / 5.0, constraint], dtype=np.float64)))
            records["x"].append(x.astype(np.float32)); records["y"].append(target); records["baseline"].append(baseline); records["group"].append(family); records["split"].append(code)
    return {key: np.asarray(value, dtype=np.float32 if key in {"x", "y", "baseline"} else np.int8 if key == "split" else None) for key, value in records.items()}, manifest


def build_p109(rng: np.random.Generator) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    records = {"x": [], "y": [], "baseline": [], "group": [], "split": []}
    manifest = []
    for family_index in range(100):
        family = f"STACK-{family_index:03d}"
        cte1, cte2 = rng.uniform(3.0, 28.0, size=2)
        e1, e2 = rng.uniform(20.0, 180.0, size=2)
        t1, t2 = rng.uniform(0.08, 0.9, size=2) * 1e-3
        tg1, tg2 = rng.uniform(75.0, 190.0, size=2)
        code = split_for_group(family)
        manifest.append({"family": family, "cte_ppm_K": [float(cte1), float(cte2)], "modulus_GPa": [float(e1), float(e2)], "thickness_m": [float(t1), float(t2)], "Tg_C": [float(tg1), float(tg2)], "split": SPLIT_NAMES[code]})
        for _ in range(40):
            low, high = float(rng.uniform(-55.0, 5.0)), float(rng.uniform(110.0, 190.0))
            temperature = np.linspace(low, high, POINTS)
            reference = float(rng.uniform(15.0, 35.0))
            length = float(rng.uniform(8.0, 35.0)) * 1e-3
            dwell = float(rng.uniform(5.0, 180.0))
            e1_t = e1 * (0.12 + 0.88 / (1.0 + np.exp(np.clip((temperature - tg1) / 8.0, -20.0, 20.0))))
            e2_t = e2 * (0.12 + 0.88 / (1.0 + np.exp(np.clip((temperature - tg2) / 8.0, -20.0, 20.0))))
            delta_t = temperature - reference
            curvature = bimetal_curvature((cte1 - cte2) * 1e-6, delta_t, e1_t, e2_t, t1, t2)
            relaxation = 1.0 - 0.18 * (temperature > min(tg1, tg2)) * (1.0 - math.exp(-dwell / 80.0))
            target = (curvature * relaxation * length**2 / 8.0 * 1e6).astype(np.float32)
            baseline_curvature = bimetal_curvature((cte1 - cte2) * 1e-6, delta_t, np.full(POINTS, e1), np.full(POINTS, e2), t1, t2)
            baseline = (baseline_curvature * length**2 / 8.0 * 1e6).astype(np.float32)
            x = np.concatenate((temperature / 220.0, np.asarray([reference / 50.0, cte1 / 30.0, cte2 / 30.0, e1 / 200.0, e2 / 200.0, t1 / 1e-3, t2 / 1e-3, tg1 / 200.0, tg2 / 200.0, length / 0.04, dwell / 200.0], dtype=np.float64)))
            records["x"].append(x.astype(np.float32)); records["y"].append(target); records["baseline"].append(baseline); records["group"].append(family); records["split"].append(code)
    return {key: np.asarray(value, dtype=np.float32 if key in {"x", "y", "baseline"} else np.int8 if key == "split" else None) for key, value in records.items()}, manifest


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args()
    root = args.root.resolve(); rng = np.random.default_rng(SEED)
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    definitions = {"CAND-P-103": (*build_p103(rng), "TEAM_OWNED_INCREMENTAL_MAXWELL_CURE_SHRINKAGE_SIM", "LINEAR_THERMOELASTIC_NO_CURE_SHRINKAGE_OR_RELAXATION"), "CAND-P-109": (*build_p109(rng), "TEAM_OWNED_TEMPERATURE_DEPENDENT_BILAYER_BEAM_SIM", "CONSTANT_MODULUS_LINEAR_CTE_MISMATCH_BILAYER")}
    evidence = []
    for candidate_id, (arrays, manifest, simulator, baseline_kind) in definitions.items():
        counts, overlap = audit_split(arrays["group"], arrays["split"])
        if overlap or min(counts.values()) < 400 or not np.all(np.isfinite(arrays["y"])):
            raise RuntimeError(f"{candidate_id}:DATA_GATE:{overlap}:{counts}")
        output = root / "data" / "staged_physics_p103_p109_exact_v1" / f"{candidate_id}.npz"; output.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(output, **arrays)
        contract = contracts[candidate_id]
        metadata = {"schema": "cimc.forge200.physics-trajectory-exact-dataset.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "candidate_id": candidate_id, "truth_class": "PHYSICS_SIM", "public_claim_scope": "SIM_ONLY", "simulator": simulator, "generation_seed": SEED, "records": len(arrays["y"]), "trajectory_points": POINTS, "family_manifest_root_sha256": hashlib.sha256(canonical_bytes(manifest)).hexdigest(), "split_method": "SHA256_MATERIAL_FAMILY_70_15_15", "split_counts": counts, "cross_split_component_overlap": overlap, "cross_split_family_overlap": overlap, "baseline_execution": baseline_kind, "input_contract": contract["input_contract"], "target_label": contract["target_label"], "primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"], "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(), "source_license": "TEAM_OWNED_GENERATED_SIMULATION", "experimental_records": 0, "teacher_outputs": 0, "authority": 0, "board_accepted": False, "countable_model": False, "generator_sha256": sha256_file(Path(__file__)), "sha256": sha256_file(output)}
        write_json(output.with_suffix(".metadata.json"), metadata); evidence.append(metadata)
    write_json(root / "evidence" / "physics_p103_p109_exact_staging.v1.json", {"schema": "cimc.forge200.physics-p103-p109-exact-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "records": evidence, "authority_nonzero": 0, "board_actions": 0})
    print(json.dumps({"status": "PASS", "tasks": {item["candidate_id"]: item["split_counts"] for item in evidence}}, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
