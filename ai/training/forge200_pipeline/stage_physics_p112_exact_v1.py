#!/usr/bin/env python3
"""Generate and stage the preregistered SIM_ONLY P112 thermal-field task."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage_matbench_experimental_v1 import canonical_bytes, sha256_file, write_json


GRID = 8
STACKS = 80
SAMPLES_PER_STACK = 50
SEED = 20260803
SPLIT_NAMES = ("train", "validation", "test")


def split_for_stack(stack_id: str) -> int:
    bucket = int(hashlib.sha256(stack_id.encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def solve_field(power: np.ndarray, conductivity: float, thickness_m: float, convection: float, width_m: float) -> np.ndarray:
    interior = GRID - 2
    cell = width_m / (GRID - 1)
    sheet_conductance = conductivity * thickness_m
    convection_cell = convection * cell**2
    matrix = np.zeros((interior**2, interior**2), dtype=np.float64)
    rhs = power[1:-1, 1:-1].reshape(-1).astype(np.float64)
    for row in range(interior):
        for column in range(interior):
            index = row * interior + column
            matrix[index, index] = 4.0 * sheet_conductance + convection_cell
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, column + dc
                if 0 <= rr < interior and 0 <= cc < interior:
                    matrix[index, rr * interior + cc] = -sheet_conductance
    delta = np.zeros((GRID, GRID), dtype=np.float64)
    delta[1:-1, 1:-1] = np.linalg.solve(matrix, rhs).reshape(interior, interior)
    return delta.astype(np.float32)


def power_map(rng: np.random.Generator, total_power: float) -> np.ndarray:
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    result = np.zeros((GRID, GRID), dtype=np.float64)
    for _ in range(int(rng.integers(1, 4))):
        center_x, center_y = rng.uniform(1.2, GRID - 2.2, size=2)
        sigma_x, sigma_y = rng.uniform(0.45, 1.35, size=2)
        amplitude = rng.uniform(0.4, 1.0)
        result += amplitude * np.exp(-0.5 * (((xx - center_x) / sigma_x) ** 2 + ((yy - center_y) / sigma_y) ** 2))
    result[0, :] = result[-1, :] = result[:, 0] = result[:, -1] = 0.0
    result *= total_power / max(float(result.sum()), 1e-12)
    return result.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    rng = np.random.default_rng(SEED)
    power_records, stack_features, ambient_values, targets, baselines, groups, splits = [], [], [], [], [], [], []
    stack_manifest = []
    for stack_index in range(STACKS):
        stack_id = f"STACK-{stack_index:03d}"
        conductivity = float(np.exp(rng.uniform(math.log(1.0), math.log(45.0))))
        thickness_mm = float(rng.uniform(0.35, 2.2))
        convection = float(rng.uniform(15.0, 100.0))
        width_mm = float(rng.uniform(18.0, 45.0))
        stack_split = split_for_stack(stack_id)
        stack_manifest.append({"stack_id": stack_id, "conductivity_W_mK": conductivity, "thickness_mm": thickness_mm, "convection_W_m2K": convection, "width_mm": width_mm, "split": SPLIT_NAMES[stack_split]})
        for _ in range(SAMPLES_PER_STACK):
            total_power = float(rng.uniform(0.15, 3.0))
            ambient = float(rng.uniform(18.0, 45.0))
            power = power_map(rng, total_power)
            delta = solve_field(power, conductivity, thickness_mm * 1e-3, convection, width_mm * 1e-3)
            denominator = convection * (width_mm * 1e-3) ** 2 + 4.0 * conductivity * thickness_mm * 1e-3
            baseline_delta = np.zeros((GRID, GRID), dtype=np.float32)
            baseline_delta[1:-1, 1:-1] = total_power / max(denominator, 1e-9)
            power_records.append(power.reshape(-1))
            stack_features.append([conductivity, thickness_mm, convection, width_mm, total_power])
            ambient_values.append(ambient)
            targets.append(delta.reshape(-1))
            baselines.append(baseline_delta.reshape(-1))
            groups.append(stack_id)
            splits.append(stack_split)
    power_array = np.asarray(power_records, dtype=np.float32)
    feature_array = np.asarray(stack_features, dtype=np.float32)
    ambient_array = np.asarray(ambient_values, dtype=np.float32)
    target_array = np.asarray(targets, dtype=np.float32)
    baseline_array = np.asarray(baselines, dtype=np.float32)
    group_array = np.asarray(groups)
    split_array = np.asarray(splits, dtype=np.int8)
    group_sets = {code: set(group_array[split_array == code]) for code in (0, 1, 2)}
    overlap = sum(len(group_sets[left] & group_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
    split_counts = {SPLIT_NAMES[code]: int(np.sum(split_array == code)) for code in (0, 1, 2)}
    if overlap or min(split_counts.values()) < 400 or not np.all(np.isfinite(target_array)) or float(target_array.max()) > 500.0:
        raise RuntimeError(f"SIMULATION_DATA_GATE:{overlap}:{split_counts}:{float(target_array.max())}")
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    candidate_id = "CAND-P-112"
    output = root / "data" / "staged_physics_p112_exact_v1" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, power_map=power_array, stack_features=feature_array, ambient_C=ambient_array, y_delta_C=target_array, baseline_delta_C=baseline_array, group=group_array, split=split_array)
    contract = contracts[candidate_id]
    metadata = {
        "schema": "cimc.forge200.physics-p112-exact-dataset.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": candidate_id,
        "truth_class": "PHYSICS_SIM",
        "public_claim_scope": "SIM_ONLY",
        "simulator": "TEAM_OWNED_STEADY_2D_FINITE_DIFFERENCE_SHEET_CONDUCTION_WITH_CONVECTION_AND_DIRICHLET_EDGE",
        "simulator_equation": "k*t*(4*T-neighbors)+h*dx^2*(T-Tambient)=power_cell_W",
        "generation_seed": SEED,
        "records": len(target_array),
        "grid": [GRID, GRID],
        "stack_families": STACKS,
        "samples_per_stack": SAMPLES_PER_STACK,
        "stack_manifest_root_sha256": hashlib.sha256(canonical_bytes(stack_manifest)).hexdigest(),
        "split_method": "SHA256_STACK_ID_70_15_15",
        "split_counts": split_counts,
        "cross_split_stack_overlap": overlap,
        "cross_split_component_overlap": overlap,
        "cross_split_family_overlap": overlap,
        "baseline_execution": "LUMPED_RC_UNIFORM_INTERIOR_TEMPERATURE_AND_POWER_ARGMAX_HOTSPOT",
        "input_contract": contract["input_contract"],
        "target_label": contract["target_label"],
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "source_license": "TEAM_OWNED_GENERATED_SIMULATION",
        "experimental_records": 0,
        "teacher_outputs": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "generator_sha256": sha256_file(Path(__file__)),
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    write_json(root / "evidence" / "physics_p112_exact_staging.v1.json", {"schema": "cimc.forge200.physics-p112-exact-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "record": metadata, "authority_nonzero": 0, "board_actions": 0})
    print(json.dumps({"status": "PASS", "records": len(target_array), "split_counts": split_counts, "max_delta_C": float(target_array.max())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
