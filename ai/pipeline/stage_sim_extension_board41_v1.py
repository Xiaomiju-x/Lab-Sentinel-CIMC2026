#!/usr/bin/env python3
"""Stage family-split SIM_ONLY datasets for board-facing P001-P041."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from stage_matbench_experimental_v1 import canonical_bytes, sha256_file, write_json


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def split_for(candidate_id: str, family_id: str) -> int:
    bucket = int(hashlib.sha256(f"{candidate_id}|{family_id}".encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def basis(x: np.ndarray, domain: str) -> np.ndarray:
    phase = {"thermal": 0.1, "thermal_field": 0.3, "vision": 0.5, "vision_mask": 0.7, "sensor": 0.9, "power": 1.1, "mechanical": 1.3, "clock": 1.5}[domain]
    return np.column_stack(
        (
            x[:, :6],
            x[:, 0] * x[:, 8],
            x[:, 1] * x[:, 9],
            np.sin(np.pi * (x[:, 2] + phase * x[:, 10])),
            np.cos(np.pi * (x[:, 3] - phase * x[:, 11])),
            np.sqrt(np.maximum(x[:, 4], 0.0)) * x[:, 12],
            np.exp(-x[:, 5] / (0.1 + x[:, 13])),
            sigmoid(8.0 * (x[:, 6] - x[:, 14])),
            x[:, 7] / (0.2 + x[:, 15]),
            x[:, 8] ** 2,
            x[:, 9] * x[:, 10],
            np.sin(2.0 * np.pi * x[:, 14] * x[:, 15]),
        )
    )


def make_target(candidate_id: str, domain: str, x: np.ndarray, output_dimensions: int) -> np.ndarray:
    seed = int(hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    phi = basis(x, domain)
    weights = rng.normal(0.0, 0.8, size=(phi.shape[1], output_dimensions))
    logits = phi @ weights + rng.normal(0.0, 0.25, size=output_dimensions)
    values = sigmoid(logits)
    if domain in {"thermal_field", "vision_mask"} and output_dimensions > 3:
        values = 0.65 * values + 0.35 * np.roll(values, 1, axis=1)
    if domain == "clock":
        values[:, 0] = 2.0 * values[:, 0] - 1.0
    return values.astype(np.float32)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "contracts" / "sim_extension_board41.v1.json"
    extension = json.loads(contract_path.read_text(encoding="utf-8"))
    if extension["status"] != "PRETRAIN_FROZEN" or extension["authority"] != 0:
        raise RuntimeError("BOARD41_CONTRACT_GATE")
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        task_contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    task_ids = [task["candidate_id"] for task in extension["tasks"]]
    if task_ids != [f"CAND-P-{index:03d}" for index in range(1, 42)]:
        raise RuntimeError("BOARD41_ID_GATE")

    output_root = root / "data" / "staged_sim_extension_board41_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    receipt_records = []
    common = extension["dataset"]
    for task in extension["tasks"]:
        candidate_id = task["candidate_id"]
        rng = np.random.default_rng(common["generation_seed"] + int(candidate_id[-3:]))
        x_rows, groups, splits = [], [], []
        for family_index in range(common["families_per_task"]):
            family_id = f"{candidate_id}-BOARD-SIMFAM-{family_index:03d}"
            family = rng.uniform(0.02, 0.98, size=8)
            split = split_for(candidate_id, family_id)
            for _ in range(common["records_per_family"]):
                x_rows.append(np.concatenate((family, rng.uniform(0.0, 1.0, size=8))).astype(np.float32))
                groups.append(family_id)
                splits.append(split)
        x = np.asarray(x_rows, dtype=np.float32)
        group = np.asarray(groups)
        split = np.asarray(splits, dtype=np.int8)
        y = make_target(candidate_id, task["domain"], x.astype(np.float64), task["output_dimensions"])
        train = split == 0
        design_train = np.column_stack((np.ones(np.sum(train)), x[train, :6]))
        penalty = np.eye(design_train.shape[1]) * 0.25
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(design_train.T @ design_train + penalty, design_train.T @ y[train])
        baseline = (np.column_stack((np.ones(len(x)), x[:, :6])) @ coefficients).astype(np.float32)
        sets = {code: set(group[split == code].tolist()) for code in (0, 1, 2)}
        overlap = sum(len(sets[left] & sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        counts = {name: int(np.sum(split == code)) for code, name in enumerate(("train", "validation", "test"))}
        if overlap or min(counts.values()) < 120 or not np.isfinite(y).all():
            raise RuntimeError(f"{candidate_id}:DATA_GATE:{overlap}:{counts}")
        dataset = output_root / f"{candidate_id}.npz"
        np.savez_compressed(dataset, x=x, y=y, baseline=baseline, group=group, split=split)
        original = task_contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.sim-extension-board-dataset.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_SIM_EXTENSION_DATA_FROZEN",
            "candidate_id": candidate_id,
            "truth_class": "PHYSICS_SIM",
            "public_claim_scope": "SIM_ONLY",
            "simulator": f"TEAM_OWNED_{task['domain'].upper()}_COUPLED_EQUATION_SURROGATE",
            "simulator_boundary": "Synthetic board-domain benchmark; not measured GD32, sensor, camera, actuator, or process truth.",
            "generation_seed": common["generation_seed"] + int(candidate_id[-3:]),
            "records": len(y),
            "families": common["families_per_task"],
            "input_dimensions": x.shape[1],
            "output_dimensions": y.shape[1],
            "output_semantics": [f"normalized_SIM_ONLY_{original['target_label']}_{index}" for index in range(y.shape[1])],
            "split_method": common["split"],
            "split_counts": counts,
            "cross_split_family_overlap": overlap,
            "baseline_execution": "TRAIN_ONLY_RIDGE_ON_FIRST_SIX_INPUTS_REDUCED_ORDER_SURROGATE",
            "baseline_coefficients_sha256": hashlib.sha256(canonical_bytes(coefficients.tolist())).hexdigest(),
            "original_input_contract": original["input_contract"],
            "original_target_label": original["target_label"],
            "original_primary_metric": original["primary_metric"],
            "original_source_gate": original["source_gate"],
            "original_task_contract_status": "UNCHANGED_FAIL_CLOSED",
            "task_contract_sha256": hashlib.sha256(canonical_bytes(original)).hexdigest(),
            "extension_contract": {"path": "contracts/sim_extension_board41.v1.json", "sha256": sha256_file(contract_path)},
            "experimental_records": 0,
            "teacher_outputs": 0,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
            "generator_sha256": sha256_file(Path(__file__)),
            "sha256": sha256_file(dataset),
        }
        write_json(dataset.with_suffix(".metadata.json"), metadata)
        receipt_records.append(metadata)
    receipt = {"schema": "cimc.forge200.sim-extension-board41-staging.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS_41_SIM_EXTENSION_DATASETS_FROZEN", "task_count": 41, "records": receipt_records, "original_exact_contract_promotions": 0, "authority_nonzero": 0, "board_actions": 0}
    write_json(root / "evidence" / "sim_extension_board41_staging.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "tasks": 41, "records": sum(item["records"] for item in receipt_records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
