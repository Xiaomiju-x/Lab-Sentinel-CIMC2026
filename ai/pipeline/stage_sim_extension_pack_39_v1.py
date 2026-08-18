#!/usr/bin/env python3
"""Stage 39 independent, family-split SIM_ONLY surrogate tasks."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stage_matbench_experimental_v1 import canonical_bytes, sha256_file, write_json


SPLIT_NAMES = ("train", "validation", "test")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_for(candidate_id: str, family_id: str) -> int:
    digest = hashlib.sha256(f"{candidate_id}|{family_id}".encode("ascii")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -20.0, 20.0)))


def nonlinear_basis(x: np.ndarray, domain: str) -> np.ndarray:
    phase = {
        "kinetics": 0.15,
        "kinetics_trajectory": 0.35,
        "transport": 0.55,
        "transport_profile": 0.75,
        "device": 0.95,
        "imaging": 1.15,
        "multiphysics": 1.35,
        "phase_simplex": 1.55,
    }[domain]
    return np.column_stack(
        [
            x[:, 0],
            x[:, 1],
            x[:, 2],
            x[:, 3],
            x[:, 4],
            x[:, 5],
            x[:, 0] * x[:, 8],
            x[:, 1] * x[:, 9],
            np.sin(np.pi * (x[:, 4] + phase * x[:, 10])),
            np.exp(-2.0 * x[:, 5]) * (0.4 + x[:, 11]),
            np.sqrt(np.maximum(x[:, 6], 0.0)) * x[:, 12],
            sigmoid(7.0 * (x[:, 7] - x[:, 13])),
            x[:, 8] ** 2,
            x[:, 9] * x[:, 10],
            np.cos(np.pi * (x[:, 11] - phase * x[:, 2])),
            np.exp(-x[:, 12] / (0.1 + x[:, 13])),
            x[:, 14] / (0.2 + x[:, 15]),
            np.sin(2.0 * np.pi * x[:, 14] * x[:, 15]),
        ]
    ).astype(np.float64)


def make_targets(candidate_id: str, domain: str, x: np.ndarray, output_dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(candidate_id.encode("ascii")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    basis = nonlinear_basis(x, domain)
    weights = rng.normal(0.0, 0.75, size=(basis.shape[1], max(output_dim, 4)))
    bias = rng.normal(0.0, 0.25, size=max(output_dim, 4))
    logits = basis @ weights + bias
    if domain == "kinetics_trajectory":
        amplitude = sigmoid(logits[:, 0])
        rate = 0.25 + 4.5 * sigmoid(logits[:, 1])
        exponent = 0.65 + 1.7 * sigmoid(logits[:, 2])
        time = np.linspace(0.0, 1.0, output_dim)
        y = amplitude[:, None] * (1.0 - np.exp(-rate[:, None] * time[None, :] ** exponent[:, None]))
        if candidate_id == "CAND-P-083":
            y = amplitude[:, None] * np.exp(-rate[:, None] * time[None, :] ** exponent[:, None])
        return np.clip(y, 0.0, 1.0)
    if domain == "transport_profile":
        diffusivity = 0.04 + 0.45 * sigmoid(logits[:, 0])
        exposure = 0.1 + 3.0 * sigmoid(logits[:, 1])
        position = np.linspace(0.0, 1.0, output_dim - 1)
        profile = np.exp(-(position[None, :] ** 2) / (diffusivity[:, None] * exposure[:, None]))
        saturation = sigmoid(logits[:, 2])[:, None]
        return np.concatenate((np.clip(profile, 0.0, 1.0), saturation), axis=1)
    if domain == "phase_simplex":
        phase_logits = logits[:, :4]
        phase_logits -= np.max(phase_logits, axis=1, keepdims=True)
        fractions = np.exp(phase_logits)
        fractions /= np.sum(fractions, axis=1, keepdims=True)
        interval = (0.1 + 0.9 * sigmoid(logits[:, 4]))[:, None]
        return np.concatenate((fractions, interval), axis=1)
    values = sigmoid(logits[:, :output_dim])
    if candidate_id == "CAND-P-066":
        values[:, :3] = 2.0 * values[:, :3] - 1.0
    return values


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    extension_path = root / "contracts" / "sim_extension_pack_39.v1.json"
    extension = load_json(extension_path)
    if extension["status"] != "PRETRAIN_FROZEN" or extension["authority"] != 0:
        raise RuntimeError("SIM_EXTENSION_CONTRACT_GATE")
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}

    common = extension["dataset"]
    task_ids = [task["candidate_id"] for task in extension["tasks"]]
    if len(task_ids) != 39 or len(set(task_ids)) != 39:
        raise RuntimeError("SIM_EXTENSION_TASK_COUNT_GATE")

    output_root = root / "data" / "staged_sim_extension_pack_39_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for task in extension["tasks"]:
        candidate_id = task["candidate_id"]
        contract = contracts[candidate_id]
        seed = common["generation_seed"] + int(candidate_id[-3:])
        rng = np.random.default_rng(seed)
        x_rows: list[np.ndarray] = []
        groups: list[str] = []
        splits: list[int] = []
        for family_index in range(common["families_per_task"]):
            family_id = f"{candidate_id}-SIMFAM-{family_index:03d}"
            family = rng.uniform(0.02, 0.98, size=8)
            split = split_for(candidate_id, family_id)
            for _ in range(common["records_per_family"]):
                condition = rng.uniform(0.0, 1.0, size=8)
                x_rows.append(np.concatenate((family, condition)).astype(np.float32))
                groups.append(family_id)
                splits.append(split)
        x = np.asarray(x_rows, dtype=np.float32)
        group = np.asarray(groups)
        split = np.asarray(splits, dtype=np.int8)
        y_norm = make_targets(candidate_id, task["domain"], x.astype(np.float64), len(task["outputs"]))
        offset = np.asarray(task["offset"], dtype=np.float64)
        scale = np.asarray(task["scale"], dtype=np.float64)
        y = (offset + scale * y_norm).astype(np.float32)

        train = split == 0
        baseline_x = np.column_stack((np.ones(np.sum(train)), x[train, :6]))
        ridge = 0.25 * np.eye(baseline_x.shape[1])
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(baseline_x.T @ baseline_x + ridge, baseline_x.T @ y[train])
        baseline = (np.column_stack((np.ones(len(x)), x[:, :6])) @ coefficients).astype(np.float32)

        sets = {code: set(group[split == code].tolist()) for code in (0, 1, 2)}
        overlap = sum(len(sets[a] & sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        counts = {SPLIT_NAMES[code]: int(np.sum(split == code)) for code in (0, 1, 2)}
        if overlap or min(counts.values()) < 120 or not np.all(np.isfinite(y)):
            raise RuntimeError(f"{candidate_id}:DATA_GATE:{overlap}:{counts}")

        dataset = output_root / f"{candidate_id}.npz"
        np.savez_compressed(dataset, x=x, y=y, baseline=baseline, group=group, split=split)
        metadata = {
            "schema": "cimc.forge200.sim-extension-dataset.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_SIM_EXTENSION_DATA_FROZEN",
            "candidate_id": candidate_id,
            "truth_class": extension["truth_class"],
            "public_claim_scope": extension["public_claim_scope"],
            "simulator": f"TEAM_OWNED_{task['domain'].upper()}_COUPLED_EQUATION_SURROGATE",
            "simulator_boundary": "Numerical host/ABI benchmark only; not calibrated experimental or board truth.",
            "generation_seed": seed,
            "records": len(y),
            "families": common["families_per_task"],
            "input_dimensions": x.shape[1],
            "output_dimensions": y.shape[1],
            "output_semantics": task["outputs"],
            "output_offset": task["offset"],
            "output_scale": task["scale"],
            "split_method": common["split"],
            "split_counts": counts,
            "cross_split_family_overlap": overlap,
            "baseline_execution": "TRAIN_ONLY_RIDGE_ON_FIRST_SIX_INPUTS_REDUCED_ORDER_SURROGATE",
            "baseline_coefficients_sha256": hashlib.sha256(canonical_bytes(coefficients.tolist())).hexdigest(),
            "original_input_contract": contract["input_contract"],
            "original_target_label": contract["target_label"],
            "original_primary_metric": contract["primary_metric"],
            "original_source_gate": contract["source_gate"],
            "original_task_contract_status": "UNCHANGED_FAIL_CLOSED",
            "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
            "extension_contract": {
                "path": "contracts/sim_extension_pack_39.v1.json",
                "sha256": sha256_file(extension_path),
            },
            "source_license": "TEAM_OWNED_GENERATED_SIMULATION",
            "experimental_records": 0,
            "teacher_outputs": 0,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
            "generator_sha256": sha256_file(Path(__file__)),
            "sha256": sha256_file(dataset),
        }
        write_json(dataset.with_suffix(".metadata.json"), metadata)
        records.append(metadata)

    receipt = {
        "schema": "cimc.forge200.sim-extension-pack-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_39_SIM_EXTENSION_DATASETS_FROZEN",
        "task_count": len(records),
        "records": records,
        "original_exact_contract_promotions": 0,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "sim_extension_pack_39_staging.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "tasks": len(records), "records": sum(r["records"] for r in records)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
