#!/usr/bin/env python3
"""Stage and train P088 as a low-rank masked SECOM sensor imputer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import build_package, canonical_bytes, heartbeat, sha256_file, write_json

SEEDS = [20260801, 20260802, 20260803]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stage(root: Path) -> tuple[Path, dict[str, Any]]:
    source = root / "data" / "raw" / "uci_secom" / "secom.zip"
    assignment = root / "data" / "splits" / "uci_secom.time_group_v2.assignments.tsv"
    rows = read_tsv(assignment)
    by_index = {int(row["row_index"]): row for row in rows}
    with zipfile.ZipFile(source) as archive:
        lines = archive.read("secom.data").decode("utf-8").splitlines()
    values, groups, splits = [], [], []
    for index, line in enumerate(lines):
        values.append([float(item) if item != "NaN" else np.nan for item in line.split()])
        groups.append(by_index[index]["time_group"])
        splits.append({"train": 0, "validation": 1, "test": 2}[by_index[index]["split"]])
    x = np.asarray(values, dtype=np.float32)
    split = np.asarray(splits, dtype=np.int8)
    output = root / "data" / "staged_contract_v2" / "CAND-P-088.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, x=x, groups=np.asarray(groups), split=split, candidate_id=np.asarray("CAND-P-088"), task_kind=np.asarray("masked_multivariate_regression"), authority=np.asarray(0, dtype=np.int8))
    contract = next(row for row in read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv") if row["candidate_id"] == "CAND-P-088")
    group_sets = {code: set(np.asarray(groups)[split == code]) for code in (0, 1, 2)}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    metadata = {
        "schema": "cimc.forge200.masked-imputation-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if overlap == 0 else "FAIL_CLOSED",
        "candidate_id": "CAND-P-088",
        "records": len(x),
        "features": x.shape[1],
        "observed_cells": int(np.isfinite(x).sum()),
        "missing_cells": int((~np.isfinite(x)).sum()),
        "counts": {name: int(np.sum(split == code)) for code, name in enumerate(("train", "validation", "test"))},
        "split_unit": "SECOM_calendar_day_time_group",
        "cross_split_group_overlap": overlap,
        "source_id": "uci_secom",
        "source_sha256": sha256_file(source),
        "split_sha256": sha256_file(assignment),
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "baseline_contract": contract["baseline"],
        "primary_metric_contract": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "truth_class": "OPEN_SECOM_OBSERVED_SENSOR_VALUES_SELF_SUPERVISED_MASKING",
        "mask_policy": "fixed_blind_masks_for_evaluation_random_train_only_masking",
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    if metadata["status"] != "PASS":
        raise RuntimeError("SPLIT_LEAKAGE")
    return output, metadata


def fixed_mask(observed: np.ndarray, split: np.ndarray, code: int) -> np.ndarray:
    rng = np.random.default_rng(20260801 + code)
    result = (rng.random(observed.shape) < 0.12) & observed & (split[:, None] == code)
    return result


def metrics(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray, interval_half: float) -> dict[str, float]:
    error = prediction[mask] - target[mask]
    return {
        "normalized_rmse": float(np.sqrt(np.mean(error ** 2))),
        "normalized_mae": float(np.mean(np.abs(error))),
        "interval_90_coverage": float(np.mean(np.abs(error) <= interval_half)),
        "interval_90_coverage_error": abs(float(np.mean(np.abs(error) <= interval_half)) - 0.9),
        "evaluated_cells": int(mask.sum()),
    }


def objective(value: dict[str, float], baseline: dict[str, float]) -> float:
    return float(
        0.45 * (1.0 - value["normalized_rmse"] / max(baseline["normalized_rmse"], 1e-9))
        + 0.35 * (1.0 - value["normalized_mae"] / max(baseline["normalized_mae"], 1e-9))
        + 0.20 * (baseline["interval_90_coverage_error"] - value["interval_90_coverage_error"])
    )


def validation_output_gate(target: np.ndarray, prediction: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Choose per-sensor residual shrinkage from validation cells only.

    A channel is left at the preregistered zero/median baseline unless at
    least ten blinded validation cells support a >=1% robust-error reduction.
    This prevents unstable near-constant sensors from exporting residuals.
    """
    gate = np.zeros(target.shape[1], dtype=np.float32)
    for column in range(target.shape[1]):
        selected = mask[:, column]
        if int(np.sum(selected)) < 10:
            continue
        truth = target[selected, column]
        residual = prediction[selected, column]
        baseline_loss = 0.55 * float(np.sqrt(np.mean(truth ** 2))) + 0.45 * float(np.mean(np.abs(truth)))
        candidates = []
        for alpha in (0.25, 0.5, 0.75, 1.0):
            error = alpha * residual - truth
            loss = 0.55 * float(np.sqrt(np.mean(error ** 2))) + 0.45 * float(np.mean(np.abs(error)))
            candidates.append((loss, alpha))
        best_loss, best_alpha = min(candidates)
        if best_loss < baseline_loss * 0.99:
            gate[column] = best_alpha
    return gate


def quantize(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    q, s = {}, {}
    for name, tensor in state.items():
        value = tensor.detach().cpu().numpy().astype(np.float32)
        scale = np.maximum(np.max(np.abs(value), axis=1, keepdims=True), 1e-12) / 127.0 if value.ndim == 2 else np.asarray(max(float(np.max(np.abs(value))), 1e-12) / 127.0, dtype=np.float32)
        q[name] = np.asarray(np.clip(np.rint(value / scale), -127, 127), dtype=np.int8)
        s[name] = np.asarray(scale, dtype=np.float32)
    return q, s


def run(args: argparse.Namespace) -> None:
    import torch
    from torch import nn

    root = args.root.resolve()
    dataset, metadata = stage(root)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset, allow_pickle=False)
    original = raw["x"].astype(np.float32)
    split = raw["split"].astype(np.int8)
    observed = np.isfinite(original)
    train = split == 0
    median = np.nanmedian(original[train], axis=0)
    median[~np.isfinite(median)] = 0.0
    filled = np.where(observed, original, median)
    scale = np.nanstd(original[train], axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    target = ((filled - median) / scale).astype(np.float32)
    original_missing = (~observed).astype(np.float32)
    masks = {name: fixed_mask(observed, split, code) for code, name in enumerate(("train", "validation", "test"))}
    baseline_prediction = np.zeros_like(target)
    baseline_train_error = np.abs(target[masks["train"]])
    baseline_interval = float(np.quantile(baseline_train_error, 0.90))
    baseline = {name: metrics(target, baseline_prediction, masks[name], baseline_interval) for name in masks}
    baseline_validation_objective = objective(baseline["validation"], baseline["validation"])
    baseline_test_objective = objective(baseline["test"], baseline["test"])

    class LowRankImputer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(1180, 32), nn.GELU(), nn.Linear(32, 590))
            self.register_buffer("output_gate", torch.ones(590, dtype=torch.float32))
            # The preregistered median imputer is zero in standardized space.
            # Start exactly at that baseline and learn only a residual so an
            # unlucky seed cannot create the catastrophic first-pass output.
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        def forward(self, value: Any) -> Any:
            return self.net(value) * self.output_gate

    parameter_count = sum(p.numel() for p in LowRankImputer().parameters())
    if parameter_count > 64_000:
        raise RuntimeError("PARAMETER_CAP")
    output = (args.artifact_root / "CAND-P-088").resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    train_indices = np.flatnonzero(train)

    def predict(model: Any, artificial_mask: np.ndarray) -> np.ndarray:
        model.eval()
        values = target.copy()
        values[artificial_mask] = 0.0
        input_value = np.concatenate((values, np.maximum(original_missing, artificial_mask.astype(np.float32))), axis=1).astype(np.float32)
        parts = []
        with torch.no_grad():
            for start in range(0, len(values), 256):
                parts.append(model(torch.from_numpy(input_value[start:start+256]).to(device)).cpu().numpy())
        return np.concatenate(parts)

    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = LowRankImputer().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=2e-4)
        generator = np.random.default_rng(seed)
        best_score, patience = 0.0, 0
        best_path = output / f"train_seed_{seed}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), best_path)
        for epoch in range(120):
            order = generator.permutation(train_indices)
            model.train()
            for start in range(0, len(order), 64):
                chosen = order[start:start+64]
                batch_observed = observed[chosen]
                artificial = (generator.random(batch_observed.shape) < 0.15) & batch_observed
                batch_values = target[chosen].copy()
                batch_values[artificial] = 0.0
                batch_input = np.concatenate((batch_values, np.maximum(original_missing[chosen], artificial.astype(np.float32))), axis=1).astype(np.float32)
                prediction = model(torch.from_numpy(batch_input).to(device))
                mask_tensor = torch.from_numpy(artificial).to(device)
                target_tensor = torch.from_numpy(target[chosen]).to(device)
                loss = torch.mean((prediction[mask_tensor] - target_tensor[mask_tensor]) ** 2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            train_prediction = predict(model, masks["train"])
            interval = float(np.quantile(np.abs(train_prediction[masks["train"]] - target[masks["train"]]), 0.90))
            validation_prediction = predict(model, masks["validation"])
            validation_metrics = metrics(target, validation_prediction, masks["validation"], interval)
            value = objective(validation_metrics, baseline["validation"])
            if value > best_score + 1e-5:
                best_score, patience = value, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-P-088", "TRAIN_MASKED_IMPUTER", seed, epoch)
            if epoch >= 29 and patience >= 15:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        # Freeze a per-output residual gate using validation only.  Test cells
        # are never consulted; disabled channels exactly equal the median
        # baseline in standardized space.
        with torch.no_grad():
            model.output_gate.fill_(1.0)
        validation_raw_prediction = predict(model, masks["validation"])
        selected_gate = validation_output_gate(target, validation_raw_prediction, masks["validation"])
        with torch.no_grad():
            model.output_gate.copy_(torch.from_numpy(selected_gate).to(device))
        train_prediction = predict(model, masks["train"])
        interval = float(np.quantile(np.abs(train_prediction[masks["train"]] - target[masks["train"]]), 0.90))
        test_prediction = predict(model, masks["test"])
        test_metrics = metrics(target, test_prediction, masks["test"], interval)
        test_objective = objective(test_metrics, baseline["test"])
        gated_validation_prediction = predict(model, masks["validation"])
        gated_validation_metrics = metrics(target, gated_validation_prediction, masks["validation"], interval)
        gated_validation_objective = objective(gated_validation_metrics, baseline["validation"])
        reports.append({"seed": seed, "epochs": epoch+1, "validation_objective": gated_validation_objective, "ungated_checkpoint_validation_objective": best_score, "enabled_output_channels": int(np.count_nonzero(selected_gate)), "test_objective": test_objective, "baseline_test_objective": baseline_test_objective, "beats_baseline": test_objective > baseline_test_objective + 1e-4, "test": test_metrics, "interval_half_width_normalized": interval})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best["seed"])
    state = states[best_seed]
    model = LowRankImputer().to(device)
    model.load_state_dict(state)
    sample_indices = np.flatnonzero(split == 2)[:16]
    sample_values = target[sample_indices].copy()
    sample_mask = masks["test"][sample_indices]
    sample_values[sample_mask] = 0.0
    sample_input = np.concatenate((sample_values, np.maximum(original_missing[sample_indices], sample_mask.astype(np.float32))), axis=1).astype(np.float32)
    model.eval()
    with torch.no_grad():
        fp = model(torch.from_numpy(sample_input).to(device)).cpu().numpy()
    q, s = quantize(state)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **q, **{f"scale::{name}": value for name, value in s.items()})
    payload = buffer.getvalue()
    if len(payload) > 80 * 1024:
        raise RuntimeError("W8_PAYLOAD_CAP")
    model.load_state_dict({name: torch.from_numpy(np.asarray(np.asarray(value, dtype=np.float32)*np.asarray(s[name], dtype=np.float32), dtype=np.float32)) for name, value in q.items()})
    with torch.no_grad():
        quant = model(torch.from_numpy(sample_input).to(device)).cpu().numpy()
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=sample_input, y=target[sample_indices], mask=sample_mask, fp32=fp, quantized=quant)
    model.load_state_dict(state)
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(sample_input[:1]).to(device), onnx_path, input_names=["masked_values_and_missingness"], output_names=["imputed_standardized_values"], dynamic_axes={"masked_values_and_missingness": {0: "batch"}, "imputed_standardized_values": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": "CAND-P-088", "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "best_seed": best_seed})).hexdigest()
    schema = {"task_kind": "masked_multivariate_regression", "shape": [None, 590], "authority": 0}
    package = build_package(output, "CAND-P-088", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(schema)).hexdigest(), engine_id=1)
    pass_count = sum(item["beats_baseline"] for item in reports)
    objective_values = np.asarray([item["test_objective"] for item in reports], dtype=np.float64)
    aggregate = {"mean": float(objective_values.mean()), "variance": float(objective_values.var()), "std": float(objective_values.std()), "worst": float(objective_values.min())}
    aggregate_pass = aggregate["mean"] > baseline_test_objective + 1e-4
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if aggregate_pass else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {"schema": "cimc.forge200.p088-evaluation.v2", "status": "PASS" if aggregate_pass else "FAIL_CLOSED", "candidate_id": "CAND-P-088", "baseline": baseline, "baseline_test_objective": baseline_test_objective, "validation_only_output_gate": "PER_SENSOR_ALPHA_IN_0_0P25_0P5_0P75_1_REQUIRING_10_CELLS_AND_1_PERCENT_ROBUST_ERROR_GAIN", "seed_reports": reports, "three_seed_aggregate": aggregate, "three_seed_mean_beats_baseline": aggregate_pass, "individual_seed_baseline_pass_count_reported_not_used_as_extra_gate": pass_count, "authority": 0, "board_accepted": False}
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"median": median.tolist(), "scale": scale.tolist(), "best_interval_half_width_normalized": best["interval_half_width_normalized"]})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "max_abs_standardized_error": float(np.max(np.abs(fp-quant)))})
    receipt = {"schema": "cimc.forge200.promotion-receipt.v3", "status": status, "candidate_id": "CAND-P-088", "authority": 0, "board_accepted": False, "countable_model": False, "three_seed_count": 3, "three_seed_aggregate": aggregate, "three_seed_mean_beats_baseline": aggregate_pass, "individual_seed_baseline_pass_count_reported_not_used_as_extra_gate": pass_count, "parameter_count": parameter_count, "parameter_cap": 64_000, "w8_payload_bytes": len(payload), "w8_payload_byte_cap": 80*1024, "best_seed": best_seed, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter()-started, "gpu": {"name": props.name, "vram_gib": props.total_memory/1024**3}}
    write_json(output / "promotion_receipt.json", receipt)
    print(json.dumps({"candidate_id": "CAND-P-088", "status": status, "three_seed_mean_objective": aggregate["mean"], "individual_passes_reported": pass_count, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
