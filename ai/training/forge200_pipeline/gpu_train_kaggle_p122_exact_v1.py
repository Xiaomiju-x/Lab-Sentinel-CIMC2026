#!/usr/bin/env python3
"""Train, evaluate once, W8A8-quantize, export, and package exact P122."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state, manifest, quantize_state


PARAMETER_CAP = 96_000
WEIGHT_BYTE_CAP = 100 * 1024
LANDMARK = 250.0
INTERVAL_Z90 = 1.6448536269514722


def softplus_np(value: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)


def decode(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = raw[:, 0].astype(np.float64)
    sigma = np.clip(softplus_np(raw[:, 1].astype(np.float64)) + 0.08, 0.08, 2.0)
    point = np.clip(np.exp(mu), 25.0, 5000.0)
    lower = np.clip(np.exp(mu - INTERVAL_Z90 * sigma), 1.0, 5000.0)
    upper = np.clip(np.exp(mu + INTERVAL_Z90 * sigma), 25.0, 10000.0)
    return point, lower, upper, sigma


def concordance_index(time_value: np.ndarray, event: np.ndarray, prediction: np.ndarray) -> float:
    comparable = concordant = ties = 0
    for index in np.flatnonzero(event):
        later = time_value > time_value[index]
        count = int(np.sum(later))
        if count == 0:
            continue
        comparable += count
        concordant += int(np.sum(prediction[later] > prediction[index]))
        ties += int(np.sum(prediction[later] == prediction[index]))
    return float((concordant + 0.5 * ties) / comparable) if comparable else 0.5


def survival_metrics(
    time_value: np.ndarray,
    event: np.ndarray,
    point_rul: np.ndarray,
    lower_rul: np.ndarray,
    upper_rul: np.ndarray,
) -> dict[str, float]:
    true_rul = time_value - LANDMARK
    observed = event.astype(bool)
    mae = float(np.mean(np.abs(point_rul[observed] - true_rul[observed])))
    c_index = concordance_index(time_value, observed, point_rul + LANDMARK)
    coverage = float(np.mean((true_rul[observed] >= lower_rul[observed]) & (true_rul[observed] <= upper_rul[observed])))
    width = upper_rul[observed] - lower_rul[observed]
    mae_score = 1.0 / (1.0 + mae / 250.0)
    coverage_score = max(0.0, 1.0 - abs(coverage - 0.90))
    result = {
        "RUL_MAE_cycles": mae,
        "concordance_index": c_index,
        "interval_90_coverage": coverage,
        "interval_90_mean_width_cycles": float(np.mean(width)),
        "interval_90_median_width_cycles": float(np.median(width)),
        "event_records": int(np.sum(observed)),
        "right_censored_records": int(np.sum(~observed)),
        "mae_score": mae_score,
        "coverage_score": coverage_score,
    }
    result["primary_composite"] = float(np.mean([mae_score, c_index, coverage_score]))
    return result


def damage_rate(relative_history: np.ndarray) -> np.ndarray:
    return np.maximum((relative_history[:, -1] - relative_history[:, 0]) / LANDMARK, 1e-7)


def fit_baseline(
    relative_history: np.ndarray,
    time_value: np.ndarray,
    event: np.ndarray,
    train: np.ndarray,
    validation: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rate = damage_rate(relative_history)
    train_events = train[event[train]]
    log_rate = np.log(rate[train_events])
    log_rul = np.log(np.maximum(time_value[train_events] - LANDMARK, 25.0))
    slope = float(np.cov(log_rate, log_rul, bias=True)[0, 1] / (np.var(log_rate) + 1e-8))
    slope = float(np.clip(slope, -3.0, -0.25))
    intercept = float(np.mean(log_rul) - slope * np.mean(log_rate))
    cm_rul = np.clip(np.exp(intercept + slope * np.log(rate)), 25.0, 5000.0)
    remaining_damage = np.maximum(0.20 - relative_history[:, -1], 0.005)
    linear_damage_rul = np.clip(remaining_damage / rate, 25.0, 5000.0)

    candidates = []
    for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
        point = np.exp((1.0 - alpha) * np.log(cm_rul) + alpha * np.log(linear_damage_rul))
        residual = log_rul - np.log(point[train_events])
        sigma = float(np.clip(np.std(residual), 0.15, 1.50))
        lower = np.exp(np.log(point) - INTERVAL_Z90 * sigma)
        upper = np.exp(np.log(point) + INTERVAL_Z90 * sigma)
        metrics = survival_metrics(time_value[validation], event[validation], point[validation], lower[validation], upper[validation])
        candidates.append({"alpha": alpha, "sigma_log": sigma, "validation": metrics})
    selected = max(candidates, key=lambda record: (record["validation"]["primary_composite"], -record["alpha"]))
    alpha = float(selected["alpha"])
    sigma = float(selected["sigma_log"])
    point = np.exp((1.0 - alpha) * np.log(cm_rul) + alpha * np.log(linear_damage_rul))
    mu = np.log(point)
    lower = np.exp(mu - INTERVAL_Z90 * sigma)
    upper = np.exp(mu + INTERVAL_Z90 * sigma)
    frozen = {
        "kind": "Coffin_Manson_fit_with_linear_damage_accumulation",
        "implementation": "train-observed log_RUL versus pre-landmark Rth damage-rate power law blended with deterministic remaining-damage/rate projection",
        "coffin_manson_log_intercept": intercept,
        "coffin_manson_exponent": slope,
        "alpha_grid": [record["alpha"] for record in candidates],
        "selection_split": "validation_only",
        "selected_alpha_linear_damage": alpha,
        "sigma_log_train_event_residual": sigma,
        "validation_candidates": candidates,
        "test_labels_or_metrics_read_during_selection": False,
    }
    return frozen, point.astype(np.float32), lower.astype(np.float32), upper.astype(np.float32), mu.astype(np.float32)


def fake_quantize(value: Any, scale: float) -> Any:
    import torch

    return torch.clamp(torch.round(value / scale), -127.0, 127.0) * scale


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    dataset = root / "data" / "staged_kaggle_p122_exact_v1" / "CAND-P-122.npz"
    staging_path = root / "evidence" / "kaggle_p122_exact_staging.v1.json"
    source_audit_path = root / "evidence" / "kaggle_p107_p122_source_contract_audit.v1.json"
    staging = json.loads(staging_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if (
        staging["status"] != "PASS_EXACT_SOURCE_LABEL_SPLIT_TRAINING_AUTHORIZED"
        or not staging["training_authorized"]
        or staging["split"]["cross_split_family_overlap"] != 0
        or staging["split"]["cross_split_unit_overlap"] != 0
        or staging["future_history_in_inputs"]
        or sha256_file(dataset) != staging["dataset"]["sha256"]
        or source_audit["p122"]["status"] != "EXACT_ADMITTED_TRAINING_AUTHORIZED"
    ):
        raise RuntimeError("P122_DATA_SOURCE_SPLIT_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = args.artifact_root.resolve() / "CAND-P-122"
    output.mkdir(parents=True, exist_ok=True)
    once_path = output / "frozen_test_evaluation.v1.json"
    if once_path.exists():
        raise RuntimeError("FROZEN_TEST_ALREADY_EVALUATED_REFUSE_RERUN")
    heartbeat_path = output / "heartbeat.json"
    started = time.perf_counter()

    data = np.load(dataset, allow_pickle=False)
    split = data["split"].astype(np.int8)
    train = np.flatnonzero(split == 0)
    validation = np.flatnonzero(split == 1)
    test = np.flatnonzero(split == 2)
    if len(set(data["family"][train].astype(str)) & set(data["family"][validation].astype(str))) or len(set(data["family"][train].astype(str)) & set(data["family"][test].astype(str))):
        raise RuntimeError("RUNTIME_FAMILY_LEAKAGE_GATE")
    event = data["event_observed"].astype(bool)
    time_value = data["event_or_censor_cycle"].astype(np.float32)
    relative_history = data["relative_rth_history"].astype(np.float32)
    baseline_freeze, baseline_point, baseline_lower, baseline_upper, baseline_mu = fit_baseline(relative_history, time_value, event, train, validation)
    baseline_freeze.update({
        "schema": "cimc.forge200.p122-baseline-freeze.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "CAND-P-122",
        "dataset_sha256": sha256_file(dataset),
        "training_families": sorted(set(data["family"][train].astype(str))),
        "validation_families": sorted(set(data["family"][validation].astype(str))),
        "test_families_names_frozen_but_test_labels_not_evaluated": sorted(set(data["family"][test].astype(str))),
        "authority": 0,
    })
    write_json(output / "baseline_selection_frozen_before_test.json", baseline_freeze)

    history_cycles = data["history_cycles"].astype(np.float32)
    slopes = np.diff(relative_history, axis=1) / np.diff(history_cycles)[None, :] * 100.0
    x_raw = np.concatenate(
        (
            data["static"].astype(np.float32),
            data["rth_history"].astype(np.float32),
            relative_history,
            slopes.astype(np.float32),
            relative_history[:, -1:].astype(np.float32),
        ),
        axis=1,
    )
    mean = x_raw[train].mean(axis=0)
    std = x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x_features = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    model_input = np.concatenate((x_features, baseline_mu[:, None]), axis=1).astype(np.float32)
    log_target = np.log(np.maximum(time_value - LANDMARK, 25.0)).astype(np.float32)

    class SurvivalMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(x_features.shape[1], 96)
            self.fc2 = nn.Linear(96, 48)
            self.fc3 = nn.Linear(48, 2)

        def forward(self, value: Any) -> Any:
            features, baseline = value[:, :-1], value[:, -1]
            hidden = nn.functional.gelu(self.fc1(features))
            hidden = nn.functional.gelu(self.fc2(hidden))
            raw = self.fc3(hidden)
            mu = baseline + 2.0 * torch.tanh(raw[:, 0])
            return torch.stack((mu, raw[:, 1]), dim=1)

    parameter_count = sum(parameter.numel() for parameter in SurvivalMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")

    train_set = TensorDataset(
        torch.from_numpy(model_input[train]),
        torch.from_numpy(log_target[train]),
        torch.from_numpy(event[train].astype(np.float32)),
    )

    def torch_loss(raw: Any, log_time: Any, observed: Any) -> Any:
        mu = raw[:, 0]
        sigma = torch.clamp(nn.functional.softplus(raw[:, 1]) + 0.08, 0.08, 2.0)
        z = (log_time - mu) / sigma
        event_nll = 0.5 * z.square() + torch.log(sigma)
        survival = torch.clamp(0.5 * torch.erfc(z / math.sqrt(2.0)), min=1e-7)
        censor_nll = -torch.log(survival)
        nll = observed * event_nll + (1.0 - observed) * censor_nll
        event_regression = observed * nn.functional.smooth_l1_loss(mu, log_time, reduction="none", beta=0.20)
        return torch.mean(nll + 0.12 * event_regression)

    reports: list[dict[str, Any]] = []
    states: dict[int, dict[str, Any]] = {}
    validation_outputs: dict[int, np.ndarray] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SurvivalMLP().to(device)
        with torch.no_grad():
            model.fc3.weight.zero_()
            model.fc3.bias.zero_()
            sigma_initial = float(baseline_freeze["sigma_log_train_event_residual"])
            model.fc3.bias[1] = math.log(max(math.exp(max(sigma_initial - 0.08, 1e-4)) - 1.0, 1e-4))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=4e-4)
        loader = DataLoader(train_set, batch_size=min(args.batch_size, len(train)), shuffle=True, generator=torch.Generator().manual_seed(seed))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_log_time, batch_event in loader:
                optimizer.zero_grad(set_to_none=True)
                raw = model(batch_x.to(device))
                loss = torch_loss(raw, batch_log_time.to(device), batch_event.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_raw = model(torch.from_numpy(model_input[validation]).to(device)).cpu().numpy()
            point, lower, upper, _ = decode(validation_raw)
            validation_metrics = survival_metrics(time_value[validation], event[validation], point, lower, upper)
            score = validation_metrics["primary_composite"]
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-P-122", "TRAIN_EXACT_SURVIVAL", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            validation_raw = model(torch.from_numpy(model_input[validation]).to(device)).cpu().numpy()
        point, lower, upper, _ = decode(validation_raw)
        validation_metrics = survival_metrics(time_value[validation], event[validation], point, lower, upper)
        reports.append({"seed": seed, "epochs": epoch + 1, "validation": validation_metrics})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        validation_outputs[seed] = validation_raw

    best_seed = int(max(reports, key=lambda record: record["validation"]["primary_composite"])["seed"])
    baseline_test = survival_metrics(time_value[test], event[test], baseline_point[test], baseline_lower[test], baseline_upper[test])
    test_reports = []
    test_outputs: dict[int, np.ndarray] = {}
    for report in reports:
        seed = int(report["seed"])
        model = SurvivalMLP().to(device)
        model.load_state_dict(states[seed])
        model.eval()
        with torch.no_grad():
            raw = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy()
        point, lower, upper, _ = decode(raw)
        metric = survival_metrics(time_value[test], event[test], point, lower, upper)
        test_reports.append({"seed": seed, "test": metric, "beats_baseline": metric["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        test_outputs[seed] = raw
    composites = np.asarray([record["test"]["primary_composite"] for record in test_reports], dtype=np.float64)
    aggregate = {"mean": float(composites.mean()), "variance": float(composites.var()), "std": float(composites.std()), "worst": float(composites.min())}
    aggregate_pass = aggregate["mean"] > baseline_test["primary_composite"] + 1e-4
    once = {
        "schema": "cimc.forge200.p122-frozen-test-evaluation.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "EVALUATED_ONCE_AFTER_VALIDATION_FREEZE",
        "candidate_id": "CAND-P-122",
        "dataset_sha256": sha256_file(dataset),
        "baseline_selection_sha256": sha256_file(output / "baseline_selection_frozen_before_test.json"),
        "baseline_test": baseline_test,
        "seed_test_reports": test_reports,
        "aggregate": aggregate,
        "aggregate_mean_gate": aggregate_pass,
        "test_used_for_hyperparameter_or_seed_selection": False,
        "best_seed_selected_by_validation": best_seed,
        "authority": 0,
    }
    write_json(once_path, once)

    model = SurvivalMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    fp_raw = test_outputs[best_seed]
    fp_point, fp_lower, fp_upper, _ = decode(fp_raw)
    quantized, scales = quantize_state(states[best_seed])
    with torch.no_grad():
        feature_tensor = torch.from_numpy(model_input[train, :-1]).to(device)
        hidden1 = nn.functional.gelu(model.fc1(feature_tensor))
        hidden2 = nn.functional.gelu(model.fc2(hidden1))
    activation_scales = {
        "input": max(float(np.max(np.abs(model_input[train])) / 127.0), 1e-8),
        "hidden1": max(float(hidden1.abs().max().item() / 127.0), 1e-8),
        "hidden2": max(float(hidden2.abs().max().item() / 127.0), 1e-8),
    }
    quant_model = SurvivalMLP().to(device)
    quant_model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_model.eval()

    def quant_infer(values: np.ndarray) -> np.ndarray:
        value = torch.from_numpy(values).to(device)
        with torch.no_grad():
            value = fake_quantize(value, activation_scales["input"])
            features, baseline = value[:, :-1], value[:, -1]
            hidden = fake_quantize(nn.functional.gelu(quant_model.fc1(features)), activation_scales["hidden1"])
            hidden = fake_quantize(nn.functional.gelu(quant_model.fc2(hidden)), activation_scales["hidden2"])
            raw = quant_model.fc3(hidden)
            mu = baseline + 2.0 * torch.tanh(raw[:, 0])
            return torch.stack((mu, raw[:, 1]), dim=1).cpu().numpy()

    quant_raw = quant_infer(model_input[test])
    quant_point, quant_lower, quant_upper, _ = decode(quant_raw)
    quant_metrics = survival_metrics(time_value[test], event[test], quant_point, quant_lower, quant_upper)
    fp_metrics = next(record["test"] for record in test_reports if record["seed"] == best_seed)
    quant_delta = float(fp_metrics["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    passed = aggregate_pass and quant_pass

    payload_buffer = io.BytesIO()
    np.savez_compressed(
        payload_buffer,
        **quantized,
        **{f"scale::{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()},
        **{f"activation_scale::{name}": np.asarray(value, dtype=np.float32) for name, value in activation_scales.items()},
    )
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8A8_PAYLOAD_CAP:{len(payload)}")
    golden = output / "golden_vectors.npz"
    sample = test[:64]
    np.savez_compressed(
        golden,
        input=model_input[sample],
        event_or_censor_cycle=time_value[sample],
        event_observed=event[sample].astype(np.int8),
        fp32_raw=fp_raw[:64],
        fp32_point_rul=fp_point[:64],
        fp32_interval_lower=fp_lower[:64],
        fp32_interval_upper=fp_upper[:64],
        w8a8_raw=quant_raw[:64],
        w8a8_point_rul=quant_point[:64],
    )
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(model_input[test[:1]]).to(device),
        onnx_path,
        input_names=["preprocessed_joint_geometry_alloy_and_prelandmark_rth_history"],
        output_names=["lognormal_rul_location_and_raw_scale"],
        dynamic_axes={"preprocessed_joint_geometry_alloy_and_prelandmark_rth_history": {0: "batch"}, "lognormal_rul_location_and_raw_scale": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {
        "task_kind": "right_censored_lognormal_survival_regression",
        "shape": [None, 2],
        "semantics": ["mu_log_remaining_cycles", "raw_sigma_log_remaining_cycles"],
        "postprocess": "sigma=clip(softplus(raw)+0.08,0.08,2); median_RUL=clip(exp(mu),25,5000); 90pct_interval=exp(mu+-1.64485362695*sigma)",
        "event_definition": "F1_first_published_inspection_cycle_with_steady_state_Rth_relative_increase_ge_0.20",
        "landmark_cycle": 250,
        "units": "thermal_shock_cycles",
        "authority": 0,
        "runtime_domain": "PACKAGING_RELIABILITY_SHADOW",
    }
    release_root = hashlib.sha256(canonical_bytes({
        "candidate_id": "CAND-P-122",
        "dataset_sha256": sha256_file(dataset),
        "baseline_freeze_sha256": sha256_file(output / "baseline_selection_frozen_before_test.json"),
        "test_evaluation_sha256": sha256_file(once_path),
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden),
        "three_seed_mean": aggregate["mean"],
    })).hexdigest()
    package = build_package(output, "CAND-P-122", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    audit = {
        "schema": "cimc.forge200.kaggle-p122-contract-exact-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": "CAND-P-122",
        "truth_class": "OPEN_EXPERIMENTAL_L2",
        "source_gate_match": "LICENSE_REVIEWED_OPEN_FATIGUE_DATA_WITH_PACKAGE_FAMILY_SPLIT",
        "baseline": {"kind": baseline_freeze["kind"], "validation": baseline_freeze["validation_candidates"], "test": baseline_test},
        "validation_seed_reports": reports,
        "test_seed_reports": test_reports,
        "aggregate": aggregate,
        "g3_aggregate_mean_gate": aggregate_pass,
        "quantized_best_validation_seed": {"seed": best_seed, "test": quant_metrics, "primary_composite_delta": quant_delta, "gate": quant_pass},
        "parameter_count": parameter_count,
        "w8a8_payload_bytes": len(payload),
        "activation_quantization": "fixed_symmetric_INT8_scales_calibrated_on_train_only",
        "test_evaluated_once_after_validation_freeze": True,
        "future_history_in_inputs": False,
        "teacher_or_fixture_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", audit)
    write_json(output / "source_manifest.json", staging)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "activation_scales": activation_scales})
    write_json(output / "output_schema.json", output_schema)
    write_json(output / "quantization_parity.json", {"fp32": fp_metrics, "w8a8": quant_metrics, "primary_composite_delta": quant_delta, "gate": quant_pass})
    write_json(output / "baseline_report.json", {"freeze": baseline_freeze, "test": baseline_test})
    (output / "model_card.md").write_text(
        f"# CAND-P-122 exact solder-fatigue survival model\n\n"
        f"- Status: `{status}`.\n"
        f"- Source: 1,531 independent high-power LED solder-interconnect samples; package-family split; F1 is first published stage with steady-state Rth rise >=20%.\n"
        f"- Three-seed mean composite: `{aggregate['mean']:.6f}` vs frozen Coffin-Manson + linear-damage baseline `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Validation-selected W8A8 test RUL MAE: `{quant_metrics['RUL_MAE_cycles']:.3f}` cycles; concordance `{quant_metrics['concordance_index']:.6f}`; 90% interval coverage `{quant_metrics['interval_90_coverage']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8A8 payload: `{len(payload)}` bytes.\n"
        f"- Scope is the published LED solder-interconnect thermal-shock protocol only. Authority `0`; deterministic control unavailable; unified GD32 board acceptance pending.\n",
        encoding="utf-8",
    )
    promotion = {
        "schema": "cimc.forge200.promotion-receipt.v3",
        "status": status,
        "candidate_id": "CAND-P-122",
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "host_contract_pass": passed,
        "truth_class": "OPEN_EXPERIMENTAL_L2",
        "claim_state": "HOST_EXACT_BOARD_PENDING",
        "public_claim_scope": "HIGH_POWER_LED_SOLDER_INTERCONNECT_THERMAL_SHOCK_ONLY",
        "source_gate_match": "LICENSE_REVIEWED_OPEN_FATIGUE_DATA_WITH_PACKAGE_FAMILY_SPLIT",
        "three_seed_count": 3,
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden),
        "contract_exact_audit_sha256": sha256_file(output / "contract_exact_audit.json"),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3},
    }
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    heartbeat(heartbeat_path, "CAND-P-122", "COMPLETE")
    closure = {
        "schema": "cimc.forge200.kaggle-p122-exact-closure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "PARTIAL",
        "record": audit,
        "promotion_receipt": {"path": str((output / "promotion_receipt.json").relative_to(root)).replace("\\", "/"), "sha256": sha256_file(output / "promotion_receipt.json")},
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "kaggle_p122_exact_closure.v1.json", closure)
    print(json.dumps({
        "candidate_id": "CAND-P-122",
        "status": status,
        "three_seed_mean": aggregate["mean"],
        "baseline": baseline_test["primary_composite"],
        "w8a8": quant_metrics["primary_composite"],
        "runtime_seconds": promotion["runtime_seconds"],
    }, sort_keys=True))
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=192)
    parser.add_argument("--max-epochs", type=int, default=220)
    parser.add_argument("--min-epochs", type=int, default=45)
    parser.add_argument("--early-stop-patience", type=int, default=28)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    args = parser.parse_args()
    receipt = run(args)
    return 0 if receipt["host_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
