#!/usr/bin/env python3
"""GPU corrective trainer for the five contract-restaged material tasks."""

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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
)
from scipy.stats import spearmanr

from gpu_train_job import (
    SEEDS,
    build_package,
    canonical_bytes,
    heartbeat,
    sha256_file,
    write_json,
)


ARCHITECTURES = {
    "CAND-P-069": (192, 96),
    "CAND-P-071": (192, 96),
    "CAND-P-072": (192, 96),
    "CAND-P-074": (160, 96),
    "CAND-P-075": (160, 96),
    "CAND-P-076": (160, 96),
    "CAND-P-077": (160, 96),
    "CAND-P-078": (176, 96),
    "CAND-P-086": (192, 96),
    "CAND-P-087": (40, 32),
    "CAND-P-140": (256, 160),
}
PARAMETER_CAPS = {
    "CAND-P-069": 64_000,
    "CAND-P-071": 60_000,
    "CAND-P-072": 58_000,
    "CAND-P-074": 50_000,
    "CAND-P-075": 50_000,
    "CAND-P-076": 48_000,
    "CAND-P-077": 48_000,
    "CAND-P-078": 56_000,
    "CAND-P-086": 62_000,
    "CAND-P-087": 58_000,
    "CAND-P-140": 128_000,
}
WEIGHT_BYTE_CAPS = {
    "CAND-P-069": 80 * 1024,
    "CAND-P-071": 72 * 1024,
    "CAND-P-072": 72 * 1024,
    "CAND-P-074": 64 * 1024,
    "CAND-P-075": 64 * 1024,
    "CAND-P-076": 60 * 1024,
    "CAND-P-077": 60 * 1024,
    "CAND-P-078": 68 * 1024,
    "CAND-P-086": 76 * 1024,
    "CAND-P-087": 72 * 1024,
    "CAND-P-140": 128 * 1024,
}


def probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    return {
        "auroc": float(roc_auc_score(y, probability)),
        "auprc": float(average_precision_score(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, prediction)),
        "unstable_recall": float(recall_score(y, prediction, pos_label=0, zero_division=0)),
    }


def regression_metrics(candidate_id: str, y: np.ndarray, prediction: np.ndarray, interval_half: float) -> dict[str, float]:
    prediction = prediction.reshape(-1)
    if candidate_id != "CAND-P-069":
        prediction = np.maximum(prediction, 0.0)
    y = y.reshape(-1)
    result = {
        "mae": float(np.mean(np.abs(prediction - y))),
        "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
    }
    if candidate_id in {"CAND-P-069", "CAND-P-071", "CAND-P-074", "CAND-P-075"}:
        rho = spearmanr(y, prediction).statistic
        result["spearman_rho"] = float(rho if np.isfinite(rho) else 0.0)
    if candidate_id in {"CAND-P-074", "CAND-P-075"}:
        result["log_mae"] = float(
            np.mean(np.abs(np.log1p(prediction) - np.log1p(np.maximum(y, 0.0))))
        )
    elif candidate_id == "CAND-P-072":
        result["log_mae"] = float(
            np.mean(np.abs(np.log1p(prediction) - np.log1p(np.maximum(y, 0.0))))
        )
        result["relative_mape"] = float(
            np.mean(np.abs(prediction - y) / np.maximum(np.abs(y), 1.0))
        )
    elif candidate_id == "CAND-P-078":
        easy = (y <= 2.0).astype(np.int64)
        result["easy_exfoliation_auroc"] = float(
            roc_auc_score(easy, -prediction)
        ) if len(np.unique(easy)) == 2 else 0.5
    elif candidate_id in {"CAND-P-076", "CAND-P-077"}:
        result["relative_mape"] = float(
            np.mean(np.abs(prediction - y) / np.maximum(np.abs(y), 1.0))
        )
    elif candidate_id == "CAND-P-140":
        rho = spearmanr(y, prediction).statistic
        result["spearman_rho"] = float(rho if np.isfinite(rho) else 0.0)
        result["interval_90_coverage"] = float(
            np.mean(np.abs(y - prediction) <= interval_half)
        )
        result["interval_90_coverage_error"] = abs(
            result["interval_90_coverage"] - 0.9
        )
    return result


def objective(candidate_id: str, metrics: dict[str, float], baseline: dict[str, float]) -> float:
    """Higher is better; scales are frozen from the train-only baseline."""
    if candidate_id in {"CAND-P-069", "CAND-P-071"}:
        return 0.5 * (
            1.0 - metrics["mae"] / max(baseline["mae"], 1e-9)
            + metrics["spearman_rho"] - baseline["spearman_rho"]
        )
    if candidate_id in {"CAND-P-074", "CAND-P-075"}:
        return 0.5 * (
            1.0 - metrics["log_mae"] / max(baseline["log_mae"], 1e-9)
            + metrics["spearman_rho"] - baseline["spearman_rho"]
        )
    if candidate_id == "CAND-P-072":
        return -0.5 * (
            metrics["log_mae"] / max(baseline["log_mae"], 1e-9)
            + metrics["relative_mape"] / max(baseline["relative_mape"], 1e-9)
        )
    if candidate_id == "CAND-P-078":
        return 0.5 * (
            1.0 - metrics["mae"] / max(baseline["mae"], 1e-9)
            + metrics["easy_exfoliation_auroc"] - baseline["easy_exfoliation_auroc"]
        )
    if candidate_id == "CAND-P-086":
        return float(np.mean([metrics["auroc"], metrics["macro_f1"], metrics["unstable_recall"]]))
    if candidate_id == "CAND-P-087":
        return float(np.mean([(metrics["mcc"] + 1.0) / 2.0, metrics["auprc"], metrics["balanced_accuracy"]]))
    if candidate_id in {"CAND-P-076", "CAND-P-077"}:
        return -0.5 * (
            metrics["mae"] / max(baseline["mae"], 1e-9)
            + metrics["relative_mape"] / max(baseline["relative_mape"], 1e-9)
        )
    if candidate_id == "CAND-P-140":
        return float(
            0.5 * (1.0 - metrics["mae"] / max(baseline["mae"], 1e-9))
            + 0.35 * metrics["spearman_rho"]
            - 0.15 * metrics["interval_90_coverage_error"]
        )
    raise KeyError(candidate_id)


def baseline_prediction(
    candidate_id: str,
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    baseline_group: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    if candidate_id == "CAND-P-069":
        assert baseline_group is not None
        fallback = float(np.mean(y[train]))
        means = {
            group: float(np.mean(y[train][baseline_group[train] == group]))
            for group in np.unique(baseline_group[train])
        }
        prediction = np.asarray([means.get(group, fallback) for group in baseline_group])
        kind = "train_crystal_dimensionality_chemistry_family_mean"
    elif candidate_id == "CAND-P-071":
        feature = np.nan_to_num(x[:, 133:134], nan=float(np.nanmedian(x[train, 133])))
        model = LinearRegression().fit(feature[train], y[train])
        prediction = np.maximum(model.predict(feature), 0.0)
        kind = "train_linear_correction_from_optb88vdw_PBE_gap"
    elif candidate_id == "CAND-P-072":
        feature = 1.0 / np.maximum(x[:, 122:123], 0.05)
        model = LinearRegression().fit(feature[train], np.log1p(np.maximum(y[train], 0.0)))
        prediction = np.maximum(np.expm1(model.predict(feature)), 0.0)
        kind = "train_fitted_inverse_optb88vdw_bandgap_linear"
    elif candidate_id in {"CAND-P-074", "CAND-P-075", "CAND-P-078"}:
        assert baseline_group is not None
        fallback = float(np.median(y[train]))
        medians = {
            group: float(np.median(y[train][baseline_group[train] == group]))
            for group in np.unique(baseline_group[train])
        }
        prediction = np.asarray([medians.get(group, fallback) for group in baseline_group])
        kind = "train_crystal_dimensionality_family_median"
    elif candidate_id == "CAND-P-086":
        formation = x[:, 128]
        candidates = np.quantile(formation[train], np.linspace(0.02, 0.98, 97))
        best = None
        for orientation in (-1.0, 1.0):
            for threshold in candidates:
                pred = ((formation[train] - threshold) * orientation >= 0).astype(int)
                score = balanced_accuracy_score(y[train], pred)
                if best is None or score > best[0]:
                    best = (score, orientation, float(threshold))
        assert best is not None
        scale = max(float(np.nanstd(formation[train])), 1e-3)
        logits = (formation - best[2]) * best[1] / scale * 4.0
        prediction = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        kind = "train_fitted_formation_energy_threshold"
    elif candidate_id == "CAND-P-087":
        baseline_x = x.copy()
        baseline_median = np.nanmedian(baseline_x[train], axis=0)
        baseline_median[~np.isfinite(baseline_median)] = 0.0
        baseline_x = np.where(np.isfinite(baseline_x), baseline_x, baseline_median)
        baseline_mean = baseline_x[train].mean(axis=0)
        baseline_std = baseline_x[train].std(axis=0)
        baseline_std[baseline_std < 1e-6] = 1.0
        baseline_x = np.clip(
            (baseline_x - baseline_mean) / baseline_std, -12.0, 12.0
        )
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            solver="liblinear",
            random_state=20260801,
        ).fit(baseline_x[train], y[train])
        prediction = model.predict_proba(baseline_x)[:, 1]
        kind = "regularized_logistic_regression_train_only"
    elif candidate_id == "CAND-P-076":
        feature = np.nan_to_num(x[:, 118:119], nan=float(np.nanmedian(x[train, 118])))
        model = LinearRegression().fit(feature[train], y[train])
        prediction = np.maximum(model.predict(feature), 0.0)
        kind = "density_linear_regression_train_only"
    elif candidate_id == "CAND-P-077":
        prediction = np.maximum(0.6 * x[:, 139], 0.0)
        kind = "fixed_Poisson_0p25_relation_from_bulk_modulus"
    elif candidate_id == "CAND-P-140":
        model = HistGradientBoostingRegressor(
            loss="absolute_error",
            max_iter=240,
            max_leaf_nodes=15,
            learning_rate=0.05,
            l2_regularization=1.0,
            random_state=20260801,
        ).fit(x[train], np.log1p(np.maximum(y[train], 0.0)))
        prediction = np.maximum(np.expm1(model.predict(x)), 0.0)
        kind = "composition_descriptor_hist_gradient_boosting_train_only"
    else:
        raise KeyError(candidate_id)

    if candidate_id in {"CAND-P-086", "CAND-P-087"}:
        validation_metrics = probability_metrics(y[validation], prediction[validation])
        test_metrics = probability_metrics(y[test], prediction[test])
        interval_half = 0.0
    else:
        train_residual = np.abs(y[train] - prediction[train])
        interval_half = float(np.quantile(train_residual, 0.90))
        validation_metrics = regression_metrics(candidate_id, y[validation], prediction[validation], interval_half)
        test_metrics = regression_metrics(candidate_id, y[test], prediction[test], interval_half)
    report = {
        "kind": kind,
        "fit_split": "train_only",
        "validation": validation_metrics,
        "test": test_metrics,
        "interval_half_width_train_q90": interval_half,
    }
    return prediction, test, interval_half, report


def quantize_state(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 2:
            scale = np.maximum(np.max(np.abs(array), axis=1, keepdims=True), 1e-12) / 127.0
        else:
            scale = np.asarray(max(float(np.max(np.abs(array))), 1e-12) / 127.0, dtype=np.float32)
        quantized[name] = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        scales[name] = np.asarray(scale, dtype=np.float32)
    return quantized, scales


def dequantized_state(torch: Any, quantized: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        name: torch.from_numpy(array.astype(np.float32) * scales[name])
        for name, array in quantized.items()
    }


def records_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "heartbeat.json"}:
            records.append(
                {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    return {
        "schema": "cimc.forge200.artifact-manifest.v2",
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    candidate_id = args.candidate_id
    if candidate_id not in ARCHITECTURES:
        raise RuntimeError("UNSUPPORTED_CORRECTIVE_CANDIDATE")
    dataset_path = root / "data" / "staged_contract_v2" / f"{candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or metadata["authority"] != 0:
        raise RuntimeError("DATA_GATE")
    if sha256_file(dataset_path) != metadata["sha256"]:
        raise RuntimeError("DATA_HASH")
    if metadata["cross_split_group_overlap"] != 0:
        raise RuntimeError("SPLIT_LEAKAGE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = (args.artifact_root / candidate_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"

    raw = np.load(dataset_path, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y = raw["y"]
    split = raw["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    median = np.nanmedian(x_raw[train], axis=0)
    median[~np.isfinite(median)] = 0.0
    x = np.where(np.isfinite(x_raw), x_raw, median)
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x - mean) / std, -12.0, 12.0).astype(np.float32)
    baseline_group = raw["baseline_group"].astype(str) if "baseline_group" in raw else None
    _, _, baseline_interval, baseline = baseline_prediction(
        candidate_id, x_raw, y, split, baseline_group
    )
    baseline_test = baseline["test"]
    baseline_validation = baseline["validation"]
    baseline_test_objective = objective(candidate_id, baseline_test, baseline_test)
    baseline_validation_objective = objective(candidate_id, baseline_validation, baseline_validation)

    classification = metadata["task_kind"] == "classification"
    output_count = 2 if classification else 1
    hidden1, hidden2 = ARCHITECTURES[candidate_id]

    class ContractMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if candidate_id == "CAND-P-087":
                self.skip = nn.Linear(x.shape[1], output_count)
                self.hidden = nn.Sequential(
                    nn.Linear(x.shape[1], 32), nn.GELU(), nn.Linear(32, output_count)
                )
            else:
                self.net = nn.Sequential(
                    nn.Linear(x.shape[1], hidden1),
                    nn.GELU(),
                    nn.Linear(hidden1, hidden2),
                    nn.GELU(),
                    nn.Linear(hidden2, output_count),
                )

        def forward(self, value: Any) -> Any:
            if candidate_id == "CAND-P-087":
                return self.skip(value) + self.hidden(value)
            return self.net(value)

    parameter_count = sum(p.numel() for p in ContractMLP().parameters())
    if parameter_count > PARAMETER_CAPS[candidate_id]:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")

    signed_target_scale = max(float(np.std(y[train])), 1e-3)
    if classification:
        target_all = torch.from_numpy(y.astype(np.int64))
    elif candidate_id == "CAND-P-069":
        target_all = torch.from_numpy(
            np.arcsinh(y.astype(np.float32) / signed_target_scale).reshape(-1, 1)
        )
    else:
        target_all = torch.from_numpy(
            np.log1p(np.maximum(y.astype(np.float32), 0.0)).reshape(-1, 1)
        )

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            value = model(torch.from_numpy(x[selected]).to(device)).cpu().numpy()
        if classification:
            value = np.exp(value - value.max(axis=1, keepdims=True))
            value /= value.sum(axis=1, keepdims=True)
            return value[:, 1]
        if candidate_id == "CAND-P-069":
            return np.sinh(value.reshape(-1)) * signed_target_scale
        return np.maximum(np.expm1(value.reshape(-1)), 0.0)

    reports = []
    states: dict[int, dict[str, Any]] = {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ContractMLP().to(device)
        if candidate_id == "CAND-P-087":
            logistic = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                solver="liblinear",
                random_state=20260801,
            ).fit(x[train], y[train])
            with torch.no_grad():
                model.skip.weight.zero_()
                model.skip.bias.zero_()
                model.skip.weight[1].copy_(
                    torch.from_numpy(logistic.coef_[0].astype(np.float32)).to(device)
                )
                model.skip.bias[1] = float(logistic.intercept_[0])
                model.hidden[-1].weight.zero_()
                model.hidden[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        if classification:
            counts = np.bincount(y[train], minlength=2).astype(np.float32)
            weights = np.sqrt(counts.sum() / np.maximum(counts * 2.0, 1.0))
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
        else:
            criterion = nn.SmoothL1Loss(beta=0.35)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train]), target_all[train]),
            batch_size=min(args.batch_size, len(train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        checkpoint_dir = output / f"train_seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_dir / "best.pt"
        if candidate_id == "CAND-P-087":
            initial_metrics = probability_metrics(y[validation], infer(model, validation))
            best_score = objective(candidate_id, initial_metrics, baseline_validation)
            torch.save(model.state_dict(), best_path)
        else:
            best_score = -math.inf
        patience = 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            val_prediction = infer(model, validation)
            if classification:
                val_metrics = probability_metrics(y[validation], val_prediction)
            else:
                train_prediction = infer(model, train)
                interval = float(np.quantile(np.abs(y[train] - train_prediction), 0.90))
                val_metrics = regression_metrics(candidate_id, y[validation], val_prediction, interval)
            val_score = objective(candidate_id, val_metrics, baseline_validation)
            if val_score > best_score + 1e-5:
                best_score, patience = val_score, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_CONTRACT_V2", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        train_prediction = infer(model, train)
        test_prediction = infer(model, test)
        interval = 0.0 if classification else float(
            np.quantile(np.abs(y[train] - train_prediction), 0.90)
        )
        test_metrics = (
            probability_metrics(y[test], test_prediction)
            if classification
            else regression_metrics(candidate_id, y[test], test_prediction, interval)
        )
        test_score = objective(candidate_id, test_metrics, baseline_test)
        reports.append(
            {
                "seed": seed,
                "epochs": epoch + 1,
                "validation_objective": best_score,
                "test_objective": test_score,
                "baseline_test_objective": baseline_test_objective,
                "beats_baseline": test_score > baseline_test_objective + 1e-4,
                "test": test_metrics,
                "interval_half_width_train_q90": interval,
            }
        )
        states[seed] = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }

    best_report = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best_report["seed"])
    best_state = states[best_seed]
    model = ContractMLP().to(device)
    model.load_state_dict(best_state)
    fp_prediction = infer(model, test[:64])
    quantized, scales = quantize_state(best_state)
    payload_buffer = io.BytesIO()
    np.savez_compressed(
        payload_buffer,
        **quantized,
        **{f"scale::{name}": value for name, value in scales.items()},
    )
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAPS[candidate_id]:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_prediction = infer(model, test[:64])
    parity = float(np.max(np.abs(fp_prediction - quant_prediction)))
    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(
        golden_path,
        x=x[test[:64]],
        y=y[test[:64]],
        fp32=fp_prediction,
        quantized=quant_prediction,
    )
    model.load_state_dict(best_state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(x[test[:1]]).to(device),
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {
        "task_kind": metadata["task_kind"],
        "shape": [None, output_count],
        "authority": 0,
    }
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": candidate_id,
                "dataset_sha256": metadata["sha256"],
                "task_contract_sha256": metadata["task_contract_sha256"],
                "onnx_sha256": sha256_file(onnx_path),
                "golden_sha256": sha256_file(golden_path),
                "best_seed": best_seed,
            }
        )
    ).hexdigest()
    package = build_package(
        output,
        candidate_id,
        payload,
        sha256_file(golden_path),
        release_root,
        hashlib.sha256(canonical_bytes(output_schema)).hexdigest(),
        engine_id=1,
    )
    all_seeds_pass = all(item["beats_baseline"] for item in reports)
    status = (
        "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING"
        if all_seeds_pass
        else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    )
    evaluation = {
        "schema": "cimc.forge200.contract-exact-evaluation.v2",
        "status": "PASS" if all_seeds_pass else "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "baseline_contract": metadata["baseline"],
        "primary_metric_contract": metadata["primary_metric"],
        "baseline": baseline,
        "baseline_test_objective": baseline_test_objective,
        "seed_reports": reports,
        "three_seed_count": len(reports),
        "three_seed_baseline_pass": sum(item["beats_baseline"] for item in reports),
        "worst_seed_objective": min(item["test_objective"] for item in reports),
        "authority": 0,
        "board_accepted": False,
    }
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", metadata)
    write_json(
        output / "preprocessing_train_only.json",
        {"median": median.tolist(), "mean": mean.tolist(), "std": std.tolist()},
    )
    write_json(
        output / "quantization_parity.json",
        {"scheme": "W8_per_output_channel", "max_abs_output_error": parity},
    )
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v2",
        "status": status,
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "three_seed_count": len(reports),
        "three_seed_baseline_pass": evaluation["three_seed_baseline_pass"],
        "parameter_count": parameter_count,
        "parameter_cap": PARAMETER_CAPS[candidate_id],
        "w8_payload_bytes": len(payload),
        "w8_payload_byte_cap": WEIGHT_BYTE_CAPS[candidate_id],
        "best_seed": best_seed,
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden_path),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3},
    }
    write_json(output / "promotion_receipt.json", receipt)
    (output / "model_card.md").write_text(
        f"# {candidate_id} corrective model card\n\n"
        f"- Status: `{status}`\n"
        f"- Three-seed baseline passes: `{evaluation['three_seed_baseline_pass']}/3`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        "- Authority: `0`; unified GD32 board evidence is pending.\n",
        encoding="utf-8",
    )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "passes": evaluation["three_seed_baseline_pass"], "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--min-epochs", type=int, default=45)
    parser.add_argument("--early-stop-patience", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / args.candidate_id
        write_json(
            output / "failure.json",
            {
                "schema": "cimc.forge200.job-failure.v2",
                "status": "FAIL_CLOSED",
                "candidate_id": args.candidate_id,
                "authority": 0,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
