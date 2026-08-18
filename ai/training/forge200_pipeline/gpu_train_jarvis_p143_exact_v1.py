#!/usr/bin/env python3
"""Train, quantize, export, and package the source-bound P143 IR model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json


PARAMETER_CAP = 128_000
WEIGHT_BYTE_CAP = 128 * 1024


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(y_true: np.ndarray, score: np.ndarray) -> float:
    left, right = average_ranks(y_true), average_ranks(score)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator > 0.0 else 0.0


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    scores = []
    for label in (False, True):
        tp = int(np.sum((y_true == label) & (y_pred == label)))
        fp = int(np.sum((y_true != label) & (y_pred == label)))
        fn = int(np.sum((y_true == label) & (y_pred != label)))
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def metrics(y_log: np.ndarray, prediction_log: np.ndarray, active_threshold_log: float) -> dict[str, float]:
    mae = float(np.mean(np.abs(prediction_log - y_log)))
    scale = max(float(np.std(y_log)), 1e-6)
    truth_active = y_log > active_threshold_log
    predicted_active = prediction_log > active_threshold_log
    result = {
        "log_intensity_MAE": mae,
        "active_macro_F1": macro_f1(truth_active, predicted_active),
        "Spearman_rho": spearman(y_log, prediction_log),
        "MAE_skill_vs_test_std": 1.0 - mae / scale,
        "active_threshold_raw": float(np.expm1(active_threshold_log)),
    }
    result["primary_composite"] = float(np.mean([result["MAE_skill_vs_test_std"], result["active_macro_F1"], result["Spearman_rho"]]))
    return result


def frequency_bin_baseline(
    frequency: np.ndarray,
    y_log: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.unique(np.quantile(frequency[train], np.linspace(0.0, 1.0, 13)))
    if len(edges) < 5:
        raise RuntimeError("BASELINE_FREQUENCY_BIN_GATE")
    bins = np.clip(np.searchsorted(edges[1:-1], frequency, side="right"), 0, len(edges) - 2)
    global_mean = float(np.mean(y_log[train]))
    means = np.asarray(
        [float(np.mean(y_log[train][bins[train] == index])) if np.any(bins[train] == index) else global_mean for index in range(len(edges) - 1)],
        dtype=np.float32,
    )
    return means[bins], edges.astype(np.float32), means


def quantize_state(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quantized, scales = {}, {}
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
    return {name: torch.from_numpy(value.astype(np.float32) * scales[name]) for name, value in quantized.items()}


def records_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "heartbeat.json"}:
            records.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    candidate_id = "CAND-P-143"
    dataset = root / "data" / "staged_jarvis_p143_exact_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("authority") != 0 or metadata.get("cross_split_group_overlap") != 0:
        raise RuntimeError("DATA_GATE")
    if sha256_file(dataset) != metadata["sha256"]:
        raise RuntimeError("DATA_HASH_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = (args.artifact_root / candidate_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"

    raw = np.load(dataset, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y_raw = raw["y"].astype(np.float32)
    y_log = np.log1p(y_raw).astype(np.float32)
    split = raw["split"].astype(np.int8)
    frequency = raw["frequency_proxy"].astype(np.float32)
    active_threshold = float(raw["active_threshold"])
    active_threshold_log = float(np.log1p(active_threshold))
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean, std = x_raw[train].mean(axis=0), x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    baseline_prediction, baseline_edges, baseline_means = frequency_bin_baseline(frequency, y_log, train)
    baseline_validation = metrics(y_log[validation], baseline_prediction[validation], active_threshold_log)
    baseline_test = metrics(y_log[test], baseline_prediction[test], active_threshold_log)

    class IRIntensityMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 160), nn.GELU(), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))

        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(parameter.numel() for parameter in IRIntensityMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    inactive_weight = float(np.sum(y_log[train] > active_threshold_log) / max(np.sum(y_log[train] <= active_threshold_log), 1))
    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = IRIntensityMLP().to(device)
        with torch.no_grad():
            model.net[-1].bias.fill_(float(np.mean(y_log[train])))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y_log[train].reshape(-1, 1))),
            batch_size=min(args.batch_size, len(train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x)
                regression = nn.functional.smooth_l1_loss(prediction, batch_y, beta=0.5)
                active = (batch_y > active_threshold_log).float()
                logits = (prediction - active_threshold_log) * 4.0
                class_loss = nn.functional.binary_cross_entropy_with_logits(logits, active, reduction="none")
                weights = torch.where(active > 0.5, torch.ones_like(active), torch.full_like(active, inactive_weight))
                loss = regression + 0.15 * torch.mean(class_loss * weights)
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.from_numpy(x[validation]).to(device)).cpu().numpy().reshape(-1)
            current = metrics(y_log[validation], validation_prediction, active_threshold_log)
            score = float(current["primary_composite"])
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_JARVIS_IR_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1)
        test_metrics = metrics(y_log[test], prediction, active_threshold_log)
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": test_metrics, "beats_baseline": test_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    composites = np.asarray([item["test"]["primary_composite"] for item in reports], dtype=np.float64)
    mean_composite = float(composites.mean())
    aggregate_pass = mean_composite > float(baseline_test["primary_composite"]) + 1e-4
    best_report = max(reports, key=lambda item: item["validation_primary_composite"])
    best_seed = int(best_report["seed"])
    model = IRIntensityMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    with torch.no_grad():
        fp_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1)
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    model.eval()
    with torch.no_grad():
        quant_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1)
    quant_metrics = metrics(y_log[test], quant_prediction, active_threshold_log)
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    status_pass = aggregate_pass and quant_pass

    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(
        golden_path,
        x=x[test[:64]],
        y_raw=y_raw[test[:64]],
        y_log=y_log[test[:64]],
        fp32_log_prediction=fp_prediction[:64],
        quantized_log_prediction=quant_prediction[:64],
        quantized_raw_intensity=np.maximum(np.expm1(quant_prediction[:64]), 0.0),
    )
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(x[test[:1]]).to(device),
        onnx_path,
        input_names=["ir_material_features"],
        output_names=["log1p_max_ir_mode_intensity"],
        dynamic_axes={"ir_material_features": {0: "batch"}, "log1p_max_ir_mode_intensity": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {
        "shape": [None, 1],
        "semantics": "log1p_max_ir_mode_intensity",
        "postprocess": "max(expm1(value),0); active iff raw_intensity > 0.1",
        "authority": 0,
    }
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden_path), "three_seed_mean_composite": mean_composite})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden_path), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest(), engine_id=1)
    evaluation = {
        "schema": "cimc.forge200.jarvis-ir-contract-exact-evaluation.v1",
        "status": "PASS" if status_pass else "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "baseline_contract": metadata["baseline"],
        "primary_metric_contract": metadata["primary_metric"],
        "baseline": {"kind": metadata["baseline_execution"], "validation": baseline_validation, "test": baseline_test},
        "seed_reports": reports,
        "g3_aggregate_mean_gate": aggregate_pass,
        "g4": {"mean_composite": mean_composite, "variance_composite": float(composites.var()), "worst_composite": float(composites.min())},
        "quantized_best_seed": {"seed": best_seed, "test": quant_metrics, "metric_delta": quant_delta, "gate": quant_pass},
        "authority": 0,
        "board_accepted": False,
    }
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "baseline_report.json", evaluation["baseline"])
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "baseline_frequency_edges": baseline_edges.tolist(), "baseline_log_intensity_means": baseline_means.tolist(), "active_threshold_raw": active_threshold})
    write_json(output / "output_schema.json", output_schema)
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "primary_composite_delta": quant_delta, "gate": quant_pass})
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status_pass else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v3",
        "status": status,
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "three_seed_count": 3,
        "g3_aggregate_mean_gate": aggregate_pass,
        "parameter_count": parameter_count,
        "parameter_cap": PARAMETER_CAP,
        "w8_payload_bytes": len(payload),
        "w8_payload_byte_cap": WEIGHT_BYTE_CAP,
        "best_seed_by_validation": best_seed,
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden_path),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3},
    }
    write_json(output / "promotion_receipt.json", receipt)
    (output / "model_card.md").write_text(
        f"# {candidate_id} exact JARVIS IR-intensity model\n\n"
        f"- Status: `{status}`.\n"
        f"- Scope: published per-material maximum IR-mode intensity plus a fixed `>0.1` active class; it does not reconstruct a full IR spectrum.\n"
        f"- Three-seed mean composite: `{mean_composite:.6f}`; preregistered frequency-bin baseline: `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        f"- Truth: `{metadata['truth_class']}`; no experimental-performance claim.\n"
        f"- Authority: `0`; unified GD32 board evidence remains pending.\n",
        encoding="utf-8",
    )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "mean_composite": mean_composite, "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / "CAND-P-143"
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": "CAND-P-143", "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
