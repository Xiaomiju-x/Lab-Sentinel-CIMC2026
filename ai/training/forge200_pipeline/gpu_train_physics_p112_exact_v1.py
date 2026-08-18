#!/usr/bin/env python3
"""Train, quantize, export, and package the SIM_ONLY P112 field model."""

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
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state, manifest, quantize_state


PARAMETER_CAP = 82_000
WEIGHT_CAP = 100 * 1024
GRID = 8


def metrics(y: np.ndarray, prediction: np.ndarray, power: np.ndarray) -> dict[str, float]:
    error = prediction - y
    rmse = float(np.sqrt(np.mean(error**2)))
    peak_mae = float(np.mean(np.abs(np.max(prediction, axis=1) - np.max(y, axis=1))))
    true_index = np.argmax(y, axis=1)
    pred_index = np.argmax(prediction, axis=1)
    true_xy = np.column_stack((true_index // GRID, true_index % GRID))
    pred_xy = np.column_stack((pred_index // GRID, pred_index % GRID))
    fde = float(np.mean(np.linalg.norm(pred_xy - true_xy, axis=1)))
    result = {
        "field_RMSE_C": rmse,
        "hotspot_FDE_px": fde,
        "peak_MAE_C": peak_mae,
        "field_score": 1.0 / (1.0 + rmse / 10.0),
        "hotspot_score": 1.0 / (1.0 + fde),
        "peak_score": 1.0 / (1.0 + peak_mae / 10.0),
    }
    result["primary_composite"] = float(np.mean([result["field_score"], result["hotspot_score"], result["peak_score"]]))
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    dataset = root / "data" / "staged_physics_p112_exact_v1" / "CAND-P-112.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or metadata["truth_class"] != "PHYSICS_SIM" or metadata["cross_split_stack_overlap"] != 0 or sha256_file(dataset) != metadata["sha256"]:
        raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset, allow_pickle=False)
    power = raw["power_map"].astype(np.float32)
    physical = raw["stack_features"].astype(np.float32)
    y = raw["y_delta_C"].astype(np.float32)
    baseline_prediction = raw["baseline_delta_C"].astype(np.float32)
    split = raw["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    x_raw = np.column_stack((power, physical)).astype(np.float32)
    mean, std = x_raw[train].mean(axis=0), x_raw[train].std(axis=0)
    std[std < 1e-7] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    y_scale = np.maximum(np.quantile(y[train], 0.95, axis=0).astype(np.float32), 0.1)
    y_scaled = y / y_scale
    output = args.artifact_root.resolve() / "CAND-P-112"
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    started = time.perf_counter()

    class FieldMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 160), nn.GELU(), nn.Linear(160, 96), nn.GELU(), nn.Linear(96, GRID * GRID), nn.Softplus())

        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(parameter.numel() for parameter in FieldMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    train_dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y_scaled[train]))
    reports, states = [], {}
    baseline_validation = metrics(y[validation], baseline_prediction[validation], power[validation])
    baseline_test = metrics(y[test], baseline_prediction[test], power[test])
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = FieldMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
        loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x.to(device))
                loss = nn.functional.smooth_l1_loss(prediction, batch_y.to(device), beta=0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_prediction = model(torch.from_numpy(x[validation]).to(device)).cpu().numpy() * y_scale
            current = metrics(y[validation], val_prediction, power[validation])
            if current["primary_composite"] > best_score + 1e-5:
                best_score, patience = current["primary_composite"], 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-P-112", "TRAIN_PHYSICS_P112_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            test_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy() * y_scale
        report = metrics(y[test], test_prediction, power[test])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": report, "beats_baseline": report["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    composites = np.asarray([record["test"]["primary_composite"] for record in reports])
    aggregate = {"mean": float(composites.mean()), "variance": float(composites.var()), "std": float(composites.std()), "worst": float(composites.min())}
    aggregate_pass = aggregate["mean"] > baseline_test["primary_composite"] + 1e-4
    best_seed = int(max(reports, key=lambda record: record["validation_primary_composite"])["seed"])
    model = FieldMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    with torch.no_grad():
        fp_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy() * y_scale
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    with torch.no_grad():
        quant_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy() * y_scale
    quant_metrics = metrics(y[test], quant_prediction, power[test])
    selected_fp_metrics = metrics(y[test], fp_prediction, power[test])
    quant_delta = selected_fp_metrics["primary_composite"] - quant_metrics["primary_composite"]
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    passed = aggregate_pass and quant_pass
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=x[test[:64]], y_delta_C=y[test[:64]], fp32_delta_C=fp_prediction[:64], quantized_delta_C=quant_prediction[:64], ambient_C=raw["ambient_C"][test[:64]])
    model.load_state_dict(states[best_seed])
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(x[test[:1]]).to(device), onnx_path, input_names=["power_map_and_stack_features"], output_names=["temperature_rise_field_C_scaled"], dynamic_axes={"power_map_and_stack_features": {0: "batch"}, "temperature_rise_field_C_scaled": {0: "batch"}}, opset_version=17, dynamo=False)
    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"task_kind": "field_regression", "shape": [None, GRID, GRID], "semantics": "package_temperature_field_C_and_hotspot_xy", "model_output": "temperature_rise_scaled", "postprocess": "reshape_8x8_multiply_y_scale_add_ambient_then_argmax", "authority": 0, "public_claim_scope": "SIM_ONLY"}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": "CAND-P-112", "dataset_sha256": metadata["sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "mean": aggregate["mean"]})).hexdigest()
    package = build_package(output, "CAND-P-112", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING_SIM_ONLY" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE_SIM_ONLY"
    audit = {"schema": "cimc.forge200.physics-p112-contract-exact-audit.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status, "candidate_id": "CAND-P-112", "truth_class": "PHYSICS_SIM", "public_claim_scope": "SIM_ONLY", "baseline": {"kind": metadata["baseline_execution"], "validation": baseline_validation, "test": baseline_test}, "seed_reports": reports, "aggregate": aggregate, "g3_aggregate_mean_gate": aggregate_pass, "quantized_best_seed": {"seed": best_seed, "test": quant_metrics, "metric_delta": quant_delta, "gate": quant_pass}, "parameter_count": parameter_count, "w8_payload_bytes": len(payload), "authority": 0, "board_accepted": False, "countable_model": False}
    write_json(output / "contract_exact_audit.json", audit)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "y_scale_train_q95": y_scale.tolist()})
    write_json(output / "output_schema.json", output_schema)
    write_json(output / "quantization_parity.json", {"primary_composite_delta": quant_delta, "gate": quant_pass})
    (output / "model_card.md").write_text(f"# CAND-P-112 SIM_ONLY model\n\n- Status: `{status}`.\n- Truth: `PHYSICS_SIM`; no experimental or board-performance claim.\n- Three-seed mean composite: `{aggregate['mean']:.6f}`; lumped-RC baseline: `{baseline_test['primary_composite']:.6f}`.\n- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n- Authority: `0`; unified GD32 board acceptance remains pending.\n", encoding="utf-8")
    promotion = {"schema": "cimc.forge200.promotion-receipt.v3", "status": status, "candidate_id": "CAND-P-112", "authority": 0, "board_accepted": False, "countable_model": False, "host_contract_pass": passed, "truth_class": "PHYSICS_SIM", "three_seed_count": 3, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3}}
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    heartbeat(heartbeat_path, "CAND-P-112", "COMPLETE")
    write_json(root / "evidence" / "physics_p112_exact_closure.v1.json", {"schema": "cimc.forge200.physics-p112-exact-closure.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "PARTIAL", "record": audit, "authority_nonzero": 0, "board_actions": 0})
    print(json.dumps({"candidate_id": "CAND-P-112", "status": status, "mean_composite": aggregate["mean"], "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True))
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=14)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    receipt = run(args)
    return 0 if receipt["host_contract_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
