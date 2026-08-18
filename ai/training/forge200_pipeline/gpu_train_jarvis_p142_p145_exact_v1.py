#!/usr/bin/env python3
"""Train and package exact JARVIS P142/P145 property models."""

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
SUPPORTED = {"CAND-P-142", "CAND-P-145"}


def regression_ridge(x: np.ndarray, y: np.ndarray, alpha: float = 1e-3) -> tuple[np.ndarray, float]:
    design = np.column_stack((x.astype(np.float64), np.ones(len(x), dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ y.astype(np.float64))
    return solution[:-1].astype(np.float32), float(solution[-1])


def binary_auroc(y_true: np.ndarray, score: np.ndarray) -> float:
    positive = y_true.astype(bool)
    p, n = int(np.sum(positive)), int(np.sum(~positive))
    if not p or not n:
        return 0.5
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1, dtype=np.float64)
    return float((np.sum(ranks[positive]) - p * (p + 1) / 2) / (p * n))


def grouped_slme_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    ndcg_values, top5_values = [], []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) < 2:
            continue
        truth = y[selected]
        score = prediction[selected]
        true_order = np.argsort(-truth, kind="mergesort")
        pred_order = np.argsort(-score, kind="mergesort")
        cutoff = min(10, len(selected))
        relevance = np.expm1(np.clip(truth, 0.0, 34.0) / 10.0)
        discount = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.sum(relevance[pred_order[:cutoff]] * discount))
        idcg = float(np.sum(relevance[true_order[:cutoff]] * discount))
        ndcg_values.append(dcg / max(idcg, 1e-12))
        top = min(5, len(selected))
        top5_values.append(len(set(true_order[:top]) & set(pred_order[:top])) / top)
    mae = float(np.mean(np.abs(prediction - y)))
    scale = max(float(np.std(y)), 1e-6)
    mae_skill = 1.0 - mae / scale
    result = {
        "SLME_MAE_percent": mae,
        "NDCG_at_10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "top5_recall": float(np.mean(top5_values)) if top5_values else 0.0,
        "ranked_chemical_systems": len(ndcg_values),
        "MAE_skill_vs_test_std": mae_skill,
    }
    result["primary_composite"] = float(np.mean([result["MAE_skill_vs_test_std"], result["NDCG_at_10"], result["top5_recall"]]))
    return result


def spillage_metrics(y: np.ndarray, prediction: np.ndarray, _groups: np.ndarray) -> dict[str, float]:
    truth_class = y >= 0.5
    order = np.argsort(-prediction, kind="mergesort")[: min(20, len(y))]
    mae = float(np.mean(np.abs(prediction - y)))
    scale = max(float(np.std(y)), 1e-6)
    result = {
        "spillage_MAE": mae,
        "candidate_AUROC": binary_auroc(truth_class, prediction),
        "top20_precision": float(np.mean(truth_class[order])),
        "MAE_skill_vs_test_std": 1.0 - mae / scale,
        "candidate_threshold": 0.5,
    }
    result["primary_composite"] = float(np.mean([result["MAE_skill_vs_test_std"], result["candidate_AUROC"], result["top20_precision"]]))
    return result


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
    candidate_id = args.candidate_id
    if candidate_id not in SUPPORTED:
        raise RuntimeError(f"UNSUPPORTED_CANDIDATE:{candidate_id}")
    dataset = root / "data" / "staged_jarvis_p142_p145_exact_v1" / f"{candidate_id}.npz"
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
    y = raw["y"].astype(np.float32)
    groups = raw["groups"].astype(str)
    split = raw["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean, std = x_raw[train].mean(axis=0), x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    indices = metadata["feature_indices"]
    if candidate_id == "CAND-P-142":
        baseline_indices = [indices[name] for name in ("optb88vdw_bandgap", "bandgap_squared", "bandgap_cubed", "bandgap_fourth")]
        metrics = grouped_slme_metrics
        residual_scale_value = max(float(np.std(y[train])) * 0.6, 1.0)
    else:
        baseline_indices = [indices[name] for name in ("high_Z_fraction", "inverse_bandgap_plus_0p15", "space_group_number_scaled", "SOC_Z4_proxy")]
        metrics = spillage_metrics
        residual_scale_value = max(float(np.std(y[train])) * 0.6, 0.1)
    raw_weight, raw_bias = regression_ridge(x_raw[train][:, baseline_indices], y[train], alpha=2e-2)
    baseline_weight = np.zeros(x.shape[1], dtype=np.float32)
    for local_index, feature_index in enumerate(baseline_indices):
        baseline_weight[feature_index] = raw_weight[local_index] * std[feature_index]
    baseline_bias = float(raw_bias + np.sum(raw_weight * mean[baseline_indices]))
    baseline_prediction = x @ baseline_weight + baseline_bias
    baseline_validation = metrics(y[validation], baseline_prediction[validation], groups[validation])
    baseline_test = metrics(y[test], baseline_prediction[test], groups[test])

    class PropertyMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 160), nn.GELU(), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))
            self.register_buffer("baseline_weight", torch.from_numpy(baseline_weight.reshape(1, -1)))
            self.register_buffer("baseline_bias", torch.tensor([baseline_bias], dtype=torch.float32))
            self.register_buffer("residual_scale", torch.tensor([residual_scale_value], dtype=torch.float32))

        def forward(self, value: Any) -> Any:
            baseline = value @ self.baseline_weight.t() + self.baseline_bias
            return baseline + self.net(value) * self.residual_scale

    parameter_count = sum(parameter.numel() for parameter in PropertyMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = PropertyMLP().to(device)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        criterion = nn.SmoothL1Loss(beta=max(float(np.std(y[train])) * 0.2, 0.05))
        loader = DataLoader(TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train].reshape(-1, 1))), batch_size=min(args.batch_size, len(train)), shuffle=True, generator=torch.Generator().manual_seed(seed))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
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
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.from_numpy(x[validation]).to(device)).cpu().numpy().reshape(-1)
            current = metrics(y[validation], np.maximum(validation_prediction, 0.0), groups[validation])
            score = float(current["primary_composite"])
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_JARVIS_PROPERTY_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1)
        test_metrics = metrics(y[test], np.maximum(prediction, 0.0), groups[test])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": test_metrics, "beats_baseline": test_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    composites = np.asarray([item["test"]["primary_composite"] for item in reports], dtype=np.float64)
    mean_composite = float(composites.mean())
    aggregate_pass = mean_composite > float(baseline_test["primary_composite"]) + 1e-4
    best_report = max(reports, key=lambda item: item["validation_primary_composite"])
    best_seed = int(best_report["seed"])
    model = PropertyMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    with torch.no_grad():
        fp_prediction = np.maximum(model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1), 0.0)
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    model.eval()
    with torch.no_grad():
        quant_prediction = np.maximum(model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1), 0.0)
    quant_metrics = metrics(y[test], quant_prediction, groups[test])
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    status_pass = aggregate_pass and quant_pass

    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(golden_path, x=x[test[:64]], y=y[test[:64]], fp32_prediction=fp_prediction[:64], quantized_prediction=quant_prediction[:64])
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    input_name = "slme_material_features" if candidate_id == "CAND-P-142" else "spillage_material_features"
    output_name = "slme_percent" if candidate_id == "CAND-P-142" else "spin_orbit_spillage_score"
    torch.onnx.export(model, torch.from_numpy(x[test[:1]]).to(device), onnx_path, input_names=[input_name], output_names=[output_name], dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"shape": [None, 1], "semantics": output_name, "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden_path), "three_seed_mean_composite": mean_composite})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden_path), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest(), engine_id=1)
    evaluation = {
        "schema": "cimc.forge200.jarvis-property-contract-exact-evaluation.v1",
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
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "baseline_feature_indices": baseline_indices, "baseline_raw_weight": raw_weight.tolist(), "baseline_raw_bias": raw_bias})
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
    (output / "model_card.md").write_text(f"# {candidate_id} exact JARVIS property model\n\n- Status: `{status}`.\n- Three-seed mean composite: `{mean_composite:.6f}`; preregistered baseline: `{baseline_test['primary_composite']:.6f}`.\n- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n- Truth: `{metadata['truth_class']}`; no experimental-performance claim.\n- Authority: `0`; unified GD32 board evidence remains pending.\n", encoding="utf-8")
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "mean_composite": mean_composite, "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True, choices=sorted(SUPPORTED))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / args.candidate_id
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": args.candidate_id, "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
