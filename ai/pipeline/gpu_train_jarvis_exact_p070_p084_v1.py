#!/usr/bin/env python3
"""Train and package exact JARVIS P070/P084 residual MLPs on one local GPU."""

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


LIMITS = {
    "CAND-P-070": {"hidden": (128, 64), "parameters": 70_000, "weight_bytes": 88 * 1024},
    "CAND-P-084": {"hidden": (128, 64), "parameters": 66_000, "weight_bytes": 80 * 1024},
}


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


def p070_rank_ndcg(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    values = []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) < 2:
            continue
        relevance = np.max(y[selected]) - y[selected]
        if float(np.max(relevance)) <= 0:
            continue
        order = np.argsort(prediction[selected], kind="mergesort")
        ideal = np.argsort(-relevance, kind="mergesort")
        discount = 1.0 / np.log2(np.arange(2, len(selected) + 2))
        dcg = float(np.sum((np.power(2.0, relevance[order]) - 1.0) * discount))
        idcg = float(np.sum((np.power(2.0, relevance[ideal]) - 1.0) * discount))
        values.append(dcg / max(idcg, 1e-12))
    return (float(np.mean(values)) if values else 0.0), len(values)


def metrics(
    candidate_id: str,
    y: np.ndarray,
    prediction: np.ndarray,
    groups: np.ndarray,
    weak_threshold: float,
) -> dict[str, float | int]:
    prediction = prediction.reshape(-1)
    result: dict[str, float | int] = {
        "mae": float(np.mean(np.abs(prediction - y))),
        "rmse": float(np.sqrt(np.mean((prediction - y) ** 2))),
    }
    if candidate_id == "CAND-P-070":
        rank, count = p070_rank_ndcg(y, prediction, groups)
        result["defect_rank_ndcg"] = rank
        result["rankable_pristine_groups"] = count
    else:
        weak = (y <= weak_threshold).astype(np.int8)
        if len(np.unique(weak)) == 2:
            score = -prediction
            order = np.argsort(score, kind="mergesort")
            ranks = np.empty(len(score), dtype=np.float64)
            start = 0
            while start < len(order):
                end = start + 1
                while end < len(order) and score[order[end]] == score[order[start]]:
                    end += 1
                ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
                start = end
            positives = weak == 1
            pos_count = int(np.sum(positives))
            neg_count = len(weak) - pos_count
            result["weak_interface_auroc"] = float((np.sum(ranks[positives]) - pos_count * (pos_count + 1) / 2.0) / (pos_count * neg_count))
        else:
            result["weak_interface_auroc"] = 0.5
        result["weak_interface_threshold_train_q25_J_per_m2"] = weak_threshold
    return result


def objective(candidate_id: str, current: dict[str, Any], baseline: dict[str, Any]) -> float:
    primary = 1.0 - float(current["mae"]) / max(float(baseline["mae"]), 1e-9)
    if candidate_id == "CAND-P-070":
        secondary = float(current["defect_rank_ndcg"]) - float(baseline["defect_rank_ndcg"])
    else:
        secondary = float(current["weak_interface_auroc"]) - float(baseline["weak_interface_auroc"])
    return 0.75 * primary + 0.25 * secondary


def baseline_model(
    candidate_id: str,
    raw: Any,
    x: np.ndarray,
    y: np.ndarray,
    split: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    train = split == 0
    if candidate_id == "CAND-P-070":
        # Contract baseline: pristine formation-energy descriptors plus a
        # defect-species constant.  Chemical potential is an explicit input.
        baseline_x = np.concatenate((raw["baseline_features"], x[:, 118:236]), axis=1).astype(np.float64)
        train_x = baseline_x[train]
        train_y = y[train].astype(np.float64)
        center_x = train_x.mean(axis=0)
        center_y = float(train_y.mean())
        centered = train_x - center_x
        regularizer = np.eye(centered.shape[1], dtype=np.float64) * 10.0
        coefficients = np.linalg.solve(centered.T @ centered + regularizer, centered.T @ (train_y - center_y))
        intercept = center_y - float(np.dot(center_x, coefficients))
        prediction = (baseline_x @ coefficients + intercept).astype(np.float32)
        raw_weight = np.zeros(x.shape[1], dtype=np.float32)
        raw_weight[-4] = float(coefficients[0])  # bulk energy per atom
        raw_weight[-5] = float(coefficients[1])  # chemical potential
        raw_weight[118:236] = coefficients[2:].astype(np.float32)
        raw_bias = float(intercept)
        kind = "train_only_ridge_pristine_energy_chemical_potential_plus_defect_species_constant"
    else:
        prediction = raw["baseline_pred"].astype(np.float32)
        matches = []
        for left in range(x.shape[1]):
            for right in range(left, x.shape[1]):
                if np.allclose(x[:, left] + x[:, right], prediction, rtol=1e-6, atol=1e-6):
                    matches.append((left, right))
        if len(matches) != 1:
            raise RuntimeError(f"P084_BASELINE_COLUMN_GATE:{matches}")
        raw_weight = np.zeros(x.shape[1], dtype=np.float32)
        raw_weight[matches[0][0]] += 1.0
        raw_weight[matches[0][1]] += 1.0
        raw_bias = 0.0
        kind = "fixed_sum_of_two_published_surface_energies"
    normalized_weight = raw_weight * std
    normalized_bias = raw_bias + float(np.dot(raw_weight, mean))
    reconstructed = (x - mean) / std @ normalized_weight + normalized_bias
    if not np.allclose(reconstructed, prediction, rtol=2e-5, atol=2e-5):
        raise RuntimeError(f"BASELINE_EMBEDDING_PARITY:{float(np.max(np.abs(reconstructed - prediction)))}")
    return prediction, normalized_weight.astype(np.float32), float(normalized_bias), {"kind": kind, "fit_split": "train_only" if candidate_id == "CAND-P-070" else "fixed_no_fit"}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    candidate_id = args.candidate_id
    if candidate_id not in LIMITS:
        raise RuntimeError("UNSUPPORTED_CANDIDATE")
    dataset = root / "data" / "staged_jarvis_exact_v1" / f"{candidate_id}.npz"
    metadata_path = dataset.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("authority") != 0:
        raise RuntimeError("DATA_GATE")
    if sha256_file(dataset) != metadata["sha256"] or metadata["cross_split_group_overlap"] != 0:
        raise RuntimeError("DATA_HASH_OR_SPLIT_GATE")
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
    split = raw["split"].astype(np.int8)
    groups = raw["groups"].astype(str)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean = x_raw[train].mean(axis=0)
    std = x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    baseline_prediction, baseline_weight, baseline_bias, baseline_info = baseline_model(
        candidate_id, raw, x_raw, y, split, mean, std
    )
    weak_threshold = float(np.quantile(y[train], 0.25))
    baseline_validation = metrics(candidate_id, y[validation], baseline_prediction[validation], groups[validation], weak_threshold)
    baseline_test = metrics(candidate_id, y[test], baseline_prediction[test], groups[test], weak_threshold)

    hidden1, hidden2 = LIMITS[candidate_id]["hidden"]
    residual_scale = max(float(np.std((y - baseline_prediction)[train])), 0.05)

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], hidden1),
                nn.GELU(),
                nn.Linear(hidden1, hidden2),
                nn.GELU(),
                nn.Linear(hidden2, 1),
            )
            self.register_buffer("baseline_weight", torch.from_numpy(baseline_weight.reshape(1, -1)))
            self.register_buffer("baseline_bias", torch.tensor([baseline_bias], dtype=torch.float32))
            self.register_buffer("residual_scale", torch.tensor([residual_scale], dtype=torch.float32))

        def forward(self, value: Any) -> Any:
            baseline = value @ self.baseline_weight.t() + self.baseline_bias
            return baseline + self.net(value) * self.residual_scale

    parameter_count = sum(parameter.numel() for parameter in ResidualMLP().parameters())
    if parameter_count > LIMITS[candidate_id]["parameters"]:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(selected), 1024):
                batch = torch.from_numpy(x[selected[start : start + 1024]]).to(device)
                outputs.append(model(batch).cpu().numpy().reshape(-1))
        return np.concatenate(outputs)

    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ResidualMLP().to(device)
        # Start exactly at the preregistered baseline; learning can only be
        # promoted if the three-seed aggregate later improves it.
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        criterion = nn.SmoothL1Loss(beta=max(0.10 * float(np.std(y[train])), 0.05))
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train].reshape(-1, 1))),
            batch_size=min(args.batch_size, len(train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = 0.0, 0
        torch.save(model.state_dict(), checkpoint)
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
            validation_prediction = infer(model, validation)
            validation_metrics = metrics(candidate_id, y[validation], validation_prediction, groups[validation], weak_threshold)
            score = objective(candidate_id, validation_metrics, baseline_validation)
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_JARVIS_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        test_prediction = infer(model, test)
        test_metrics = metrics(candidate_id, y[test], test_prediction, groups[test], weak_threshold)
        test_score = objective(candidate_id, test_metrics, baseline_test)
        reports.append(
            {
                "seed": seed,
                "epochs": epoch + 1,
                "validation_objective": best_score,
                "test_objective": test_score,
                "test": test_metrics,
                "beats_baseline_primary_mae": float(test_metrics["mae"]) < float(baseline_test["mae"]),
            }
        )
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    mean_objective = float(np.mean([item["test_objective"] for item in reports]))
    mean_mae = float(np.mean([item["test"]["mae"] for item in reports]))
    aggregate_pass = mean_objective > 1e-4 and mean_mae < float(baseline_test["mae"])
    best_report = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best_report["seed"])
    model = ResidualMLP().to(device)
    model.load_state_dict(states[best_seed])
    fp_prediction = infer(model, test)
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > LIMITS[candidate_id]["weight_bytes"]:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_prediction = infer(model, test)
    quant_metrics = metrics(candidate_id, y[test], quant_prediction, groups[test], weak_threshold)
    quant_objective = objective(candidate_id, quant_metrics, baseline_test)
    quant_primary_pass = float(quant_metrics["mae"]) < float(baseline_test["mae"])
    quant_degradation = (float(quant_metrics["mae"]) - float(best_report["test"]["mae"])) / max(float(best_report["test"]["mae"]), 1e-9)
    quant_pass = quant_primary_pass and quant_degradation <= 0.03
    status_pass = aggregate_pass and quant_pass

    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(golden_path, x=x[test[:64]], y=y[test[:64]], fp32=fp_prediction[:64], quantized=quant_prediction[:64])
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(x[test[:1]]).to(device),
        onnx_path,
        input_names=["input"],
        output_names=["prediction"],
        dynamic_axes={"input": {0: "batch"}, "prediction": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"task_kind": "regression", "shape": [None, 1], "authority": 0}
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": candidate_id,
                "dataset_sha256": metadata["sha256"],
                "task_contract_sha256": metadata["task_contract_sha256"],
                "onnx_sha256": sha256_file(onnx_path),
                "golden_sha256": sha256_file(golden_path),
                "best_seed_by_validation": best_seed,
                "three_seed_mean_objective": mean_objective,
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
    evaluation = {
        "schema": "cimc.forge200.jarvis-contract-exact-evaluation.v1",
        "status": "PASS" if status_pass else "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "baseline_contract": metadata["baseline"],
        "primary_metric_contract": metadata["primary_metric"],
        "baseline": {**baseline_info, "validation": baseline_validation, "test": baseline_test},
        "seed_reports": reports,
        "three_seed_count": 3,
        "g3_aggregate_mean_gate": aggregate_pass,
        "g4": {
            "mean_objective": mean_objective,
            "variance_objective": float(np.var([item["test_objective"] for item in reports])),
            "worst_objective": float(min(item["test_objective"] for item in reports)),
            "mean_mae": mean_mae,
            "variance_mae": float(np.var([item["test"]["mae"] for item in reports])),
            "worst_mae": float(max(item["test"]["mae"] for item in reports)),
        },
        "quantized_test": quant_metrics,
        "quantized_objective": quant_objective,
        "quantization_relative_mae_degradation": quant_degradation,
        "quantization_gate": quant_pass,
        "authority": 0,
        "board_accepted": False,
    }
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "baseline_report.json", evaluation["baseline"])
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist()})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "relative_mae_degradation": quant_degradation, "gate": quant_pass})
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
        "parameter_cap": LIMITS[candidate_id]["parameters"],
        "w8_payload_bytes": len(payload),
        "w8_payload_byte_cap": LIMITS[candidate_id]["weight_bytes"],
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
        f"# {candidate_id} exact JARVIS model card\n\n"
        f"- Status: `{status}`\n"
        f"- G3 three-seed aggregate mean gate: `{aggregate_pass}`.\n"
        f"- G4 mean/variance/worst objective: `{evaluation['g4']['mean_objective']:.6f}` / `{evaluation['g4']['variance_objective']:.6f}` / `{evaluation['g4']['worst_objective']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        "- Authority: `0`; unified GD32 board evidence remains pending.\n",
        encoding="utf-8",
    )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "mean_objective": mean_objective, "mean_mae": mean_mae, "baseline_mae": baseline_test["mae"], "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=320)
    parser.add_argument("--min-epochs", type=int, default=80)
    parser.add_argument("--early-stop-patience", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / args.candidate_id
        write_json(
            output / "failure.json",
            {
                "schema": "cimc.forge200.job-failure.v3",
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
