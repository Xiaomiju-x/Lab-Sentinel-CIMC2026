#!/usr/bin/env python3
"""Train, quantize, export, and package teMatDb contracts P141/P144."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json


PARAMETER_CAP = 128_000
WEIGHT_BYTE_CAP = 128 * 1024
SUPPORTED = {"CAND-P-073", "CAND-P-141", "CAND-P-144"}


def ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    design = np.column_stack((x.astype(np.float64), np.ones(len(x), dtype=np.float64)))
    penalty = np.eye(design.shape[1], dtype=np.float64) * alpha
    penalty[-1, -1] = 0.0
    solution = np.linalg.solve(design.T @ design + penalty, design.T @ y.astype(np.float64))
    return solution[:-1].astype(np.float32), float(solution[-1])


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


def spearman(y: np.ndarray, prediction: np.ndarray) -> float:
    left, right = average_ranks(y), average_ranks(prediction)
    left -= left.mean()
    right -= right.mean()
    denominator = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / denominator) if denominator else 0.0


def seebeck_metrics(y: np.ndarray, prediction: np.ndarray, _groups: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(prediction - y)))
    scale = max(float(np.std(y)), 1e-6)
    result = {
        "Seebeck_MAE_uV_per_K": mae,
        "sign_accuracy": float(np.mean((prediction >= 0.0) == (y >= 0.0))),
        "Spearman_rho": spearman(y, prediction),
        "MAE_skill_vs_test_std": 1.0 - mae / scale,
    }
    result["primary_composite"] = float(np.mean([result["MAE_skill_vs_test_std"], result["sign_accuracy"], (result["Spearman_rho"] + 1.0) / 2.0]))
    return result


def thermoelectric_rank_metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    denominator = np.maximum(np.abs(y), 1e-6)
    mape = float(np.mean(np.abs(prediction - y) / denominator) * 100.0)
    ndcg_values, top5_values = [], []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) < 3:
            continue
        truth, score = y[selected], prediction[selected]
        true_order = np.argsort(-truth, kind="mergesort")
        pred_order = np.argsort(-score, kind="mergesort")
        cutoff = min(10, len(selected))
        relevance = np.log1p(np.maximum(truth, 0.0))
        discount = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.sum(relevance[pred_order[:cutoff]] * discount))
        idcg = float(np.sum(relevance[true_order[:cutoff]] * discount))
        ndcg_values.append(dcg / max(idcg, 1e-12))
        top = min(5, len(selected))
        top5_values.append(len(set(true_order[:top]) & set(pred_order[:top])) / top)
    result = {
        "power_factor_MAPE_percent": mape,
        "NDCG_at_10": float(np.mean(ndcg_values)) if ndcg_values else 0.0,
        "top5_recall": float(np.mean(top5_values)) if top5_values else 0.0,
        "ranked_compound_families": len(ndcg_values),
        "MAPE_score": 1.0 / (1.0 + mape / 100.0),
    }
    result["primary_composite"] = float(np.mean([result["MAPE_score"], result["NDCG_at_10"], result["top5_recall"]]))
    return result


def optical_onset_metrics(y: np.ndarray, prediction: np.ndarray, _groups: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - y)
    mae = float(np.mean(error))
    p95 = float(np.quantile(error, 0.95))
    result = {
        "absorption_onset_MAE_eV": mae,
        "p95_error_eV": p95,
        "MAE_score": 1.0 / (1.0 + mae),
        "p95_score": 1.0 / (1.0 + p95),
    }
    result["primary_composite"] = float(np.mean([result["MAE_score"], result["p95_score"]]))
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


def manifest(output: Path) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_manifest.json", "heartbeat.json"}):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    candidate_id = args.candidate_id
    dataset = root / "data" / args.dataset_dir / f"{candidate_id}.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or metadata["cross_split_component_overlap"] != 0 or metadata["cross_split_family_overlap"] != 0 or sha256_file(dataset) != metadata["sha256"]:
        raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = args.artifact_root.resolve() / candidate_id
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    started = time.perf_counter()

    raw = np.load(dataset, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y_raw = raw["y"].astype(np.float32)
    groups = raw["family"].astype(str)
    split = raw["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean, std = x_raw[train].mean(axis=0), x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    if candidate_id == "CAND-P-141":
        y_transformed = np.arcsinh(y_raw / 200.0).astype(np.float32)
        baseline_weight, baseline_bias = ridge(x[train], y_transformed[train], alpha=3.0)
        baseline_transformed = x @ baseline_weight + baseline_bias
        inverse: Callable[[np.ndarray], np.ndarray] = lambda value: np.sinh(np.clip(value, -6.0, 6.0)) * 200.0
        metric_function = seebeck_metrics
        output_semantics = "Seebeck_coefficient_uV_per_K"
        input_name = "thermoelectric_composition_temperature_features_and_ridge_prior"
        weight_cap = 128 * 1024
    elif candidate_id == "CAND-P-144":
        y_transformed = np.log1p(np.maximum(y_raw, 0.0)).astype(np.float32)
        baseline_raw = raw["baseline_pred"].astype(np.float32)
        baseline_transformed = np.log1p(np.maximum(baseline_raw, 0.0)).astype(np.float32)
        baseline_weight, baseline_bias = np.zeros(x.shape[1], dtype=np.float32), 0.0
        inverse = lambda value: np.maximum(np.expm1(np.clip(value, 0.0, 16.0)), 0.0)
        metric_function = thermoelectric_rank_metrics
        output_semantics = "thermoelectric_power_factor_uW_per_cmK2"
        input_name = "thermoelectric_measured_transport_composition_features_and_physics_prior"
        weight_cap = 128 * 1024
    else:
        y_transformed = np.log1p(np.maximum(y_raw, 0.0)).astype(np.float32)
        baseline_raw = raw["baseline_pred"].astype(np.float32)
        baseline_transformed = np.log1p(np.maximum(baseline_raw, 0.0)).astype(np.float32)
        baseline_weight, baseline_bias = np.zeros(x.shape[1], dtype=np.float32), 0.0
        inverse = lambda value: np.maximum(np.expm1(np.clip(value, 0.0, 8.0)), 0.0)
        metric_function = optical_onset_metrics
        output_semantics = "optical_absorption_onset_eV"
        input_name = "composition_structure_spectrum_metadata_and_bandgap_prior"
        weight_cap = 128 * 1024
    model_input = np.concatenate((x, baseline_transformed[:, None]), axis=1).astype(np.float32)
    baseline_prediction = inverse(baseline_transformed)
    baseline_validation = metric_function(y_raw[validation], baseline_prediction[validation], groups[validation])
    baseline_test = metric_function(y_raw[test], baseline_prediction[test], groups[test])

    class ResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 48), nn.GELU(), nn.Linear(48, 1))

        def forward(self, value: Any) -> Any:
            return value[:, -1:] + self.net(value[:, :-1])

    parameter_count = sum(parameter.numel() for parameter in ResidualMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    reports, states = [], {}
    train_dataset = TensorDataset(torch.from_numpy(model_input[train]), torch.from_numpy(y_transformed[train].reshape(-1, 1)))
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ResidualMLP().to(device)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
        loader = DataLoader(train_dataset, batch_size=min(args.batch_size, len(train)), shuffle=True, generator=torch.Generator().manual_seed(seed))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x.to(device))
                loss = nn.functional.smooth_l1_loss(prediction, batch_y.to(device), beta=0.25)
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_transformed = model(torch.from_numpy(model_input[validation]).to(device)).cpu().numpy().reshape(-1)
            current = metric_function(y_raw[validation], inverse(validation_transformed), groups[validation])
            score = float(current["primary_composite"])
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, args.heartbeat_phase, seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            test_transformed = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy().reshape(-1)
        report = metric_function(y_raw[test], inverse(test_transformed), groups[test])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": report, "beats_baseline": report["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    composites = np.asarray([report["test"]["primary_composite"] for report in reports])
    aggregate = {"mean": float(composites.mean()), "variance": float(composites.var()), "std": float(composites.std()), "worst": float(composites.min())}
    aggregate_pass = aggregate["mean"] > baseline_test["primary_composite"] + 1e-4
    best_report = max(reports, key=lambda report: report["validation_primary_composite"])
    best_seed = int(best_report["seed"])
    model = ResidualMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    with torch.no_grad():
        fp_transformed = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy().reshape(-1)
    fp_prediction = inverse(fp_transformed)
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > min(WEIGHT_BYTE_CAP, weight_cap):
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    model.eval()
    with torch.no_grad():
        quant_transformed = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy().reshape(-1)
    quant_prediction = inverse(quant_transformed)
    quant_metrics = metric_function(y_raw[test], quant_prediction, groups[test])
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    passed = aggregate_pass and quant_pass

    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=model_input[test[:64]], y=y_raw[test[:64]], fp32=fp_prediction[:64], quantized=quant_prediction[:64])
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(model_input[test[:1]]).to(device), onnx_path, input_names=[input_name], output_names=[f"transformed_{output_semantics}"], dynamic_axes={input_name: {0: "batch"}, f"transformed_{output_semantics}": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    postprocess = "sinh(value)*200" if candidate_id == "CAND-P-141" else "max(expm1(value),0)"
    output_schema = {"task_kind": "regression", "shape": [None, 1], "semantics": output_semantics, "model_output_transform": "asinh_y_over_200" if candidate_id == "CAND-P-141" else "log1p", "postprocess": postprocess, "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "three_seed_mean": aggregate["mean"]})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    audit = {
        "schema": args.audit_schema,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": candidate_id,
        "baseline": {"kind": metadata["baseline_execution"], "validation": baseline_validation, "test": baseline_test},
        "seed_reports": reports,
        "aggregate": aggregate,
        "g3_aggregate_mean_gate": aggregate_pass,
        "quantized_best_seed": {"seed": best_seed, "test": quant_metrics, "metric_delta": quant_delta, "gate": quant_pass},
        "parameter_count": parameter_count,
        "w8_payload_bytes": len(payload),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", audit)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "baseline_report.json", audit["baseline"])
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "baseline_weight": baseline_weight.tolist(), "baseline_bias": baseline_bias, "transform": output_schema["model_output_transform"]})
    write_json(output / "output_schema.json", output_schema)
    write_json(output / "quantization_parity.json", {"primary_composite_delta": quant_delta, "gate": quant_pass})
    (output / "model_card.md").write_text(
        f"# {candidate_id} exact {args.dataset_label} model\n\n"
        f"- Status: `{status}`.\n"
        f"- Truth: `{metadata['truth_class']}`; target scope: {metadata['target_scope']}.\n"
        f"- Three-seed mean composite: `{aggregate['mean']:.6f}`; preregistered baseline: `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        f"- Authority: `0`; unified GD32 board acceptance remains pending.\n",
        encoding="utf-8",
    )
    promotion = {"schema": "cimc.forge200.promotion-receipt.v3", "status": status, "candidate_id": candidate_id, "authority": 0, "board_accepted": False, "countable_model": False, "host_contract_pass": passed, "three_seed_count": 3, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "contract_exact_audit_sha256": sha256_file(output / "contract_exact_audit.json"), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3}}
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    write_json(root / "evidence" / f"{args.evidence_prefix}_{candidate_id[-4:].lower()}_exact_closure.v1.json", {"schema": args.closure_schema, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "PARTIAL", "record": audit, "authority_nonzero": 0, "board_actions": 0})
    print(json.dumps({"candidate_id": candidate_id, "status": status, "mean_composite": aggregate["mean"], "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True))
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-id", choices=sorted(SUPPORTED), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-epochs", type=int, default=70)
    parser.add_argument("--min-epochs", type=int, default=24)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    parser.add_argument("--dataset-dir", default="staged_tematdb_p141_p144_exact_v1")
    parser.add_argument("--dataset-label", default="teMatDb")
    parser.add_argument("--evidence-prefix", default="tematdb")
    parser.add_argument("--heartbeat-phase", default="TRAIN_TEMATDB_EXACT")
    parser.add_argument("--audit-schema", default="cimc.forge200.tematdb-contract-exact-audit.v1")
    parser.add_argument("--closure-schema", default="cimc.forge200.tematdb-exact-closure.v1")
    args = parser.parse_args()
    try:
        receipt = run(args)
        return 0 if receipt["host_contract_pass"] else 2
    except Exception as exc:
        output = args.artifact_root.resolve() / args.candidate_id
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": args.candidate_id, "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
