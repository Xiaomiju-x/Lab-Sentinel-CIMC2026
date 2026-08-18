#!/usr/bin/env python3
"""Train and package the exact JARVIS chemical-system ranker P138."""

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


def ranking_metrics(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> dict[str, float | int]:
    ndcg_values, top5_values, pairwise_values = [], [], []
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        if len(selected) < 3:
            continue
        truth = y[selected]
        predicted = score[selected]
        true_order = np.argsort(truth, kind="mergesort")
        pred_order = np.argsort(predicted, kind="mergesort")
        cutoff = min(10, len(selected))
        relevance = np.exp(-truth / 0.25)
        discount = 1.0 / np.log2(np.arange(2, cutoff + 2))
        dcg = float(np.sum(relevance[pred_order[:cutoff]] * discount))
        idcg = float(np.sum(relevance[true_order[:cutoff]] * discount))
        ndcg_values.append(dcg / max(idcg, 1e-12))
        top = min(5, len(selected))
        top5_values.append(len(set(true_order[:top]) & set(pred_order[:top])) / top)
        correct, total = 0, 0
        for left in range(len(selected)):
            for right in range(left + 1, len(selected)):
                true_delta = truth[left] - truth[right]
                pred_delta = predicted[left] - predicted[right]
                if true_delta == 0:
                    continue
                correct += int(true_delta * pred_delta > 0)
                total += 1
        if total:
            pairwise_values.append(correct / total)
    result = {
        "NDCG_at_10": float(np.mean(ndcg_values)),
        "top5_recall": float(np.mean(top5_values)),
        "pairwise_accuracy": float(np.mean(pairwise_values)),
        "ranked_chemical_systems": len(ndcg_values),
        "ranking_score_vs_ehull_MAE_not_primary": float(np.mean(np.abs(score - y))),
        "relevance_definition": "exp(-published_ehull_eV_per_atom/0.25)",
    }
    result["primary_composite"] = float(np.mean([result["NDCG_at_10"], result["top5_recall"], result["pairwise_accuracy"]]))
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
    candidate_id = "CAND-P-138"
    dataset = root / "data" / "staged_jarvis_p138_exact_v1" / f"{candidate_id}.npz"
    metadata_path = dataset.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
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
    baseline_score = raw["baseline_score"].astype(np.float32)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean = x_raw[train].mean(axis=0)
    std = x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    baseline_validation = ranking_metrics(y[validation], baseline_score[validation], groups[validation])
    baseline_test = ranking_metrics(y[test], baseline_score[test], groups[test])

    rank_target = np.zeros(len(y), dtype=np.float32)
    for group in np.unique(groups):
        selected = np.flatnonzero(groups == group)
        order = selected[np.argsort(y[selected], kind="mergesort")]
        rank_target[order] = np.arange(len(order), dtype=np.float32) / max(len(order) - 1, 1)
    baseline_weight = np.zeros(x.shape[1], dtype=np.float32)
    baseline_weight[-6] = std[-6]
    baseline_bias = float(mean[-6])

    class RankMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], 192),
                nn.GELU(),
                nn.Linear(192, 96),
                nn.GELU(),
                nn.Linear(96, 1),
            )
            self.register_buffer("baseline_weight", torch.from_numpy(baseline_weight.reshape(1, -1)))
            self.register_buffer("baseline_bias", torch.tensor([baseline_bias], dtype=torch.float32))
            self.register_buffer("residual_scale", torch.tensor([0.35], dtype=torch.float32))

        def forward(self, value: Any) -> Any:
            baseline = value @ self.baseline_weight.t() + self.baseline_bias
            return baseline + self.net(value) * self.residual_scale

    parameter_count = sum(parameter.numel() for parameter in RankMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(selected), 2048):
                values.append(model(torch.from_numpy(x[selected[start : start + 2048]]).to(device)).cpu().numpy().reshape(-1))
        return np.maximum(np.concatenate(values), 0.0)

    reports, states = [], {}
    started = time.perf_counter()
    target = rank_target
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = RankMLP().to(device)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        criterion = nn.SmoothL1Loss(beta=0.20)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(target[train].reshape(-1, 1))),
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
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                raw_validation = model(torch.from_numpy(x[validation]).to(device)).cpu().numpy().reshape(-1)
            current = ranking_metrics(y[validation], raw_validation, groups[validation])
            score = float(current["primary_composite"])
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_JARVIS_P138_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            raw_test = model(torch.from_numpy(x[test]).to(device)).cpu().numpy().reshape(-1)
        test_metrics = ranking_metrics(y[test], raw_test, groups[test])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": test_metrics, "beats_baseline": test_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    composites = np.asarray([item["test"]["primary_composite"] for item in reports], dtype=np.float64)
    mean_composite = float(composites.mean())
    aggregate_pass = mean_composite > float(baseline_test["primary_composite"]) + 1e-4
    best_report = max(reports, key=lambda item: item["validation_primary_composite"])
    best_seed = int(best_report["seed"])
    model = RankMLP().to(device)
    model.load_state_dict(states[best_seed])

    def transformed(selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            value = model(torch.from_numpy(x[selected]).to(device)).cpu().numpy().reshape(-1)
        return value

    fp_prediction = transformed(test)
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_prediction = transformed(test)
    quant_metrics = ranking_metrics(y[test], quant_prediction, groups[test])
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    status_pass = aggregate_pass and quant_pass

    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(golden_path, x=x[test[:64]], y=y[test[:64]], fp32_rank_score=fp_prediction[:64], quantized_rank_score=quant_prediction[:64])
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(x[test[:1]]).to(device), onnx_path, input_names=["host_and_competing_phase_features"], output_names=["stability_rank_score"], dynamic_axes={"host_and_competing_phase_features": {0: "batch"}, "stability_rank_score": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"shape": [None, 1], "semantics": "within_candidate_set_stability_rank_score", "ranking": "ascending", "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden_path), "three_seed_mean_composite": mean_composite})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden_path), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest(), engine_id=1)
    evaluation = {
        "schema": "cimc.forge200.jarvis-p138-contract-exact-evaluation.v1",
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
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "baseline_report.json", evaluation["baseline"])
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "target_transform": "within_chemical_system_normalized_rank", "embedded_baseline_feature": "formation_energy_percentile_within_candidate_system"})
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
        f"# {candidate_id} host-stability ranker\n\n- Status: `{status}`\n- Three-seed mean ranking composite: `{mean_composite:.6f}`; baseline: `{baseline_test['primary_composite']:.6f}`.\n- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n- Scope: computed thermodynamic screening only, not experimental phosphor performance.\n- Authority: `0`; unified GD32 board evidence remains pending.\n",
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / "CAND-P-138"
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": "CAND-P-138", "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
