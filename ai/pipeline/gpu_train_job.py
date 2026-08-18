#!/usr/bin/env python3
"""Resumable CUDA trainer for canonical Forge200 staged datasets.

This runner is intentionally narrow: it accepts only queue jobs admitted by
the frozen data gate and canonical NPZ datasets produced by
stage_admitted_data.py.  Unsupported modalities are rejected before CUDA is
used.  All packages remain authority=0 and board-pending.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = [20260801, 20260802, 20260803]
HEADER_BYTES = 256


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def heartbeat(path: Path, candidate_id: str, phase: str, seed: int | None = None, epoch: int | None = None) -> None:
    write_json(
        path,
        {
            "schema": "cimc.forge200.job-heartbeat.v1",
            "candidate_id": candidate_id,
            "phase": phase,
            "seed": seed,
            "epoch": epoch,
            "utc": datetime.now(timezone.utc).isoformat(),
        },
    )


def load_job(root: Path, candidate_id: str) -> dict[str, Any]:
    queue = json.loads((root / "queue" / "dual_5090_queue.v1.json").read_text(encoding="utf-8"))
    for job in queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]:
        if job["candidate_id"] == candidate_id:
            return job
    raise RuntimeError(f"unknown candidate: {candidate_id}")


def validate_admission(root: Path, job: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    if job.get("authority") != 0:
        raise RuntimeError("AUTHORITY_NONZERO")
    if job.get("admission_state") != "ADMITTED":
        raise RuntimeError("BLOCKED_PRE_GPU:" + job["data_binding"]["full_data_state"])
    dataset = root / job["staged_dataset"]
    metadata_path = root / job["staged_metadata"]
    if not dataset.is_file() or sha256_file(dataset) != job["staged_dataset_sha256"]:
        raise RuntimeError("STAGED_DATASET_HASH_MISMATCH")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("authority") != 0:
        raise RuntimeError("STAGED_METADATA_NOT_ADMITTED")
    if metadata.get("cross_split_group_overlap") != 0:
        raise RuntimeError("SPLIT_LEAKAGE")
    return dataset, metadata


def preprocess(data: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    x = data["x"].astype(np.float32)
    y = data["y"]
    split = data["split"].astype(np.int8)
    train = split == 0
    median = np.nanmedian(x[train], axis=0)
    median[~np.isfinite(median)] = 0.0
    x = np.where(np.isfinite(x), x, median)
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    return x.astype(np.float32), y, split, {"median": median, "mean": mean, "std": std}


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability.argmax(axis=1)
    classes = probability.shape[1]
    recalls, f1s = [], []
    for cls in range(classes):
        tp = int(np.sum((prediction == cls) & (y == cls)))
        fp = int(np.sum((prediction == cls) & (y != cls)))
        fn = int(np.sum((prediction != cls) & (y == cls)))
        recalls.append(tp / max(tp + fn, 1))
        precision = tp / max(tp + fp, 1)
        recall = recalls[-1]
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    confidence = probability.max(axis=1)
    correct = (prediction == y).astype(np.float32)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "brier": float(np.mean(np.sum((probability - np.eye(classes)[y]) ** 2, axis=1))),
        "ece_10bin": ece,
    }


def reranking_validation_composite(
    y: np.ndarray,
    probability: np.ndarray,
    query_id: np.ndarray,
    special_match: np.ndarray,
) -> float:
    scores = probability[:, 1]
    ndcg_values, reciprocal_values, special_values = [], [], []
    for query in np.unique(query_id):
        selected = np.flatnonzero(query_id == query)
        order = selected[np.argsort(-scores[selected], kind="mergesort")]
        relevance = y[order].astype(np.float64)
        relevant = int(np.sum(y[selected]))
        if relevant <= 0:
            continue
        top = relevance[:10]
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        ndcg_values.append(float(np.sum(top * discounts)) / max(float(np.sum(discounts[: min(relevant, 10)])), 1e-12))
        positions = np.flatnonzero(relevance)
        reciprocal_values.append(1.0 / float(positions[0] + 1))
        special_values.append(float(special_match[order[0]] == 1))
    if not ndcg_values:
        return 0.0
    return float(np.mean([np.mean(ndcg_values), np.mean(reciprocal_values), np.mean(special_values)]))


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction.reshape(-1) - y.reshape(-1)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2)))}


def quantize_state(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    quantized, scales = {}, {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        scale = max(float(np.max(np.abs(array))) / 127.0, 1e-12)
        quantized[name] = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        scales[name] = scale
    return quantized, scales


def dequantized_state(torch: Any, quantized: dict[str, np.ndarray], scales: dict[str, float]) -> dict[str, Any]:
    return {name: torch.from_numpy(array.astype(np.float32) * scales[name]) for name, array in quantized.items()}


def build_package(
    output: Path,
    candidate_id: str,
    payload: bytes,
    golden_sha: str,
    release_root: str,
    output_schema_sha: str,
    engine_id: int = 1,
) -> dict[str, Any]:
    payload_sha = hashlib.sha256(payload).digest()
    header = bytearray(HEADER_BYTES)
    struct.pack_into("<4sHHHHBBHQQIII", header, 0, b"ICMF", 1, 256, engine_id, 1, 0, 1, 0, 1, len(payload), 0, 0, 0)
    header[44:76] = candidate_id.encode("utf-8")[:31].ljust(32, b"\0")
    header[76:108] = payload_sha
    header[108:140] = bytes.fromhex(golden_sha)
    header[140:172] = bytes.fromhex(release_root)
    header[172:204] = bytes.fromhex(output_schema_sha)
    package_path = output / "w8_or_w8a8.bin"
    package_path.write_bytes(bytes(header) + payload)
    return {
        "path": package_path.name,
        "sha256": sha256_file(package_path),
        "bytes": package_path.stat().st_size,
        "payload_sha256": payload_sha.hex(),
        "authority": 0,
        "engine_id": engine_id,
        "board_accepted": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    job = load_job(root, args.candidate_id)
    if args.staged_subdir:
        if not args.staged_subdir.replace("_", "").isalnum():
            raise RuntimeError("UNSAFE_STAGED_SUBDIRECTORY")
        dataset_path = root / "data" / args.staged_subdir / f"{args.candidate_id}.npz"
        metadata_path = dataset_path.with_suffix(".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["status"] != "PASS" or metadata["authority"] != 0:
            raise RuntimeError("LOCAL_STAGED_DATA_GATE")
        if sha256_file(dataset_path) != metadata["sha256"]:
            raise RuntimeError("LOCAL_STAGED_DATA_HASH")
        job = dict(job)
        job["staged_dataset_sha256"] = metadata["sha256"]
    else:
        dataset_path, metadata = validate_admission(root, job)
    output = (args.artifact_root / args.candidate_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    heartbeat(heartbeat_path, args.candidate_id, "IMPORT_TORCH")
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_LOCAL_4050_NOT_AUTHORIZED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    vram_gib = props.total_memory / (1024**3)
    if not args.allow_local_vram_override and vram_gib + 0.1 < float(job["estimated_vram_gib"]):
        raise RuntimeError(f"VRAM_GATE:{vram_gib:.2f}GiB")
    data = np.load(dataset_path, allow_pickle=False)
    if int(data["authority"]) != 0 or str(data["candidate_id"]) != args.candidate_id:
        raise RuntimeError("DATASET_IDENTITY_OR_AUTHORITY_GATE")
    x, y, split, prep = preprocess(data)
    task_kind = str(data["task_kind"])
    validation_ranking = bool(metadata.get("checkpoint_selection") == "VALIDATION_RANKING_COMPOSITE_V1")
    if validation_ranking and not all(name in data for name in ("query_id", "special_match")):
        raise RuntimeError("VALIDATION_RANKING_FIELDS_MISSING")
    output_count = 1 if task_kind == "regression" else int(np.max(y)) + 1
    hidden = args.hidden_size
    residual_prior_index = metadata.get("residual_prior_feature_index")
    if residual_prior_index is not None:
        residual_prior_index = int(residual_prior_index)
        if task_kind != "regression" or not (0 <= residual_prior_index < x.shape[1]):
            raise RuntimeError("INVALID_RESIDUAL_PRIOR_CONTRACT")

    class ForgeMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, output_count))
            if residual_prior_index is not None:
                nn.init.zeros_(self.net[-1].weight)
                nn.init.zeros_(self.net[-1].bias)

        def forward(self, value: Any) -> Any:
            result = self.net(value)
            if residual_prior_index is not None:
                prior = value[:, residual_prior_index : residual_prior_index + 1] * float(prep["std"][residual_prior_index]) + float(prep["mean"][residual_prior_index])
                result = result + prior
            return result

    def infer(model: Any, indices: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            result = model(torch.from_numpy(x[indices]).to(device)).cpu().numpy()
        if task_kind != "regression":
            result = np.exp(result - result.max(axis=1, keepdims=True))
            result /= result.sum(axis=1, keepdims=True)
        return result

    train_indices, validation_indices, test_indices = (np.flatnonzero(split == value) for value in (0, 1, 2))
    if min(len(train_indices), len(validation_indices), len(test_indices)) < 16:
        raise RuntimeError("EMPTY_OR_TOO_SMALL_SPLIT")
    baseline = (
        {"kind": "train_mean", **regression_metrics(y[test_indices], np.full(len(test_indices), float(np.mean(y[train_indices]))))}
        if task_kind == "regression"
        else {"kind": "train_majority", **classification_metrics(y[test_indices], np.eye(output_count)[np.full(len(test_indices), int(np.bincount(y[train_indices]).argmax()))])}
    )
    seed_reports, best_state, best_score, best_seed = [], None, math.inf, None
    seed_test_outputs: dict[str, np.ndarray] = {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ForgeMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        if task_kind == "regression":
            criterion = nn.SmoothL1Loss()
            target_tensor = torch.from_numpy(y.astype(np.float32)).reshape(-1, 1)
        else:
            counts = np.bincount(y[train_indices], minlength=output_count).astype(np.float32)
            weights = counts.sum() / np.maximum(counts * output_count, 1.0)
            criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))
            target_tensor = torch.from_numpy(y.astype(np.int64))
        loader_generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train_indices]), target_tensor[train_indices]),
            batch_size=min(args.batch_size, len(train_indices)),
            shuffle=True,
            generator=loader_generator,
        )
        checkpoint_dir = output / f"train_seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "last.pt"
        start_epoch, patience, local_best = 0, 0, math.inf
        if args.resume and state_path.is_file():
            checkpoint = torch.load(state_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            local_best = float(checkpoint["local_best"])
        epoch = start_epoch - 1
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch_x.to(device)), batch_y.to(device))
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                optimizer.step()
            validation_output = infer(model, validation_indices)
            if validation_ranking:
                validation_metric = 1.0 - reranking_validation_composite(
                    y[validation_indices],
                    validation_output,
                    data["query_id"][validation_indices],
                    data["special_match"][validation_indices],
                )
            else:
                validation_metric = (
                    regression_metrics(y[validation_indices], validation_output)["mae"]
                    if task_kind == "regression"
                    else 1.0 - classification_metrics(y[validation_indices], validation_output)["balanced_accuracy"]
                )
            if validation_metric < local_best - 1e-6:
                local_best, patience = validation_metric, 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                patience += 1
            if epoch % args.checkpoint_epochs == 0 or epoch + 1 == args.max_epochs:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "local_best": local_best}, state_path)
            heartbeat(heartbeat_path, args.candidate_id, "TRAIN", seed, epoch)
            if patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True))
        validation_output = infer(model, validation_indices)
        test_output = infer(model, test_indices)
        validation_metrics = regression_metrics(y[validation_indices], validation_output) if task_kind == "regression" else classification_metrics(y[validation_indices], validation_output)
        test_metrics = regression_metrics(y[test_indices], test_output) if task_kind == "regression" else classification_metrics(y[test_indices], test_output)
        if validation_ranking:
            score = 1.0 - reranking_validation_composite(
                y[validation_indices],
                validation_output,
                data["query_id"][validation_indices],
                data["special_match"][validation_indices],
            )
        else:
            score = validation_metrics["mae"] if task_kind == "regression" else 1.0 - validation_metrics["balanced_accuracy"]
        seed_reports.append({"seed": seed, "validation": validation_metrics, "test": test_metrics, "epochs": epoch + 1})
        seed_test_outputs[f"seed_{seed}"] = test_output.astype(np.float32, copy=False)
        if score < best_score:
            best_score, best_seed = score, seed
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None and best_seed is not None
    prediction_evidence = {
        "indices": test_indices.astype(np.int64),
        "y": y[test_indices],
        "authority": np.asarray(0, dtype=np.int8),
        **seed_test_outputs,
    }
    for optional_name in (
        "baseline_prediction",
        "baseline_probability",
        "sequence_id",
        "token_position",
        "reason_code",
        "bad_answer",
        "query_id",
        "baseline_score",
        "special_match",
        "domain_id",
        "ood_label",
        "stale_label",
    ):
        if optional_name in data:
            prediction_evidence[optional_name] = data[optional_name][test_indices]
    np.savez_compressed(output / "three_seed_test_predictions.npz", **prediction_evidence)
    model = ForgeMLP().to(device)
    model.load_state_dict(best_state)
    fp32_test = infer(model, test_indices)
    quantized, scales = quantize_state(best_state)
    quant_buffer = io.BytesIO()
    np.savez_compressed(quant_buffer, **quantized, **{f"scale::{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()})
    payload = quant_buffer.getvalue()
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_test = infer(model, test_indices)
    prediction_path = output / "three_seed_test_predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as stored:
        prediction_evidence = {name: stored[name] for name in stored.files}
    prediction_evidence["quantized_best_seed"] = quant_test.astype(np.float32, copy=False)
    np.savez_compressed(prediction_path, **prediction_evidence)
    parity = float(np.max(np.abs(fp32_test - quant_test)))
    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(golden_path, x=x[test_indices[:32]], fp32=fp32_test[:32], quantized=quant_test[:32], y=y[test_indices[:32]])
    golden_sha = sha256_file(golden_path)
    heartbeat(heartbeat_path, args.candidate_id, "EXPORT")
    model.load_state_dict(best_state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(x[test_indices[:1]]).to(device), onnx_path, input_names=["input"], output_names=["logits"], dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"task_kind": task_kind, "shape": [None, output_count], "authority": 0}
    output_schema_sha = hashlib.sha256(canonical_bytes(output_schema)).hexdigest()
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": args.candidate_id,
                "dataset_sha256": job["staged_dataset_sha256"],
                "task_contract_sha256": metadata["task_contract_sha256"],
                "onnx_sha256": sha256_file(onnx_path),
                "golden_sha256": golden_sha,
                "best_seed": best_seed,
            }
        )
    ).hexdigest()
    package = build_package(output, args.candidate_id, payload, golden_sha, release_root, output_schema_sha)
    calibration = {
        "schema": "cimc.forge200.calibration-ood.v1",
        "task_kind": task_kind,
        "quant_max_abs_error": parity,
        "ood_score": "max_abs_train_fitted_zscore",
        "ood_test_p95": float(np.percentile(np.max(np.abs(x[test_indices]), axis=1), 95)),
    }
    evaluation = {
        "schema": "cimc.forge200.grouped-evaluation.v1",
        "candidate_id": args.candidate_id,
        "truth_class": metadata["truth_class"],
        "baseline": baseline,
        "seed_reports": seed_reports,
        "best_seed": best_seed,
        "quantized_test": regression_metrics(y[test_indices], quant_test) if task_kind == "regression" else classification_metrics(y[test_indices], quant_test),
        "split_counts": {"train": len(train_indices), "validation": len(validation_indices), "test": len(test_indices)},
        "group_overlap": 0,
    }
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "calibration_ood.json", calibration)
    write_json(output / "preprocessing_train_only.json", {name: value.tolist() for name, value in prep.items()})
    write_json(output / "task_contract.json", {"candidate_id": args.candidate_id, "task_contract_sha256": metadata["task_contract_sha256"], "authority": 0})
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "split_manifest.json", {"split_sha256": metadata["split_sha256"], "cross_split_group_overlap": 0})
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "ablation.json", {"status": "CANONICAL_MLP_ONLY", "feature_contract": metadata["feature_contract"]})
    model_card = f"""# {args.candidate_id} model card

- Status: `HOST_GPU_TRAINED_BOARD_PENDING`
- Truth class: `{metadata['truth_class']}`
- Best seed: `{best_seed}`; all three fixed seeds are reported.
- Release root: `{release_root}`
- ONNX SHA-256: `{sha256_file(onnx_path)}`
- Package SHA-256: `{package['sha256']}`
- Quantized parity max absolute error: `{parity:.8g}`
- Authority: `0`; this model cannot control the furnace or deterministic safety chain.
- Board evidence: absent. This artifact is not counted as deployed until unified GD32 acceptance.
"""
    (output / "model_card.md").write_text(model_card, encoding="utf-8")
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v1",
        "status": "HOST_GPU_TRAINED_BOARD_PENDING",
        "candidate_id": args.candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "best_seed": best_seed,
        "three_seed_count": len(seed_reports),
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": golden_sha,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": vram_gib},
    }
    write_json(output / "promotion_receipt.json", receipt)
    heartbeat(heartbeat_path, args.candidate_id, "COMPLETE")
    records = []
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_manifest.json", "transfer_manifest.json"}
        and not item.name.startswith("worker_attempt_")
    ):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})
    print(json.dumps({"status": receipt["status"], "candidate_id": args.candidate_id, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--staged-subdir")
    parser.add_argument("--allow-local-vram-override", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--checkpoint-epochs", type=int, default=5)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        failure = {
            "schema": "cimc.forge200.job-failure.v1",
            "status": "FAIL_CLOSED",
            "candidate_id": args.candidate_id,
            "authority": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(args.artifact_root.resolve() / args.candidate_id / "failure.json", failure)
        print(json.dumps(failure, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
