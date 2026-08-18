#!/usr/bin/env python3
"""Build and train S033 from frozen S027 router outputs and held-out labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from gpu_train_job import build_package, canonical_bytes, heartbeat, sha256_file, write_json

SEEDS = [20260801, 20260802, 20260803]


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    classes = probability.shape[1]
    onehot = np.eye(classes)[y]
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = (prediction == y).astype(np.float32)
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    domain_ece = []
    for domain in range(6):
        mask = y == domain
        if not np.any(mask):
            continue
        domain_ece.append(abs(float(np.mean(probability[mask, domain])) - float(np.mean(prediction[mask] == domain))))
    abstain_target = (y == 6).astype(np.int64)
    return {
        "accuracy": float(np.mean(prediction == y)),
        "brier": float(np.mean(np.sum((probability - onehot) ** 2, axis=1))),
        "ece_10bin": ece,
        "abstention_auroc": float(roc_auc_score(abstain_target, probability[:, 6])),
        "worst_domain_ece": max(domain_ece) if domain_ece else 1.0,
    }


def objective(value: dict[str, float], baseline: dict[str, float]) -> float:
    return float(
        0.30 * (1.0 - value["brier"] / max(baseline["brier"], 1e-9))
        + 0.25 * (1.0 - value["ece_10bin"] / max(baseline["ece_10bin"], 1e-3))
        + 0.25 * (value["abstention_auroc"] - baseline["abstention_auroc"])
        + 0.20 * (1.0 - value["worst_domain_ece"] / max(baseline["worst_domain_ece"], 1e-3))
    )


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
    router_root = args.router_artifact.resolve()
    router_receipt = json.loads((router_root / "promotion_receipt.json").read_text(encoding="utf-8"))
    router_source = json.loads((router_root / "source_manifest.json").read_text(encoding="utf-8"))
    dataset = root / router_source["path"]
    if sha256_file(dataset) != router_source["sha256"] or router_source["cross_split_group_overlap"] != 0:
        raise RuntimeError("ROUTER_DATA_GATE")
    if router_receipt["authority"] != 0 or router_receipt["board_accepted"]:
        raise RuntimeError("ROUTER_AUTHORITY_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y8 = raw["y"].astype(np.int64)
    y = np.where(y8 < 6, y8, 6).astype(np.int64)
    split = raw["split"].astype(np.int8)
    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    preprocessing = json.loads((router_root / "preprocessing_train_only.json").read_text(encoding="utf-8"))
    median = np.asarray(preprocessing["median"], dtype=np.float32)
    mean = np.asarray(preprocessing["mean"], dtype=np.float32)
    std = np.asarray(preprocessing["std"], dtype=np.float32)
    x_clean = np.where(np.isfinite(x_raw), x_raw, median)
    x = ((x_clean - mean) / std).astype(np.float32)

    class RouterMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 8))
        def forward(self, value: Any) -> Any:
            return self.net(value)

    router = RouterMLP().to(device)
    router.load_state_dict(torch.load(router_root / f"train_seed_{router_receipt['best_seed']}" / "best.pt", map_location=device, weights_only=True))
    router.eval()
    logits_parts = []
    with torch.no_grad():
        for start in range(0, len(x), 1024):
            logits_parts.append(router(torch.from_numpy(x[start:start+1024]).to(device)).cpu().numpy())
    logits = np.concatenate(logits_parts)
    router_probability = softmax(logits)
    centroids = []
    for domain in range(6):
        centroid = x_raw[indices["train"]][y8[indices["train"]] == domain].mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-9)
        centroids.append(centroid)
    centroids = np.asarray(centroids, dtype=np.float32)
    normalized_x = x_raw / np.maximum(np.linalg.norm(x_raw, axis=1, keepdims=True), 1e-9)
    retrieval_scores = normalized_x @ centroids.T
    entropy = -np.sum(router_probability * np.log(np.maximum(router_probability, 1e-9)), axis=1, keepdims=True)
    ordered = np.sort(router_probability, axis=1)
    margin = (ordered[:, -1] - ordered[:, -2]).reshape(-1, 1)
    ood_z = np.max(np.abs(x), axis=1, keepdims=True)
    features = np.concatenate((logits, router_probability, retrieval_scores, entropy, margin, ood_z), axis=1).astype(np.float32)
    feature_mean = features[indices["train"]].mean(axis=0)
    feature_std = features[indices["train"]].std(axis=0)
    feature_std[feature_std < 1e-6] = 1.0
    features = np.clip((features - feature_mean) / feature_std, -12.0, 12.0).astype(np.float32)

    temperatures = np.linspace(0.4, 4.0, 145)
    train = indices["train"]
    best_temperature = min(
        temperatures,
        key=lambda temperature: float(-np.mean(np.log(np.maximum(softmax(logits[train] / temperature)[np.arange(len(train)), y8[train]], 1e-9)))),
    )
    base8 = softmax(logits / best_temperature)
    baseline_probability = np.concatenate((base8[:, :6], base8[:, 6:].sum(axis=1, keepdims=True)), axis=1)
    baseline = {name: metrics(y[selected], baseline_probability[selected]) for name, selected in indices.items()}
    baseline_validation_objective = objective(baseline["validation"], baseline["validation"])
    baseline_test_objective = objective(baseline["test"], baseline["test"])

    class CalibrationMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(features.shape[1], 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 7))
        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(p.numel() for p in CalibrationMLP().parameters())
    if parameter_count > 32_000:
        raise RuntimeError("PARAMETER_CAP")
    output = (args.artifact_root / "CAND-S-033").resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            value = model(torch.from_numpy(features[selected]).to(device)).cpu().numpy()
        return softmax(value)

    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CalibrationMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=3e-4)
        counts = np.bincount(y[train], minlength=7).astype(np.float32)
        weights = np.sqrt(counts.sum() / np.maximum(counts * 7.0, 1.0))
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device), label_smoothing=0.01)
        best_score, patience = -math.inf, 0
        best_path = output / f"train_seed_{seed}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(seed)
        for epoch in range(160):
            order = generator.permutation(train)
            model.train()
            for start in range(0, len(order), 256):
                chosen = order[start:start+256]
                loss = criterion(model(torch.from_numpy(features[chosen]).to(device)), torch.from_numpy(y[chosen]).to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation_metrics = metrics(y[indices["validation"]], infer(model, indices["validation"]))
            value = objective(validation_metrics, baseline["validation"])
            if value > best_score + 1e-6:
                best_score, patience = value, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-S-033", "TRAIN_POSTROUTER", seed, epoch)
            if epoch >= 39 and patience >= 18:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics = metrics(y[indices["test"]], infer(model, indices["test"]))
        test_objective = objective(test_metrics, baseline["test"])
        reports.append({"seed": seed, "epochs": epoch+1, "validation_objective": best_score, "test_objective": test_objective, "baseline_test_objective": baseline_test_objective, "beats_baseline": test_objective > baseline_test_objective + 1e-4, "test": test_metrics})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best["seed"])
    state = states[best_seed]
    model = CalibrationMLP().to(device)
    model.load_state_dict(state)
    selected = indices["test"][:64]
    fp = infer(model, selected)
    q, s = quantize(state)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **q, **{f"scale::{name}": value for name, value in s.items()})
    payload = buffer.getvalue()
    model.load_state_dict({name: torch.from_numpy(np.asarray(np.asarray(value, dtype=np.float32)*np.asarray(s[name], dtype=np.float32), dtype=np.float32)) for name, value in q.items()})
    quant = infer(model, selected)
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=features[selected], y=y[selected], fp32=fp, quantized=quant)
    model.load_state_dict(state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(features[selected[:1]]).to(device), onnx_path, input_names=["router_retrieval_ood_features"], output_names=["calibrated_logits"], dynamic_axes={"router_retrieval_ood_features": {0: "batch"}, "calibrated_logits": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == "CAND-S-033")
    task_hash = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    source_content = {"router_release_root": router_receipt["release_root"], "router_package_sha256": router_receipt["package"]["sha256"], "dataset_sha256": router_source["sha256"], "feature_preprocessing_fit": "train_only", "temperature_fit": "train_only", "feature_count": features.shape[1]}
    source_manifest = {"schema": "cimc.forge200.s033-postrouter-source.v1", "status": "PASS", "candidate_id": "CAND-S-033", "records": len(y), "counts": {name: len(selected_indices) for name, selected_indices in indices.items()}, "cross_split_group_overlap": 0, "task_contract_sha256": task_hash, "truth_class": "POST_GPU_ROUTER_OUTPUTS_WITH_SOURCE_FAMILY_HELDOUT_LABELS", "baseline_contract": contract["baseline"], "primary_metric_contract": contract["primary_metric"], "temperature": float(best_temperature), **source_content, "content_root_sha256": hashlib.sha256(canonical_bytes(source_content)).hexdigest(), "authority": 0, "board_accepted": False, "countable_model": False}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": "CAND-S-033", "source_root": source_manifest["content_root_sha256"], "task_contract_sha256": task_hash, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "best_seed": best_seed})).hexdigest()
    schema = {"task_kind": "domain_and_abstain_probability", "shape": [None, 7], "authority": 0}
    package = build_package(output, "CAND-S-033", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(schema)).hexdigest(), engine_id=1)
    pass_count = sum(item["beats_baseline"] for item in reports)
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if pass_count == 3 else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {"schema": "cimc.forge200.s033-evaluation.v1", "status": "PASS" if pass_count == 3 else "FAIL_CLOSED", "candidate_id": "CAND-S-033", "baseline": baseline, "baseline_test_objective": baseline_test_objective, "seed_reports": reports, "three_seed_baseline_pass": pass_count, "authority": 0, "board_accepted": False}
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", source_manifest)
    write_json(output / "preprocessing_train_only.json", {"mean": feature_mean.tolist(), "std": feature_std.tolist()})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "max_abs_probability_error": float(np.max(np.abs(fp-quant)))})
    receipt = {"schema": "cimc.forge200.promotion-receipt.v2", "status": status, "candidate_id": "CAND-S-033", "authority": 0, "board_accepted": False, "countable_model": False, "three_seed_count": 3, "three_seed_baseline_pass": pass_count, "parameter_count": parameter_count, "parameter_cap": 32_000, "w8_payload_bytes": len(payload), "best_seed": best_seed, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter()-started, "gpu": {"name": props.name, "vram_gib": props.total_memory/1024**3}}
    write_json(output / "promotion_receipt.json", receipt)
    print(json.dumps({"candidate_id": "CAND-S-033", "status": status, "passes": pass_count, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--router-artifact", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
