#!/usr/bin/env python3
"""Train S001 against its exact deterministic keyword-router baseline."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import build_package, canonical_bytes, heartbeat, sha256_file, write_json

SEEDS = [20260801, 20260802, 20260803]


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - value.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    order = np.argsort(-probability, axis=1)
    rank = np.argmax(order == y[:, None], axis=1) + 1
    prediction = order[:, 0]
    confidence = probability[np.arange(len(y)), prediction]
    correct = prediction == y
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (confidence >= lower) & (confidence < lower + 0.1)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return {"top1_accuracy": float(correct.mean()), "mrr": float(np.mean(1.0 / rank)), "ece_10bin": ece}


def objective(value: dict[str, float]) -> float:
    return 0.45 * value["top1_accuracy"] + 0.35 * value["mrr"] + 0.20 * (1.0 - value["ece_10bin"])


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
    dataset = root / "data" / "staged_router_contract_v2" / "CAND-S-001.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or sha256_file(dataset) != metadata["sha256"] or metadata["cross_split_group_overlap"] != 0:
        raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset, allow_pickle=False)
    x, y, split = raw["x"].astype(np.float32), raw["y"].astype(np.int64), raw["split"].astype(np.int8)
    baseline_probability = raw["baseline_probability"].astype(np.float32)
    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    mean = x[indices["train"]].mean(axis=0)
    std = x[indices["train"]].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    baseline = {name: metrics(y[selected], baseline_probability[selected]) for name, selected in indices.items()}
    baseline_test_objective = objective(baseline["test"])

    class ExpertSelector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Linear(128, 6))
        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(p.numel() for p in ExpertSelector().parameters())
    if parameter_count > 42_000:
        raise RuntimeError("PARAMETER_CAP")
    output = (args.artifact_root / "CAND-S-001").resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return softmax(model(torch.from_numpy(x[selected]).to(device)).cpu().numpy())

    reports, states = [], {}
    started = time.perf_counter()
    train = indices["train"]
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ExpertSelector().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=4e-4)
        best_score, patience = -math.inf, 0
        best_path = output / f"train_seed_{seed}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(seed)
        for epoch in range(140):
            order = generator.permutation(train)
            model.train()
            for start in range(0, len(order), 512):
                selected = order[start:start + 512]
                logits = model(torch.from_numpy(x[selected]).to(device))
                loss = nn.functional.cross_entropy(logits, torch.from_numpy(y[selected]).to(device), label_smoothing=0.015)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation = metrics(y[indices["validation"]], infer(model, indices["validation"]))
            score = objective(validation)
            if score > best_score + 1e-6:
                best_score, patience = score, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-S-001", "TRAIN_EXACT_ROUTER", seed, epoch)
            if epoch >= 30 and patience >= 16:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test = metrics(y[indices["test"]], infer(model, indices["test"]))
        test_score = objective(test)
        beats = (
            test["top1_accuracy"] > baseline["test"]["top1_accuracy"]
            and test["mrr"] > baseline["test"]["mrr"]
            and test_score > baseline_test_objective + 1e-4
        )
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_objective": best_score, "test_objective": test_score, "baseline_test_objective": baseline_test_objective, "beats_baseline": beats, "test": test})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    best = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best["seed"])
    state = states[best_seed]
    model = ExpertSelector().to(device)
    model.load_state_dict(state)
    chosen = indices["test"][:64]
    fp = infer(model, chosen)
    q, s = quantize(state)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **q, **{f"scale::{name}": value for name, value in s.items()})
    payload = buffer.getvalue()
    model.load_state_dict({name: torch.from_numpy(np.asarray(np.asarray(value, dtype=np.float32) * np.asarray(s[name], dtype=np.float32), dtype=np.float32)) for name, value in q.items()})
    quant = infer(model, chosen)
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=x[chosen], y=y[chosen], fp32=fp, quantized=quant)
    model.load_state_dict(state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(x[chosen[:1]]).to(device), onnx_path, input_names=["query_domain_capability_features"], output_names=["expert_logits"], dynamic_axes={"query_domain_capability_features": {0: "batch"}, "expert_logits": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": "CAND-S-001", "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "best_seed": best_seed})).hexdigest()
    schema = {"task_kind": "expert_probability", "shape": [None, 6], "authority": 0}
    package = build_package(output, "CAND-S-001", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(schema)).hexdigest(), engine_id=1)
    pass_count = sum(item["beats_baseline"] for item in reports)
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if pass_count == 3 else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    write_json(output / "eval_contract_exact.json", {"schema": "cimc.forge200.s001-evaluation.v2", "status": "PASS" if pass_count == 3 else "FAIL_CLOSED", "candidate_id": "CAND-S-001", "baseline": baseline, "baseline_test_objective": baseline_test_objective, "seed_reports": reports, "three_seed_baseline_pass": pass_count, "authority": 0, "board_accepted": False})
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist()})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "max_abs_probability_error": float(np.max(np.abs(fp - quant)))})
    receipt = {"schema": "cimc.forge200.promotion-receipt.v2", "status": status, "candidate_id": "CAND-S-001", "authority": 0, "board_accepted": False, "countable_model": False, "three_seed_count": 3, "three_seed_baseline_pass": pass_count, "parameter_count": parameter_count, "parameter_cap": 42_000, "w8_payload_bytes": len(payload), "best_seed": best_seed, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3}}
    write_json(output / "promotion_receipt.json", receipt)
    print(json.dumps({"candidate_id": "CAND-S-001", "status": status, "passes": pass_count, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
