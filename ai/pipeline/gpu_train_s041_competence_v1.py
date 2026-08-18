#!/usr/bin/env python3
"""Train the post-GPU S041 competence estimator with model-family holdout."""

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

from gpu_train_job import build_package, canonical_bytes, heartbeat, sha256_file, write_json

SEEDS = [20260801, 20260802, 20260803]


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(probability.reshape(-1), 1e-6, 1 - 1e-6)
    count = max(1, int(math.ceil(len(y) * 0.8)))
    selected = np.argsort(-probability)[:count]
    selective_risk = float(np.mean(1 - y[selected]))
    oracle_errors = max(0, count - int(np.sum(y)))
    oracle_risk = oracle_errors / count
    ece = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return {
        "brier": float(np.mean((probability - y) ** 2)),
        "selective_risk_at_80pct_coverage": selective_risk,
        "oracle_risk_at_80pct_coverage": oracle_risk,
        "regret_vs_oracle": selective_risk - oracle_risk,
        "ece_10bin": ece,
    }


def objective(value: dict[str, float], baseline: dict[str, float]) -> float:
    return -float(np.mean([
        value["brier"] / max(baseline["brier"], 1e-9),
        value["selective_risk_at_80pct_coverage"] / max(baseline["selective_risk_at_80pct_coverage"], 1e-3),
        value["regret_vs_oracle"] / max(baseline["regret_vs_oracle"], 1e-3),
    ]))


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
    dataset = root / "data" / "staged_postgpu" / "CAND-S-041.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or sha256_file(dataset) != metadata["sha256"] or metadata["cross_split_group_overlap"] != 0:
        raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset, allow_pickle=False)
    x = raw["x"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    split = raw["split"].astype(np.int8)
    baseline_probability = raw["baseline_probability"].astype(np.float32)
    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    mean, std = x[indices["train"]].mean(axis=0), x[indices["train"]].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x - mean) / std, -10.0, 10.0).astype(np.float32)
    baseline = {name: metrics(y[selected], baseline_probability[selected]) for name, selected in indices.items()}
    baseline_validation_objective = objective(baseline["validation"], baseline["validation"])
    baseline_test_objective = objective(baseline["test"], baseline["test"])

    class CompetenceMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))
        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(p.numel() for p in CompetenceMLP().parameters())
    if parameter_count > 48_000:
        raise RuntimeError("PARAMETER_CAP")
    output = (args.artifact_root / "CAND-S-041").resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"

    def infer(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(x[selected]).to(device))).cpu().numpy().reshape(-1)

    reports, states = [], {}
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CompetenceMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=3e-4)
        best_score, patience = -math.inf, 0
        best_path = output / f"train_seed_{seed}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(seed)
        train = indices["train"]
        positive_weight = float(np.sum(y[train] == 0) / max(np.sum(y[train] == 1), 1))
        for epoch in range(180):
            order = generator.permutation(train)
            model.train()
            for start in range(0, len(order), 64):
                chosen = order[start:start+64]
                logits = model(torch.from_numpy(x[chosen]).to(device)).reshape(-1)
                target = torch.from_numpy(y[chosen]).to(device)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(positive_weight, device=device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            validation = metrics(y[indices["validation"]], infer(model, indices["validation"]))
            value = objective(validation, baseline["validation"])
            if value > best_score + 1e-6:
                best_score, patience = value, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, "CAND-S-041", "TRAIN_POSTGPU", seed, epoch)
            if epoch >= 39 and patience >= 20:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics = metrics(y[indices["test"]], infer(model, indices["test"]))
        test_objective = objective(test_metrics, baseline["test"])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_objective": best_score, "test_objective": test_objective, "baseline_test_objective": baseline_test_objective, "beats_baseline": test_objective > baseline_test_objective + 1e-4, "test": test_metrics})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    best = max(reports, key=lambda item: item["validation_objective"])
    best_seed = int(best["seed"])
    state = states[best_seed]
    model = CompetenceMLP().to(device)
    model.load_state_dict(state)
    sample = x[indices["test"][:32]]
    fp = infer(model, indices["test"][:32])
    q, s = quantize(state)
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **q, **{f"scale::{name}": value for name, value in s.items()})
    payload = payload_buffer.getvalue()
    dequantized = {name: torch.from_numpy(np.asarray(np.asarray(value, dtype=np.float32) * np.asarray(s[name], dtype=np.float32), dtype=np.float32)) for name, value in q.items()}
    model.load_state_dict(dequantized)
    quant = infer(model, indices["test"][:32])
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=sample, y=y[indices["test"][:32]], fp32=fp, quantized=quant)
    model.load_state_dict(state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(sample[:1]).to(device), onnx_path, input_names=["model_state"], output_names=["pass_logit"], dynamic_axes={"model_state": {0: "batch"}, "pass_logit": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": "CAND-S-041", "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "best_seed": best_seed})).hexdigest()
    schema = {"task_kind": "binary_probability", "shape": [None, 1], "authority": 0}
    package = build_package(output, "CAND-S-041", payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(schema)).hexdigest(), engine_id=1)
    pass_count = sum(item["beats_baseline"] for item in reports)
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if pass_count == 3 else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {"schema": "cimc.forge200.s041-evaluation.v1", "status": "PASS" if pass_count == 3 else "FAIL_CLOSED", "candidate_id": "CAND-S-041", "baseline": baseline, "baseline_test_objective": baseline_test_objective, "seed_reports": reports, "three_seed_baseline_pass": pass_count, "authority": 0, "board_accepted": False}
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist()})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "max_abs_probability_error": float(np.max(np.abs(fp-quant)))})
    receipt = {"schema": "cimc.forge200.promotion-receipt.v2", "status": status, "candidate_id": "CAND-S-041", "authority": 0, "board_accepted": False, "countable_model": False, "three_seed_count": 3, "three_seed_baseline_pass": pass_count, "parameter_count": parameter_count, "parameter_cap": 48_000, "w8_payload_bytes": len(payload), "best_seed": best_seed, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter()-started, "gpu": {"name": props.name, "vram_gib": props.total_memory/1024**3}}
    write_json(output / "promotion_receipt.json", receipt)
    print(json.dumps({"candidate_id": "CAND-S-041", "status": status, "passes": pass_count, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))


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
