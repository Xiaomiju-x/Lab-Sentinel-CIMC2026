#!/usr/bin/env python3
"""Train the S019 condition-aware linear reranker on RTX4050."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_reranker_exact_v1 import ranking_metrics
from gpu_train_job import SEEDS, build_package, canonical_bytes, dequantized_state, quantize_state, sha256_file, write_json


CID = "CAND-S-019"


def manifest(output: Path) -> dict[str, Any]:
    records = [{"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json")]
    return {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    root, output = args.root.resolve(), args.artifact_root.resolve() / CID; output.mkdir(parents=True, exist_ok=True)
    dataset_path = root / "data" / "staged_reranker_exact_v1" / f"{CID}.npz"; metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")); data = np.load(dataset_path, allow_pickle=False)
    if metadata["status"] != "PASS" or metadata["authority"] != 0 or sha256_file(dataset_path) != metadata["sha256"]:
        raise RuntimeError("S019_DATA_GATE")
    import torch
    from torch import nn
    if not torch.cuda.is_available(): raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device); props = torch.cuda.get_device_properties(device)
    x = np.column_stack((data["baseline_score"].astype(np.float32), data["special_match"].astype(np.float32)))
    y = data["y"].astype(np.float32); split = data["split"].astype(np.int8); qid = data["query_id"].astype(np.int64); special = data["special_match"].astype(np.uint8)
    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    baseline = ranking_metrics(CID, y[indices["test"]], x[indices["test"], 0], qid[indices["test"]], special[indices["test"]])

    class LinearReranker(nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__(); self.linear = nn.Linear(2, 1)
            generator = torch.Generator().manual_seed(seed)
            with torch.no_grad():
                self.linear.weight[:] = torch.tensor([[1.0, 0.0]]) + .01 * torch.randn((1, 2), generator=generator)
                self.linear.bias.zero_()
        def forward(self, value: Any) -> Any: return self.linear(value).squeeze(-1)

    def scores(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad(): return model(torch.from_numpy(x[selected]).to(device)).cpu().numpy()

    reports, states, best_state, best_seed, best_validation = [], {}, None, None, -1.0; started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); rng = np.random.default_rng(seed); model = LinearReranker(seed).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.03, weight_decay=1e-3)
        train = indices["train"]; positive = train[y[train] == 1]; pair_pos, pair_neg = [], []
        for pos in positive:
            negatives = train[(qid[train] == qid[pos]) & (y[train] == 0)]
            if len(negatives):
                for neg in rng.choice(negatives, size=min(4, len(negatives)), replace=False): pair_pos.append(pos); pair_neg.append(int(neg))
        pair_pos, pair_neg = np.asarray(pair_pos), np.asarray(pair_neg)
        local_best, local_state, patience = -1.0, None, 0
        for epoch in range(160):
            model.train(); optimizer.zero_grad(set_to_none=True); logits = model(torch.from_numpy(x[train]).to(device)); targets = torch.from_numpy(y[train]).to(device)
            positive_weight = torch.tensor(float(np.sum(y[train] == 0) / max(np.sum(y[train] == 1), 1)), device=device)
            bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=positive_weight)
            pos_score = model(torch.from_numpy(x[pair_pos]).to(device)); neg_score = model(torch.from_numpy(x[pair_neg]).to(device)); ranking = nn.functional.softplus(-(pos_score - neg_score)).mean()
            loss = bce + .7 * ranking; loss.backward(); optimizer.step()
            validation_scores = scores(model, indices["validation"]); evaluation = ranking_metrics(CID, y[indices["validation"]], validation_scores, qid[indices["validation"]], special[indices["validation"]])
            if evaluation["composite"] > local_best + 1e-7:
                local_best, patience = evaluation["composite"], 0; local_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            else: patience += 1
            if patience >= 20: break
        assert local_state is not None; model.load_state_dict(local_state); test_scores = scores(model, indices["test"]); test_metrics = ranking_metrics(CID, y[indices["test"]], test_scores, qid[indices["test"]], special[indices["test"]])
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_composite": local_best, "test": test_metrics}); states[f"seed_{seed}"] = test_scores.astype(np.float32)
        if local_best > best_validation: best_validation, best_seed, best_state = local_best, seed, local_state
    assert best_state is not None and best_seed is not None
    model = LinearReranker(best_seed).to(device); model.load_state_dict(best_state); fp_scores = scores(model, indices["test"])
    quantized, scales = quantize_state(best_state); buffer = io.BytesIO(); np.savez_compressed(buffer, **quantized, **{f"scale::{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()})
    model.load_state_dict(dequantized_state(torch, quantized, scales)); quant_scores = scores(model, indices["test"]); quant_metrics = ranking_metrics(CID, y[indices["test"]], quant_scores, qid[indices["test"]], special[indices["test"]])
    golden = output / "golden_vectors.npz"; take = indices["test"][:64]; np.savez_compressed(golden, x=x[take], y=y[take], fp32=fp_scores[:64], quantized=quant_scores[:64], authority=np.asarray(0, dtype=np.int8))
    np.savez_compressed(output / "three_seed_test_scores.npz", y=y[indices["test"]], query_id=qid[indices["test"]], special_match=special[indices["test"]], baseline_score=x[indices["test"], 0], quantized_best_seed=quant_scores, authority=np.asarray(0, dtype=np.int8), **states)
    model.load_state_dict(best_state); model.eval(); onnx_path = output / "fp32.onnx"; torch.onnx.export(model, torch.from_numpy(x[take[:1]]).to(device), onnx_path, input_names=["rerank_features"], output_names=["score"], dynamic_axes={"rerank_features": {0: "batch"}, "score": {0: "batch"}}, opset_version=17)
    import onnx; onnx.checker.check_model(onnx.load(onnx_path))
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": CID, "dataset_sha256": sha256_file(dataset_path), "best_seed": best_seed, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden)})).hexdigest()
    package = build_package(output, CID, buffer.getvalue(), sha256_file(golden), release_root, hashlib.sha256(canonical_bytes({"task_kind": "linear_reranker", "shape": [None, 1], "authority": 0})).hexdigest(), engine_id=1)
    composites = np.asarray([item["test"]["composite"] for item in reports]); mean_pass = float(np.mean(composites)) > baseline["composite"] + 1e-6; best_report = next(item for item in reports if item["seed"] == best_seed); quant_delta = best_report["test"]["composite"] - quant_metrics["composite"]; quant_pass = quant_delta <= .02
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if mean_pass and quant_pass else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {"schema": "cimc.forge200.s019-linear-reranker-evaluation.v1", "status": status, "candidate_id": CID, "baseline": baseline, "seed_reports": reports, "three_seed_mean_composite": float(np.mean(composites)), "three_seed_variance_composite": float(np.var(composites)), "three_seed_worst_composite": float(np.min(composites)), "aggregate_mean_beats_preregistered_baseline": mean_pass, "quantized_best_seed": quant_metrics, "quantized_best_seed_metric_delta": quant_delta, "quantization_pass": quant_pass, "authority": 0, "board_accepted": False, "countable_model": False}
    write_json(output / "contract_exact_evaluation.v1.json", evaluation); write_json(output / "eval_grouped.json", evaluation); write_json(output / "source_manifest.json", metadata); write_json(output / "split_manifest.json", {"split_sha256": metadata["split_sha256"], "cross_split_group_overlap": 0}); write_json(output / "output_schema.json", {"task_kind": "linear_reranker", "shape": [None, 1], "authority": 0})
    (output / "model_card.md").write_text(f"# {CID} condition-aware linear reranker\n\n- Status: `{status}`.\n- Three-seed mean `{float(np.mean(composites)):.6f}` vs baseline `{baseline['composite']:.6f}`.\n- Authority `0`; board acceptance pending.\n", encoding="utf-8")
    promotion = {"schema": "cimc.forge200.promotion-receipt.v1", "status": status, "candidate_id": CID, "authority": 0, "board_accepted": False, "countable_model": False, "exact_contract_baseline_pending": not status.endswith("PASS_BOARD_PENDING"), "best_seed": best_seed, "three_seed_count": 3, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / (1024**3)}}
    write_json(output / "promotion_receipt.json", promotion); write_json(output / "artifact_manifest.json", manifest(output)); print(json.dumps({"status": status, "mean": float(np.mean(composites)), "baseline": baseline["composite"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True)); return 0 if status.endswith("PASS_BOARD_PENDING") else 2


if __name__ == "__main__": raise SystemExit(main())
