#!/usr/bin/env python3
"""Train 50-way retrieval encoders against the frozen BM25 contract baseline."""

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

from gpu_train_job import (
    SEEDS,
    build_package,
    canonical_bytes,
    heartbeat,
    sha256_file,
    write_json,
)


CAPS = {
    "CAND-S-009": (64_000, 80 * 1024),
    "CAND-S-010": (64_000, 80 * 1024),
    "CAND-S-011": (68_000, 84 * 1024),
    "CAND-S-012": (64_000, 80 * 1024),
    "CAND-S-013": (64_000, 80 * 1024),
    "CAND-S-014": (64_000, 80 * 1024),
    "CAND-S-029": (192_000, 192 * 1024),
}


def ranking_metrics(scores: np.ndarray, labels: np.ndarray, domains: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    ranked = np.take_along_axis(labels, order, axis=1)
    rank = np.argmax(ranked, axis=1) + 1
    discount = 1.0 / np.log2(np.arange(2, labels.shape[1] + 2))
    per_domain = {
        int(domain): float(np.mean(rank[domains == domain] <= 20))
        for domain in np.unique(domains)
    }
    return {
        "mrr_at_10": float(np.mean(np.where(rank <= 10, 1.0 / rank, 0.0))),
        "recall_at_10": float(np.mean(rank <= 10)),
        "recall_at_20": float(np.mean(rank <= 20)),
        "ndcg_at_10": float(np.mean(np.sum(ranked[:, :10] * discount[:10], axis=1))),
        "worst_domain_recall_at_20": min(per_domain.values()),
        "domain_recall_at_20": per_domain,
    }


def score(candidate_id: str, metrics: dict[str, Any]) -> float:
    if candidate_id == "CAND-S-029":
        return float(np.mean([metrics["mrr_at_10"], metrics["recall_at_20"], metrics["worst_domain_recall_at_20"]]))
    return float(np.mean([metrics["recall_at_10"], metrics["mrr_at_10"], metrics["ndcg_at_10"]]))


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
    return {
        name: torch.from_numpy(
            np.asarray(
                np.asarray(value, dtype=np.float32)
                * np.asarray(scales[name], dtype=np.float32),
                dtype=np.float32,
            )
        )
        for name, value in quantized.items()
    }


def records_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "heartbeat.json"}:
            records.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    candidate_id = args.candidate_id
    if candidate_id not in CAPS:
        raise RuntimeError("UNSUPPORTED_ENCODER")
    root = args.root.resolve()
    dataset_path = root / "data" / "staged_rag_contract_v2" / f"{candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or metadata["authority"] != 0 or metadata["cross_split_group_overlap"] != 0:
        raise RuntimeError("DATA_GATE")
    if sha256_file(dataset_path) != metadata["sha256"]:
        raise RuntimeError("DATA_HASH")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    raw = np.load(dataset_path, allow_pickle=False)
    xq = raw["x_query"].astype(np.float32)
    xp = raw["x_passage"].astype(np.float32)
    y = raw["y"].astype(np.int64)
    split = raw["split"].astype(np.int8)
    domains = raw["domain_id"].astype(np.int8)
    baseline_scores = raw["baseline_score"].astype(np.float32)
    if len(y) % 50 or not np.all(y.reshape(-1, 50).sum(axis=1) == 1):
        raise RuntimeError("TOP50_GROUP_CONTRACT")
    query_split = split.reshape(-1, 50)[:, 0]
    query_domains = domains.reshape(-1, 50)[:, 0]
    labels = y.reshape(-1, 50)
    grouped_baseline = baseline_scores.reshape(-1, 50)
    query_features = xq.reshape(-1, 50, xq.shape[1])[:, 0]
    passage_features = xp.reshape(-1, 50, xp.shape[1])
    indices = {name: np.flatnonzero(query_split == code) for code, name in enumerate(("train", "validation", "test"))}
    baseline = {name: ranking_metrics(grouped_baseline[selected], labels[selected], query_domains[selected]) for name, selected in indices.items()}
    baseline_validation_score = score(candidate_id, baseline["validation"])
    baseline_test_score = score(candidate_id, baseline["test"])

    class ResidualEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Sequential(nn.Linear(256, 96), nn.GELU(), nn.Linear(96, 64))
            self.residual_scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, value: Any) -> Any:
            projected = self.projection(value)
            combined = torch.cat((value * torch.nn.functional.softplus(self.residual_scale), projected), dim=-1)
            return nn.functional.normalize(combined, dim=-1)

    parameter_count = sum(item.numel() for item in ResidualEncoder().parameters())
    parameter_cap, byte_cap = CAPS[candidate_id]
    if parameter_count > parameter_cap:
        raise RuntimeError("PARAMETER_CAP")
    output = (args.artifact_root / candidate_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    reports, states = [], {}
    started = time.perf_counter()

    def evaluate(model: Any, selected: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        model.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(selected), args.eval_batch_queries):
                chosen = selected[start : start + args.eval_batch_queries]
                q = model(torch.from_numpy(query_features[chosen]).to(device))
                p = model(torch.from_numpy(passage_features[chosen].reshape(-1, 256)).to(device)).reshape(len(chosen), 50, -1)
                scores.append(torch.einsum("bd,bkd->bk", q, p).cpu().numpy())
        value = np.concatenate(scores)
        return ranking_metrics(value, labels[selected], query_domains[selected]), value

    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ResidualEncoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        best_score, patience = -math.inf, 0
        best_path = output / f"train_seed_{seed}" / "best.pt"
        best_path.parent.mkdir(parents=True, exist_ok=True)
        generator = np.random.default_rng(seed)
        for epoch in range(args.max_epochs):
            model.train()
            order = generator.permutation(indices["train"])
            for start in range(0, len(order), args.batch_queries):
                chosen = order[start : start + args.batch_queries]
                q = model(torch.from_numpy(query_features[chosen]).to(device))
                p = model(torch.from_numpy(passage_features[chosen].reshape(-1, 256)).to(device)).reshape(len(chosen), 50, -1)
                logits = 14.0 * torch.einsum("bd,bkd->bk", q, p)
                target = torch.from_numpy(np.argmax(labels[chosen], axis=1)).to(device)
                loss = nn.functional.cross_entropy(logits, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            validation_metrics, _ = evaluate(model, indices["validation"])
            validation_score = score(candidate_id, validation_metrics)
            if validation_score > best_score + 1e-6:
                best_score, patience = validation_score, 0
                torch.save(model.state_dict(), best_path)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_RETRIEVAL_V2", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        test_metrics, _ = evaluate(model, indices["test"])
        test_score = score(candidate_id, test_metrics)
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_score": best_score, "test_score": test_score, "baseline_test_score": baseline_test_score, "beats_baseline": test_score > baseline_test_score + 1e-4, "test": test_metrics})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    best = max(reports, key=lambda item: item["validation_score"])
    best_seed = int(best["seed"])
    best_state = states[best_seed]
    model = ResidualEncoder().to(device)
    model.load_state_dict(best_state)
    sample = query_features[indices["test"][:64]]
    model.eval()
    with torch.no_grad():
        fp = model(torch.from_numpy(sample).to(device)).cpu().numpy()
    quantized, scales = quantize_state(best_state)
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{name}": value for name, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > byte_cap:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    with torch.no_grad():
        quant = model(torch.from_numpy(sample).to(device)).cpu().numpy()
    parity = float(np.max(np.abs(fp - quant)))
    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(golden_path, x=sample, fp32=fp, quantized=quant)
    model.load_state_dict(best_state)
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(sample[:1]).to(device), onnx_path, input_names=["features"], output_names=["embedding"], dynamic_axes={"features": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"task_kind": "retrieval_embedding", "shape": [None, 320], "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden_path), "best_seed": best_seed})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden_path), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest(), engine_id=4)
    pass_count = sum(item["beats_baseline"] for item in reports)
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if pass_count == 3 else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {"schema": "cimc.forge200.retrieval-contract-evaluation.v2", "status": "PASS" if pass_count == 3 else "FAIL_CLOSED", "candidate_id": candidate_id, "truth_class": metadata["truth_class"], "claim_state": metadata["claim_state"], "baseline": baseline, "baseline_test_score": baseline_test_score, "seed_reports": reports, "three_seed_baseline_pass": pass_count, "authority": 0, "board_accepted": False}
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "max_abs_embedding_error": parity})
    torch.save(best_state, output / "best_fp32_state.pt")
    receipt = {"schema": "cimc.forge200.promotion-receipt.v2", "status": status, "candidate_id": candidate_id, "authority": 0, "board_accepted": False, "countable_model": False, "three_seed_count": 3, "three_seed_baseline_pass": pass_count, "parameter_count": parameter_count, "parameter_cap": parameter_cap, "w8_payload_bytes": len(payload), "w8_payload_byte_cap": byte_cap, "best_seed": best_seed, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden_path), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3}}
    write_json(output / "promotion_receipt.json", receipt)
    (output / "model_card.md").write_text(f"# {candidate_id} retrieval encoder\n\n- Status: `{status}`\n- Three-seed BM25 baseline passes: `{pass_count}/3`.\n- Expanded corpus: `{metadata['source_documents']}` documents / `{metadata['source_chunks']}` chunks.\n- Authority: `0`; board evidence pending.\n", encoding="utf-8")
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "passes": pass_count, "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-queries", type=int, default=32)
    parser.add_argument("--eval-batch-queries", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=25)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        write_json(args.artifact_root.resolve() / args.candidate_id / "failure.json", {"schema": "cimc.forge200.job-failure.v2", "status": "FAIL_CLOSED", "candidate_id": args.candidate_id, "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
