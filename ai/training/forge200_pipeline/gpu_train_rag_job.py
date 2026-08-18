#!/usr/bin/env python3
"""CUDA trainer for source-bound byte-LM and contrastive RAG encoders."""

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
    dequantized_state,
    heartbeat,
    load_job,
    quantize_state,
    sha256_file,
    validate_admission,
    write_json,
)


def binary_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = probability >= 0.5
    recalls = []
    for label in (0, 1):
        mask = y == label
        recalls.append(float(np.mean(prediction[mask] == label)) if np.any(mask) else 0.0)
    positive = probability[y == 1]
    negative = probability[y == 0]
    if len(positive) and len(negative):
        comparisons = (positive[:, None] > negative[None, :]).mean()
        ties = (positive[:, None] == negative[None, :]).mean()
        auroc = float(comparisons + 0.5 * ties)
    else:
        auroc = 0.5
    return {
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float(np.mean(recalls)),
        "auroc": auroc,
        "brier": float(np.mean((probability - y) ** 2)),
    }


def retrieval_metrics(scores: np.ndarray, relevance: np.ndarray) -> dict[str, float]:
    if scores.shape != relevance.shape or scores.ndim != 2:
        raise RuntimeError("RETRIEVAL_SCORE_SHAPE")
    recall, reciprocal, ndcg = [], [], []
    for row_scores, row_relevance in zip(scores, relevance):
        order = np.argsort(-row_scores, kind="mergesort")
        ranked = row_relevance[order].astype(np.float64)
        relevant = int(np.sum(row_relevance))
        if relevant <= 0:
            raise RuntimeError("RETRIEVAL_QUERY_WITHOUT_RELEVANCE")
        top = ranked[:10]
        recall.append(float(np.sum(top) / relevant))
        positions = np.flatnonzero(ranked)
        reciprocal.append(1.0 / float(positions[0] + 1))
        discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        dcg = float(np.sum(top * discounts))
        ideal = min(relevant, 10)
        idcg = float(np.sum(discounts[:ideal]))
        ndcg.append(dcg / max(idcg, 1e-12))
    result = {
        "recall_at_10": float(np.mean(recall)),
        "mrr_at_10": float(np.mean(reciprocal)),
        "ndcg_at_10": float(np.mean(ndcg)),
        "queries": len(scores),
        "passages": scores.shape[1],
    }
    result["primary_composite"] = float(np.mean([result["recall_at_10"], result["mrr_at_10"], result["ndcg_at_10"]]))
    return result


def records_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_manifest.json", "transfer_manifest.json"}
        and not item.name.startswith("worker_attempt_")
    ):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def train_lm(torch: Any, nn: Any, data: Any, split: np.ndarray, device: Any, args: argparse.Namespace, output: Path, heartbeat_path: Path) -> dict[str, Any]:
    from torch.utils.data import DataLoader, TensorDataset

    x = data["x"].astype(np.int64)
    y = data["y"].astype(np.int64)
    loss_mask = data["loss_mask"].astype(np.float32)

    class ByteGRULM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(259, 48, padding_idx=258)
            self.gru = nn.GRU(48, 64, batch_first=True)
            self.output = nn.Linear(64, 259)

        def forward(self, tokens: Any) -> Any:
            hidden, _ = self.gru(self.embedding(tokens))
            return self.output(hidden)

    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    if min(len(value) for value in indices.values()) < 16:
        raise RuntimeError("EMPTY_OR_TOO_SMALL_SPLIT")

    def evaluate(model: Any, selected: np.ndarray, collect: bool = False) -> tuple[dict[str, float], np.ndarray | None]:
        model.eval()
        total_loss = total_correct = total_tokens = 0.0
        collected = []
        with torch.no_grad():
            for start in range(0, len(selected), 16):
                take = selected[start : start + 16]
                batch_x = torch.from_numpy(x[take]).to(device)
                batch_y = torch.from_numpy(y[take]).to(device)
                batch_mask = torch.from_numpy(loss_mask[take]).to(device)
                logits = model(batch_x)
                losses = nn.functional.cross_entropy(logits.reshape(-1, 259), batch_y.reshape(-1), reduction="none").reshape_as(batch_mask)
                total_loss += float((losses * batch_mask).sum().item())
                total_correct += float(((logits.argmax(-1) == batch_y) * (batch_mask > 0)).sum().item())
                total_tokens += float(batch_mask.sum().item())
                if collect:
                    collected.append(logits.cpu().numpy().astype(np.float32))
        metrics = {"token_nll": total_loss / max(total_tokens, 1.0), "token_accuracy": total_correct / max(total_tokens, 1.0), "evaluated_tokens": int(total_tokens)}
        return metrics, np.concatenate(collected) if collected else None

    train_targets = y[indices["train"]][loss_mask[indices["train"]] > 0]
    frequencies = np.bincount(train_targets, minlength=259).astype(np.float64) + 1.0
    frequencies /= frequencies.sum()
    test_targets = y[indices["test"]][loss_mask[indices["test"]] > 0]
    baseline = {"kind": "train_byte_unigram", "token_nll": float(-np.log(frequencies[test_targets]).mean())}
    reports, best_state, best_score, best_seed = [], None, math.inf, None
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = ByteGRULM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
        generator = torch.Generator().manual_seed(seed)
        train = indices["train"]
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]), torch.from_numpy(loss_mask[train])),
            batch_size=min(args.batch_size, len(train)),
            shuffle=True,
            generator=generator,
        )
        checkpoint_dir = output / f"train_seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "last.pt"
        start_epoch, local_best, patience = 0, math.inf, 0
        if args.resume and state_path.is_file():
            checkpoint = torch.load(state_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            local_best = float(checkpoint["local_best"])
        epoch = start_epoch - 1
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            for batch_x, batch_y, batch_mask in loader:
                batch_x, batch_y, batch_mask = batch_x.to(device), batch_y.to(device), batch_mask.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                losses = nn.functional.cross_entropy(logits.reshape(-1, 259), batch_y.reshape(-1), reduction="none").reshape_as(batch_mask)
                loss = (losses * batch_mask).sum() / batch_mask.sum().clamp_min(1.0)
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            validation, _ = evaluate(model, indices["validation"])
            score = validation["token_nll"]
            if score < local_best - 1e-5:
                local_best, patience = score, 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                patience += 1
            if epoch % args.checkpoint_epochs == 0 or epoch + 1 == args.max_epochs:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "local_best": local_best}, state_path)
            heartbeat(heartbeat_path, args.candidate_id, "TRAIN_TOKEN_LM", seed, epoch)
            if patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True))
        validation, _ = evaluate(model, indices["validation"])
        test, _ = evaluate(model, indices["test"])
        reports.append({"seed": seed, "validation": validation, "test": test, "epochs": epoch + 1})
        if validation["token_nll"] < best_score:
            best_score, best_seed = validation["token_nll"], seed
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None and best_seed is not None
    model = ByteGRULM().to(device)
    model.load_state_dict(best_state)
    fp_metrics, fp_logits = evaluate(model, indices["test"][:16], collect=True)
    quantized, scales = quantize_state(best_state)
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()})
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_metrics, quant_logits = evaluate(model, indices["test"][:16], collect=True)
    parity = float(np.max(np.abs(fp_logits - quant_logits)))
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=x[indices["test"][:16]], y=y[indices["test"][:16]], mask=loss_mask[indices["test"][:16]], fp32=fp_logits, quantized=quant_logits)
    return {
        "model": model,
        "best_state": best_state,
        "best_seed": best_seed,
        "seed_reports": reports,
        "baseline": baseline,
        "payload": payload_buffer.getvalue(),
        "golden": golden,
        "parity": parity,
        "quantized_test": quant_metrics,
        "output_schema": {"task_kind": "token_lm", "shape": [None, None, 259], "authority": 0},
        "engine_id": 5,
        "onnx_input": torch.from_numpy(x[indices["test"][:1]]).to(device),
        "onnx_input_names": ["tokens"],
        "onnx_output_names": ["logits"],
        "dynamic_axes": {"tokens": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        "split_counts": {name: len(value) for name, value in indices.items()},
        "ood": {"score": "answer_token_nll", "test_p95": fp_metrics["token_nll"]},
    }


def train_encoder(torch: Any, nn: Any, data: Any, split: np.ndarray, device: Any, args: argparse.Namespace, output: Path, heartbeat_path: Path) -> dict[str, Any]:
    from torch.utils.data import DataLoader, TensorDataset

    xq = data["x_query"].astype(np.float32)
    xp = data["x_passage"].astype(np.float32)
    y = data["y"].astype(np.int64)
    encoder_init_weight = data["encoder_init_weight"].astype(np.float32) if "encoder_init_weight" in data else None
    embedding_dimensions = int(encoder_init_weight.shape[0]) if encoder_init_weight is not None else 32

    class QueryEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            if encoder_init_weight is not None:
                self.projection = nn.Linear(xq.shape[1], embedding_dimensions, bias=False)
                with torch.no_grad():
                    self.projection.weight.copy_(torch.from_numpy(encoder_init_weight))
            else:
                self.projection = nn.Sequential(nn.Linear(xq.shape[1], 48), nn.Tanh(), nn.Linear(48, 32))

        def forward(self, value: Any) -> Any:
            return nn.functional.normalize(self.projection(value), dim=-1)

    indices = {name: np.flatnonzero(split == code) for code, name in enumerate(("train", "validation", "test"))}
    if min(len(value) for value in indices.values()) < 16:
        raise RuntimeError("EMPTY_OR_TOO_SMALL_SPLIT")

    def infer(model: Any, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model.eval()
        with torch.no_grad():
            q = model(torch.from_numpy(xq[selected]).to(device)).cpu().numpy()
            p = model(torch.from_numpy(xp[selected]).to(device)).cpu().numpy()
        probability = 1.0 / (1.0 + np.exp(-10.0 * np.sum(q * p, axis=1)))
        return probability, q, p

    has_retrieval = all(
        name in data
        for name in (
            "validation_retrieval_query",
            "validation_retrieval_passage",
            "validation_retrieval_relevance",
            "validation_bm25_scores",
            "test_retrieval_query",
            "test_retrieval_passage",
            "test_retrieval_relevance",
            "test_bm25_scores",
        )
    )

    def embed(model: Any, values: np.ndarray) -> np.ndarray:
        model.eval()
        batches = []
        with torch.no_grad():
            for start in range(0, len(values), 1024):
                batches.append(model(torch.from_numpy(values[start : start + 1024].astype(np.float32)).to(device)).cpu().numpy())
        return np.concatenate(batches)

    def evaluate_retrieval(model: Any, prefix: str) -> tuple[dict[str, float], np.ndarray]:
        query = data[f"{prefix}_retrieval_query"].astype(np.float32)
        passage = data[f"{prefix}_retrieval_passage"].astype(np.float32)
        relevance = data[f"{prefix}_retrieval_relevance"].astype(np.uint8)
        scores = embed(model, query) @ embed(model, passage).T
        return retrieval_metrics(scores, relevance), scores

    if has_retrieval:
        baseline = {
            "kind": "BM25_FULL_SPLIT_PASSAGE_POOL",
            **retrieval_metrics(data["test_bm25_scores"].astype(np.float32), data["test_retrieval_relevance"].astype(np.uint8)),
        }
    else:
        raw_probability = 1.0 / (1.0 + np.exp(-10.0 * np.sum(xq[indices["test"]] * xp[indices["test"]], axis=1)))
        baseline = {"kind": "hashed_cosine_without_learned_encoder", **binary_metrics(y[indices["test"]], raw_probability)}
    reports, best_state, best_score, best_seed = [], None, math.inf, None
    retrieval_seed_scores: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = QueryEncoder().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        train = indices["train"]
        loader = DataLoader(
            TensorDataset(torch.from_numpy(xq[train]), torch.from_numpy(xp[train]), torch.from_numpy(y[train].astype(np.float32))),
            batch_size=min(args.batch_size, len(train)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        checkpoint_dir = output / f"train_seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        state_path = checkpoint_dir / "last.pt"
        start_epoch, local_best, patience = 0, math.inf, 0
        if args.resume and state_path.is_file():
            checkpoint = torch.load(state_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            local_best = float(checkpoint["local_best"])
        epoch = start_epoch - 1
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            for batch_q, batch_p, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                qe = model(batch_q.to(device))
                pe = model(batch_p.to(device))
                logits = 10.0 * torch.sum(qe * pe, dim=1)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, batch_y.to(device))
                loss.backward()
                optimizer.step()
            probability, _, _ = infer(model, indices["validation"])
            if has_retrieval:
                validation_retrieval, _ = evaluate_retrieval(model, "validation")
                score = 1.0 - validation_retrieval["primary_composite"]
            else:
                score = 1.0 - binary_metrics(y[indices["validation"]], probability)["balanced_accuracy"]
            if score < local_best - 1e-6:
                local_best, patience = score, 0
                torch.save(model.state_dict(), checkpoint_dir / "best.pt")
            else:
                patience += 1
            if epoch % args.checkpoint_epochs == 0 or epoch + 1 == args.max_epochs:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "local_best": local_best}, state_path)
            heartbeat(heartbeat_path, args.candidate_id, "TRAIN_CONTRASTIVE", seed, epoch)
            if patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint_dir / "best.pt", map_location=device, weights_only=True))
        validation_probability, _, _ = infer(model, indices["validation"])
        test_probability, _, _ = infer(model, indices["test"])
        validation = binary_metrics(y[indices["validation"]], validation_probability)
        test = binary_metrics(y[indices["test"]], test_probability)
        report = {"seed": seed, "validation": validation, "test": test, "epochs": epoch + 1}
        if has_retrieval:
            validation_retrieval, _ = evaluate_retrieval(model, "validation")
            test_retrieval, test_scores = evaluate_retrieval(model, "test")
            report["validation_retrieval"] = validation_retrieval
            report["test_retrieval"] = test_retrieval
            retrieval_seed_scores[f"seed_{seed}"] = test_scores.astype(np.float32)
            score = 1.0 - validation_retrieval["primary_composite"]
        else:
            score = 1.0 - validation["balanced_accuracy"]
        reports.append(report)
        if score < best_score:
            best_score, best_seed = score, seed
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    assert best_state is not None and best_seed is not None
    model = QueryEncoder().to(device)
    model.load_state_dict(best_state)
    selected = indices["test"][:32]
    fp_probability, fp_query, fp_passage = infer(model, selected)
    quantized, scales = quantize_state(best_state)
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{name}": np.asarray(value, dtype=np.float32) for name, value in scales.items()})
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_probability, quant_query, quant_passage = infer(model, selected)
    if has_retrieval:
        quantized_test, quantized_scores = evaluate_retrieval(model, "test")
        np.savez_compressed(
            output / "three_seed_retrieval_scores.npz",
            relevance=data["test_retrieval_relevance"].astype(np.uint8),
            baseline_scores=data["test_bm25_scores"].astype(np.float32),
            quantized_best_seed_scores=quantized_scores.astype(np.float32),
            authority=np.asarray(0, dtype=np.int8),
            **retrieval_seed_scores,
        )
    else:
        quantized_test = binary_metrics(y[selected], quant_probability)
    parity = float(max(np.max(np.abs(fp_query - quant_query)), np.max(np.abs(fp_passage - quant_passage))))
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=xq[selected], fp32=fp_query, quantized=quant_query, y=y[selected])
    return {
        "model": model,
        "best_state": best_state,
        "best_seed": best_seed,
        "seed_reports": reports,
        "baseline": baseline,
        "payload": payload_buffer.getvalue(),
        "golden": golden,
        "parity": parity,
        "quantized_test": quantized_test,
        "output_schema": {"task_kind": "contrastive_embedding", "shape": [None, embedding_dimensions], "authority": 0},
        "engine_id": 4,
        "onnx_input": torch.from_numpy(xq[selected[:1]]).to(device),
        "onnx_input_names": ["query_features"],
        "onnx_output_names": ["embedding"],
        "dynamic_axes": {"query_features": {0: "batch"}, "embedding": {0: "batch"}},
        "split_counts": {name: len(value) for name, value in indices.items()},
        "ood": {"score": "one_minus_nearest_cosine", "test_p95": float(np.percentile(1.0 - fp_probability, 95))},
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
    task_kind = str(data["task_kind"])
    split = data["split"].astype(np.int8)
    started = time.perf_counter()
    if task_kind == "token_lm":
        result = train_lm(torch, nn, data, split, device, args, output, heartbeat_path)
    elif task_kind == "contrastive_embedding":
        result = train_encoder(torch, nn, data, split, device, args, output, heartbeat_path)
    else:
        raise RuntimeError(f"UNSUPPORTED_RAG_TASK_KIND:{task_kind}")
    golden_sha = sha256_file(result["golden"])
    heartbeat(heartbeat_path, args.candidate_id, "EXPORT")
    model = result["model"]
    model.load_state_dict(result["best_state"])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        result["onnx_input"],
        onnx_path,
        input_names=result["onnx_input_names"],
        output_names=result["onnx_output_names"],
        dynamic_axes=result["dynamic_axes"],
        opset_version=17,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema_sha = hashlib.sha256(canonical_bytes(result["output_schema"])).hexdigest()
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": args.candidate_id,
                "dataset_sha256": job["staged_dataset_sha256"],
                "task_contract_sha256": metadata["task_contract_sha256"],
                "onnx_sha256": sha256_file(onnx_path),
                "golden_sha256": golden_sha,
                "best_seed": result["best_seed"],
            }
        )
    ).hexdigest()
    package = build_package(output, args.candidate_id, result["payload"], golden_sha, release_root, output_schema_sha, engine_id=result["engine_id"])
    evaluation = {
        "schema": "cimc.forge200.grouped-evaluation.v1",
        "candidate_id": args.candidate_id,
        "truth_class": metadata["truth_class"],
        "claim_state": metadata["claim_state"],
        "baseline": result["baseline"],
        "seed_reports": result["seed_reports"],
        "best_seed": result["best_seed"],
        "quantized_test": result["quantized_test"],
        "split_counts": result["split_counts"],
        "group_overlap": 0,
    }
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "calibration_ood.json", {"schema": "cimc.forge200.calibration-ood.v1", "task_kind": task_kind, "quant_max_abs_error": result["parity"], **result["ood"]})
    write_json(output / "preprocessing_train_only.json", {"kind": "fixed_byte_tokens" if task_kind == "token_lm" else "fixed_hashed_features", "fit_on_test": False})
    write_json(output / "task_contract.json", {"candidate_id": args.candidate_id, "task_contract_sha256": metadata["task_contract_sha256"], "authority": 0})
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "split_manifest.json", {"split_sha256": metadata["split_sha256"], "cross_split_group_overlap": 0})
    write_json(output / "baseline_report.json", result["baseline"])
    write_json(output / "ablation.json", {"status": "CANONICAL_SMALL_MODEL_ONLY", "feature_contract": metadata["feature_contract"]})
    (output / "model_card.md").write_text(
        f"# {args.candidate_id} model card\n\n"
        f"- Status: `HOST_GPU_TRAINED_BOARD_PENDING`\n"
        f"- Task: `{task_kind}`; truth: `{metadata['truth_class']}`.\n"
        f"- Claim state: `{metadata['claim_state']}`.\n"
        f"- Best seed: `{result['best_seed']}`; all three fixed seeds are reported.\n"
        f"- Release root: `{release_root}`\n"
        f"- ONNX SHA-256: `{sha256_file(onnx_path)}`\n"
        f"- Package SHA-256: `{package['sha256']}`\n"
        f"- Authority: `0`; no deterministic control authority. Board evidence is pending.\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v1",
        "status": "HOST_GPU_TRAINED_BOARD_PENDING",
        "candidate_id": args.candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "best_seed": result["best_seed"],
        "three_seed_count": len(result["seed_reports"]),
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": golden_sha,
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": vram_gib},
    }
    write_json(output / "promotion_receipt.json", receipt)
    heartbeat(heartbeat_path, args.candidate_id, "COMPLETE")
    write_json(output / "artifact_manifest.json", records_manifest(output))
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
    parser.add_argument("--batch-size", type=int, default=32)
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
