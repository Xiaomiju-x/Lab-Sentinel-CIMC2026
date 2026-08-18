#!/usr/bin/env python3
"""Corrective Qwen-distilled nano-transformer trainer for CAND-G-001..026.

The first GPU pass is retained as evidence but used a 51k-parameter byte GRU
and did not satisfy the frozen nano-LM or task-metric contracts.  This job
trains the 0.4--1.8M W8 route, consumes train-only 4B/1.7B teacher caches, and
emits fail-closed host artifacts.  Exact legacy-baseline and board promotion
remain separate gates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import struct
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_teacher_distillation_cache import ContractTokenizer
from gpu_train_job import (
    SEEDS,
    canonical_bytes,
    heartbeat,
    sha256_file,
    write_json,
)
from nanolm_architecture import (
    CONTEXT_TOKENS,
    MAX_GENERATION_TOKENS,
    build_model,
    config_for_candidate,
    parameter_count,
)


HEADER_BYTES = 256
ENGINE_ID = 5
W8_GROUP_SIZE = 32


def quantize_nanolm_state(
    state: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Symmetric W8 with group-32 scales for every matrix row."""

    quantized: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 2:
            rows, columns = array.shape
            groups = (columns + W8_GROUP_SIZE - 1) // W8_GROUP_SIZE
            padded = np.zeros((rows, groups * W8_GROUP_SIZE), dtype=np.float32)
            padded[:, :columns] = array
            grouped = padded.reshape(rows, groups, W8_GROUP_SIZE)
            scale = np.maximum(np.max(np.abs(grouped), axis=2), 1e-12) / 127.0
            quantized_grouped = np.clip(
                np.rint(grouped / scale[:, :, None]), -127, 127
            ).astype(np.int8)
            quantized[name] = quantized_grouped.reshape(rows, -1)[:, :columns]
            scales[name] = scale.astype(np.float32)
        else:
            scale_value = max(float(np.max(np.abs(array))), 1e-12) / 127.0
            quantized[name] = np.clip(
                np.rint(array / scale_value), -127, 127
            ).astype(np.int8)
            scales[name] = np.asarray(scale_value, dtype=np.float32)
    return quantized, scales


def dequantize_nanolm_state(
    torch: Any,
    quantized: dict[str, np.ndarray],
    scales: dict[str, np.ndarray],
) -> dict[str, Any]:
    state = {}
    for name, array in quantized.items():
        scale = scales[name]
        if array.ndim == 2:
            rows, columns = array.shape
            groups = scale.shape[1]
            padded = np.zeros((rows, groups * W8_GROUP_SIZE), dtype=np.float32)
            padded[:, :columns] = array.astype(np.float32)
            value = (
                padded.reshape(rows, groups, W8_GROUP_SIZE)
                * scale[:, :, None]
            ).reshape(rows, -1)[:, :columns]
        else:
            value = array.astype(np.float32) * float(scale)
        state[name] = torch.from_numpy(value)
    return state


def records_manifest(output: Path) -> dict[str, Any]:
    records = []
    for path in sorted(
        item
        for item in output.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_manifest.json", "transfer_manifest.json"}
        and not item.name.startswith("worker_attempt_")
    ):
        records.append(
            {
                "path": str(path.relative_to(output)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return {
        "schema": "cimc.forge200.artifact-manifest.v2",
        "records": records,
        "bytes": sum(item["bytes"] for item in records),
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }


def load_contract(root: Path, candidate_id: str) -> dict[str, str]:
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["candidate_id"] == candidate_id:
                return row
    raise RuntimeError(f"task contract missing: {candidate_id}")


def load_cache(
    cache_root: Path,
    candidate_id: str,
    dataset_sha256: str,
    split: np.ndarray,
    role: str,
) -> tuple[Any, dict[str, Any]]:
    directory = cache_root.resolve() / candidate_id
    receipt_path = directory / "receipt.json"
    cache_path = directory / "teacher_cache.npz"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_TRAIN_ONLY_TEACHER_CACHE":
        raise RuntimeError(f"{role}_CACHE_STATUS:{receipt.get('status')}")
    if receipt.get("dataset_sha256") != dataset_sha256:
        raise RuntimeError(f"{role}_CACHE_DATASET_HASH")
    if receipt.get("validation_records_seen") != 0 or receipt.get("test_records_seen") != 0:
        raise RuntimeError(f"{role}_CACHE_LEAKAGE")
    if receipt.get("teacher_promoted_to_ground_truth") is not False:
        raise RuntimeError(f"{role}_CACHE_TRUTH_PROMOTION")
    if sha256_file(cache_path) != receipt.get("cache_sha256"):
        raise RuntimeError(f"{role}_CACHE_HASH")
    cache = np.load(cache_path, allow_pickle=False)
    if int(cache["authority"]) != 0 or str(cache["truth_class"]) != "TEACHER_CANDIDATE":
        raise RuntimeError(f"{role}_CACHE_AUTHORITY_OR_TRUTH")
    processed = cache["processed_mask"].astype(bool)
    if not np.all(processed[split == 0]) or np.any(processed[split != 0]):
        raise RuntimeError(f"{role}_CACHE_TRAIN_ONLY_MASK")
    return cache, receipt


def sparse_teacher_loss(
    torch: Any,
    logits: Any,
    ids: Any,
    probabilities: Any,
    mask: Any,
    sample_weight: Any,
) -> Any:
    log_probability = torch.log_softmax(logits.float(), dim=-1)
    selected = torch.gather(log_probability, dim=-1, index=ids.long())
    position_loss = -(selected * probabilities.float()).sum(dim=-1)
    weighted_mask = mask.float() * sample_weight[:, None]
    return (position_loss * weighted_mask).sum() / weighted_mask.sum().clamp_min(1.0)


def token_f1(reference: list[int], prediction: list[int]) -> float:
    reference = [item for item in reference if item not in {0, 2}]
    prediction = [item for item in prediction if item not in {0, 2}]
    if not reference and not prediction:
        return 1.0
    if not reference or not prediction:
        return 0.0
    ref_counts: dict[int, int] = {}
    pred_counts: dict[int, int] = {}
    for item in reference:
        ref_counts[item] = ref_counts.get(item, 0) + 1
    for item in prediction:
        pred_counts[item] = pred_counts.get(item, 0) + 1
    overlap = sum(min(value, pred_counts.get(key, 0)) for key, value in ref_counts.items())
    precision = overlap / len(prediction)
    recall = overlap / len(reference)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def evaluation_subset(indices: np.ndarray, is_refusal: np.ndarray, limit: int = 120) -> np.ndarray:
    half = limit // 2
    positive = indices[is_refusal[indices] == 0][:half]
    negative = indices[is_refusal[indices] == 1][:half]
    selected = np.concatenate((positive, negative))
    return np.sort(selected)


def greedy_generate(
    torch: Any,
    model: Any,
    prompt_tokens: np.ndarray,
    prompt_lengths: np.ndarray,
    max_tokens: int = MAX_GENERATION_TOKENS,
) -> np.ndarray:
    """Generate from each example's real prompt boundary.

    ``prompt_tokens`` is a fixed-width ABI array padded on the right.  Treating
    the last padded column as the generation boundary silently changed both the
    causal position and the attended history.  Grouping equal real lengths
    keeps inference batched while matching the training boundary exactly.
    """
    model.eval()
    device = next(model.parameters()).device
    lengths = np.asarray(prompt_lengths, dtype=np.int64)
    if len(lengths) != len(prompt_tokens) or np.any(lengths <= 0):
        raise RuntimeError("PROMPT_LENGTH_CONTRACT")
    result = np.zeros((len(prompt_tokens), max_tokens), dtype=np.int64)
    with torch.inference_mode():
        for length in np.unique(lengths):
            selected = np.flatnonzero(lengths == length)
            tokens = torch.from_numpy(
                prompt_tokens[selected, : int(length)].astype(np.int64)
            ).to(device)
            generated = torch.zeros(
                (len(selected), max_tokens), dtype=torch.long, device=device
            )
            finished = torch.zeros(len(selected), dtype=torch.bool, device=device)
            for step in range(max_tokens):
                logits = model(tokens)
                next_token = logits[:, -1].argmax(dim=-1)
                next_token = torch.where(finished, torch.zeros_like(next_token), next_token)
                generated[:, step] = next_token
                finished = finished | (next_token == 2)
                tokens = torch.cat((tokens, next_token[:, None]), dim=1)
                if bool(finished.all()):
                    break
            result[selected] = generated.cpu().numpy()
    return result


def generation_metrics(
    tokenizer: ContractTokenizer,
    predictions: np.ndarray,
    targets: np.ndarray,
    target_lengths: np.ndarray,
    is_refusal: np.ndarray,
) -> dict[str, float]:
    f1_values = []
    positive_citation = []
    refusal_correct = []
    negative_unsupported = []
    exact = []
    grounded_claim = []
    for prediction, target, length, refusal in zip(
        predictions, targets, target_lengths, is_refusal
    ):
        predicted_ids = prediction.tolist()
        if 2 in predicted_ids:
            predicted_ids = predicted_ids[: predicted_ids.index(2) + 1]
        reference_ids = target[: int(length)].tolist()
        predicted_text = tokenizer.decode(predicted_ids).strip()
        reference_text = tokenizer.decode(reference_ids).strip()
        f1_values.append(token_f1(reference_ids, predicted_ids))
        exact.append(float(predicted_text == reference_text))
        predicted_refusal = predicted_text.startswith("REFUSE")
        refusal_correct.append(float(predicted_refusal == bool(refusal)))
        if not refusal:
            positive_citation.append(float("[1]" in predicted_text))
            def claim_words(value: str) -> list[str]:
                value = re.sub(r"^\[1\]\s*", "", value.strip())
                value = re.split(r"\s+(?:UNCERTAINTY|SOURCE_STATE):", value, maxsplit=1)[0]
                return re.findall(r"[A-Za-z0-9]+(?:[.+-][A-Za-z0-9]+)*", value.lower())

            reference_claim = claim_words(reference_text)
            predicted_claim = claim_words(predicted_text)
            ref_counter = {word: reference_claim.count(word) for word in set(reference_claim)}
            pred_counter = {word: predicted_claim.count(word) for word in set(predicted_claim)}
            overlap = sum(min(count, pred_counter.get(word, 0)) for word, count in ref_counter.items())
            precision = overlap / max(len(predicted_claim), 1)
            recall = overlap / max(len(reference_claim), 1)
            grounded_claim.append(2 * precision * recall / max(precision + recall, 1e-12))
        else:
            negative_unsupported.append(float(not predicted_refusal))
    token_score = float(np.mean(f1_values))
    citation = float(np.mean(positive_citation)) if positive_citation else 0.0
    refusal_accuracy = float(np.mean(refusal_correct))
    unsupported = float(np.mean(negative_unsupported)) if negative_unsupported else 1.0
    primary = 0.40 * token_score + 0.20 * citation + 0.25 * refusal_accuracy + 0.15 * (1.0 - unsupported)
    return {
        "answer_token_f1": token_score,
        "answer_exact": float(np.mean(exact)),
        "positive_citation_exact": citation,
        "grounded_claim_f1": float(np.mean(grounded_claim)) if grounded_claim else 0.0,
        "refusal_accuracy": refusal_accuracy,
        "unsupported_negative_rate": unsupported,
        "primary_composite": primary,
        "evaluated_examples": len(predictions),
    }


def evaluate_generation(
    torch: Any,
    model: Any,
    data: Any,
    tokenizer: ContractTokenizer,
    selected: np.ndarray,
    is_refusal: np.ndarray,
) -> dict[str, float]:
    """Evaluate the pre-registered autoregressive metric on one frozen split.

    Checkpoint selection must follow the generation contract, not teacher-forced
    NLL.  In particular, a checkpoint with slightly lower NLL can collapse to a
    non-refusing answer policy.  Only validation examples are used here; the test
    split remains untouched until the final three-seed report.
    """
    take = evaluation_subset(selected, is_refusal, limit=64)
    predictions = greedy_generate(
        torch,
        model,
        data["prompt_tokens"][take],
        data["prompt_length"][take],
        MAX_GENERATION_TOKENS,
    )
    return generation_metrics(
        tokenizer,
        predictions,
        data["target_tokens"][take],
        data["target_length"][take],
        is_refusal[take],
    )


def evaluate_nll(
    torch: Any,
    nn: Any,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    loss_mask: np.ndarray,
    selected: np.ndarray,
    device: Any,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_loss = total_correct = total_tokens = 0.0
    with torch.inference_mode():
        for start in range(0, len(selected), batch_size):
            take = selected[start : start + batch_size]
            batch_x = torch.from_numpy(x[take]).to(device)
            batch_y = torch.from_numpy(y[take]).to(device)
            batch_mask = torch.from_numpy(loss_mask[take]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch_x)
            losses = nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                batch_y.reshape(-1),
                reduction="none",
            ).reshape_as(batch_mask)
            total_loss += float((losses * batch_mask).sum().item())
            total_correct += float(
                ((logits.argmax(-1) == batch_y) * (batch_mask > 0)).sum().item()
            )
            total_tokens += float(batch_mask.sum().item())
    return {
        "answer_token_nll": total_loss / max(total_tokens, 1.0),
        "answer_token_accuracy": total_correct / max(total_tokens, 1.0),
        "evaluated_tokens": int(total_tokens),
    }


def build_package(
    output: Path,
    candidate_id: str,
    payload: bytes,
    tensor_count: int,
    golden_sha: str,
    release_root: str,
    output_schema_sha: str,
    config: Any,
) -> dict[str, Any]:
    payload_sha = hashlib.sha256(payload).digest()
    kv_bytes = config.n_layers * 2 * config.context_tokens * config.d_model
    arena_bytes = 512 * 1024
    scratch_bytes = 256 * 1024
    if arena_bytes + kv_bytes > 2_621_440:
        raise RuntimeError("ARENA_KV_HARD_GATE")
    header = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<4sHHHHBBHQQIII",
        header,
        0,
        b"ICMF",
        1,
        HEADER_BYTES,
        ENGINE_ID,
        1,
        0,
        1,  # tied-embedding flag
        tensor_count,
        1,
        len(payload),
        scratch_bytes,
        arena_bytes,
        kv_bytes,
    )
    header[44:76] = candidate_id.encode("utf-8")[:31].ljust(32, b"\0")
    header[76:108] = payload_sha
    header[108:140] = bytes.fromhex(golden_sha)
    header[140:172] = bytes.fromhex(release_root)
    header[172:204] = bytes.fromhex(output_schema_sha)
    package_path = output / "w8_or_w8a8.bin"
    package_path.write_bytes(bytes(header) + payload)
    if package_path.stat().st_size > 2_097_152:
        raise RuntimeError(f"ONLINE_NANOLM_PACKAGE_HARD_GATE:{package_path.stat().st_size}")
    return {
        "path": package_path.name,
        "sha256": sha256_file(package_path),
        "bytes": package_path.stat().st_size,
        "payload_sha256": payload_sha.hex(),
        "authority": 0,
        "engine_id": ENGINE_ID,
        "opset": 1,
        "scratch_bytes": scratch_bytes,
        "arena_bytes": arena_bytes,
        "kv_bytes": kv_bytes,
        "arena_plus_kv_bytes": arena_bytes + kv_bytes,
        "board_accepted": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    candidate_id = args.candidate_id
    config = config_for_candidate(candidate_id)
    contract = load_contract(root, candidate_id)
    staged_subdir = args.staged_subdir or "staged_nanolm_v2"
    if not staged_subdir.replace("_", "").isalnum():
        raise RuntimeError("UNSAFE_STAGED_SUBDIRECTORY")
    dataset_path = root / "data" / staged_subdir / f"{candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    allowed_status = {"PASS_CORRECTIVE_DATASET_TEACHER_PENDING"}
    if args.supervised_only:
        allowed_status.add("PASS_CONTRACT_SHAPED_SOURCE_SUPERVISED")
    if metadata.get("status") not in allowed_status:
        raise RuntimeError("NANOLM_V2_DATASET_STATUS")
    if metadata.get("cross_split_group_overlap") != 0 or metadata.get("authority") != 0:
        raise RuntimeError("NANOLM_V2_SPLIT_OR_AUTHORITY")
    dataset_sha = sha256_file(dataset_path)
    if dataset_sha != metadata.get("sha256"):
        raise RuntimeError("NANOLM_V2_DATASET_HASH")
    data = np.load(dataset_path, allow_pickle=False)
    split = data["split"].astype(np.int8)
    if args.supervised_only:
        teacher = bridge = None
        teacher_receipt = bridge_receipt = {
            "status": "NOT_USED_SOURCE_SUPERVISED_ONLY",
            "cache_sha256": "NONE_SOURCE_SUPERVISED_ONLY",
            "teacher_promoted_to_ground_truth": False,
            "validation_records_seen": 0,
            "test_records_seen": 0,
        }
    else:
        teacher, teacher_receipt = load_cache(
            args.teacher_cache_root, candidate_id, dataset_sha, split, "PRIMARY_TEACHER"
        )
        bridge, bridge_receipt = load_cache(
            args.bridge_cache_root, candidate_id, dataset_sha, split, "BRIDGE_REVIEWER"
        )
    output = args.artifact_root.resolve() / candidate_id
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    heartbeat(heartbeat_path, candidate_id, "IMPORT_TORCH")

    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_LOCAL_4050_NOT_AUTHORIZED")
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    props = torch.cuda.get_device_properties(device)
    x = data["x"].astype(np.int64)
    y = data["y"].astype(np.int64)
    loss_mask = data["loss_mask"].astype(np.float32)
    is_refusal = data["is_refusal"].astype(np.uint8)
    indices = {
        name: np.flatnonzero(split == code)
        for code, name in enumerate(("train", "validation", "test"))
    }
    if min(len(value) for value in indices.values()) < 16:
        raise RuntimeError("NANOLM_V2_SPLIT_TOO_SMALL")
    tokenizer = ContractTokenizer(root / metadata["tokenizer_path"])
    train_targets = y[indices["train"]][loss_mask[indices["train"]] > 0]
    frequencies = np.bincount(train_targets, minlength=config.vocab_size).astype(np.float64) + 1.0
    frequencies /= frequencies.sum()
    test_targets = y[indices["test"]][loss_mask[indices["test"]] > 0]
    unigram_nll = float(-np.log(frequencies[test_targets]).mean())
    reference_baseline = {
        "contract_name": contract["baseline"],
        "execution_state": "REFERENCE_NON_REFUSING_EXTRACT_PROXY_NOT_PROMOTION_AUTHORITY",
        "answer_token_f1": 0.5,
        "positive_citation_exact": 1.0,
        "refusal_accuracy": 0.5,
        "unsupported_negative_rate": 1.0,
        "primary_composite": 0.525,
        "unigram_answer_token_nll": unigram_nll,
    }
    if args.supervised_only:
        primary_nll = bridge_nll = np.zeros(len(split), dtype=np.float32)
        agreement = np.ones(len(split), dtype=np.float32)
    else:
        primary_nll = teacher["sequence_nll"].astype(np.float32)
        bridge_nll = bridge["sequence_nll"].astype(np.float32)
        agreement = np.exp(-np.abs(primary_nll - bridge_nll) / 2.0).clip(0.25, 1.0).astype(np.float32)
    reports = []
    seed_prediction_arrays: dict[str, np.ndarray] = {}
    seed_evaluation_indices: np.ndarray | None = None
    best_score = -math.inf
    best_seed = None
    best_state = None
    started = time.perf_counter()
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = build_model(torch, nn, config).to(device)
        primary_head = nn.Linear(config.d_model, 64, bias=False).to(device)
        bridge_head = nn.Linear(config.d_model, 64, bias=False).to(device)
        optimizer = torch.optim.AdamW(
            list(model.parameters())
            + list(primary_head.parameters())
            + list(bridge_head.parameters()),
            lr=args.learning_rate,
            weight_decay=0.01,
        )
        checkpoint_dir = output / f"train_seed_{seed}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_path = checkpoint_dir / "last.pt"
        start_epoch, local_best, local_best_nll, patience = 0, -math.inf, math.inf, 0
        if args.resume and last_path.is_file():
            checkpoint = torch.load(last_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            primary_head.load_state_dict(checkpoint["primary_head"])
            bridge_head.load_state_dict(checkpoint["bridge_head"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            local_best = float(checkpoint.get("local_best_generation", -math.inf))
            local_best_nll = float(checkpoint.get("local_best_nll", math.inf))
            patience = int(checkpoint.get("patience", 0))
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            TensorDataset(torch.from_numpy(indices["train"])),
            batch_size=min(args.batch_size, len(indices["train"])),
            shuffle=True,
            generator=generator,
        )
        epochs_completed = start_epoch
        for epoch in range(start_epoch, args.max_epochs):
            model.train()
            primary_head.train()
            bridge_head.train()
            for (batch_indices_cpu,) in loader:
                batch_indices = batch_indices_cpu.numpy()
                batch_x = torch.from_numpy(x[batch_indices]).to(device)
                batch_y = torch.from_numpy(y[batch_indices]).to(device)
                batch_mask = torch.from_numpy(loss_mask[batch_indices]).to(device)
                sample_weight = torch.from_numpy(agreement[batch_indices]).to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits, hidden = model(batch_x, return_hidden=True)
                    raw_losses = nn.functional.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        batch_y.reshape(-1),
                        reduction="none",
                    ).reshape_as(batch_mask)
                    per_example = (raw_losses * batch_mask).sum(dim=1) / batch_mask.sum(dim=1).clamp_min(1.0)
                    supervised = (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
                    if args.supervised_only:
                        loss = supervised
                    else:
                        primary_soft = sparse_teacher_loss(
                            torch,
                            logits,
                            torch.from_numpy(teacher["soft_ids"][batch_indices]).to(device),
                            torch.from_numpy(teacher["soft_probabilities"][batch_indices]).to(device),
                            torch.from_numpy(teacher["soft_mask"][batch_indices]).to(device),
                            sample_weight,
                        )
                        bridge_soft = sparse_teacher_loss(
                            torch,
                            logits,
                            torch.from_numpy(bridge["soft_ids"][batch_indices]).to(device),
                            torch.from_numpy(bridge["soft_probabilities"][batch_indices]).to(device),
                            torch.from_numpy(bridge["soft_mask"][batch_indices]).to(device),
                            sample_weight,
                        )
                        pooled = (hidden * batch_mask[:, :, None]).sum(dim=1) / batch_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                        primary_hidden = nn.functional.mse_loss(
                            primary_head(pooled).float(),
                            torch.from_numpy(teacher["hidden64"][batch_indices].astype(np.float32)).to(device),
                        )
                        bridge_hidden = nn.functional.mse_loss(
                            bridge_head(pooled).float(),
                            torch.from_numpy(bridge["hidden64"][batch_indices].astype(np.float32)).to(device),
                        )
                        pair_count = len(batch_indices) // 2
                        if pair_count:
                            student_diff = per_example[: 2 * pair_count : 2] - per_example[1 : 2 * pair_count : 2]
                            teacher_values = torch.from_numpy(
                                ((primary_nll + bridge_nll) * 0.5)[batch_indices]
                            ).to(device)
                            teacher_diff = teacher_values[: 2 * pair_count : 2] - teacher_values[1 : 2 * pair_count : 2]
                            direction = torch.sign(teacher_diff).masked_fill(teacher_diff == 0, 1.0)
                            ranking = nn.functional.softplus(-direction * student_diff).mean()
                        else:
                            ranking = supervised * 0.0
                        loss = (
                            supervised
                            + 0.15 * primary_soft
                            + 0.10 * bridge_soft
                            + 0.01 * primary_hidden
                            + 0.01 * bridge_hidden
                            + 0.02 * ranking
                        )
                if not torch.isfinite(loss):
                    raise RuntimeError("NANOLM_NONFINITE_LOSS")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            validation = evaluate_nll(
                torch,
                nn,
                model,
                x,
                y,
                loss_mask,
                indices["validation"],
                device,
                args.batch_size,
            )
            validation_generation = evaluate_generation(
                torch,
                model,
                data,
                tokenizer,
                indices["validation"],
                is_refusal,
            )
            score = validation_generation["grounded_claim_f1"] if args.grounded_claim_selection else validation_generation["primary_composite"]
            nll = validation["answer_token_nll"]
            improved = score > local_best + 1e-6 or (
                abs(score - local_best) <= 1e-6 and nll < local_best_nll - 1e-4
            )
            if improved:
                local_best, local_best_nll, patience = score, nll, 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "primary_head": primary_head.state_dict(),
                        "bridge_head": bridge_head.state_dict(),
                        "validation_generation": validation_generation,
                        "validation_nll": validation,
                    },
                    checkpoint_dir / "best.pt",
                )
            else:
                patience += 1
            epochs_completed = epoch + 1
            if epoch % args.checkpoint_epochs == 0 or epoch + 1 == args.max_epochs:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "primary_head": primary_head.state_dict(),
                        "bridge_head": bridge_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "epoch": epoch,
                        "local_best_generation": local_best,
                        "local_best_nll": local_best_nll,
                        "patience": patience,
                    },
                    last_path,
                )
            heartbeat(heartbeat_path, candidate_id, "TRAIN_NANOTRANSFORMER_V2", seed, epoch)
            if (
                epochs_completed >= args.min_epochs
                and patience >= args.early_stop_patience
            ):
                break
        checkpoint = torch.load(
            checkpoint_dir / "best.pt", map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        pre_qat_validation = evaluate_nll(
            torch, nn, model, x, y, loss_mask, indices["validation"], device, args.batch_size
        )
        # Projected QAT: the release FP state itself stays on the exact W8
        # group grid.  This lets optimization absorb quantization error rather
        # than redefining parity after export.
        projected, projected_scales = quantize_nanolm_state(model.state_dict())
        model.load_state_dict(
            dequantize_nanolm_state(torch, projected, projected_scales)
        )
        qat_optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.qat_learning_rate, weight_decay=0.0
        )
        qat_best_validation = evaluate_nll(
            torch, nn, model, x, y, loss_mask, indices["validation"], device, args.batch_size
        )
        qat_best_generation = evaluate_generation(
            torch, model, data, tokenizer, indices["validation"], is_refusal
        )
        qat_best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        for qat_epoch in range(args.qat_epochs):
            model.train()
            for (batch_indices_cpu,) in loader:
                batch_indices = batch_indices_cpu.numpy()
                batch_x = torch.from_numpy(x[batch_indices]).to(device)
                batch_y = torch.from_numpy(y[batch_indices]).to(device)
                batch_mask = torch.from_numpy(loss_mask[batch_indices]).to(device)
                sample_weight = torch.from_numpy(agreement[batch_indices]).to(device)
                qat_optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(batch_x)
                    raw_losses = nn.functional.cross_entropy(
                        logits.reshape(-1, config.vocab_size),
                        batch_y.reshape(-1),
                        reduction="none",
                    ).reshape_as(batch_mask)
                    per_example = (raw_losses * batch_mask).sum(dim=1) / batch_mask.sum(dim=1).clamp_min(1.0)
                    qat_loss = (per_example * sample_weight).sum() / sample_weight.sum().clamp_min(1.0)
                qat_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                qat_optimizer.step()
                projected, projected_scales = quantize_nanolm_state(model.state_dict())
                model.load_state_dict(
                    dequantize_nanolm_state(torch, projected, projected_scales)
                )
            qat_validation = evaluate_nll(
                torch,
                nn,
                model,
                x,
                y,
                loss_mask,
                indices["validation"],
                device,
                args.batch_size,
            )
            qat_generation = evaluate_generation(
                torch, model, data, tokenizer, indices["validation"], is_refusal
            )
            qat_score = qat_generation["grounded_claim_f1"] if args.grounded_claim_selection else qat_generation["primary_composite"]
            qat_best_score = qat_best_generation["grounded_claim_f1"] if args.grounded_claim_selection else qat_best_generation["primary_composite"]
            if qat_score > qat_best_score + 1e-6 or (
                abs(qat_score - qat_best_score) <= 1e-6
                and qat_validation["answer_token_nll"]
                < qat_best_validation["answer_token_nll"] - 1e-4
            ):
                qat_best_validation = qat_validation
                qat_best_generation = qat_generation
                qat_best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            heartbeat(
                heartbeat_path,
                candidate_id,
                "QAT_W8_GROUP32",
                seed,
                qat_epoch,
            )
        model.load_state_dict(qat_best_state)
        validation = evaluate_nll(
            torch, nn, model, x, y, loss_mask, indices["validation"], device, args.batch_size
        )
        test = evaluate_nll(
            torch, nn, model, x, y, loss_mask, indices["test"], device, args.batch_size
        )
        eval_indices = evaluation_subset(indices["test"], is_refusal)
        predictions = greedy_generate(
            torch,
            model,
            data["prompt_tokens"][eval_indices],
            data["prompt_length"][eval_indices],
            MAX_GENERATION_TOKENS,
        )
        if seed_evaluation_indices is None:
            seed_evaluation_indices = eval_indices.astype(np.int64, copy=True)
        elif not np.array_equal(seed_evaluation_indices, eval_indices):
            raise RuntimeError("THREE_SEED_EVALUATION_INDEX_DRIFT")
        seed_prediction_arrays[f"seed_{seed}"] = predictions.astype(np.uint16, copy=False)
        generated = generation_metrics(
            tokenizer,
            predictions,
            data["target_tokens"][eval_indices],
            data["target_length"][eval_indices],
            is_refusal[eval_indices],
        )
        reports.append(
            {
                "seed": seed,
                "epochs": epochs_completed,
                "qat_epochs": args.qat_epochs,
                "pre_qat_validation": pre_qat_validation,
                "validation": validation,
                "test": test,
                "generation": generated,
            }
        )
        validation_generation = evaluate_generation(
            torch, model, data, tokenizer, indices["validation"], is_refusal
        )
        selection_score = validation_generation["grounded_claim_f1"] if args.grounded_claim_selection else validation_generation["primary_composite"]
        if selection_score > best_score:
            best_score, best_seed = selection_score, seed
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    assert best_state is not None and best_seed is not None
    if seed_evaluation_indices is None or len(seed_prediction_arrays) != len(SEEDS):
        raise RuntimeError("THREE_SEED_PREDICTION_EVIDENCE_INCOMPLETE")
    model = build_model(torch, nn, config).to(device)
    model.load_state_dict(best_state)
    golden_indices = evaluation_subset(indices["test"], is_refusal, limit=16)
    fp_tokens = greedy_generate(
        torch,
        model,
        data["prompt_tokens"][golden_indices],
        data["prompt_length"][golden_indices],
        MAX_GENERATION_TOKENS,
    )
    quantized, scales = quantize_nanolm_state(best_state)
    dequantized = dequantize_nanolm_state(torch, quantized, scales)
    model.load_state_dict(dequantized)
    quant_tokens = greedy_generate(
        torch,
        model,
        data["prompt_tokens"][golden_indices],
        data["prompt_length"][golden_indices],
        MAX_GENERATION_TOKENS,
    )
    quantized_test_tokens = greedy_generate(
        torch,
        model,
        data["prompt_tokens"][seed_evaluation_indices],
        data["prompt_length"][seed_evaluation_indices],
        MAX_GENERATION_TOKENS,
    )
    np.savez_compressed(
        output / "three_seed_test_predictions.npz",
        indices=seed_evaluation_indices,
        target_tokens=data["target_tokens"][seed_evaluation_indices].astype(np.uint16),
        target_length=data["target_length"][seed_evaluation_indices].astype(np.uint16),
        is_refusal=is_refusal[seed_evaluation_indices].astype(np.uint8),
        quantized_best_seed=quantized_test_tokens.astype(np.uint16, copy=False),
        authority=np.asarray(0, dtype=np.int8),
        **seed_prediction_arrays,
    )
    token_parity = float(np.mean(fp_tokens == quant_tokens))
    sequence_parity = float(np.mean(np.all(fp_tokens == quant_tokens, axis=1)))
    if token_parity < 0.95:
        raise RuntimeError(f"W8_TOKEN_PARITY_GATE:{token_parity:.6f}")
    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(
        golden_path,
        indices=golden_indices,
        prompt_tokens=data["prompt_tokens"][golden_indices],
        target_tokens=data["target_tokens"][golden_indices],
        fp32_generated=fp_tokens,
        w8_generated=quant_tokens,
        authority=np.asarray(0, dtype=np.int8),
    )
    payload_buffer = io.BytesIO()
    np.savez_compressed(
        payload_buffer,
        **quantized,
        **{
            f"scale::{name}": np.asarray(value, dtype=np.float32)
            for name, value in scales.items()
        },
        quant_group_size=np.asarray(W8_GROUP_SIZE, dtype=np.uint16),
    )
    payload = payload_buffer.getvalue()
    output_schema = {
        "schema": "cimc.icmat.nanollm-output.v2",
        "task_kind": "nano_transformer_lm",
        "shape": [1, CONTEXT_TOKENS, config.vocab_size],
        "architecture": config.to_dict(),
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "max_generation_tokens": MAX_GENERATION_TOKENS,
        "weight_quantization": f"SYMMETRIC_W8_GROUP{W8_GROUP_SIZE}",
        "authority": 0,
    }
    output_schema_sha = hashlib.sha256(canonical_bytes(output_schema)).hexdigest()
    golden_sha = sha256_file(golden_path)
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": candidate_id,
                "dataset_sha256": dataset_sha,
                "teacher_cache_sha256": teacher_receipt["cache_sha256"],
                "bridge_cache_sha256": bridge_receipt["cache_sha256"],
                "tokenizer_sha256": metadata["tokenizer_sha256"],
                "golden_sha256": golden_sha,
                "best_seed": best_seed,
                "architecture": config.to_dict(),
            }
        )
    ).hexdigest()
    package = build_package(
        output,
        candidate_id,
        payload,
        len(quantized),
        golden_sha,
        release_root,
        output_schema_sha,
        config,
    )
    model.load_state_dict(best_state)
    model.eval()
    onnx_path = output / "fp32.onnx"
    heartbeat(heartbeat_path, candidate_id, "EXPORT_ONNX")
    torch.onnx.export(
        model,
        torch.from_numpy(x[indices["test"][:1]]).to(device),
        onnx_path,
        input_names=["tokens"],
        output_names=["logits"],
        opset_version=17,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    mean_primary = float(
        np.mean([item["generation"]["primary_composite"] for item in reports])
    )
    proxy_pass = all(
        item["generation"]["primary_composite"] > reference_baseline["primary_composite"]
        for item in reports
    )
    evaluation = {
        "schema": "cimc.forge200.nanollm-grouped-evaluation.v2",
        "candidate_id": candidate_id,
        "truth_class": metadata["truth_class"],
        "claim_state": metadata["claim_state"],
        "contract_primary_metric": contract["primary_metric"],
        "contract_baseline": contract["baseline"],
        "baseline": reference_baseline,
        "baseline_proxy_pass": proxy_pass,
        "checkpoint_selection": "validation_primary_generation_composite_then_nll_no_test_access",
        "exact_contract_baseline_pending": True,
        "seed_reports": reports,
        "mean_primary_composite": mean_primary,
        "best_seed": best_seed,
        "split_counts": {name: len(value) for name, value in indices.items()},
        "group_overlap": 0,
    }
    write_json(output / "eval_grouped.json", evaluation)
    write_json(
        output / "quantization_parity.json",
        {
            "schema": "cimc.forge200.nanollm-w8-parity.v1",
            "token_parity": token_parity,
            "sequence_parity": sequence_parity,
            "golden_examples": len(golden_indices),
            "status": "PASS" if token_parity >= 0.95 else "FAIL",
        },
    )
    write_json(output / "architecture.json", config.to_dict())
    write_json(output / "output_schema.json", output_schema)
    write_json(
        output / "distillation_receipt.json",
        {
            "schema": "cimc.forge200.nanollm-distillation.v1",
            "primary_teacher": teacher_receipt,
            "bridge_reviewer": bridge_receipt,
            "teacher_truth_class": "TEACHER_CANDIDATE",
            "teacher_promoted_to_ground_truth": False,
            "validation_records_seen": 0,
            "test_records_seen": 0,
            "losses": ["source_bound_supervised_ce"] if args.supervised_only else [
                "source_bound_supervised_ce",
                "qwen3_4b_sparse_logit_distillation",
                "qwen3_1p7b_sparse_logit_review",
                "dual_hidden_projection_distillation",
                "teacher_sequence_ranking",
            ],
            "training_mode": "SOURCE_SUPERVISED_ONLY_NO_TEACHER" if args.supervised_only else "DUAL_TEACHER_DISTILLATION",
            "authority": 0,
        },
    )
    write_json(
        output / "source_manifest.json",
        {**metadata, "dataset_sha256": dataset_sha},
    )
    write_json(
        output / "split_manifest.json",
        {
            "split_unit": metadata["split_unit"],
            "cross_split_group_overlap": 0,
            "teacher_validation_records_seen": 0,
            "teacher_test_records_seen": 0,
        },
    )
    write_json(output / "baseline_report.json", reference_baseline)
    write_json(
        output / "ablation.json",
        {
            "status": "TRAINED_CORRECTIVE_ABLATION_PENDING_EXACT_LEGACY_BASELINE",
            "components": ["4B_logits", "1.7B_review", "dual_hidden", "ranking", "refusal"],
        },
    )
    (output / "model_card.md").write_text(
        f"# {candidate_id} corrective nano-LM model card\n\n"
        f"- Architecture: `{config.architecture}`; family `{config.family}`; parameters `{parameter_count(config)}`.\n"
        f"- Quantization: `W8`; package bytes `{package['bytes']}`; engine `{ENGINE_ID}`.\n"
        f"- Data truth: `{metadata['truth_class']}`; teacher outputs remain `TEACHER_CANDIDATE`.\n"
        f"- Three fixed seeds: `{SEEDS}`; best seed `{best_seed}`.\n"
        f"- W8 token parity: `{token_parity:.6f}`; sequence parity `{sequence_parity:.6f}`.\n"
        f"- Exact contract baseline: pending; board: pending; countable model: false.\n"
        f"- Authority: `0`; no deterministic control or actuation authority.\n",
        encoding="utf-8",
    )
    receipt = {
        "schema": "cimc.forge200.nanollm-promotion-receipt.v2",
        "status": "HOST_GPU_TRAINED_CORRECTIVE_EXACT_BASELINE_AND_BOARD_PENDING",
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "exact_contract_baseline_pending": True,
        "best_seed": best_seed,
        "three_seed_count": len(reports),
        "parameter_count": parameter_count(config),
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": golden_sha,
        "tokenizer_sha256": metadata["tokenizer_sha256"],
        "teacher_cache_sha256": teacher_receipt["cache_sha256"],
        "bridge_cache_sha256": bridge_receipt["cache_sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {
            "name": props.name,
            "vram_gib": props.total_memory / (1024**3),
        },
    }
    write_json(output / "promotion_receipt.json", receipt)
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    write_json(output / "artifact_manifest.json", records_manifest(output))
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--teacher-cache-root", type=Path)
    parser.add_argument("--bridge-cache-root", type=Path)
    parser.add_argument("--staged-subdir")
    parser.add_argument("--supervised-only", action="store_true")
    parser.add_argument("--grounded-claim-selection", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--checkpoint-epochs", type=int, default=2)
    parser.add_argument("--early-stop-patience", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument("--qat-epochs", type=int, default=4)
    parser.add_argument("--qat-learning-rate", type=float, default=2e-5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        output = args.artifact_root.resolve() / args.candidate_id
        output.mkdir(parents=True, exist_ok=True)
        failure = {
            "schema": "cimc.forge200.nanollm-job-failure.v2",
            "status": "FAIL_CLOSED",
            "candidate_id": args.candidate_id,
            "authority": 0,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(output / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
