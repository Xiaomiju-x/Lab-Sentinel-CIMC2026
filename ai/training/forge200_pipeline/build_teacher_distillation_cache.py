#!/usr/bin/env python3
"""Create train-only Qwen3 distillation caches for corrective nano-LMs.

Teacher logits, hidden projections and sequence scores are explicitly marked
``TEACHER_CANDIDATE``.  They supplement source-bound supervised targets and
are never promoted to material/process ground truth.  Validation and test
examples are not tokenized or shown to the teacher.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanolm_architecture import CONTEXT_TOKENS, MAX_GENERATION_TOKENS, VOCAB_SIZE


PAD = 0
BOS = 1
EOS = 2
BYTE_BASE = 3
FIRST_PIECE = BYTE_BASE + 256


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class ContractTokenizer:
    def __init__(self, contract_path: Path) -> None:
        self.contract_path = contract_path
        self.contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if self.contract.get("vocab_size") != VOCAB_SIZE:
            raise RuntimeError("student tokenizer vocabulary gate")
        self.pieces = [base64.b64decode(item["base64"]) for item in self.contract["pieces"]]
        if len(self.pieces) != VOCAB_SIZE:
            raise RuntimeError("student tokenizer piece count gate")
        trie: dict[int, Any] = {}
        for token_id, piece in enumerate(self.pieces[FIRST_PIECE:], start=FIRST_PIECE):
            node = trie
            for value in piece:
                node = node.setdefault(value, {})
            node[-1] = token_id
        self.trie = trie

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        output: list[int] = []
        offset = 0
        while offset < len(raw):
            node = self.trie
            cursor = offset
            matched_end = matched_id = None
            while cursor < len(raw) and raw[cursor] in node:
                node = node[raw[cursor]]
                cursor += 1
                if -1 in node:
                    matched_end, matched_id = cursor, node[-1]
            if matched_id is None:
                output.append(BYTE_BASE + raw[offset])
                offset += 1
            else:
                output.append(int(matched_id))
                offset = int(matched_end)
        return output

    def token_bytes(self, token_id: int) -> bytes:
        if BYTE_BASE <= token_id < FIRST_PIECE:
            return bytes([token_id - BYTE_BASE])
        if FIRST_PIECE <= token_id < len(self.pieces):
            return self.pieces[token_id]
        return b""

    def decode(self, token_ids: Iterable[int]) -> str:
        raw = b"".join(self.token_bytes(int(token_id)) for token_id in token_ids)
        return raw.decode("utf-8", errors="replace")

    def byte_offset_to_token(self, token_ids: list[int], byte_offset: int) -> int | None:
        cursor = 0
        for index, token_id in enumerate(token_ids):
            piece = self.token_bytes(token_id)
            if not piece:
                continue
            if cursor <= byte_offset < cursor + len(piece):
                return index
            cursor += len(piece)
        return None


def model_manifest(model_path: Path) -> tuple[list[dict[str, Any]], str]:
    records = []
    for path in sorted(item for item in model_path.rglob("*") if item.is_file()):
        # ModelScope's resumable-download bookkeeping is not part of the model.
        if "._____temp" in path.name:
            continue
        records.append(
            {
                "path": str(path.relative_to(model_path)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records, hashlib.sha256(canonical_bytes(records)).hexdigest()


def projection_matrix(torch: Any, hidden_size: int, device: Any) -> tuple[Any, str]:
    rng = np.random.default_rng(20260803)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(hidden_size, 64))
    signs /= math.sqrt(hidden_size)
    digest = hashlib.sha256(signs.tobytes(order="C")).hexdigest()
    return torch.from_numpy(signs).to(device), digest


def mapped_distribution(
    torch: Any,
    teacher_tokenizer: Any,
    student_tokenizer: ContractTokenizer,
    logits: Any,
    top_k: int,
    temperature: float,
) -> tuple[list[int], list[float]]:
    probability = torch.softmax(logits.float() / temperature, dim=-1)
    values, indices = torch.topk(probability, k=top_k * 4)
    mapped: dict[int, float] = {}
    for value, token_id in zip(values.tolist(), indices.tolist()):
        text = teacher_tokenizer.decode(
            [int(token_id)], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        encoded = student_tokenizer.encode(text)
        if encoded:
            mapped[encoded[0]] = mapped.get(encoded[0], 0.0) + float(value)
        if len(mapped) >= top_k:
            # Continue a few extra entries only when collisions dominate.
            continue
    ranked = sorted(mapped.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    total = sum(value for _, value in ranked)
    if not ranked or total <= 0:
        return [], []
    return [item[0] for item in ranked], [item[1] / total for item in ranked]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--model-revision", default="master")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max-train-examples", type=int, default=0)
    parser.add_argument("--skip-model-file-hashes", action="store_true")
    parser.add_argument("--model-manifest-cache", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    model_path = args.model_path.resolve()
    dataset_path = root / "data" / "staged_nanolm_v2" / f"{args.candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS_CORRECTIVE_DATASET_TEACHER_PENDING":
        raise RuntimeError("corrective dataset state gate")
    if metadata.get("teacher_may_view_validation_or_test") is not False:
        raise RuntimeError("teacher leakage contract gate")
    data = np.load(dataset_path, allow_pickle=False)
    if int(data["authority"]) != 0 or str(data["candidate_id"]) != args.candidate_id:
        raise RuntimeError("dataset identity or authority gate")
    train_indices = np.flatnonzero(data["split"] == 0)
    selected = train_indices[: args.max_train_examples] if args.max_train_examples else train_indices
    output = args.output_root.resolve() / args.candidate_id
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    write_json(
        heartbeat_path,
        {
            "candidate_id": args.candidate_id,
            "stage": "LOAD_TEACHER",
            "utc": datetime.now(timezone.utc).isoformat(),
            "authority": 0,
        },
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, use_fast=True
    )
    if not teacher_tokenizer.is_fast:
        raise RuntimeError("teacher tokenizer offsets require a fast tokenizer")
    if teacher_tokenizer.pad_token_id is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    teacher_tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    student_tokenizer = ContractTokenizer(root / metadata["tokenizer_path"])
    projection, projection_sha = projection_matrix(
        torch, int(model.config.hidden_size), device
    )
    records = len(data["split"])
    soft_ids = np.zeros((records, MAX_GENERATION_TOKENS, args.top_k), dtype=np.uint16)
    soft_probabilities = np.zeros(
        (records, MAX_GENERATION_TOKENS, args.top_k), dtype=np.float16
    )
    soft_mask = np.zeros((records, MAX_GENERATION_TOKENS), dtype=np.uint8)
    hidden64 = np.zeros((records, 64), dtype=np.float16)
    hidden_mask = np.zeros(records, dtype=np.uint8)
    sequence_nll = np.zeros(records, dtype=np.float32)
    processed_mask = np.zeros(records, dtype=np.uint8)
    started = time.perf_counter()
    for batch_start in range(0, len(selected), args.batch_size):
        batch_indices = selected[batch_start : batch_start + args.batch_size]
        prompts = [
            student_tokenizer.decode(
                data["prompt_tokens"][index, : int(data["prompt_length"][index])]
            )
            for index in batch_indices
        ]
        targets = [
            student_tokenizer.decode(
                data["target_tokens"][index, : int(data["target_length"][index])]
            )
            for index in batch_indices
        ]
        texts = [prompt + target for prompt, target in zip(prompts, targets)]
        encoded = teacher_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=768,
            return_offsets_mapping=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        offsets = encoded.pop("offset_mapping").cpu().numpy()
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            base_output = model.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            last_hidden = base_output.last_hidden_state
        for local_index, record_index in enumerate(batch_indices):
            prompt_chars = len(prompts[local_index])
            target = targets[local_index]
            target_ids = data["target_tokens"][
                record_index, : int(data["target_length"][record_index])
            ].astype(np.int64).tolist()
            teacher_positions = []
            for position, (start_char, end_char) in enumerate(offsets[local_index]):
                if (
                    int(encoded["attention_mask"][local_index, position]) == 1
                    and end_char > prompt_chars
                    and position > 0
                ):
                    teacher_positions.append(position)
            if not teacher_positions:
                raise RuntimeError(f"{record_index}: teacher target alignment empty")
            pooled = last_hidden[local_index, teacher_positions].float().mean(dim=0)
            hidden64[record_index] = (pooled @ projection).cpu().numpy().astype(np.float16)
            hidden_mask[record_index] = 1
            losses = []
            for position in teacher_positions:
                token_id = encoded["input_ids"][local_index, position]
                with torch.inference_mode():
                    logits = model.lm_head(last_hidden[local_index, position - 1]).float()
                    log_probability = torch.log_softmax(logits, dim=-1)
                losses.append(float(-log_probability[token_id].item()))
                relative_char = max(int(offsets[local_index, position, 0]) - prompt_chars, 0)
                byte_offset = len(target[:relative_char].encode("utf-8"))
                student_position = student_tokenizer.byte_offset_to_token(
                    target_ids, byte_offset
                )
                if student_position is None or student_position >= MAX_GENERATION_TOKENS:
                    continue
                ids, probabilities = mapped_distribution(
                    torch,
                    teacher_tokenizer,
                    student_tokenizer,
                    logits,
                    args.top_k,
                    args.temperature,
                )
                if ids:
                    soft_ids[record_index, student_position, : len(ids)] = ids
                    soft_probabilities[record_index, student_position, : len(ids)] = probabilities
                    soft_mask[record_index, student_position] = 1
            sequence_nll[record_index] = float(np.mean(losses))
            processed_mask[record_index] = 1
        write_json(
            heartbeat_path,
            {
                "candidate_id": args.candidate_id,
                "stage": "TEACHER_FORWARD",
                "processed": min(batch_start + len(batch_indices), len(selected)),
                "selected": len(selected),
                "utc": datetime.now(timezone.utc).isoformat(),
                "authority": 0,
            },
        )
    cache_path = output / "teacher_cache.npz"
    np.savez_compressed(
        cache_path,
        soft_ids=soft_ids,
        soft_probabilities=soft_probabilities,
        soft_mask=soft_mask,
        hidden64=hidden64,
        hidden_mask=hidden_mask,
        sequence_nll=sequence_nll,
        processed_mask=processed_mask,
        split=data["split"],
        candidate_id=np.asarray(args.candidate_id),
        truth_class=np.asarray("TEACHER_CANDIDATE"),
        authority=np.asarray(0, dtype=np.int8),
    )
    if args.skip_model_file_hashes:
        model_records, model_root = [], "PILOT_HASH_DEFERRED"
    elif args.model_manifest_cache and args.model_manifest_cache.is_file():
        cached_manifest = json.loads(
            args.model_manifest_cache.read_text(encoding="utf-8")
        )
        model_records = cached_manifest["records"]
        model_root = cached_manifest["content_root_sha256"]
    else:
        model_records, model_root = model_manifest(model_path)
        if args.model_manifest_cache:
            write_json(
                args.model_manifest_cache,
                {
                    "schema": "cimc.forge200.teacher-model-files.v1",
                    "model_id": args.model_id,
                    "revision": args.model_revision,
                    "records": model_records,
                    "content_root_sha256": model_root,
                },
            )
    full = len(selected) == len(train_indices)
    receipt = {
        "schema": "cimc.forge200.teacher-distillation-cache.v1",
        "status": "PASS_TRAIN_ONLY_TEACHER_CACHE" if full else "PILOT_TRAIN_ONLY_TEACHER_CACHE_NOT_PROMOTABLE",
        "candidate_id": args.candidate_id,
        "teacher_model_id": args.model_id,
        "teacher_model_revision": args.model_revision,
        "teacher_model_path_role": "REMOTE_EPHEMERAL_CACHE_NOT_RELEASE_PAYLOAD",
        "teacher_model_files": model_records,
        "teacher_model_content_root_sha256": model_root,
        "teacher_truth_class": "TEACHER_CANDIDATE",
        "teacher_promoted_to_ground_truth": False,
        "dataset_path": str(dataset_path.relative_to(root)).replace("\\", "/"),
        "dataset_sha256": sha256_file(dataset_path),
        "tokenizer_sha256": sha256_file(root / metadata["tokenizer_path"]),
        "projection_seed": 20260803,
        "projection_sha256": projection_sha,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "train_records_total": len(train_indices),
        "train_records_processed": int(processed_mask.sum()),
        "validation_records_seen": 0,
        "test_records_seen": 0,
        "soft_positions": int(soft_mask.sum()),
        "hidden_records": int(hidden_mask.sum()),
        "cache_path": "teacher_cache.npz",
        "cache_bytes": cache_path.stat().st_size,
        "cache_sha256": sha256_file(cache_path),
        "runtime_seconds": time.perf_counter() - started,
        "authority": 0,
    }
    write_json(output / "receipt.json", receipt)
    write_json(
        heartbeat_path,
        {
            "candidate_id": args.candidate_id,
            "stage": "COMPLETE",
            "status": receipt["status"],
            "utc": datetime.now(timezone.utc).isoformat(),
            "authority": 0,
        },
    )
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
