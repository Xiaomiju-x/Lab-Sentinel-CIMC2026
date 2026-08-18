#!/usr/bin/env python3
"""Build the corrective unified tokenizer and 26 source-bound nano-LM sets.

The original GPU pass used a 259-symbol byte LM and only next-byte NLL.  That
artifact is retained, but it does not satisfy the frozen 0.4--1.8M nano-LM,
24-token answer, citation, or refusal contracts.  This builder creates a
train-only fitted 2048-piece reversible tokenizer and explicit positive /
fail-closed refusal examples without treating teacher output as truth.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from nanolm_architecture import (
    CONTEXT_TOKENS,
    MAX_GENERATION_TOKENS,
    VOCAB_SIZE,
    config_for_candidate,
)


PAD = 0
BOS = 1
EOS = 2
BYTE_BASE = 3
FIRST_PIECE = BYTE_BASE + 256
VOCAB_SPEC = "ICMAT_GREEDY_PIECE_2048_V1"
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}

DOMAIN_G_TASKS: dict[str, tuple[str, tuple[str, ...]]] = {
    "CAND-G-001": ("PHOSPHOR", ()),
    "CAND-G-002": ("FURNACE", ()),
    "CAND-G-003": ("SEMIMAT", ()),
    "CAND-G-004": ("METROLOGY", ()),
    "CAND-G-005": ("PACKAGING", ()),
    "CAND-G-006": ("FABQUALITY", ()),
    "CAND-G-007": ("PHOSPHOR", ("phase", "crystal", "transition")),
    "CAND-G-008": ("PHOSPHOR", ("dop", "site", "substitut", "occup")),
    "CAND-G-009": ("PHOSPHOR", ("atmosphere", "oxygen", "vacan", "defect", "valence")),
    "CAND-G-010": ("PHOSPHOR", ("energy transfer", "emission", "excitation", "luminescen")),
    "CAND-G-011": ("PHOSPHOR", ("thermal", "quench", "temperature")),
    "CAND-G-012": ("FURNACE", ("sinter", "kinetic", "densif", "grain")),
    "CAND-G-013": ("FURNACE", ("impurit", "precursor", "purity", "powder")),
    "CAND-G-014": ("PHOSPHOR", ("host", "composition", "transfer")),
    "CAND-G-015": ("METROLOGY", ("xrd", "diffraction", "phase")),
    "CAND-G-016": ("METROLOGY", ("photoluminescence", "spectrum", "spectral", "normalization")),
    "CAND-G-017": ("METROLOGY", ("sem", "eds", "microstructure", "morphology")),
    "CAND-G-018": ("METROLOGY", ("multimodal", "conflict", "comparison", "correlation")),
    "CAND-G-019": ("SEMIMAT", ("defect", "dop", "vacan", "charge")),
    "CAND-G-020": ("SEMIMAT", ("thin film", "interface", "adhesion", "stress")),
    "CAND-G-021": ("SEMIMAT", ("dielectric", "thermal", "permittivity", "loss")),
    "CAND-G-022": ("SEMIMAT", ("optoelectronic", "integration", "device", "interface")),
    "CAND-G-023": ("PACKAGING", ("cure", "rheolog", "underfill", "viscos")),
    "CAND-G-024": ("PACKAGING", ("solder", "interconnect", "joint", "metallurg", "diffusion")),
    "CAND-G-025": ("PACKAGING", ("warpage", "reliability", "delamination", "moisture", "stress")),
    "CAND-G-026": ("FABQUALITY", ("virtual metrology", "yield", "process", "wafer")),
}


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def compact_sentence(text: str, words: int = 14) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    terms = sentence.split()
    return " ".join(terms[:words]).strip(" ,;:")


def candidate_rows(rows: list[dict[str, Any]], domain: str, keywords: tuple[str, ...]) -> list[dict[str, Any]]:
    selected = []
    for row in rows:
        haystack = f"{row['title']} {row['section']} {row['text']}".lower()
        if row["domain"] == domain and (not keywords or any(word in haystack for word in keywords)):
            selected.append(row)
    return selected


def cross_row(pool: list[dict[str, Any]], current: dict[str, Any], index: int) -> dict[str, Any]:
    for offset in range(1, len(pool) + 1):
        other = pool[(index + offset) % len(pool)]
        if other["split"] == current["split"] and other["pmcid"] != current["pmcid"]:
            return other
    current_text = current["text"].lower()
    for offset in range(1, len(pool) + 1):
        other = pool[(index + offset) % len(pool)]
        probe = compact_sentence(other["text"], words=8).lower()
        if (
            other["split"] == current["split"]
            and other["chunk_id"] != current["chunk_id"]
            and probe not in current_text
        ):
            return other
    raise RuntimeError(f"no same-split cross-document negative for {current['chunk_id']}")


def make_examples(
    rows: list[dict[str, Any]],
    negative_pool: list[dict[str, Any]],
    domain: str,
    focus: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        evidence = row["text"][:900].strip()
        positive_prompt = (
            f"DOMAIN {domain}\nQUESTION Give one concise finding about {focus}. "
            "Use only Evidence [1], cite [1], and refuse if unsupported.\n"
            f"SOURCE[1] {row['chunk_id']}\nEVIDENCE[1] {evidence}\nANSWER "
        )
        positive_target = f"[1] {compact_sentence(row['text'])}"
        peer = cross_row(negative_pool, row, index)
        unsupported = compact_sentence(peer["text"], words=8)
        refusal_prompt = (
            f"DOMAIN {domain}\nQUESTION Does Evidence [1] establish: {unsupported}? "
            "Use only Evidence [1], cite [1], and refuse if unsupported.\n"
            f"SOURCE[1] {row['chunk_id']}\nEVIDENCE[1] {evidence}\nANSWER "
        )
        common = {
            "group": row["pmcid"],
            "split": row["split"],
            "source_chunk_id": row["chunk_id"],
            "source_sha256": row["source_sha256"],
        }
        examples.append({**common, "prompt": positive_prompt, "target": positive_target, "is_refusal": 0})
        examples.append(
            {
                **common,
                "prompt": refusal_prompt,
                "target": "REFUSE Evidence [1] does not support the requested claim.",
                "is_refusal": 1,
                "negative_source_chunk_id": peer["chunk_id"],
            }
        )
    return examples


def piece_candidates(text: str) -> Iterable[bytes]:
    # Leading whitespace stays attached to the following lexical unit so a
    # 24-token response can still carry a short finding and citation.
    pattern = r"\s+[A-Za-z0-9_:+./%°µ-]+|[A-Za-z0-9_:+./%°µ-]+|\s+|[^\w\s]"
    for match in re.finditer(pattern, text, flags=re.UNICODE):
        raw = match.group(0).encode("utf-8")
        if 2 <= len(raw) <= 48:
            yield raw


class GreedyPieceTokenizer:
    def __init__(self, pieces: list[bytes]) -> None:
        if len(pieces) != VOCAB_SIZE:
            raise ValueError(f"vocabulary must have {VOCAB_SIZE} entries")
        self.pieces = pieces
        trie: dict[int, Any] = {}
        for token_id, piece in enumerate(pieces[FIRST_PIECE:], start=FIRST_PIECE):
            node = trie
            for value in piece:
                node = node.setdefault(value, {})
            node[-1] = token_id
        self.trie = trie

    def encode(self, text: str) -> list[int]:
        raw = text.encode("utf-8")
        result: list[int] = []
        offset = 0
        while offset < len(raw):
            node = self.trie
            cursor = offset
            matched_end = None
            matched_id = None
            while cursor < len(raw) and raw[cursor] in node:
                node = node[raw[cursor]]
                cursor += 1
                if -1 in node:
                    matched_end = cursor
                    matched_id = node[-1]
            if matched_id is None:
                result.append(BYTE_BASE + raw[offset])
                offset += 1
            else:
                result.append(matched_id)
                offset = int(matched_end)
        return result

    def decode(self, token_ids: Iterable[int]) -> str:
        raw = bytearray()
        for token_id in token_ids:
            if token_id in {PAD, BOS, EOS}:
                continue
            if BYTE_BASE <= token_id < FIRST_PIECE:
                raw.append(token_id - BYTE_BASE)
            elif FIRST_PIECE <= token_id < len(self.pieces):
                raw.extend(self.pieces[token_id])
            else:
                raise ValueError(f"token id outside vocabulary: {token_id}")
        return raw.decode("utf-8", errors="replace")


def build_tokenizer(examples: dict[str, list[dict[str, Any]]]) -> GreedyPieceTokenizer:
    counts: Counter[bytes] = Counter()
    for candidate_examples in examples.values():
        for item in candidate_examples:
            if item["split"] != "train":
                continue
            counts.update(piece_candidates(item["prompt"]))
            counts.update(piece_candidates(item["target"]))
    room = VOCAB_SIZE - FIRST_PIECE
    ranked = sorted(counts.items(), key=lambda item: (-item[1], -len(item[0]), item[0]))
    selected = [piece for piece, count in ranked if count >= 2][:room]
    if len(selected) != room:
        raise RuntimeError(f"insufficient train-only pieces: {len(selected)} != {room}")
    pieces = [b"<PAD>", b"<BOS>", b"<EOS>"]
    pieces.extend(bytes([value]) for value in range(256))
    pieces.extend(selected)
    if len(set(pieces[FIRST_PIECE:])) != room:
        raise RuntimeError("duplicate learned tokenizer pieces")
    return GreedyPieceTokenizer(pieces)


def encode_example(tokenizer: GreedyPieceTokenizer, item: dict[str, Any]) -> dict[str, np.ndarray | int]:
    target = tokenizer.encode(item["target"])[: MAX_GENERATION_TOKENS - 1] + [EOS]
    # The generation ABI always reserves the final 24 positions, even when a
    # particular reference answer is shorter.  Training and autoregressive
    # host/board inference therefore see exactly the same prompt boundary.
    max_prompt = CONTEXT_TOKENS - MAX_GENERATION_TOKENS
    prompt = [BOS] + tokenizer.encode(item["prompt"])
    prompt = prompt[:max_prompt]
    if not prompt or prompt[0] != BOS:
        raise RuntimeError("prompt BOS contract")
    sequence = prompt + target
    x = np.full(CONTEXT_TOKENS, PAD, dtype=np.int64)
    y = np.full(CONTEXT_TOKENS, PAD, dtype=np.int64)
    mask = np.zeros(CONTEXT_TOKENS, dtype=np.float32)
    usable = min(len(sequence) - 1, CONTEXT_TOKENS)
    x[:usable] = sequence[:usable]
    y[:usable] = sequence[1 : usable + 1]
    answer_start = len(prompt) - 1
    mask[answer_start:usable] = 1.0
    prompt_array = np.full(CONTEXT_TOKENS - MAX_GENERATION_TOKENS, PAD, dtype=np.int64)
    stored_prompt = prompt[: len(prompt_array)]
    prompt_array[: len(stored_prompt)] = stored_prompt
    target_array = np.full(MAX_GENERATION_TOKENS, PAD, dtype=np.int64)
    target_array[: len(target)] = target
    return {
        "x": x,
        "y": y,
        "loss_mask": mask,
        "prompt_tokens": prompt_array,
        "prompt_length": len(stored_prompt),
        "target_tokens": target_array,
        "target_length": len(target),
    }


def task_hashes(root: Path) -> dict[str, str]:
    rows = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    result = {}
    for row in rows:
        result[row["candidate_id"]] = hashlib.sha256(canonical_bytes(row)).hexdigest()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v1.jsonl"
    rows = read_jsonl(corpus_path)
    hashes = task_hashes(root)
    examples: dict[str, list[dict[str, Any]]] = {}
    for candidate_id, (domain, keywords) in DOMAIN_G_TASKS.items():
        selected = candidate_rows(rows, domain, keywords)
        negative_pool = [row for row in rows if row["domain"] == domain]
        focus = "domain evidence" if not keywords else "/".join(keywords)
        examples[candidate_id] = make_examples(selected, negative_pool, domain, focus)
    tokenizer = build_tokenizer(examples)
    learned = [
        {
            "id": index,
            "base64": base64.b64encode(piece).decode("ascii"),
            "bytes": len(piece),
        }
        for index, piece in enumerate(tokenizer.pieces)
    ]
    tokenizer_core = {
        "schema": "cimc.icmat.nanollm-tokenizer.v1",
        "name": VOCAB_SPEC,
        "status": "FROZEN_FOR_CORRECTIVE_GPU_TRAINING_BOARD_PENDING",
        "fit_split": "train_only",
        "vocab_size": VOCAB_SIZE,
        "special_ids": {"pad": PAD, "bos": BOS, "eos": EOS},
        "byte_fallback": {"base_id": BYTE_BASE, "count": 256},
        "learned_piece_base_id": FIRST_PIECE,
        "selection": "frequency_desc_length_desc_bytes_lexicographic_min_count_2",
        "encoding": "longest_piece_first_then_byte_fallback",
        "pieces": learned,
        "authority": 0,
    }
    tokenizer_sha = hashlib.sha256(canonical_bytes(tokenizer_core)).hexdigest()
    tokenizer_contract = {
        **tokenizer_core,
        "content_sha256": tokenizer_sha,
        "roundtrip_fixture": {
            text: tokenizer.decode(tokenizer.encode(text))
            for text in ("Evidence [1]", "烧结 temperature 725 °C", "REFUSE unsupported")
        },
    }
    if any(key != value for key, value in tokenizer_contract["roundtrip_fixture"].items()):
        raise RuntimeError("tokenizer roundtrip failure")
    tokenizer_path = root / "contracts" / "nanolm_tokenizer.v1.json"
    write_json(tokenizer_path, tokenizer_contract)
    stage_root = root / "data" / "staged_nanolm_v2"
    stage_root.mkdir(parents=True, exist_ok=True)
    manifest_records = []
    corpus_sha = sha256_file(corpus_path)
    for candidate_id, items in examples.items():
        encoded = [encode_example(tokenizer, item) for item in items]
        arrays = {
            key: np.asarray([item[key] for item in encoded])
            for key in ("x", "y", "loss_mask", "prompt_tokens", "prompt_length", "target_tokens", "target_length")
        }
        arrays.update(
            {
                "groups": np.asarray([item["group"] for item in items]),
                "split": np.asarray([SPLIT_CODE[item["split"]] for item in items], dtype=np.int8),
                "is_refusal": np.asarray([item["is_refusal"] for item in items], dtype=np.uint8),
                "candidate_id": np.asarray(candidate_id),
                "task_kind": np.asarray("nano_transformer_lm"),
                "truth_class": np.asarray("SOURCE_BOUND_QA_PLUS_STRUCTURE_DERIVED_REFUSAL"),
                "authority": np.asarray(0, dtype=np.int8),
            }
        )
        data_path = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(data_path, **arrays)
        split_counts = {
            name: int(np.sum(arrays["split"] == code)) for name, code in SPLIT_CODE.items()
        }
        group_sets = {
            code: set(arrays["groups"][arrays["split"] == code].tolist()) for code in SPLIT_CODE.values()
        }
        overlap = sum(len(group_sets[a] & group_sets[b]) for a in group_sets for b in group_sets if a < b)
        config = config_for_candidate(candidate_id).to_dict()
        metadata = {
            "schema": "cimc.forge200.staged-nanolm.v2",
            "status": "PASS_CORRECTIVE_DATASET_TEACHER_PENDING",
            "candidate_id": candidate_id,
            "task_kind": "nano_transformer_lm",
            "truth_class": "LITERATURE_CURATED_EXPERIMENT_SOURCE_BOUND_QA_AND_STRUCTURE_DERIVED_REFUSAL",
            "claim_state": "SOURCE_BOUND_OR_EXACT_CONTROLLED_NEGATIVE_NOT_INDEPENDENT_GROUND_TRUTH",
            "path": str(data_path.relative_to(root)).replace("\\", "/"),
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "records": len(items),
            "positive_records": int(np.sum(arrays["is_refusal"] == 0)),
            "refusal_records": int(np.sum(arrays["is_refusal"] == 1)),
            "split_counts": split_counts,
            "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_EXAMPLE_DERIVATION",
            "cross_split_group_overlap": overlap,
            "tokenizer_path": str(tokenizer_path.relative_to(root)).replace("\\", "/"),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "tokenizer_content_sha256": tokenizer_sha,
            "source_path": str(corpus_path.relative_to(root)).replace("\\", "/"),
            "source_sha256": corpus_sha,
            "task_contract_sha256": hashes[candidate_id],
            "architecture": config,
            "teacher_outputs": 0,
            "teacher_may_view_validation_or_test": False,
            "metric_coverage": [
                "answer_token_nll",
                "answer_token_f1",
                "citation_exact",
                "refusal_exact",
                "unsupported_answer_rate",
                "three_seed_variance",
                "quantized_token_parity",
            ],
            "authority": 0,
        }
        if overlap or min(split_counts.values()) < 16:
            raise RuntimeError(f"{candidate_id}: split gate failed {split_counts} overlap={overlap}")
        metadata_path = stage_root / f"{candidate_id}.metadata.json"
        write_json(metadata_path, metadata)
        manifest_records.append(metadata)
    manifest = {
        "schema": "cimc.forge200.nanollm-staging.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_CORRECTIVE_DATASETS_TEACHER_PENDING",
        "candidate_count": len(manifest_records),
        "candidates": [item["candidate_id"] for item in manifest_records],
        "records": sum(item["records"] for item in manifest_records),
        "tokenizer_sha256": sha256_file(tokenizer_path),
        "teacher_outputs": 0,
        "teacher_promoted_to_ground_truth": 0,
        "authority_nonzero": 0,
        "content_root_sha256": hashlib.sha256(canonical_bytes(manifest_records)).hexdigest(),
    }
    write_json(stage_root / "manifest.v2.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
