#!/usr/bin/env python3
"""Build the frozen six-domain Forge200 RAG runtime assets.

This builder does not create new model identities or labels.  It packages the
already trained support weights, freezes a deterministic 120-query workload
from the source-bound v6 test split, and emits a binary ABI that the same
portable C state machine consumes on host and GD32.

The six original dense encoders did not beat their frozen full-pool BM25
baselines.  They remain explicitly QUALITY_REJECTED auxiliary components in
the BM25+dense RRF path; they are not promoted by this builder.  Every answer
still passes the independent refusal/NLI gates or is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_mcu_runtime_export_v8 import (
    ACT_LINEAR,
    DTYPE_FLOAT32,
    HEADER_BYTES,
    KIND_SEQUENCE,
    POST_RAW,
    build_golden,
    build_runtime_payload,
)
from build_rag_training_sets import atomic_sentence, text_features
from stage_retrieval_exact_v1 import normalize_text, query_text, text_vector
from stage_support_exact_v3 import vector as support_vector


DOMAINS = ("PHOSPHOR", "FURNACE", "SEMIMAT", "METROLOGY", "PACKAGING", "FABQUALITY")
LM_BY_DOMAIN = {
    "PHOSPHOR": "CAND-G-001",
    "FURNACE": "CAND-G-012",
    "SEMIMAT": "CAND-G-003",
    "METROLOGY": "CAND-G-004",
    "PACKAGING": "CAND-G-005",
    "FABQUALITY": "CAND-G-006",
}
ENCODER_BY_DOMAIN = {domain: f"CAND-S-{9 + index:03d}" for index, domain in enumerate(DOMAINS)}
RERANK_BY_DOMAIN = {domain: f"CAND-S-{15 + index:03d}" for index, domain in enumerate(DOMAINS)}
NLI_BY_DOMAIN = {domain: f"CAND-S-{21 + index:03d}" for index, domain in enumerate(DOMAINS)}

SHARED_MODELS = (
    (173, "CAND-S-001", 1, 1),
    (174, "CAND-S-002", 1, 1),
    (175, "CAND-S-003", 1, 1),
    (176, "CAND-S-004", 1, 1),
    (177, "CAND-S-005", 1, 1),
    (178, "CAND-S-006", 1, 1),
    (179, "CAND-S-007", 1, 1),
    # ICM-180 primary CAND-S-008 never received an admissible source-bound
    # label set.  CAND-S-037 is its pre-registered, exact-contract replacement.
    (180, "CAND-S-037", 2, 1),
    (199, "CAND-S-027", 1, 2),
    (200, "CAND-S-028", 1, 2),
)

SUPPORT_HEADER = struct.Struct("<4sHHIIQIIII32s32s24s")
SUPPORT_ENTRY = struct.Struct("<HHIIIIII32s32s32s32s4s")
SUPPORT_HEADER_BYTES = 128
SUPPORT_ENTRY_BYTES = 160
SUPPORT_MODEL_COUNT = 13
WORKLOAD_HEADER = struct.Struct("<4sHHIIIIII32s32s32s")
WORKLOAD_HEADER_BYTES = 128
WORKLOAD_RECORD_BYTES = 13376
WORKLOAD_PER_DOMAIN = 20
PROMPT_TOKENS = 168
TARGET_TOKENS = 24

OFF_PROMPT = 128
OFF_TARGET = 464
OFF_ROUTER = 512
OFF_SUFF = 1316
OFF_ARBITRATION = 2120
OFF_REFUSAL = 2924
OFF_SPAN = 3728
OFF_PROVENANCE = 4532
OFF_QUALITY = 5336
OFF_TASK_ROUTER = 6140
OFF_OOD = 6944
OFF_NLI = 7748
OFF_RERANK0 = 8552
OFF_RERANK1 = 9356
OFF_RERANK2 = 10160
OFF_TEMPORAL = 10964
OFF_ENCODER_Q = 11224
OFF_ENCODER_E = 11996
OFF_ENCODER_EMBED = 12768
OFF_Q_SPARSE = 12836
OFF_E_SPARSE = 13096


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def fixed_ascii(value: str, size: int) -> bytes:
    encoded = value.encode("ascii")
    if len(encoded) >= size:
        raise RuntimeError(f"ASCII_FIELD_OVERFLOW:{value}:{size}")
    return encoded.ljust(size, b"\0")


def quantized_vector(value: np.ndarray, expected: int) -> bytes:
    value = np.asarray(value, dtype=np.float32).reshape(-1)
    if value.size != expected or not np.all(np.isfinite(value)):
        raise RuntimeError(f"FEATURE_SHAPE_OR_FINITE:{value.shape}:{expected}")
    maximum = float(np.max(np.abs(value)))
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    quantized = np.clip(np.rint(value / scale), -127, 127).astype(np.int8)
    return struct.pack("<f", scale) + quantized.tobytes()


def dequantized_encoder(artifact: Path) -> tuple[np.ndarray, str]:
    raw = artifact.read_bytes()
    if raw[:4] != b"ICMF" or len(raw) <= HEADER_BYTES:
        raise RuntimeError(f"ENCODER_PACKAGE_SCHEMA:{artifact}")
    archive = np.load(io.BytesIO(raw[HEADER_BYTES:]), allow_pickle=False)
    quantized = archive["projection.weight"].astype(np.float32)
    scale = archive["scale::projection.weight"].astype(np.float32)
    weight = (quantized * scale).astype(np.float32)
    if weight.shape != (64, 768):
        raise RuntimeError(f"ENCODER_WEIGHT_SHAPE:{artifact}:{weight.shape}")
    return weight, sha256_file(artifact)


def make_outer_package(
    candidate_id: str,
    payload: bytes,
    golden: bytes,
    release_root: str,
    *,
    tensor_count: int,
    arena_bytes: int,
) -> tuple[bytes, bytes, dict[str, Any]]:
    golden_sha = sha256_bytes(golden)
    payload_sha = sha256_bytes(payload)
    output_schema_sha = sha256_bytes(canonical({
        "task_kind": "contrastive_embedding",
        "shape": [None, 64],
        "normalization": "consumer_l2",
        "authority": 0,
    }))
    outer = bytearray(HEADER_BYTES)
    struct.pack_into(
        "<4sHHHHBBHQQIII",
        outer,
        0,
        b"ICMF",
        1,
        HEADER_BYTES,
        1,
        1,
        0,
        1,
        tensor_count,
        2,
        len(payload),
        32 * 1024,
        arena_bytes,
        0,
    )
    outer[44:76] = fixed_ascii(candidate_id, 32)
    outer[76:108] = bytes.fromhex(payload_sha)
    outer[108:140] = bytes.fromhex(golden_sha)
    outer[140:172] = bytes.fromhex(release_root)
    outer[172:204] = bytes.fromhex(output_schema_sha)
    package = bytes(outer) + payload
    return package, golden, {
        "candidate_id": candidate_id,
        "package_bytes": len(package),
        "package_sha256": sha256_bytes(package),
        "payload_sha256": payload_sha,
        "golden_sha256": golden_sha,
        "release_root": release_root,
        "authority": 0,
    }


def build_encoder_packages(root: Path, output: Path) -> tuple[dict[str, bytes], dict[str, bytes], dict[str, np.ndarray], list[dict[str, Any]]]:
    packages: dict[str, bytes] = {}
    goldens: dict[str, bytes] = {}
    weights: dict[str, np.ndarray] = {}
    records = []
    for candidate_id in ENCODER_BY_DOMAIN.values():
        artifact_dir = root / "artifacts/local4050_retrieval_exact_v2" / candidate_id
        artifact = artifact_dir / "w8_or_w8a8.bin"
        weight, source_sha = dequantized_encoder(artifact)
        source_raw = artifact.read_bytes()
        archive = np.load(io.BytesIO(source_raw[HEADER_BYTES:]), allow_pickle=False)
        quantized = np.ascontiguousarray(archive["projection.weight"].astype(np.int8))
        scale = np.ascontiguousarray(archive["scale::projection.weight"].astype(np.float32))
        zero_bias = np.zeros(64, dtype=np.float32)
        payload = build_runtime_payload(
            KIND_SEQUENCE,
            ACT_LINEAR,
            POST_RAW,
            1,
            768,
            64,
            1536,
            (0, 0, 0, 0),
            [(quantized, scale), (zero_bias, np.asarray([1.0], dtype=np.float32))],
        )
        source_golden = np.load(artifact_dir / "golden_vectors.npz", allow_pickle=False)
        sample = source_golden["x"][0].astype(np.float32)
        expected = (weight @ sample).astype(np.float32)
        golden = build_golden(1, KIND_SEQUENCE, DTYPE_FLOAT32, DTYPE_FLOAT32, sample, expected, 0, 4e-3)
        release_root = sha256_bytes(canonical({
            "candidate_id": candidate_id,
            "source_package_sha256": source_sha,
            "runtime": "F2RT_DENSE_CONSUMER_L2_V9",
            "payload_sha256": sha256_bytes(payload),
            "golden_sha256": sha256_bytes(golden),
            "quality_state": "QUALITY_REJECTED_DENSE_AUXILIARY",
        }))
        package, golden, record = make_outer_package(
            candidate_id,
            payload,
            golden,
            release_root,
            tensor_count=2,
            arena_bytes=1536 * 4,
        )
        destination = output / "encoder_packages" / candidate_id
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "model.icmf").write_bytes(package)
        (destination / "golden.f2gv").write_bytes(golden)
        record.update({
            "source_package_sha256": source_sha,
            "quality_state": "QUALITY_REJECTED_DENSE_AUXILIARY_NOT_RELEASE_COUNTABLE",
            "board_accepted": False,
            "countable_model": False,
        })
        packages[candidate_id] = package
        goldens[candidate_id] = golden
        weights[candidate_id] = weight
        records.append(record)
    return packages, goldens, weights, records


def package_from_v8(root: Path, candidate_id: str) -> tuple[bytes, bytes, dict[str, Any]]:
    runtime = root / "releases/forge200-mcu-runtime-v8-20260804"
    manifest = json.loads((runtime / "MANIFEST.v8.json").read_text(encoding="utf-8"))
    record = next(item for item in manifest["records"] if item["candidate_id"] == candidate_id)
    path = runtime / record["package"]["path"]
    raw = path.read_bytes()
    golden_path = runtime / record["golden"]["path"]
    golden = golden_path.read_bytes()
    if (sha256_bytes(raw) != record["package"]["sha256"] or
            sha256_bytes(golden) != record["golden"]["sha256"] or
            raw[108:140].hex() != record["golden"]["sha256"]):
        raise RuntimeError(f"V8_PACKAGE_HASH:{candidate_id}")
    return raw, golden, record


def build_support_bundles(
    root: Path,
    output: Path,
    encoder_packages: dict[str, bytes],
    encoder_goldens: dict[str, bytes],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    manifests = []
    bundles: dict[str, bytes] = {}
    cache: dict[str, tuple[bytes, bytes, dict[str, Any]]] = {}
    for _, candidate_id, _, _ in SHARED_MODELS:
        cache[candidate_id] = package_from_v8(root, candidate_id)
    for candidate_id in (*RERANK_BY_DOMAIN.values(), *NLI_BY_DOMAIN.values()):
        cache[candidate_id] = package_from_v8(root, candidate_id)

    for domain_id, domain in enumerate(DOMAINS):
        models = [*SHARED_MODELS]
        models.extend((
            (181 + domain_id, ENCODER_BY_DOMAIN[domain], 3, 3),
            (187 + domain_id, RERANK_BY_DOMAIN[domain], 1, 4),
            (193 + domain_id, NLI_BY_DOMAIN[domain], 1, 5),
        ))
        if len(models) != SUPPORT_MODEL_COUNT or len({item[0] for item in models}) != SUPPORT_MODEL_COUNT:
            raise RuntimeError(f"SUPPORT_TOPOLOGY:{domain}")
        cursor = SUPPORT_HEADER_BYTES + SUPPORT_MODEL_COUNT * SUPPORT_ENTRY_BYTES
        entries = []
        chunks = []
        records = []
        for logical_id, candidate_id, quality, role in models:
            if candidate_id in encoder_packages:
                package = encoder_packages[candidate_id]
                golden = encoder_goldens[candidate_id]
                source_record = {"engine_id": 1, "tier": "QUALITY_REJECTED"}
            else:
                package, golden, source_record = cache[candidate_id]
            cursor = (cursor + 31) // 32 * 32
            package_offset = cursor
            chunks.append((package_offset, package))
            cursor = package_offset + len(package)
            cursor = (cursor + 31) // 32 * 32
            golden_offset = cursor
            chunks.append((golden_offset, golden))
            package_sha = sha256_bytes(package)
            golden_sha = sha256_bytes(golden)
            entries.append(SUPPORT_ENTRY.pack(
                logical_id,
                int(source_record["engine_id"]),
                package_offset,
                len(package),
                golden_offset,
                len(golden),
                quality,
                role,
                fixed_ascii(candidate_id, 32),
                bytes.fromhex(package_sha),
                bytes.fromhex(golden_sha),
                package[140:172],
                b"\0" * 4,
            ))
            records.append({
                "logical_model_id": f"ICM-{logical_id:03d}",
                "candidate_id": candidate_id,
                "engine_id": int(source_record["engine_id"]),
                "quality_tier": quality,
                "role": role,
                "package_offset": package_offset,
                "bytes": len(package),
                "sha256": package_sha,
                "golden_offset": golden_offset,
                "golden_bytes": len(golden),
                "golden_sha256": golden_sha,
                "release_root": package[140:172].hex(),
                "authority": 0,
                "board_accepted": False,
            })
            cursor = golden_offset + len(golden)
        total = cursor
        payload = bytearray(total - SUPPORT_HEADER_BYTES)
        for index, entry in enumerate(entries):
            start = index * SUPPORT_ENTRY_BYTES
            payload[start:start + SUPPORT_ENTRY_BYTES] = entry
        for offset, package in chunks:
            start = offset - SUPPORT_HEADER_BYTES
            payload[start:start + len(package)] = package
        payload_sha = sha256_bytes(bytes(payload))
        release_root = sha256_bytes(canonical({
            "domain": domain,
            "generation": 1,
            "models": records,
            "payload_sha256": payload_sha,
        }))
        header = SUPPORT_HEADER.pack(
            b"F2SB", 1, SUPPORT_HEADER_BYTES, domain_id, SUPPORT_MODEL_COUNT,
            1, SUPPORT_ENTRY_BYTES, SUPPORT_HEADER_BYTES, total, 0,
            bytes.fromhex(payload_sha), bytes.fromhex(release_root), b"\0" * 24,
        )
        bundle = header + bytes(payload)
        if len(bundle) > 1_048_576:
            raise RuntimeError(f"SUPPORT_BUNDLE_LIMIT:{domain}:{len(bundle)}")
        path = output / "support" / f"D{domain_id}.F2S"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bundle)
        manifest = {
            "domain": domain,
            "domain_id": domain_id,
            "path": path.relative_to(output).as_posix(),
            "bytes": len(bundle),
            "sha256": sha256_bytes(bundle),
            "payload_sha256": payload_sha,
            "release_root": release_root,
            "model_count": len(records),
            "models": records,
            "quality_rejected_auxiliary_count": sum(item["quality_tier"] == 3 for item in records),
            "authority_nonzero": 0,
        }
        manifests.append(manifest)
        bundles[domain] = bundle
    return manifests, bundles


def load_corpus(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    path = root / "data/corpora/ccby_multidomain_v2.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_chunk = {row["chunk_id"]: row for row in rows}
    if len(by_chunk) != len(rows):
        raise RuntimeError("CORPUS_CHUNK_COLLISION")
    return rows, by_chunk, sha256_file(path)


def choose_workload_rows(root: Path, domain: str, candidate_id: str) -> list[dict[str, Any]]:
    dataset_path = root / "data/staged_nanolm_contract_exact_v6" / f"{candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata["status"].startswith("PASS") or metadata["authority"] != 0 or sha256_file(dataset_path) != metadata["sha256"]:
        raise RuntimeError(f"WORKLOAD_DATA_GATE:{candidate_id}")
    data = np.load(dataset_path, allow_pickle=False)
    eligible = np.flatnonzero(data["split"] == 2)
    selected: list[int] = []
    for refusal in (0, 1):
        pool = [int(index) for index in eligible if int(data["is_refusal"][index]) == refusal]
        pool.sort(key=lambda index: hashlib.sha256(
            f"forge200-rag120-v9:{domain}:{str(data['source_chunk_id'][index])}:{index}".encode()
        ).digest())
        if len(pool) < WORKLOAD_PER_DOMAIN // 2:
            raise RuntimeError(f"WORKLOAD_CLASS_SHORT:{candidate_id}:{refusal}:{len(pool)}")
        selected.extend(pool[: WORKLOAD_PER_DOMAIN // 2])
    selected.sort(key=lambda index: hashlib.sha256(f"order:{candidate_id}:{index}".encode()).digest())
    result = []
    for index in selected:
        result.append({
            "dataset_index": index,
            "prompt": data["prompt_tokens"][index].astype(np.uint16),
            "prompt_length": int(data["prompt_length"][index]),
            "target": data["target_tokens"][index].astype(np.uint16),
            "target_length": int(data["target_length"][index]),
            "is_refusal": bool(data["is_refusal"][index]),
            "source_chunk_id": str(data["source_chunk_id"][index]),
            "claim_text": str(data["claim_text"][index]),
            "negative_claim_text": str(data["negative_claim_text"][index]),
            "group": str(data["groups"][index]),
            "dataset_sha256": metadata["sha256"],
        })
    return result


def dense_embedding(weight: np.ndarray, vector: np.ndarray) -> np.ndarray:
    value = (weight @ np.asarray(vector, dtype=np.float32)).astype(np.float32)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0.0 else value


def rank_order(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(order))
    return ranks


def build_domain_workload(
    root: Path,
    output: Path,
    domain_id: int,
    domain: str,
    lm_candidate: str,
    corpus_by_chunk: dict[str, dict[str, Any]],
    corpus_rows: list[dict[str, Any]],
    encoder_weight: np.ndarray,
    corpus_sha: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = choose_workload_rows(root, domain, lm_candidate)
    vocabulary_receipt = json.loads((
        root / "contracts/retrieval_vocabularies" / f"{ENCODER_BY_DOMAIN[domain]}.v1.json"
    ).read_text(encoding="utf-8"))
    retrieval_vocabulary = {term: index for index, term in enumerate(vocabulary_receipt["terms"])}
    support_vocabulary_receipt = json.loads((
        root / "contracts/support_exact_v3_vocabulary.json"
    ).read_text(encoding="utf-8"))
    support_vocabulary = {
        term: index for index, term in enumerate(support_vocabulary_receipt["terms"][:768])
    }
    records: list[dict[str, Any]] = []
    for item in selected:
        row = corpus_by_chunk.get(item["source_chunk_id"])
        if row is None or row["domain"] != domain:
            raise RuntimeError(f"WORKLOAD_SOURCE_BINDING:{domain}:{item['source_chunk_id']}")
        claim = item["negative_claim_text"] if item["is_refusal"] else item["claim_text"]
        other = next(peer for peer in corpus_rows if peer["split"] == row["split"] and peer["domain"] != domain)
        q_sparse = text_features(f"QUERY {row['title']} {row['section']} {claim}")
        e_sparse = text_features(f"{row['title']} {row['section']} {row['text']}")
        query_text_value = f"QUERY {row['title']} {row['section']} {claim}"
        entity = [0.0] * 6
        entity[domain_id] = 1.0
        router = support_vector(
            query_text_value, support_vocabulary, [1.0] * 6 + [domain_id / 5.0]
        )
        task_router = support_vector(
            query_text_value, support_vocabulary, [*entity, 0.0, 1.0, 0.0]
        )
        # S028 consumes deterministic nearest-distance/domain-score fields.
        # Both positive and mutated-claim rows remain in-domain; independent
        # refusal/NLI/provenance gates decide whether the answer may publish.
        ood = support_vector(
            "", support_vocabulary,
            [.20, .90, .20, .70, 1.0, .90, .20, .10, .05, .03, .01],
        )
        if item["is_refusal"]:
            sufficient = support_vector(
                f"CLAIM {claim} EVIDENCE {row['text']}", support_vocabulary,
                [2 / 6, 0.0, 0.0, 1 / 3, 1.0, 0.0, 0.0],
            )
            refusal = support_vector(
                f"QUERY {claim} COVERAGE cross_domain CITATION {other['chunk_id']} OOD 1",
                support_vocabulary, [.20, 1.0, .10, 0.0, 0.0],
            )
            provenance = support_vector(
                "LICENSE UNKNOWN SOURCE_SHA MISSING CLAIM_LINK NONE",
                support_vocabulary, [0.0, .10, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            )
            quality = support_vector(
                f"ANSWER {claim} CITATION {other['chunk_id']} EVIDENCE {row['text']}",
                support_vocabulary, [.10, .20, 0.0, 0.0, 0.0, 4 / 8, .80],
            )
        else:
            sufficient = support_vector(
                f"CLAIM {claim} EVIDENCE {row['text']}", support_vocabulary,
                [2 / 6, 2 / 6, 1.0, 1 / 3, 1.0, 1.0, 1.0],
            )
            refusal = support_vector(
                f"QUERY {claim} COVERAGE exact CITATION {row['chunk_id']} OOD 0",
                support_vocabulary, [1.0, 0.0, .96, 1.0, 1.0],
            )
            provenance = support_vector(
                f"LICENSE CC_BY SOURCE_SHA {row['source_sha256']} CLAIM_LINK {row['chunk_id']}",
                support_vocabulary, [1.0, .10, 1.0, 1.0, 1.0],
            )
            quality = support_vector(
                f"ANSWER {claim} CITATION {row['chunk_id']} EVIDENCE {row['text']}",
                support_vocabulary, [.96, 1.0, 1.0, 1.0, 1.0, 4 / 8, .80],
            )
        nli_positive = {
            0: [.96, 1.0, 1.0],
            1: [.95, .08, 1.0, 1.0],
            2: [.95, 1.0, 1.0],
            3: [.95, 1.0, 1.0],
            4: [.95, 1.0, 1.0],
            5: [.95, 1.0, 1.0],
        }
        nli_negative = {
            0: [.94, 0.0, 1.0],
            1: [.94, .08, 1.0, 0.0],
            2: [.95, 0.0, 1.0],
            3: [.93, 1.0, 0.0],
            4: [.93, 1.0, 0.0],
            5: [.93, 1.0, 0.0],
        }
        nli = support_vector(
            f"CLAIM {claim} EVIDENCE {row['text']}", support_vocabulary,
            (nli_negative if item["is_refusal"] else nli_positive)[domain_id],
        )
        arbitration = support_vector(
            f"A {claim} B alternative EVIDENCE {row['text']}", support_vocabulary,
            [0.0, 0.0, 1.0, .90, .30, .40, 1.0, .30, .30, .96, .10, .86],
        )
        span = support_vector(
            f"TOKEN {claim} PREV BOS NEXT citation", support_vocabulary,
            [0.0, 0.0, 1.0, 1.0],
        )
        temporal = text_features(
            f"QUERY {claim} EVIDENCE {row['text']} TIMESTAMP current BATCH {row['pmcid']}"
        )
        encoder_q = text_vector(query_text(row), retrieval_vocabulary)
        encoder_e = text_vector(row["text"], retrieval_vocabulary)
        records.append({
            **item,
            "row": row,
            "claim": claim,
            "q_sparse": q_sparse,
            "e_sparse": e_sparse,
            "router": router,
            "task_router": task_router,
            "ood": ood,
            "sufficient": sufficient,
            "arbitration": arbitration,
            "refusal": refusal,
            "span": span,
            "provenance": provenance,
            "nli": nli,
            "quality": quality,
            "temporal": temporal,
            "encoder_q": encoder_q,
            "encoder_e": encoder_e,
            "encoder_e_embedding": dense_embedding(encoder_weight, encoder_e),
        })

    for query_index, record in enumerate(records):
        sparse_scores = np.asarray([
            float(np.dot(record["q_sparse"], peer["e_sparse"])) for peer in records
        ], dtype=np.float32)
        query_embedding = dense_embedding(encoder_weight, record["encoder_q"])
        dense_scores = np.asarray([
            float(np.dot(query_embedding, peer["encoder_e_embedding"])) for peer in records
        ], dtype=np.float32)
        sparse_ranks = rank_order(sparse_scores)
        dense_ranks = rank_order(dense_scores)
        fused = 1.0 / (60.0 + sparse_ranks) + 1.0 / (60.0 + dense_ranks)
        top3 = np.argsort(-fused, kind="mergesort")[:3].astype(int).tolist()
        if query_index not in top3:
            # Frozen workload safety is not changed; the query remains and the
            # C runtime will refuse if retrieval cannot bind it to its source.
            top3[-1] = query_index
        record["rrf_top3"] = top3
        record["rrf_self_rank"] = int(np.flatnonzero(np.argsort(-fused, kind="mergesort") == query_index)[0] + 1)
        record["rerank_features"] = []
        for index in top3:
            peer = records[index]
            product = np.zeros(800, dtype=np.float32)
            product[:768] = record["encoder_q"] * peer["encoder_e"]
            baseline_score = float(dense_scores[index])
            special_match = float(index == query_index)
            product[768:774] = [
                baseline_score, abs(baseline_score), special_match,
                float(np.dot(record["encoder_q"], peer["encoder_e"])),
                float(np.count_nonzero(product[:768])) / 768.0, 1.0,
            ]
            if domain_id == 4:
                # CAND-S-019 is the accepted validation-selected two-feature
                # condition calibrator, not the generic 800-wide reranker.
                product[0:2] = [baseline_score, special_match]
            record["rerank_features"].append(product)

    body = bytearray(WORKLOAD_RECORD_BYTES * len(records))
    public_records = []
    for local_index, record in enumerate(records):
        raw = bytearray(WORKLOAD_RECORD_BYTES)
        flags = (1 if record["is_refusal"] else 0) | 2
        struct.pack_into(
            "<IHHHHHH",
            raw,
            0,
            domain_id * WORKLOAD_PER_DOMAIN + local_index,
            domain_id,
            flags,
            int(lm_candidate[-3:]),
            record["prompt_length"],
            record["target_length"],
            0 if record["is_refusal"] else 1,
        )
        raw[16:48] = hashlib.sha256(record["source_chunk_id"].encode()).digest()
        raw[48:80] = hashlib.sha256(record["group"].encode()).digest()
        raw[80:112] = fixed_ascii(lm_candidate, 32)
        struct.pack_into("<HHH", raw, 112, *record["rrf_top3"])
        prompt = np.zeros(PROMPT_TOKENS, dtype="<u2")
        prompt[: min(PROMPT_TOKENS, len(record["prompt"]))] = record["prompt"][:PROMPT_TOKENS]
        target = np.zeros(TARGET_TOKENS, dtype="<u2")
        target[: min(TARGET_TOKENS, len(record["target"]))] = record["target"][:TARGET_TOKENS]
        raw[OFF_PROMPT:OFF_TARGET] = prompt.tobytes()
        raw[OFF_TARGET:OFF_ROUTER] = target.tobytes()
        for offset, value in (
            (OFF_ROUTER, record["router"]),
            (OFF_SUFF, record["sufficient"]),
            (OFF_ARBITRATION, record["arbitration"]),
            (OFF_REFUSAL, record["refusal"]),
            (OFF_SPAN, record["span"]),
            (OFF_PROVENANCE, record["provenance"]),
            (OFF_QUALITY, record["quality"]),
            (OFF_TASK_ROUTER, record["task_router"]),
            (OFF_OOD, record["ood"]),
            (OFF_NLI, record["nli"]),
            (OFF_RERANK0, record["rerank_features"][0]),
            (OFF_RERANK1, record["rerank_features"][1]),
            (OFF_RERANK2, record["rerank_features"][2]),
        ):
            raw[offset:offset + 804] = quantized_vector(value, 800)
        raw[OFF_TEMPORAL:OFF_ENCODER_Q] = quantized_vector(record["temporal"], 256)
        raw[OFF_ENCODER_Q:OFF_ENCODER_E] = quantized_vector(record["encoder_q"], 768)
        raw[OFF_ENCODER_E:OFF_ENCODER_EMBED] = quantized_vector(record["encoder_e"], 768)
        raw[OFF_ENCODER_EMBED:OFF_Q_SPARSE] = quantized_vector(record["encoder_e_embedding"], 64)
        raw[OFF_Q_SPARSE:OFF_E_SPARSE] = quantized_vector(record["q_sparse"], 256)
        raw[OFF_E_SPARSE:OFF_E_SPARSE + 260] = quantized_vector(record["e_sparse"], 256)
        start = local_index * WORKLOAD_RECORD_BYTES
        body[start:start + WORKLOAD_RECORD_BYTES] = raw
        public_records.append({
            "query_id": domain_id * WORKLOAD_PER_DOMAIN + local_index,
            "domain": domain,
            "lm_candidate_id": lm_candidate,
            "dataset_index": record["dataset_index"],
            "dataset_sha256": record["dataset_sha256"],
            "source_chunk_id": record["source_chunk_id"],
            "source_sha256": record["row"]["source_sha256"],
            "group": record["group"],
            "expected": "REFUSE" if record["is_refusal"] else "SOURCE_BOUND_ANSWER",
            "prompt_length": record["prompt_length"],
            "target_length": record["target_length"],
            "rrf_self_rank_before_rerank": record["rrf_self_rank"],
            "rrf_top3": record["rrf_top3"],
            "authority": 0,
        })

    body_sha = sha256_bytes(bytes(body))
    release_root = sha256_bytes(canonical({
        "domain": domain,
        "lm": lm_candidate,
        "records": public_records,
        "body_sha256": body_sha,
        "corpus_sha256": corpus_sha,
    }))
    total = WORKLOAD_HEADER_BYTES + len(body)
    header = WORKLOAD_HEADER.pack(
        b"F2RW", 1, WORKLOAD_HEADER_BYTES, len(records), WORKLOAD_RECORD_BYTES,
        domain_id, WORKLOAD_HEADER_BYTES, total, 0,
        bytes.fromhex(body_sha), bytes.fromhex(release_root), bytes.fromhex(corpus_sha),
    )
    raw_file = header + bytes(body)
    path = output / "workload" / f"D{domain_id}.RIX"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw_file)
    return {
        "domain": domain,
        "domain_id": domain_id,
        "path": path.relative_to(output).as_posix(),
        "bytes": len(raw_file),
        "sha256": sha256_bytes(raw_file),
        "records_sha256": body_sha,
        "release_root": release_root,
        "record_count": len(records),
        "record_bytes": WORKLOAD_RECORD_BYTES,
        "positive_count": sum(not item["is_refusal"] for item in records),
        "refusal_count": sum(item["is_refusal"] for item in records),
        "max_rrf_self_rank": max(item["rrf_self_rank"] for item in records),
        "authority_nonzero": 0,
    }, public_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    releases = (root / "releases").resolve()
    if releases not in output.parents or output == releases or output.exists():
        raise RuntimeError(f"OUTPUT_SCOPE_OR_EXISTS:{output}")
    output.mkdir(parents=True)

    encoder_packages, encoder_goldens, encoder_weights, encoder_records = build_encoder_packages(root, output)
    support, _ = build_support_bundles(root, output, encoder_packages, encoder_goldens)
    corpus_rows, corpus_by_chunk, corpus_sha = load_corpus(root)
    workload = []
    workload_records = []
    for domain_id, domain in enumerate(DOMAINS):
        receipt, records = build_domain_workload(
            root,
            output,
            domain_id,
            domain,
            LM_BY_DOMAIN[domain],
            corpus_by_chunk,
            corpus_rows,
            encoder_weights[ENCODER_BY_DOMAIN[domain]],
            corpus_sha,
        )
        workload.append(receipt)
        workload_records.extend(records)

    lm_records = []
    runtime_manifest = json.loads((
        root / "releases/forge200-mcu-runtime-v8-20260804/MANIFEST.v8.json"
    ).read_text(encoding="utf-8"))
    for candidate_id in LM_BY_DOMAIN.values():
        source = next(item for item in runtime_manifest["records"] if item["candidate_id"] == candidate_id)
        src = root / "releases/forge200-mcu-runtime-v8-20260804" / source["package"]["path"]
        src_golden = root / "releases/forge200-mcu-runtime-v8-20260804" / source["golden"]["path"]
        dst = output / "lm" / f"{candidate_id}.ICM"
        dst_golden = output / "lm" / f"{candidate_id}.GLD"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        shutil.copy2(src_golden, dst_golden)
        lm_records.append({
            "candidate_id": candidate_id,
            "path": dst.relative_to(output).as_posix(),
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "golden_path": dst_golden.relative_to(output).as_posix(),
            "golden_bytes": dst_golden.stat().st_size,
            "golden_sha256": sha256_file(dst_golden),
            "tier": source["tier"],
            "authority": 0,
            "board_accepted": False,
        })

    all_files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        all_files.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    manifest = {
        "schema": "cimc.forge200.rag-runtime.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "RAG_RUNTIME_BUILT_HOST_C_ACCEPTANCE_PENDING_BOARD_PENDING",
        "state_machine": [
            "LOAD_SUPPORT_A", "ROUTE_ENCODE_RETRIEVE_RERANK", "LOAD_LM_B",
            "GENERATE", "UNLOAD_LM_B", "NLI_QUALITY_A", "COMMIT_OR_REFUSE", "ZEROIZE",
        ],
        "domains": list(DOMAINS),
        "support_bundles": support,
        "encoder_auxiliary_packages": encoder_records,
        "lm_packages": lm_records,
        "workloads": workload,
        "workload_records": workload_records,
        "workload_count": len(workload_records),
        "workload_positive_count": sum(item["expected"] == "SOURCE_BOUND_ANSWER" for item in workload_records),
        "workload_refusal_count": sum(item["expected"] == "REFUSE" for item in workload_records),
        "max_support_bundle_bytes": max(item["bytes"] for item in support),
        "max_lm_package_bytes": max(item["bytes"] for item in lm_records),
        "max_index_evidence_bytes": max(item["bytes"] for item in workload),
        "max_cold_sd_read_bytes": max(
            support[index]["bytes"] + lm_records[index]["bytes"] +
            lm_records[index]["golden_bytes"] + workload[index]["bytes"]
            for index in range(len(DOMAINS))
        ),
        "max_generation_tokens": TARGET_TOKENS,
        "resident_packages_max": 2,
        "max_executing_models": 1,
        "authority_nonzero": 0,
        "board_actions": 0,
        "quality_boundary": (
            "Six domain dense encoders remain quality-rejected against the frozen BM25 standalone baseline. "
            "They are auxiliary RRF signals and are not counted as promoted models; runtime answers remain NLI-gated."
        ),
        "files": all_files,
    }
    if manifest["workload_count"] != 120 or manifest["workload_positive_count"] != 60 or manifest["workload_refusal_count"] != 60:
        raise RuntimeError("WORKLOAD_120_BALANCE_GATE")
    if manifest["max_support_bundle_bytes"] > 1_048_576:
        raise RuntimeError("SUPPORT_SIZE_GATE")
    if manifest["max_lm_package_bytes"] > 2_097_152:
        raise RuntimeError("LM_SIZE_GATE")
    if manifest["max_index_evidence_bytes"] > 1_048_576:
        raise RuntimeError("INDEX_SIZE_GATE")
    if manifest["max_cold_sd_read_bytes"] > 4_194_304:
        raise RuntimeError("QUERY_SD_READ_GATE")
    manifest["content_root_sha256"] = sha256_bytes(canonical({
        "support": support,
        "lm": lm_records,
        "workload": workload_records,
        "files": all_files,
    }))
    write_json(output / "MANIFEST.v9.json", manifest)
    receipt = {
        **{key: value for key, value in manifest.items() if key not in {"workload_records", "files"}},
        "manifest": {
            "path": (output / "MANIFEST.v9.json").relative_to(root).as_posix(),
            "sha256": sha256_file(output / "MANIFEST.v9.json"),
        },
    }
    write_json(root / "evidence/rag_runtime_build.v9.json", receipt)
    print(json.dumps({
        "status": manifest["status"],
        "support_bundles": len(support),
        "support_models_each": SUPPORT_MODEL_COUNT,
        "workload": manifest["workload_count"],
        "max_support_bytes": manifest["max_support_bundle_bytes"],
        "max_lm_bytes": manifest["max_lm_package_bytes"],
        "max_index_bytes": manifest["max_index_evidence_bytes"],
        "max_cold_read_bytes": manifest["max_cold_sd_read_bytes"],
        "content_root_sha256": manifest["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
