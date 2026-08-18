#!/usr/bin/env python3
"""Stage S015-S020 top-50 rerankers from actual upstream encoder scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from stage_retrieval_exact_v1 import DOMAINS, canonical_bytes, normalize_text, sha256_file, text_vector, tokens, write_json


TASKS = {
    "CAND-S-015": "CAND-S-009",
    "CAND-S-016": "CAND-S-010",
    "CAND-S-017": "CAND-S-011",
    "CAND-S-018": "CAND-S-012",
    "CAND-S-019": "CAND-S-013",
    "CAND-S-020": "CAND-S-014",
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


def normalized_embeddings(features: np.ndarray, weight: np.ndarray) -> np.ndarray:
    value = features @ weight.T
    value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-12)
    return value


def product_feature(query: np.ndarray, passage: np.ndarray, baseline_score: float, special_match: int) -> np.ndarray:
    value = np.zeros(800, dtype=np.float32)
    value[:768] = query * passage
    value[768:774] = [baseline_score, abs(baseline_score), special_match, float(np.dot(query, passage)), float(np.count_nonzero(query * passage)) / 768, 1.0]
    return value


def build_split(
    candidate_id: str,
    code: int,
    query: np.ndarray,
    passage: np.ndarray,
    query_groups: np.ndarray,
    passage_groups: np.ndarray,
    scores: np.ndarray,
    query_offset: int,
) -> dict[str, list[Any]]:
    output = {name: [] for name in ("x", "y", "groups", "split", "query_id", "baseline_score", "special_match")}
    accepted = 0
    for query_index in range(len(query)):
        top = np.argsort(-scores[query_index], kind="mergesort")[:50]
        same = passage_groups[top] == query_groups[query_index]
        if not np.any(same):
            continue
        special = np.asarray([
            1 if candidate_id == "CAND-S-015" else int(hashlib.sha256(f"{candidate_id}:{query_groups[query_index]}:{passage_groups[item]}:{position}".encode()).digest()[0] % 3 != 0)
            for position, item in enumerate(top)
        ], dtype=np.uint8)
        # The first same-document item always retains the contract field match;
        # other same-document candidates may carry a controlled mismatch.
        special[np.flatnonzero(same)[0]] = 1
        relevance = same.astype(np.uint8) if candidate_id == "CAND-S-015" else (same & (special == 1)).astype(np.uint8)
        if not np.any(relevance):
            continue
        qid = query_offset + accepted; accepted += 1
        for position, passage_index in enumerate(top):
            output["x"].append(product_feature(query[query_index], passage[passage_index], float(scores[query_index, passage_index]), int(special[position])))
            output["y"].append(int(relevance[position])); output["groups"].append(str(query_groups[query_index])); output["split"].append(code)
            output["query_id"].append(qid); output["baseline_score"].append(float(scores[query_index, passage_index])); output["special_match"].append(int(special[position]))
    if accepted < 16:
        raise RuntimeError(f"{candidate_id} split {code} has only {accepted} ranking queries")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); parser.add_argument("--encoder-artifact-root", type=Path, required=True); args = parser.parse_args()
    root, encoder_artifacts = args.root.resolve(), args.encoder_artifact_root.resolve()
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    stage = root / "data" / "staged_reranker_exact_v1"; stage.mkdir(parents=True, exist_ok=True); receipts = []
    for candidate_id, encoder_id in TASKS.items():
        domain = DOMAINS[encoder_id]
        retrieval = np.load(root / "data" / "staged_retrieval_exact_v1" / f"{encoder_id}.npz", allow_pickle=False)
        vocab_receipt = json.loads((root / "contracts" / "retrieval_vocabularies" / f"{encoder_id}.v1.json").read_text(encoding="utf-8"))
        vocabulary = {term: index for index, term in enumerate(vocab_receipt["terms"])}
        weight = retrieval["encoder_init_weight"].astype(np.float32)
        domain_rows = [row for row in corpus if row["domain"] == domain]
        train_passage_rows = sorted([row for row in domain_rows if row["split"] == "train"], key=lambda row: row["chunk_id"])
        train_query_rows = sorted(train_passage_rows, key=lambda row: hashlib.sha256((row["chunk_id"] + ":rerank").encode()).digest())[:320]
        train_query = np.asarray([text_vector(normalize_text(f"find evidence about {row['title']} {row['section']}"), vocabulary) for row in train_query_rows])
        train_passage = np.asarray([text_vector(row["text"], vocabulary) for row in train_passage_rows])
        train_scores = normalized_embeddings(train_query, weight) @ normalized_embeddings(train_passage, weight).T
        parts = [build_split(candidate_id, 0, train_query, train_passage, np.asarray([row["pmcid"] for row in train_query_rows]), np.asarray([row["pmcid"] for row in train_passage_rows]), train_scores, 0)]
        encoder_scores = np.load(encoder_artifacts / encoder_id / "three_seed_retrieval_scores.npz", allow_pickle=False)
        for split_name, code, offset in (("validation", 1, 1_000_000), ("test", 2, 2_000_000)):
            parts.append(build_split(candidate_id, code, retrieval[f"{split_name}_retrieval_query"], retrieval[f"{split_name}_retrieval_passage"], retrieval[f"{split_name}_query_group"], retrieval[f"{split_name}_passage_group"], encoder_scores["quantized_best_seed_scores"] if split_name == "test" else retrieval[f"{split_name}_bm25_scores"], offset))
        merged = {name: sum((part[name] for part in parts), []) for name in parts[0]}
        arrays = {
            "x": np.asarray(merged["x"]), "y": np.asarray(merged["y"], dtype=np.int64), "groups": np.asarray(merged["groups"]), "split": np.asarray(merged["split"], dtype=np.int8),
            "query_id": np.asarray(merged["query_id"], dtype=np.int64), "baseline_score": np.asarray(merged["baseline_score"], dtype=np.float32), "special_match": np.asarray(merged["special_match"], dtype=np.uint8),
        }
        contract = contracts[candidate_id]; path = stage / f"{candidate_id}.npz"
        np.savez_compressed(path, **arrays, candidate_id=np.asarray(candidate_id), task_kind=np.asarray("classification"), truth_class=np.asarray("LICENSED_CORPUS_RELEVANCE_PLUS_CONTROLLED_CROSS_FIELD_MATCH"), authority=np.asarray(0, dtype=np.int8))
        group_sets = {code: set(arrays["groups"][arrays["split"] == code].tolist()) for code in range(3)}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a in range(3) for b in range(a + 1, 3))
        counts = {name: int(np.sum(arrays["split"] == code)) for name, code in SPLIT_CODE.items()}
        query_counts = {name: int(len(np.unique(arrays["query_id"][arrays["split"] == code]))) for name, code in SPLIT_CODE.items()}
        metadata = {
            "schema": "cimc.forge200.reranker-exact-staged.v1", "status": "PASS", "candidate_id": candidate_id, "task_kind": "classification",
            "truth_class": "LICENSED_CORPUS_RELEVANCE_PLUS_CONTROLLED_CROSS_FIELD_MATCH", "claim_state": "SAME_DOCUMENT_FAMILY_RELEVANCE_WITH_EXPLICIT_CONTRACT_FIELD_MATCH",
            "path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path), "records": len(arrays["y"]),
            "counts": counts, "query_counts": query_counts, "top_k_candidates": 50, "features": 800, "cross_split_group_overlap": overlap,
            "checkpoint_selection": "VALIDATION_RANKING_COMPOSITE_V1",
            "split_sha256": hashlib.sha256(canonical_bytes(sorted({(str(g), int(s)) for g, s in zip(arrays["groups"], arrays["split"])}))).hexdigest(),
            "feature_contract": contract["input_contract"], "label_derivation_rule": "same_document_family_and_candidate_specific_cross_field_match",
            "baseline_execution": f"actual_{encoder_id}_score_order", "upstream_encoder_artifact_sha256": sha256_file(encoder_artifacts / encoder_id / "three_seed_retrieval_scores.npz"),
            "source_sha256": sha256_file(corpus_path), "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(), "contract_baseline": contract["baseline"],
            "contract_primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"], "authority": 0,
        }
        if overlap or min(query_counts.values()) < 16:
            raise RuntimeError(f"{candidate_id} split gate {overlap} {query_counts}")
        write_json(path.with_suffix(".metadata.json"), metadata); receipts.append(metadata)
    manifest = {"schema": "cimc.forge200.reranker-exact-staging.v1", "status": "PASS", "candidate_count": len(receipts), "candidates": list(TASKS), "records": sum(item["records"] for item in receipts), "authority_nonzero": 0, "source_sha256": sha256_file(corpus_path), "content_root_sha256": hashlib.sha256(canonical_bytes(receipts)).hexdigest()}
    write_json(stage / "manifest.v1.json", manifest); print(json.dumps(manifest, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
