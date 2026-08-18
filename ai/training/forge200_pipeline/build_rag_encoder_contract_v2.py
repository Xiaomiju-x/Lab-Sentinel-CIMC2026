#!/usr/bin/env python3
"""Build leakage-safe 50-way retrieval sets from the expanded CC BY corpus."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FEATURES = 256
DOMAIN_IDS = {
    "PHOSPHOR": 0,
    "FURNACE": 1,
    "SEMIMAT": 2,
    "METROLOGY": 3,
    "PACKAGING": 4,
    "FABQUALITY": 5,
}
ENCODER_TASKS = {
    "CAND-S-009": "PHOSPHOR",
    "CAND-S-010": "FURNACE",
    "CAND-S-011": "SEMIMAT",
    "CAND-S-012": "METROLOGY",
    "CAND-S-013": "PACKAGING",
    "CAND-S-014": "FABQUALITY",
    "CAND-S-029": "SHARED",
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.UNICODE
)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def terms(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]


def stable_index(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % FEATURES, 1.0 if digest[4] & 1 else -1.0


def idf_map(rows: list[dict[str, Any]]) -> tuple[dict[str, float], float]:
    train = [row for row in rows if row["split"] == "train"]
    frequency: Counter[str] = Counter()
    for row in train:
        frequency.update(set(terms(row["text"])))
    count = max(len(train), 1)
    return (
        {term: math.log((count + 1.0) / (value + 0.5)) + 1.0 for term, value in frequency.items()},
        math.log((count + 1.0) / 0.5) + 1.0,
    )


def vectorize(text: str, idf: dict[str, float], unknown_idf: float) -> np.ndarray:
    counts = Counter(terms(text))
    vector = np.zeros(FEATURES, dtype=np.float32)
    for term, count in counts.items():
        index, sign = stable_index(term)
        vector[index] += sign * (1.0 + math.log(count)) * idf.get(term, unknown_idf)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def bm25_score(
    query_terms: list[str], passage_counts: Counter[str], passage_length: int,
    idf: dict[str, float], unknown_idf: float, average_length: float,
) -> float:
    score = 0.0
    k1, b = 1.2, 0.75
    for term in set(query_terms):
        tf = passage_counts.get(term, 0)
        if not tf:
            continue
        denominator = tf + k1 * (1.0 - b + b * passage_length / max(average_length, 1.0))
        score += idf.get(term, unknown_idf) * tf * (k1 + 1.0) / denominator
    return score


def ranking_metrics(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    order = np.argsort(-scores, axis=1)
    ranked = np.take_along_axis(labels, order, axis=1)
    rank = np.argmax(ranked, axis=1) + 1
    discount = 1.0 / np.log2(np.arange(2, labels.shape[1] + 2))
    return {
        "mrr_at_10": float(np.mean(np.where(rank <= 10, 1.0 / rank, 0.0))),
        "recall_at_10": float(np.mean(rank <= 10)),
        "recall_at_20": float(np.mean(rank <= 20)),
        "ndcg_at_10": float(np.mean(np.sum(ranked[:, :10] * discount[:10], axis=1))),
        "worst_domain_recall_at_20": float(np.mean(rank <= 20)),
    }


def select_queries(rows: list[dict[str, Any]], domain: str, per_split: dict[str, int]) -> list[dict[str, Any]]:
    selected = []
    for split, limit in per_split.items():
        eligible = [
            row for row in rows
            if row["split"] == split and (domain == "SHARED" or row["domain"] == domain)
        ]
        eligible.sort(key=lambda row: hashlib.sha256(row["chunk_id"].encode()).hexdigest())
        selected.extend(eligible[:limit])
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--candidates", default=",".join(ENCODER_TASKS))
    parser.add_argument("--train-queries", type=int, default=400)
    parser.add_argument("--validation-queries", type=int, default=120)
    parser.add_argument("--test-queries", type=int, default=120)
    parser.add_argument("--candidates-per-query", type=int, default=50)
    args = parser.parse_args()
    root = args.root.resolve()
    requested = [item.strip() for item in args.candidates.split(",") if item.strip()]
    if any(item not in ENCODER_TASKS for item in requested):
        raise RuntimeError("UNKNOWN_ENCODER_CANDIDATE")
    if args.candidates_per_query != 50:
        raise RuntimeError("CONTRACT_REQUIRES_TOP50_CANDIDATES")
    ledger_path = root / "data" / "ledgers" / "ccby_multidomain_corpus.v2.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    corpus_path = root / ledger["corpus_path"]
    if ledger["status"] != "PASS" or sha256_file(corpus_path) != ledger["corpus_sha256"]:
        raise RuntimeError("CORPUS_V2_GATE")
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
    if any(row["authority"] != 0 for row in rows):
        raise RuntimeError("AUTHORITY_NONZERO")
    contracts = {row["candidate_id"]: row for row in read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")}
    split_pmcids = defaultdict(set)
    for row in rows:
        split_pmcids[row["split"]].add(row["pmcid"])
    overlap = sum(len(split_pmcids[a] & split_pmcids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    if overlap:
        raise RuntimeError("PMCID_SPLIT_LEAKAGE")

    all_by_split = {
        split: sorted(
            [row for row in rows if row["split"] == split],
            key=lambda row: row["chunk_id"],
        )
        for split in SPLIT_CODE
    }
    passage_cache = {
        row["chunk_id"]: Counter(terms(row["text"])) for row in rows
    }
    average_length = float(
        np.mean([sum(value.values()) for value in passage_cache.values()])
    )
    outputs = []
    for candidate_id in requested:
        domain = ENCODER_TASKS[candidate_id]
        namespace_rows = rows if domain == "SHARED" else [row for row in rows if row["domain"] == domain]
        idf, unknown_idf = idf_map(namespace_rows)
        queries = select_queries(
            rows,
            domain,
            {"train": args.train_queries, "validation": args.validation_queries, "test": args.test_queries},
        )
        x_query, x_passage, labels, query_ids, splits, domain_ids, baseline_scores = [], [], [], [], [], [], []
        for query_index, query in enumerate(queries):
            split = query["split"]
            pool = all_by_split[split]
            same_domain = [row for row in pool if row["domain"] == query["domain"] and row["pmcid"] != query["pmcid"]]
            cross_domain = [row for row in pool if row["domain"] != query["domain"]]
            seed = int(hashlib.sha256(query["chunk_id"].encode()).hexdigest()[:16], 16)
            negatives = []
            for source, count, salt in ((same_domain, 25, 17), (cross_domain, 24, 43)):
                if not source:
                    raise RuntimeError("NEGATIVE_POOL_EMPTY")
                start = (seed + salt) % len(source)
                for offset in range(count):
                    negatives.append(source[(start + offset * 7919) % len(source)])
            candidates = [query] + negatives
            permutation = np.random.default_rng(seed).permutation(len(candidates))
            candidates = [candidates[index] for index in permutation]
            query_text = f"{query['title']} {query['section']}"
            query_terms = terms(query_text)
            q_vector = vectorize(query_text, idf, unknown_idf)
            for candidate in candidates:
                p_counts = passage_cache[candidate["chunk_id"]]
                x_query.append(q_vector)
                x_passage.append(vectorize(candidate["text"], idf, unknown_idf))
                labels.append(int(candidate["chunk_id"] == query["chunk_id"]))
                query_ids.append(f"{candidate_id}:{query['chunk_id']}")
                splits.append(SPLIT_CODE[split])
                domain_ids.append(DOMAIN_IDS[query["domain"]])
                baseline_scores.append(
                    bm25_score(query_terms, p_counts, sum(p_counts.values()), idf, unknown_idf, average_length)
                )
        xq = np.asarray(x_query, dtype=np.float32)
        xp = np.asarray(x_passage, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)
        split_array = np.asarray(splits, dtype=np.int8)
        baseline = np.asarray(baseline_scores, dtype=np.float32)
        query_array = np.asarray(query_ids)
        counts = {name: int(np.sum(split_array == code)) for name, code in SPLIT_CODE.items()}
        baseline_report = {}
        for split_name, code in SPLIT_CODE.items():
            selected = np.flatnonzero(split_array == code)
            baseline_report[split_name] = ranking_metrics(
                baseline[selected].reshape(-1, 50), y[selected].reshape(-1, 50)
            )
        output = root / "data" / "staged_rag_contract_v2" / f"{candidate_id}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output,
            x_query=xq,
            x_passage=xp,
            y=y,
            query_id=query_array,
            domain_id=np.asarray(domain_ids, dtype=np.int8),
            baseline_score=baseline,
            split=split_array,
            candidate_id=np.asarray(candidate_id),
            task_kind=np.asarray("retrieval_embedding"),
            authority=np.asarray(0, dtype=np.int8),
        )
        contract = contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.rag-retrieval-staged.v2",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS",
            "candidate_id": candidate_id,
            "domain": domain,
            "source_id": "europe_pmc_ccby_multidomain_v2",
            "source_sha256": ledger["corpus_sha256"],
            "source_chunks": ledger["chunk_count"],
            "source_documents": ledger["document_count"],
            "split_unit": "PMCID_DOCUMENT_FAMILY",
            "cross_split_group_overlap": overlap,
            "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
            "baseline_contract": contract["baseline"],
            "primary_metric_contract": contract["primary_metric"],
            "parameter_cap": contract["parameter_cap"],
            "feature_contract": "train_only_hashed_TFIDF_unigram_bigram_256_symmetric_query_passage",
            "candidates_per_query": 50,
            "query_counts": {name: counts[name] // 50 for name in counts},
            "records": len(y),
            "counts": counts,
            "baseline": baseline_report,
            "teacher_outputs": 0,
            "expert_labels": 0,
            "truth_class": "STRUCTURE_DERIVED_SAME_CHUNK_RETRIEVAL_NOT_INDEPENDENT_RELEVANCE_JUDGMENT",
            "claim_state": "SOURCE_BOUND_PROXY_BENCHMARK",
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
            "path": str(output.relative_to(root)).replace("\\", "/"),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        write_json(output.with_suffix(".metadata.json"), metadata)
        outputs.append(metadata)
    content = {"records": outputs, "corpus_sha256": ledger["corpus_sha256"]}
    manifest = {
        "schema": "cimc.forge200.rag-encoder-staging.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_count": len(outputs),
        "authority_nonzero": 0,
        "board_accepted": 0,
        **content,
        "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }
    manifest_path = root / "evidence" / ("rag_encoder_staging_" + hashlib.sha256(args.candidates.encode()).hexdigest()[:12] + ".v2.json")
    write_json(manifest_path, manifest)
    print(json.dumps({"status": "PASS", "candidates": [item["candidate_id"] for item in outputs], "records": sum(item["records"] for item in outputs), "content_root_sha256": manifest["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
