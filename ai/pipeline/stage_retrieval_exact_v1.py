#!/usr/bin/env python3
"""Stage six domain retrieval encoders with executable full-pool BM25 baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


DOMAINS = {
    "CAND-S-009": "PHOSPHOR",
    "CAND-S-010": "FURNACE",
    "CAND-S-011": "SEMIMAT",
    "CAND-S-012": "METROLOGY",
    "CAND-S-013": "PACKAGING",
    "CAND-S-014": "FABQUALITY",
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.I | re.UNICODE)
REWRITES = (
    (r"\bphotoluminescence\b", "light emission"),
    (r"\bluminescence\b", "light emission"),
    (r"\bphosphor\b", "emissive ceramic"),
    (r"\bdop(?:ed|ing|ant)?\b", "activator substitution"),
    (r"\bsinter(?:ed|ing)?\b", "thermal consolidation"),
    (r"\bfurnace\b", "thermal chamber"),
    (r"\btemperature\b", "thermal condition"),
    (r"\bsemiconductor\b", "electronic material"),
    (r"\bband\s*gap\b", "electronic energy gap"),
    (r"\bx[- ]?ray diffraction\b|\bxrd\b", "crystal pattern measurement"),
    (r"\bmetrology\b|\bcharacteri[sz]ation\b", "measurement analysis"),
    (r"\bpackaging\b", "device assembly"),
    (r"\breliability\b", "durability"),
    (r"\bsolder\b", "metal joint"),
    (r"\bwafer\b", "substrate"),
    (r"\byield\b", "production quality"),
    (r"\bfabrication\b", "manufacturing"),
)


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


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize_text(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = re.sub(pattern, replacement, text, flags=re.I)
    return text


def query_text(row: dict[str, Any]) -> str:
    return normalize_text(f"find evidence about {row['title']} {row['section']}")


def text_vector(text: str, vocabulary: dict[str, int]) -> np.ndarray:
    value = np.zeros(len(vocabulary), dtype=np.float32)
    for token in tokens(normalize_text(text)):
        index = vocabulary.get(token)
        if index is not None:
            value[index] += 1.0
    norm = float(np.linalg.norm(value))
    return value / norm if norm else value


def fit_domain_vocabulary(rows: list[dict[str, Any]], size: int = 768) -> tuple[dict[str, int], list[str], np.ndarray]:
    counts: Counter[str] = Counter(); document_frequency: Counter[str] = Counter(); documents = 0
    for row in rows:
        if row["split"] != "train":
            continue
        for text in (normalize_text(row["text"]), query_text(row)):
            terms = tokens(text); counts.update(terms); document_frequency.update(set(terms)); documents += 1
    ordered = [term for term, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])) if count >= 3][:size]
    if len(ordered) != size:
        raise RuntimeError(f"domain vocabulary too small: {len(ordered)}")
    vocabulary = {term: index for index, term in enumerate(ordered)}
    idf = np.asarray([math.log(1.0 + (documents + 1) / (document_frequency[term] + 1)) for term in ordered], dtype=np.float32)
    idf /= max(float(np.max(idf)), 1e-12)
    return vocabulary, ordered, idf


def encoder_initial_weight(ordered: list[str], idf: np.ndarray, dimensions: int = 64) -> np.ndarray:
    weight = np.zeros((dimensions, len(ordered)), dtype=np.float32)
    for index, term in enumerate(ordered):
        digest = hashlib.sha256(term.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        weight[bucket, index] = sign * float(idf[index])
    return weight


def bm25_scores(queries: list[str], passages: list[str]) -> np.ndarray:
    passage_tokens = [tokens(text) for text in passages]
    document_frequency: Counter[str] = Counter()
    for terms in passage_tokens:
        document_frequency.update(set(terms))
    count = len(passages); average_length = np.mean([len(item) for item in passage_tokens])
    scores = np.zeros((len(queries), count), dtype=np.float32)
    for query_index, query in enumerate(queries):
        for term in set(tokens(query)):
            df = document_frequency.get(term, 0)
            if not df:
                continue
            idf = math.log(1.0 + (count - df + .5) / (df + .5))
            for passage_index, terms in enumerate(passage_tokens):
                tf = terms.count(term)
                if tf:
                    denominator = tf + 1.5 * (1 - .75 + .75 * len(terms) / max(average_length, 1))
                    scores[query_index, passage_index] += idf * tf * 2.5 / denominator
    return scores


def retrieval_arrays(rows: list[dict[str, Any]], split_name: str, vocabulary: dict[str, int]) -> dict[str, np.ndarray]:
    passages = sorted([row for row in rows if row["split"] == split_name], key=lambda row: row["chunk_id"])
    query_rows = sorted(passages, key=lambda row: hashlib.sha256((row["chunk_id"] + ":query").encode()).digest())
    query_rows = query_rows[: min(160, len(query_rows))]
    query_strings = [query_text(row) for row in query_rows]
    passage_strings = [row["text"] for row in passages]
    relevance = np.asarray([[int(query["pmcid"] == passage["pmcid"]) for passage in passages] for query in query_rows], dtype=np.uint8)
    if np.any(relevance.sum(axis=1) == 0):
        raise RuntimeError(f"retrieval relevance empty for {split_name}")
    return {
        f"{split_name}_retrieval_query": np.asarray([text_vector(text, vocabulary) for text in query_strings]),
        f"{split_name}_retrieval_passage": np.asarray([text_vector(text, vocabulary) for text in passage_strings]),
        f"{split_name}_retrieval_relevance": relevance,
        f"{split_name}_bm25_scores": bm25_scores(query_strings, passage_strings),
        f"{split_name}_query_group": np.asarray([row["pmcid"] for row in query_rows]),
        f"{split_name}_passage_group": np.asarray([row["pmcid"] for row in passages]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    with corpus_path.open("r", encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip()]
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    stage = root / "data" / "staged_retrieval_exact_v1"; stage.mkdir(parents=True, exist_ok=True)
    receipts = []
    for candidate_id, domain in DOMAINS.items():
        rows = [row for row in corpus if row["domain"] == domain]
        vocabulary, ordered_vocabulary, idf = fit_domain_vocabulary(rows)
        vocabulary_receipt = {
            "schema": "cimc.forge200.retrieval-domain-vocabulary.v1", "candidate_id": candidate_id,
            "domain": domain, "fit_split": "train_only", "terms": ordered_vocabulary,
            "idf": idf.tolist(), "authority": 0,
        }
        vocabulary_sha = hashlib.sha256(canonical_bytes(vocabulary_receipt)).hexdigest()
        vocabulary_receipt["content_sha256"] = vocabulary_sha
        write_json(root / "contracts" / "retrieval_vocabularies" / f"{candidate_id}.v1.json", vocabulary_receipt)
        xq: list[np.ndarray] = []; xp: list[np.ndarray] = []; y: list[int] = []; groups: list[str] = []; split: list[int] = []
        for split_name, code in SPLIT_CODE.items():
            selected = sorted([row for row in rows if row["split"] == split_name], key=lambda row: row["chunk_id"])
            for index, row in enumerate(selected):
                candidates = [peer for peer in selected if peer["pmcid"] != row["pmcid"]]
                negative = candidates[int.from_bytes(hashlib.sha256(row["chunk_id"].encode()).digest()[:4], "little") % len(candidates)]
                query = text_vector(query_text(row), vocabulary)
                for passage, label in ((row, 1), (negative, 0)):
                    xq.append(query); xp.append(text_vector(passage["text"], vocabulary)); y.append(label)
                    groups.append(f"{row['pmcid']}+{passage['pmcid']}"); split.append(code)
        arrays: dict[str, np.ndarray] = {
            "x_query": np.asarray(xq), "x_passage": np.asarray(xp), "y": np.asarray(y, dtype=np.int64),
            "groups": np.asarray(groups), "split": np.asarray(split, dtype=np.int8),
            "encoder_init_weight": encoder_initial_weight(ordered_vocabulary, idf),
        }
        arrays.update(retrieval_arrays(rows, "validation", vocabulary)); arrays.update(retrieval_arrays(rows, "test", vocabulary))
        data_path = stage / f"{candidate_id}.npz"
        np.savez_compressed(data_path, **arrays, candidate_id=np.asarray(candidate_id), task_kind=np.asarray("contrastive_embedding"), truth_class=np.asarray("STRUCTURE_DERIVED_LICENSED_CORPUS_RETRIEVAL"), authority=np.asarray(0, dtype=np.int8))
        group_sets = {code: set(np.asarray(groups)[np.asarray(split) == code].tolist()) for code in range(3)}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a in range(3) for b in range(a + 1, 3))
        counts = {name: int(np.sum(np.asarray(split) == code)) for name, code in SPLIT_CODE.items()}
        contract = contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.retrieval-exact-staged.v1", "status": "PASS", "candidate_id": candidate_id,
            "task_kind": "contrastive_embedding", "truth_class": "STRUCTURE_DERIVED_LICENSED_CORPUS_RETRIEVAL",
            "claim_state": "SAME_DOCUMENT_FAMILY_RELEVANCE_NOT_INDEPENDENT_EXPERT_JUDGMENT", "domain": domain,
            "path": str(data_path.relative_to(root)).replace("\\", "/"), "bytes": data_path.stat().st_size, "sha256": sha256_file(data_path),
            "records": len(y), "counts": counts, "features": len(vocabulary), "embedding_dimensions": 64,
            "cross_split_group_overlap": overlap, "split_sha256": hashlib.sha256(canonical_bytes(sorted({(g, int(s)) for g, s in zip(groups, split)}))).hexdigest(),
            "feature_contract": "TRAIN_ONLY_VOCABULARY_BOW_SHARED_QUERY_PASSAGE_ENCODER_FULL_SPLIT_RETRIEVAL",
            "label_derivation_rule": "same_PMCI_document_family_relevant_other_document_irrelevant",
            "query_rewrite_rule": "frozen_domain_synonym_map_without_test_fit", "baseline_execution": "BM25_FULL_SPLIT_PASSAGE_POOL",
            "validation_queries": int(arrays["validation_retrieval_query"].shape[0]), "validation_passages": int(arrays["validation_retrieval_passage"].shape[0]),
            "test_queries": int(arrays["test_retrieval_query"].shape[0]), "test_passages": int(arrays["test_retrieval_passage"].shape[0]),
            "vocabulary_sha256": vocabulary_sha, "source_sha256": sha256_file(corpus_path),
            "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(), "contract_baseline": contract["baseline"],
            "contract_primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"], "authority": 0,
        }
        if overlap or min(counts.values()) < 16:
            raise RuntimeError(f"{candidate_id} split gate {overlap} {counts}")
        write_json(data_path.with_suffix(".metadata.json"), metadata); receipts.append(metadata)
    manifest = {
        "schema": "cimc.forge200.retrieval-exact-staging.v1", "status": "PASS", "candidate_count": len(receipts),
        "candidates": list(DOMAINS), "records": sum(item["records"] for item in receipts), "authority_nonzero": 0,
        "source_sha256": sha256_file(corpus_path), "content_root_sha256": hashlib.sha256(canonical_bytes(receipts)).hexdigest(),
    }
    write_json(stage / "manifest.v1.json", manifest); print(json.dumps(manifest, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
