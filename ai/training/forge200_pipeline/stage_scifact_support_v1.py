#!/usr/bin/env python3
"""Stage SciFact expert annotations for Forge200 S031 and S034.

Claims/evidence annotations are CC BY 4.0 and abstracts are ODC-By 1.0.
Document-connected components are assigned before train/validation/test so a
source document cannot cross splits.  Synthetic teacher output is never used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


FEATURES = 192
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?", re.UNICODE)
NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def tokens(text: str) -> list[str]:
    raw = TOKEN_RE.findall(text.lower())
    return raw + [f"{a}::{b}" for a, b in zip(raw, raw[1:])]


def vector(text: str) -> np.ndarray:
    result = np.zeros(FEATURES, dtype=np.float32)
    for token, count in Counter(tokens(text)).items():
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % FEATURES
        sign = 1.0 if digest[4] & 1 else -1.0
        result[index] += sign * (1.0 + np.log1p(count))
    norm = float(np.linalg.norm(result))
    return result / norm if norm else result


def pair_features(claim: str, sentence: str) -> tuple[np.ndarray, dict[str, float]]:
    q, p = vector(claim), vector(sentence)
    q_words, p_words = set(TOKEN_RE.findall(claim.lower())), set(TOKEN_RE.findall(sentence.lower()))
    overlap = len(q_words & p_words) / max(len(q_words | p_words), 1)
    q_numbers, p_numbers = set(NUMBER_RE.findall(claim)), set(NUMBER_RE.findall(sentence))
    numeric_overlap = len(q_numbers & p_numbers) / max(len(q_numbers), 1) if q_numbers else 1.0
    numeric_conflict = float(bool(q_numbers and p_numbers and not (q_numbers & p_numbers)))
    negation_mismatch = float((" not " in f" {claim.lower()} ") != (" not " in f" {sentence.lower()} "))
    length_ratio = min(len(q_words), len(p_words)) / max(len(q_words), len(p_words), 1)
    extras = np.asarray([overlap, numeric_overlap, numeric_conflict, negation_mismatch, length_ratio], dtype=np.float32)
    return np.concatenate((q, p, np.abs(q - p), q * p, extras)), {
        "overlap": overlap,
        "numeric_overlap": numeric_overlap,
        "numeric_conflict": numeric_conflict,
        "negation_mismatch": negation_mismatch,
    }


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def component_split(component: str) -> int:
    bucket = int(hashlib.sha256(component.encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def contract_hashes(root: Path) -> dict[str, str]:
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["candidate_id"]: hashlib.sha256(canonical_bytes(row)).hexdigest() for row in rows}


def baseline_nli(stats: dict[str, float]) -> int:
    if stats["numeric_conflict"] or stats["negation_mismatch"]:
        return 1
    if stats["overlap"] >= 0.30:
        return 0
    return 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw" / "scifact_v1"
    data_root = raw / "data"
    archive = raw / "data.tar.gz"
    license_path = raw / "LICENSE.md"
    expected_archive = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
    expected_license = "56ee69bc639cbab8f0b2e4b67091ec5e9390aa99a1d65f0a35e4274686c9a806"
    if sha256_file(archive) != expected_archive or sha256_file(license_path) != expected_license:
        raise RuntimeError("SCIFACT_SOURCE_HASH_GATE")
    license_text = license_path.read_text(encoding="utf-8")
    if "CC BY 4.0" not in license_text or "ODC-By 1.0" not in license_text:
        raise RuntimeError("SCIFACT_LICENSE_GATE")

    corpus = {int(row["doc_id"]): row for row in read_jsonl(data_root / "corpus.jsonl")}
    claims = read_jsonl(data_root / "claims_train.jsonl") + read_jsonl(data_root / "claims_dev.jsonl")
    uf = UnionFind()
    for claim in claims:
        docs = {str(item) for item in claim.get("cited_doc_ids", [])}
        docs.update(str(item) for item in claim.get("evidence", {}))
        ordered = sorted(docs)
        for item in ordered:
            uf.find(item)
        for item in ordered[1:]:
            uf.union(ordered[0], item)

    nli_x, nli_y, nli_group, nli_split, nli_baseline, nli_origin = [], [], [], [], [], []
    span_x, span_y, span_group, span_split, span_baseline = [], [], [], [], []
    for claim in claims:
        claim_id = str(claim["id"])
        claim_text = str(claim["claim"])
        evidence = claim.get("evidence", {})
        doc_ids = sorted({int(item) for item in claim.get("cited_doc_ids", [])} | {int(item) for item in evidence})
        if not doc_ids:
            continue
        components = sorted({uf.find(str(item)) for item in doc_ids})
        component = components[0]
        split = component_split(component)
        for doc_id in doc_ids:
            document = corpus.get(doc_id)
            if not document:
                continue
            sentences = [str(item) for item in document.get("abstract", [])]
            annotations = evidence.get(str(doc_id), [])
            positive_indices: set[int] = set()
            for annotation in annotations:
                positive_indices.update(int(item) for item in annotation.get("sentences", []))
            group_id = f"{claim_id}:{doc_id}"
            for index, sentence in enumerate(sentences):
                features, stats = pair_features(claim_text, sentence)
                span_x.append(features)
                span_y.append(int(index in positive_indices))
                span_group.append(group_id)
                span_split.append(split)
                span_baseline.append(stats["overlap"])
            for annotation in annotations:
                label = 0 if annotation.get("label") == "SUPPORT" else 1
                annotated = {int(item) for item in annotation.get("sentences", [])}
                for index in sorted(annotated):
                    if 0 <= index < len(sentences):
                        features, stats = pair_features(claim_text, sentences[index])
                        nli_x.append(features)
                        nli_y.append(label)
                        nli_group.append(component)
                        nli_split.append(split)
                        nli_baseline.append(baseline_nli(stats))
                        nli_origin.append("SCIFACT_EXPERT")
                neutral = [index for index in range(len(sentences)) if index not in annotated]
                if neutral:
                    chosen = min(neutral, key=lambda value: hashlib.sha256(f"{claim_id}:{doc_id}:{value}".encode()).hexdigest())
                    features, stats = pair_features(claim_text, sentences[chosen])
                    nli_x.append(features)
                    nli_y.append(2)
                    nli_group.append(component)
                    nli_split.append(split)
                    nli_baseline.append(baseline_nli(stats))
                    nli_origin.append("CONTROLLED_NON_RATIONALE_UNKNOWN")

    output_root = root / "data" / "staged_scifact_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    contracts = contract_hashes(root)
    source_files = [archive, license_path, data_root / "claims_train.jsonl", data_root / "claims_dev.jsonl", data_root / "corpus.jsonl"]
    source_records = [{"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in source_files]
    records = []
    specifications = {
        "CAND-S-031": {
            "arrays": {"x": np.asarray(nli_x, dtype=np.float32), "y": np.asarray(nli_y, dtype=np.int64), "groups": np.asarray(nli_group), "split": np.asarray(nli_split, dtype=np.int8), "baseline_pred": np.asarray(nli_baseline, dtype=np.int64), "label_origin": np.asarray(nli_origin)},
            "truth_class": "SCIFACT_EXPERT_CC_BY_PLUS_CONTROLLED_UNKNOWN",
            "task_kind": "classification",
            "label_contract": "SUPPORT=0_CONTRADICT=1_UNKNOWN=2",
        },
        "CAND-S-034": {
            "arrays": {"x": np.asarray(span_x, dtype=np.float32), "y": np.asarray(span_y, dtype=np.int64), "query_group": np.asarray(span_group), "groups": np.asarray(span_group), "split": np.asarray(span_split, dtype=np.int8), "baseline_score": np.asarray(span_baseline, dtype=np.float32)},
            "truth_class": "SCIFACT_EXPERT_RATIONALE_SENTENCE_SPANS",
            "task_kind": "span_sentence_ranking",
            "label_contract": "expert_rationale_sentence_membership_with_no_span_groups",
        },
    }
    for candidate_id, spec in specifications.items():
        arrays = spec["arrays"]
        path = output_root / f"{candidate_id}.npz"
        np.savez_compressed(path, **arrays, candidate_id=np.asarray(candidate_id), task_kind=np.asarray(spec["task_kind"]), authority=np.asarray(0, dtype=np.int8))
        split = arrays["split"]
        groups = arrays["groups"].astype(str)
        group_sets = {code: set(groups[split == code]) for code in SPLIT_CODE.values()}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
        metadata = {
            "schema": "cimc.forge200.scifact-staged.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS" if not overlap and min(counts.values()) >= 16 else "FAIL_CLOSED",
            "candidate_id": candidate_id,
            "task_kind": spec["task_kind"],
            "truth_class": spec["truth_class"],
            "claim_state": "EXPERT_ANNOTATION_WHERE_MARKED_CONTROLLED_UNKNOWN_EXPLICITLY_SEPARATED",
            "label_contract": spec["label_contract"],
            "source_id": "allenai_scifact_v1",
            "source_urls": ["https://github.com/allenai/scifact", "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"],
            "license": {"claims_and_evidence": "CC-BY-4.0", "abstracts": "ODC-BY-1.0", "code": "APACHE-2.0"},
            "source_records": source_records,
            "split_unit": "CONNECTED_SOURCE_DOCUMENT_COMPONENT",
            "split_sha256": hashlib.sha256(canonical_bytes(sorted((group, int(code)) for group, code in zip(groups.tolist(), split.tolist())))).hexdigest(),
            "cross_split_group_overlap": overlap,
            "counts": counts,
            "records": int(len(split)),
            "features": int(arrays["x"].shape[1]),
            "feature_contract": "separate_hashed_claim_sentence_unigram_bigram_192_plus_abs_product_overlap_numeric_negation_length",
            "task_contract_sha256": contracts[candidate_id],
            "fit_preprocessing_on_train_only": True,
            "teacher_outputs": 0,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        write_json(path.with_suffix(".metadata.json"), metadata)
        records.append(metadata)
    receipt = {
        "schema": "cimc.forge200.scifact-support-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["status"] == "PASS" for item in records) else "FAIL_CLOSED",
        "records": records,
        "authority_nonzero": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }
    write_json(root / "evidence" / "scifact_support_staging.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "candidates": {item["candidate_id"]: item["records"] for item in records}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
