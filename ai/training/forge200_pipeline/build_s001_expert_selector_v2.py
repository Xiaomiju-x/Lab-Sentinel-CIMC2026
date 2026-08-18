#!/usr/bin/env python3
"""Stage S001 from source-bound CC BY queries with a PMCID-family split."""

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


DOMAIN_IDS = {
    "PHOSPHOR": 0,
    "FURNACE": 1,
    "SEMIMAT": 2,
    "METROLOGY": 3,
    "PACKAGING": 4,
    "FABQUALITY": 5,
}
KEYWORDS = {
    "PHOSPHOR": "phosphor luminescence photoluminescence emission excitation dopant rare-earth quantum yield fluorescence".split(),
    "FURNACE": "sintering furnace thermal densification ceramic kiln calcination annealing grain temperature".split(),
    "SEMIMAT": "semiconductor dielectric transistor thin-film interface defect bandgap electronic epitaxy substrate".split(),
    "METROLOGY": "diffraction microscopy spectroscopy metrology xrd sem raman xps characterization measurement".split(),
    "PACKAGING": "packaging underfill solder interconnect warpage encapsulant reliability delamination moisture package".split(),
    "FABQUALITY": "wafer deposition cmp yield process control manufacturing fault drift fab quality etch".split(),
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
FEATURES = 256
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.UNICODE)


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


def terms(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]


def stable_index(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % FEATURES, 1.0 if digest[4] & 1 else -1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-per-domain", type=int, default=900)
    parser.add_argument("--validation-per-domain", type=int, default=240)
    parser.add_argument("--test-per-domain", type=int, default=240)
    args = parser.parse_args()
    root = args.root.resolve()
    ledger_path = root / "data" / "ledgers" / "ccby_multidomain_corpus.v2.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    corpus_path = root / ledger["corpus_path"]
    if ledger["status"] != "PASS" or sha256_file(corpus_path) != ledger["corpus_sha256"]:
        raise RuntimeError("CORPUS_V2_GATE")
    corpus = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
    if any(row["authority"] != 0 or row["license"] != "CC-BY_PER_METADATA_AND_JATS_VERIFIED" for row in corpus):
        raise RuntimeError("LICENSE_OR_AUTHORITY_GATE")
    pmcids = defaultdict(set)
    for row in corpus:
        pmcids[row["split"]].add(row["pmcid"])
    overlap = sum(len(pmcids[a] & pmcids[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    if overlap:
        raise RuntimeError("PMCID_SPLIT_LEAKAGE")

    requested = {"train": args.train_per_domain, "validation": args.validation_per_domain, "test": args.test_per_domain}
    selected: list[dict[str, Any]] = []
    for split, limit in requested.items():
        for domain in DOMAIN_IDS:
            eligible = [row for row in corpus if row["split"] == split and row["domain"] == domain]
            eligible.sort(key=lambda row: hashlib.sha256(row["chunk_id"].encode()).hexdigest())
            if len(eligible) < limit:
                raise RuntimeError(f"INSUFFICIENT_ROWS:{split}:{domain}:{len(eligible)}<{limit}")
            selected.extend(eligible[:limit])

    def query(row: dict[str, Any]) -> str:
        body = re.sub(r"\s+", " ", row["text"]).strip()[:420]
        return f"Select the best available materials expert for this request: {row['title']} [{row['section']}] {body}"

    train_rows = [row for row in selected if row["split"] == "train"]
    frequency: Counter[str] = Counter()
    for row in train_rows:
        frequency.update(set(terms(query(row))))
    count = len(train_rows)
    idf = {term: math.log((count + 1.0) / (value + 0.5)) + 1.0 for term, value in frequency.items()}
    unknown_idf = math.log((count + 1.0) / 0.5) + 1.0

    x, y, split_values, pmcid_values, query_ids, baseline_probabilities = [], [], [], [], [], []
    for row in selected:
        text = query(row)
        counts = Counter(terms(text))
        vector = np.zeros(FEATURES, dtype=np.float32)
        for term, value in counts.items():
            index, sign = stable_index(term)
            vector[index] += sign * (1.0 + math.log(value)) * idf.get(term, unknown_idf)
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        available = np.ones(len(DOMAIN_IDS), dtype=np.float32)
        x.append(np.concatenate((vector, available)))
        y.append(DOMAIN_IDS[row["domain"]])
        split_values.append(SPLIT_CODE[row["split"]])
        pmcid_values.append(row["pmcid"])
        query_ids.append(row["chunk_id"])
        unigram = set(TOKEN_RE.findall(text.lower()))
        scores = np.asarray([sum(token in unigram for token in KEYWORDS[domain]) for domain in DOMAIN_IDS], dtype=np.float32)
        scores = 0.65 * scores
        scores -= scores.max()
        probability = np.exp(scores)
        baseline_probabilities.append(probability / probability.sum())

    output = root / "data" / "staged_router_contract_v2" / "CAND-S-001.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        x=np.asarray(x, dtype=np.float32),
        y=np.asarray(y, dtype=np.int64),
        split=np.asarray(split_values, dtype=np.int8),
        pmcid=np.asarray(pmcid_values),
        query_id=np.asarray(query_ids),
        baseline_probability=np.asarray(baseline_probabilities, dtype=np.float32),
        available_model_capabilities=np.ones((len(x), len(DOMAIN_IDS)), dtype=np.int8),
        authority=np.asarray(0, dtype=np.int8),
    )
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == "CAND-S-001")
    counts = {name: int(np.sum(np.asarray(split_values) == code)) for name, code in SPLIT_CODE.items()}
    source = {
        "corpus_sha256": ledger["corpus_sha256"],
        "corpus_content_root_sha256": ledger["content_root_sha256"],
        "license_gate": "PER_DOCUMENT_METADATA_PLUS_JATS_CC_BY_VERIFIED",
        "query_derivation": "SOURCE_BOUND_TITLE_SECTION_AND_TRUNCATED_PASSAGE_WITH_FIXED_NON_LABEL_TEMPLATE",
        "domain_truth": "CORPUS_CURATION_DOMAIN_METADATA",
        "split_group": "PMCID_DOCUMENT_FAMILY",
        "counts": counts,
    }
    metadata = {
        "schema": "cimc.forge200.s001-staged.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-S-001",
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "sha256": sha256_file(output),
        "records": len(x),
        "feature_count": int(np.asarray(x).shape[1]),
        "class_count": 6,
        "counts": counts,
        "cross_split_group_overlap": overlap,
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "baseline_contract": contract["baseline"],
        "primary_metric_contract": contract["primary_metric"],
        "truth_class": "SOURCE_BOUND_CC_BY_CORPUS_DOMAIN_LABEL",
        "source": source,
        "content_root_sha256": hashlib.sha256(canonical_bytes(source)).hexdigest(),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    write_json(root / "evidence" / "s001_expert_selector_staging.v2.json", metadata)
    print(json.dumps({"status": "PASS", "records": len(x), "sha256": metadata["sha256"], "counts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
