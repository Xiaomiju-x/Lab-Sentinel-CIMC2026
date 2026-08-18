#!/usr/bin/env python3
"""Build contract-shaped support-model datasets from licensed corpus records.

Labels are limited to publication namespace metadata, explicit controlled
mutations, and deterministic interface fixtures.  They are not promoted to
experimental material truth.  Every document family retains the frozen
corpus split before examples are derived.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TEXT_FEATURES = 768
STRUCT_FEATURES = 32
FEATURES = TEXT_FEATURES + STRUCT_FEATURES
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
DOMAINS = ("PHOSPHOR", "FURNACE", "SEMIMAT", "METROLOGY", "PACKAGING", "FABQUALITY")
DOMAIN_ID = {name: index for index, name in enumerate(DOMAINS)}
DOMAIN_TERMS = {
    "PHOSPHOR": ("phosphor", "luminescen", "emission", "excitation", "dop", "optical"),
    "FURNACE": ("sinter", "furnace", "thermal", "temperature", "ceramic", "heating"),
    "SEMIMAT": ("semiconductor", "bandgap", "electronic", "dielectric", "film", "device"),
    "METROLOGY": ("measurement", "xrd", "spectroscopy", "microscopy", "character", "metrology"),
    "PACKAGING": ("packaging", "solder", "underfill", "interconnect", "warpage", "reliability"),
    "FABQUALITY": ("wafer", "yield", "etch", "process control", "virtual metrology", "fabrication"),
}
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.I | re.UNICODE)


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


def atomic(text: str, words: int = 30) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return " ".join(sentence.split()[:words])


def task_contracts(root: Path) -> dict[str, dict[str, str]]:
    path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def fit_vocab(rows: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    counts: Counter[str] = Counter()
    for row in rows:
        if row["split"] == "train":
            counts.update(tokens(f"{row['title']} {row['section']} {atomic(row['text'], 48)}"))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    vocabulary = [term for term, count in ranked if count >= 4][:TEXT_FEATURES]
    if len(vocabulary) != TEXT_FEATURES:
        raise RuntimeError(f"support vocabulary too small: {len(vocabulary)}")
    return {term: index for index, term in enumerate(vocabulary)}, vocabulary


def vector(text: str, vocabulary: dict[str, int], structured: Iterable[float] = ()) -> np.ndarray:
    value = np.zeros(FEATURES, dtype=np.float32)
    for term in tokens(text):
        index = vocabulary.get(term)
        if index is not None:
            value[index] += 1.0
    norm = float(np.linalg.norm(value[:TEXT_FEATURES]))
    if norm:
        value[:TEXT_FEATURES] /= norm
    structured_values = list(structured)[:STRUCT_FEATURES]
    value[TEXT_FEATURES : TEXT_FEATURES + len(structured_values)] = structured_values
    return value


def stable_rows(rows: list[dict[str, Any]], split: str, domain: str | None, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split and (domain is None or row["domain"] == domain)]
    return sorted(selected, key=lambda row: hashlib.sha256(row["chunk_id"].encode()).digest())[:limit]


def keyword_route(text: str) -> int:
    lower = text.lower()
    scores = [sum(lower.count(term) for term in DOMAIN_TERMS[domain]) for domain in DOMAINS]
    return int(np.argmax(scores)) if max(scores) else 0


def rng_for(name: str, split_code: int) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(f"{name}:{split_code}".encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def save_dataset(
    root: Path,
    contracts: dict[str, dict[str, str]],
    candidate_id: str,
    task_kind: str,
    arrays: dict[str, np.ndarray],
    groups: list[str],
    split: list[int],
    truth_class: str,
    derivation: str,
    vocabulary_sha: str,
) -> dict[str, Any]:
    stage = root / "data" / "staged_support_exact_v3"
    stage.mkdir(parents=True, exist_ok=True)
    split_array = np.asarray(split, dtype=np.int8)
    group_array = np.asarray(groups)
    group_sets = {code: set(group_array[split_array == code].tolist()) for code in range(3)}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a in range(3) for b in range(a + 1, 3))
    counts = {name: int(np.sum(split_array == code)) for name, code in SPLIT_CODE.items()}
    if overlap or min(counts.values()) < 16:
        raise RuntimeError(f"{candidate_id} split gate: overlap={overlap} counts={counts}")
    contract = contracts[candidate_id]
    contract_sha = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    split_records = sorted({(str(group), int(code)) for group, code in zip(groups, split)})
    split_sha = hashlib.sha256(canonical_bytes(split_records)).hexdigest()
    path = stage / f"{candidate_id}.npz"
    np.savez_compressed(
        path,
        **arrays,
        groups=group_array,
        split=split_array,
        candidate_id=np.asarray(candidate_id),
        task_kind=np.asarray(task_kind),
        truth_class=np.asarray(truth_class),
        authority=np.asarray(0, dtype=np.int8),
    )
    metadata = {
        "schema": "cimc.forge200.support-exact-staged.v3",
        "status": "PASS",
        "candidate_id": candidate_id,
        "task_kind": task_kind,
        "truth_class": truth_class,
        "claim_state": "LICENSED_SOURCE_STRUCTURE_OR_EXPLICIT_CONTROLLED_INTERFACE_LABEL_NOT_EXPERIMENTAL_TRUTH",
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "records": len(split),
        "counts": counts,
        "cross_split_group_overlap": overlap,
        "split_sha256": split_sha,
        "features": FEATURES,
        "text_features": TEXT_FEATURES,
        "structured_features": STRUCT_FEATURES,
        "feature_contract": "TRAIN_ONLY_VOCABULARY_BOW_L2_PLUS_CONTRACT_STRUCTURED_FIELDS_V3",
        "vocabulary_sha256": vocabulary_sha,
        "label_derivation_rule": derivation,
        "task_contract_sha256": contract_sha,
        "contract_input": contract["input_contract"],
        "contract_target": contract["target_label"],
        "contract_baseline": contract["baseline"],
        "contract_primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "teacher_outputs": 0,
        "expert_labels": 0,
        "authority": 0,
    }
    write_json(path.with_suffix(".metadata.json"), metadata)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    corpus = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    rows = load_rows(corpus)
    contracts = task_contracts(root)
    vocabulary, ordered_vocab = fit_vocab(rows)
    vocab_receipt = {
        "schema": "cimc.forge200.support-exact-vocabulary.v3",
        "fit_split": "train_only",
        "features": TEXT_FEATURES,
        "terms": ordered_vocab,
        "source_sha256": sha256_file(corpus),
        "authority": 0,
    }
    vocabulary_sha = hashlib.sha256(canonical_bytes(vocab_receipt)).hexdigest()
    vocab_receipt["content_sha256"] = vocabulary_sha
    write_json(root / "contracts" / "support_exact_v3_vocabulary.json", vocab_receipt)
    receipts: list[dict[str, Any]] = []

    # S001: query domain plus an explicit availability mask; class 6 is abstain.
    xs: list[np.ndarray] = []
    ys: list[int] = []
    groups: list[str] = []
    splits: list[int] = []
    baseline: list[int] = []
    for split_name, code in SPLIT_CODE.items():
        for domain in DOMAINS:
            for index, row in enumerate(stable_rows(rows, split_name, domain, 240 if code == 0 else 100)):
                target = DOMAIN_ID[domain]
                mask = np.ones(6, dtype=np.float32)
                if index % 5 == 0:
                    mask[target] = 0.0
                label = target if mask[target] else 6
                query = f"{row['title']} {row['section']} {atomic(row['text'])}"
                xs.append(vector(query, vocabulary, [*mask, target / 5.0]))
                ys.append(label)
                groups.append(row["pmcid"])
                splits.append(code)
                baseline.append(keyword_route(query))
    receipts.append(save_dataset(root, contracts, "CAND-S-001", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "STRUCTURE_DERIVED_LICENSED_CORPUS_PLUS_CAPABILITY_MASK", "frozen_domain_namespace_with_explicit_available_model_capability_mask_and_abstain", vocabulary_sha))

    # S002: evidence sufficiency from required-field coverage and provenance.
    xs, ys, groups, splits, baseline = [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        rng = rng_for("S002", code)
        selected = stable_rows(rows, split_name, None, 3000 if code == 0 else 900)
        for row in selected:
            required = int(rng.integers(2, 7)); covered = int(rng.integers(0, required + 1))
            citations = int(rng.integers(0, 4)); licensed = int(rng.random() > 0.12)
            provenance = int(rng.random() > 0.16); claim_link = int(rng.random() > 0.14)
            coverage = covered / required
            label = int(coverage == 1.0 and citations > 0 and licensed and provenance and claim_link)
            structured = [required / 6, covered / 6, coverage, citations / 3, licensed, provenance, claim_link]
            xs.append(vector(atomic(row["text"]), vocabulary, structured)); ys.append(label)
            groups.append(row["pmcid"]); splits.append(code); baseline.append(int(citations >= required))
    receipts.append(save_dataset(root, contracts, "CAND-S-002", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "LICENSED_CORPUS_PLUS_CONTROLLED_EVIDENCE_COMPLETENESS", "all_required_fields_covered_and_citation_license_provenance_claim_link_valid", vocabulary_sha))

    # S003: arbitration uses confidence, provenance, and pairwise NLI; majority vote is frozen baseline.
    xs, ys, groups, splits, baseline = [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        rng = rng_for("S003", code)
        selected = stable_rows(rows, split_name, None, 3200 if code == 0 else 1000)
        for index, row in enumerate(selected):
            label = index % 3
            if label == 0: nli_a, nli_b = rng.uniform(.72, 1), rng.uniform(0, .35)
            elif label == 1: nli_a, nli_b = rng.uniform(0, .35), rng.uniform(.72, 1)
            else: nli_a, nli_b = rng.uniform(.25, .55), rng.uniform(.25, .55)
            votes = rng.integers(0, 2, size=3)
            if rng.random() < .55 and label < 2:
                votes[:] = 1 - label
            confidences = rng.uniform(.35, .98, size=3)
            provenance = rng.uniform(.2, 1, size=3)
            structured = [*votes, *confidences, *provenance, nli_a, nli_b, abs(nli_a - nli_b)]
            xs.append(vector(atomic(row["text"]), vocabulary, structured)); ys.append(label)
            groups.append(row["pmcid"]); splits.append(code)
            baseline.append(int(np.bincount(votes, minlength=2).argmax()))
    receipts.append(save_dataset(root, contracts, "CAND-S-003", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "LICENSED_CORPUS_PLUS_CONTROLLED_ARBITRATION_CASES", "supported_side_from_pairwise_NLI_with_abstain_band_confidence_and_provenance_features", vocabulary_sha))

    # S004: refusal policy combines OOD, coverage, NLI, citation, and numeric validity.
    xs, ys, groups, splits, baseline = [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        rng = rng_for("S004", code)
        selected = stable_rows(rows, split_name, None, 3000 if code == 0 else 900)
        for row in selected:
            coverage = float(rng.random()); ood = int(rng.random() < .18); nli = float(rng.random())
            citation = int(rng.random() > .12); numeric = int(rng.random() > .18)
            label = int(ood or coverage < .62 or nli < .52 or not citation or not numeric)
            xs.append(vector(atomic(row["text"]), vocabulary, [coverage, ood, nli, citation, numeric])); ys.append(label)
            groups.append(row["pmcid"]); splits.append(code); baseline.append(int(ood or not citation))
    receipts.append(save_dataset(root, contracts, "CAND-S-004", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "LICENSED_CORPUS_PLUS_CONTROLLED_REFUSAL_FLAGS", "refuse_if_OOD_low_coverage_low_NLI_missing_citation_or_numeric_mismatch", vocabulary_sha))

    # S005: BIO spans include two atomic clauses inside one sentence; the regex baseline merges them.
    xs, ys, groups, splits, baseline, sequence_ids, positions = [], [], [], [], [], [], []
    sequence_id = 0
    for split_name, code in SPLIT_CODE.items():
        selected = stable_rows(rows, split_name, None, 850 if code == 0 else 260)
        by_split = stable_rows(rows, split_name, None, len(selected) + 7)
        for index, row in enumerate(selected):
            claim_a = tokens(atomic(row["text"], 10))[:10]
            peer = by_split[(index + 7) % len(by_split)]
            claim_b = tokens(atomic(peer["text"], 8))[:8]
            answer_tokens = ["summary", ":", *claim_a, ";", "however", *claim_b, ".", "citation", ":", row["pmcid"]]
            labels = [0, 0] + ([1] + [2] * (len(claim_a) - 1)) + [0, 0] + ([1] + [2] * (len(claim_b) - 1)) + [0, 0, 0, 0]
            regex_labels = [0, 0] + ([1] + [2] * (len(claim_a) + 2 + len(claim_b) - 1)) + [0, 0, 0, 0]
            if len(answer_tokens) != len(labels) or len(labels) != len(regex_labels):
                raise RuntimeError("S005 sequence construction")
            for pos, (token, label, base) in enumerate(zip(answer_tokens, labels, regex_labels)):
                prev = answer_tokens[pos - 1] if pos else "BOS"; nxt = answer_tokens[pos + 1] if pos + 1 < len(answer_tokens) else "EOS"
                structured = [pos / len(answer_tokens), int(token in {";", ".", ":"}), int(prev in {";", ":", "however"}), int(nxt in {";", ".", "citation"})]
                xs.append(vector(f"TOKEN {token} PREV {prev} NEXT {nxt}", vocabulary, structured)); ys.append(label)
                groups.append(row["pmcid"]); splits.append(code); baseline.append(base); sequence_ids.append(sequence_id); positions.append(pos)
            sequence_id += 1
    receipts.append(save_dataset(root, contracts, "CAND-S-005", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline), "sequence_id": np.asarray(sequence_ids), "token_position": np.asarray(positions)}, groups, splits, "STRUCTURE_DERIVED_EXACT_ATOMIC_CLAIM_SPANS", "BIO_spans_from_two_source_bound_atomic_clauses_with_exact_token_boundaries", vocabulary_sha))

    # S006: seven joint risk/reason codes; whitelist ignores freshness, truth state, hash, and link validity.
    xs, ys, groups, splits, baseline = [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        selected = stable_rows(rows, split_name, None, 2800 if code == 0 else 900)
        for index, row in enumerate(selected):
            label = index % 7
            fields = np.zeros(10, dtype=np.float32)
            fields[0] = 1.0  # CC-BY
            fields[1] = .1   # normalized age
            fields[2] = 1.0  # experimental source state
            fields[3] = 1.0  # source hash
            fields[4] = 1.0  # claim link
            if label == 1: fields[1] = 1.0
            elif label == 2: fields[2] = 0.0; fields[5] = 1.0
            elif label == 3: fields[2] = 0.0; fields[6] = 1.0
            elif label == 4: fields[0] = 0.0; fields[7] = 1.0
            elif label == 5: fields[3] = 0.0; fields[8] = 1.0
            elif label == 6: fields[4] = 0.0; fields[9] = 1.0
            xs.append(vector(f"{row['title']} {row['section']}", vocabulary, fields)); ys.append(label)
            groups.append(row["pmcid"]); splits.append(code); baseline.append(0 if fields[0] else 4)
    receipts.append(save_dataset(root, contracts, "CAND-S-006", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "LICENSED_SOURCE_METADATA_PLUS_CONTROLLED_RISK_CASES", "joint_risk_reason_from_license_age_truth_class_source_hash_and_claim_link", vocabulary_sha))

    # S007: continuous quality score from independently exposed verification fields.
    xs, ys, groups, splits, baseline, bad = [], [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        rng = rng_for("S007", code)
        selected = stable_rows(rows, split_name, None, 3000 if code == 0 else 900)
        for row in selected:
            nli, coverage = float(rng.random()), float(rng.random())
            citation, numeric, refusal = (int(rng.random() > threshold) for threshold in (.18, .22, .2))
            claim_count = int(rng.integers(1, 8)); length_score = min(claim_count / 5, 1.0)
            score = .30 * nli + .20 * citation + .20 * numeric + .15 * coverage + .15 * refusal
            if nli < .25 or (not citation and claim_count > 2): score *= .35
            base = .55 * citation + .45 * length_score
            xs.append(vector(atomic(row["text"]), vocabulary, [nli, coverage, citation, numeric, refusal, claim_count / 8, length_score])); ys.append(score)
            groups.append(row["pmcid"]); splits.append(code); baseline.append(base); bad.append(int(score < .4))
    receipts.append(save_dataset(root, contracts, "CAND-S-007", "regression", {"x": np.asarray(xs), "y": np.asarray(ys, dtype=np.float32), "baseline_prediction": np.asarray(baseline, dtype=np.float32), "bad_answer": np.asarray(bad, dtype=np.uint8)}, groups, splits, "LICENSED_CORPUS_PLUS_CONTROLLED_VERIFICATION_FIELDS", "quality_score_from_NLI_citation_numeric_coverage_and_refusal_context", vocabulary_sha))

    # S027: six domains, cross-domain, and explicit OOD with entity/unit/output features.
    xs, ys, groups, splits, baseline = [], [], [], [], []
    ood_text = {0: "guitar chord garden irrigation", 1: "municipal zoning marine mammal", 2: "restaurant menu stellar orbit"}
    for split_name, code in SPLIT_CODE.items():
        selected_by_domain = {domain: stable_rows(rows, split_name, domain, 180 if code == 0 else 75) for domain in DOMAINS}
        for domain, selected in selected_by_domain.items():
            for row in selected:
                text = f"{row['title']} {row['section']} {atomic(row['text'])}"
                entity = [0.0] * 6; entity[DOMAIN_ID[domain]] = 1.0
                xs.append(vector(text, vocabulary, [*entity, 0, 1, 0])); ys.append(DOMAIN_ID[domain]); groups.append(row["pmcid"]); splits.append(code); baseline.append(keyword_route(text))
        pair_count = 300 if code == 0 else 120
        for index in range(pair_count):
            da, db = DOMAINS[index % 6], DOMAINS[(index + 2) % 6]
            a, b = selected_by_domain[da][index % len(selected_by_domain[da])], selected_by_domain[db][index % len(selected_by_domain[db])]
            text = f"{a['title']} AND {b['title']}"
            entity = [0.0] * 6; entity[DOMAIN_ID[da]] = entity[DOMAIN_ID[db]] = 1.0
            xs.append(vector(text, vocabulary, [*entity, 1, 1, 0])); ys.append(6); groups.append(f"{a['pmcid']}+{b['pmcid']}"); splits.append(code); baseline.append(keyword_route(text))
        for index in range(300 if code == 0 else 120):
            text = f"{ood_text[code]} case_{index}"
            xs.append(vector(text, vocabulary, [0] * 6 + [0, 0, 1])); ys.append(7); groups.append(f"OOD_{code}_{index}"); splits.append(code); baseline.append(keyword_route(text))
    receipts.append(save_dataset(root, contracts, "CAND-S-027", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline)}, groups, splits, "LICENSED_DOMAIN_QUERIES_PLUS_SPLIT_DISTINCT_CONTROLLED_OOD", "six_domain_cross_domain_OOD_from_query_entities_units_and_requested_output", vocabulary_sha))

    # S028: four joint in/OOD reason codes; fixed distance threshold is the baseline.
    xs, ys, groups, splits, baseline, in_domain = [], [], [], [], [], []
    for split_name, code in SPLIT_CODE.items():
        rng = rng_for("S028", code)
        count = 4500 if code == 0 else 1400
        for index in range(count):
            distance = float(rng.random()); scores = np.sort(rng.random(6))[::-1]
            top, second = float(scores[0]), float(scores[1]); margin = top - second; coverage = float(rng.random())
            if distance >= .58: label = 1
            elif top < .62 or margin < .09: label = 2
            elif coverage < .42: label = 3
            else: label = 0
            structured = [distance, top, second, margin, coverage, *scores]
            xs.append(vector("", vocabulary, structured)); ys.append(label); groups.append(f"OODCASE_{code}_{index}"); splits.append(code)
            baseline.append(0 if distance < .58 else 1); in_domain.append(int(label == 0))
    receipts.append(save_dataset(root, contracts, "CAND-S-028", "classification", {"x": np.asarray(xs), "y": np.asarray(ys), "baseline_prediction": np.asarray(baseline), "reason_code": np.asarray(ys), "in_domain": np.asarray(in_domain, dtype=np.uint8)}, groups, splits, "CONTROLLED_EMBEDDING_DISTANCE_DOMAIN_SCORE_ENTITY_COVERAGE_FIXTURE", "joint_in_domain_low_similarity_ambiguous_or_low_entity_coverage_reason", vocabulary_sha))

    manifest = {
        "schema": "cimc.forge200.support-exact-staging.v3",
        "status": "PASS",
        "candidate_count": len(receipts),
        "candidates": [item["candidate_id"] for item in receipts],
        "records": sum(item["records"] for item in receipts),
        "authority_nonzero": 0,
        "expert_labels_claimed": 0,
        "vocabulary_sha256": vocabulary_sha,
        "source_sha256": sha256_file(corpus),
        "content_root_sha256": hashlib.sha256(canonical_bytes(receipts)).hexdigest(),
    }
    write_json(root / "data" / "staged_support_exact_v3" / "manifest.v3.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
