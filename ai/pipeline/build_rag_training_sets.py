#!/usr/bin/env python3
"""Build source-bound GPU-B datasets without promoting literature to truth.

The admitted labels in this file are mechanically derivable from licensed
article structure: exact cited spans, PMCID/domain metadata, same-document
relevance, cross-document unknowns, and explicitly recorded controlled
mutations.  Tasks requiring hypotheses, measurement plans, or expert
adjudication are deliberately not materialized.
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
from typing import Any, Iterable

import numpy as np


FEATURES = 256
CONTEXT = 384
STAGED_SUBDIR = "staged"
CORPUS_SOURCE_ID = "europe_pmc_ccby_multidomain_v1"
DOMAIN_IDS = {
    "PHOSPHOR": 0,
    "FURNACE": 1,
    "SEMIMAT": 2,
    "METROLOGY": 3,
    "PACKAGING": 4,
    "FABQUALITY": 5,
}
DOMAIN_G_TASKS = {
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
ENCODER_TASKS = {
    "CAND-S-009": "PHOSPHOR",
    "CAND-S-010": "FURNACE",
    "CAND-S-011": "SEMIMAT",
    "CAND-S-012": "METROLOGY",
    "CAND-S-013": "PACKAGING",
    "CAND-S-014": "FABQUALITY",
}
RERANK_TASKS = {
    "CAND-S-015": "PHOSPHOR",
    "CAND-S-016": "FURNACE",
    "CAND-S-017": "SEMIMAT",
    "CAND-S-018": "METROLOGY",
    "CAND-S-019": "PACKAGING",
    "CAND-S-020": "FABQUALITY",
}
NLI_TASKS = {
    "CAND-S-021": "PHOSPHOR",
    "CAND-S-022": "FURNACE",
    "CAND-S-023": "SEMIMAT",
    "CAND-S-024": "METROLOGY",
    "CAND-S-025": "PACKAGING",
    "CAND-S-026": "FABQUALITY",
}
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def stable_index(token: str) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % FEATURES, 1.0 if digest[4] & 1 else -1.0


def text_features(text: str) -> np.ndarray:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", text.lower(), re.UNICODE)
    terms = tokens + [f"{a}::{b}" for a, b in zip(tokens, tokens[1:])]
    vector = np.zeros(FEATURES, dtype=np.float32)
    for term in terms:
        index, sign = stable_index(term)
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def atomic_sentence(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) >= 64]
    sentence = sentences[0] if sentences else text.strip()
    return sentence[:360].rstrip()


def mutated_numeric_claim(sentence: str, pmcid: str) -> str:
    match = re.search(r"(?<![A-Za-z])(-?\d+(?:\.\d+)?)", sentence)
    if match:
        original = float(match.group(1))
        replacement = original + max(abs(original) * 0.37, 7.0)
        text = (f"{replacement:.6g}").rstrip("0").rstrip(".")
        return sentence[: match.start()] + text + sentence[match.end() :]
    return f"The cited source identifier is PMC00000000, not {pmcid}."


def choose_cross(rows: list[dict[str, Any]], index: int, *, different_domain: bool = False) -> dict[str, Any]:
    current = rows[index]
    for offset in range(1, len(rows)):
        other = rows[(index + offset) % len(rows)]
        if other["split"] != current["split"] or other["pmcid"] == current["pmcid"]:
            continue
        if different_domain and other["domain"] == current["domain"]:
            continue
        return other
    raise RuntimeError(f"no cross-document peer for {current['chunk_id']}")


def encode_lm(prompt: str, answer: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = (prompt + "\nANSWER ").encode("utf-8")
    response = answer.encode("utf-8")
    raw = bytes() + prefix + response
    tokens = [256] + list(raw[: CONTEXT - 2]) + [257]
    x = np.full(CONTEXT, 258, dtype=np.int64)
    y = np.full(CONTEXT, 258, dtype=np.int64)
    mask = np.zeros(CONTEXT, dtype=np.float32)
    usable = min(len(tokens) - 1, CONTEXT)
    x[:usable] = tokens[:usable]
    y[:usable] = tokens[1 : usable + 1]
    answer_start = min(1 + len(prefix), CONTEXT - 1)
    mask[answer_start - 1 : usable] = 1.0
    return x, y, mask


def base_metadata(
    candidate_id: str,
    task_hash: str,
    task_kind: str,
    truth_class: str,
    corpus_sha: str,
    split_sha: str,
    counts: dict[str, int],
    records: int,
    feature_contract: str,
    engine_id: int,
) -> dict[str, Any]:
    minimum_records = 48 if task_kind == "token_lm" else 96
    return {
        "schema": "cimc.forge200.staged-dataset.v1",
        "status": "PASS" if min(counts.values()) >= 16 and records >= minimum_records else "BLOCKED_INSUFFICIENT_SPLIT_ROWS",
        "candidate_id": candidate_id,
        "task_kind": task_kind,
        "truth_class": truth_class,
        "claim_state": "SOURCE_BOUND_OR_EXPLICIT_CONTROLLED_TRANSFORM_NOT_INDEPENDENT_GROUND_TRUTH",
        "source_id": CORPUS_SOURCE_ID,
        "source_sha256": corpus_sha,
        "split_sha256": split_sha,
        "task_contract_sha256": task_hash,
        "feature_contract": feature_contract,
        "fit_preprocessing_on_train_only": task_kind not in {"token_lm", "contrastive_embedding"},
        "cross_split_group_overlap": 0,
        "counts": counts,
        "records": records,
        "minimum_records": minimum_records,
        "features": FEATURES,
        "authority": 0,
        "engine_id": engine_id,
        "teacher_outputs": 0,
        "expert_labels": 0,
    }


def save_arrays(root: Path, metadata: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    candidate_id = metadata["candidate_id"]
    path = root / "data" / STAGED_SUBDIR / f"{candidate_id}.npz"
    if metadata["status"] == "PASS":
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            **arrays,
            candidate_id=np.asarray(candidate_id),
            task_kind=np.asarray(metadata["task_kind"]),
            truth_class=np.asarray(metadata["truth_class"]),
            authority=np.asarray(0, dtype=np.int8),
        )
        metadata.update(
            {
                "path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    else:
        if path.exists():
            path.unlink()
        metadata["path"] = None
    write_json(root / "data" / STAGED_SUBDIR / f"{candidate_id}.metadata.json", metadata)
    return metadata


def stage_lm(
    root: Path,
    rows: list[dict[str, Any]],
    task_hashes: dict[str, str],
    corpus_sha: str,
    split_sha: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for candidate_id, (domain, keywords) in DOMAIN_G_TASKS.items():
        selected = []
        for row in rows:
            haystack = f"{row['title']} {row['section']} {row['text']}".lower()
            if row["domain"] == domain and (not keywords or any(keyword in haystack for keyword in keywords)):
                selected.append(row)
        xs, ys, masks, groups, splits = [], [], [], [], []
        focus = "general domain evidence" if not keywords else "/".join(keywords)
        for row in selected:
            prompt = (
                f"SOURCE {row['chunk_id']} DOMAIN {domain} SECTION {row['section']}\n"
                f"QUESTION Report only the publication evidence relevant to {focus}, with its citation."
            )
            answer = (
                f"[{row['chunk_id']}] {atomic_sentence(row['text'])} "
                "Scope: source-bound publication text; independent ground truth is not asserted."
            )
            x, y, mask = encode_lm(prompt, answer)
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            groups.append(row["pmcid"])
            splits.append(SPLIT_CODE[row["split"]])
        counts = {name: int(np.sum(np.asarray(splits) == code)) for name, code in SPLIT_CODE.items()}
        metadata = base_metadata(
            candidate_id,
            task_hashes[candidate_id],
            "token_lm",
            "LITERATURE_CURATED_EXPERIMENT_SOURCE_BOUND_QA",
            corpus_sha,
            split_sha,
            counts,
            len(xs),
            "byte_causal_lm_v1_context384_answer_only_loss_exact_citation",
            5,
        )
        metadata["selection_keywords"] = list(keywords)
        records.append(
            save_arrays(
                root,
                metadata,
                {
                    "x": np.asarray(xs, dtype=np.int64),
                    "y": np.asarray(ys, dtype=np.int64),
                    "loss_mask": np.asarray(masks, dtype=np.float32),
                    "groups": np.asarray(groups),
                    "split": np.asarray(splits, dtype=np.int8),
                },
            )
        )
    return records


def classification_rows(
    root: Path,
    candidate_id: str,
    examples: Iterable[tuple[str, int | float, str, int]],
    task_hashes: dict[str, str],
    corpus_sha: str,
    split_sha: str,
    *,
    task_kind: str = "classification",
    truth_class: str = "STRUCTURE_DERIVED_LICENSED_CORPUS",
    rule: str,
) -> dict[str, Any]:
    materialized = list(examples)
    x = np.asarray([text_features(item[0]) for item in materialized], dtype=np.float32)
    y_dtype = np.float32 if task_kind == "regression" else np.int64
    y = np.asarray([item[1] for item in materialized], dtype=y_dtype)
    groups = np.asarray([item[2] for item in materialized])
    split = np.asarray([item[3] for item in materialized], dtype=np.int8)
    counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
    metadata = base_metadata(candidate_id, task_hashes[candidate_id], task_kind, truth_class, corpus_sha, split_sha, counts, len(x), f"hashed_word_unigram_bigram_256::{rule}", 1)
    metadata["label_derivation_rule"] = rule
    return save_arrays(root, metadata, {"x": x, "y": y, "groups": groups, "split": split})


def pair_examples(rows: list[dict[str, Any]], domain: str) -> Iterable[tuple[dict[str, Any], dict[str, Any], int]]:
    selected = [row for row in rows if row["domain"] == domain]
    row_index = {row["chunk_id"]: index for index, row in enumerate(rows)}
    for row in selected:
        cross = choose_cross(rows, row_index[row["chunk_id"]], different_domain=True)
        yield row, row, 1
        yield row, cross, 0


def stage_support(
    root: Path,
    rows: list[dict[str, Any]],
    task_hashes: dict[str, str],
    corpus_sha: str,
    split_sha: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    output.append(
        classification_rows(
            root,
            "CAND-S-001",
            ((f"QUERY {row['title']} {row['section']} {atomic_sentence(row['text'])}", DOMAIN_IDS[row["domain"]], row["pmcid"], SPLIT_CODE[row["split"]]) for row in rows),
            task_hashes,
            corpus_sha,
            split_sha,
            rule="domain_label_from_frozen_corpus_namespace",
        )
    )

    claim_span_examples = []
    token_pattern = re.compile(r"[A-Za-z][A-Za-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.UNICODE)
    for row in rows:
        claim_tokens = token_pattern.findall(atomic_sentence(row["text"]))
        answer_tokens = ["Evidence", "states"] + claim_tokens + ["Citation", row["chunk_id"]]
        for position, token in enumerate(answer_tokens):
            if position < 2 or position >= 2 + len(claim_tokens):
                label = 0
            elif position == 2:
                label = 1
            else:
                label = 2
            previous = answer_tokens[position - 1] if position else "BOS"
            following = answer_tokens[position + 1] if position + 1 < len(answer_tokens) else "EOS"
            claim_span_examples.append(
                (
                    f"TOKEN {token} PREV {previous} NEXT {following} POSITION {position} SENTENCE_BOUNDARY {int(position in (0, len(answer_tokens)-1))}",
                    label,
                    row["pmcid"],
                    SPLIT_CODE[row["split"]],
                )
            )
    output.append(
        classification_rows(
            root,
            "CAND-S-005",
            claim_span_examples,
            task_hashes,
            corpus_sha,
            split_sha,
            truth_class="STRUCTURE_DERIVED_EXACT_CLAIM_SPAN",
            rule="BIO_span_from_fixed_answer_template_and_exact_source_sentence_offsets",
        )
    )

    sufficient, arbitration, refusal, provenance, quality = [], [], [], [], []
    for index, row in enumerate(rows):
        cross = choose_cross(rows, index, different_domain=True)
        code = SPLIT_CODE[row["split"]]
        claim = atomic_sentence(row["text"])
        wrong = atomic_sentence(cross["text"])
        group = f"{row['pmcid']}+{cross['pmcid']}"
        sufficient.extend(
            [
                (f"CLAIM {claim} EVIDENCE {row['text']}", 1, row["pmcid"], code),
                (f"CLAIM {claim} EVIDENCE {cross['text']}", 0, group, code),
            ]
        )
        arbitration.extend(
            [
                (f"A {claim} B {wrong} EVIDENCE {row['text']}", 0, group, code),
                (f"A {wrong} B {claim} EVIDENCE {row['text']}", 1, group, code),
                (f"A {wrong} B {mutated_numeric_claim(wrong, cross['pmcid'])} EVIDENCE {row['text']}", 2, group, code),
            ]
        )
        refusal.extend(
            [
                (f"QUERY {claim} COVERAGE exact CITATION {row['chunk_id']} OOD 0", 0, row["pmcid"], code),
                (f"QUERY {claim} COVERAGE cross_domain CITATION {cross['chunk_id']} OOD 1", 1, group, code),
            ]
        )
        provenance.extend(
            [
                (f"LICENSE CC_BY SOURCE_SHA {row['source_sha256']} CLAIM_LINK {row['chunk_id']}", 0, row["pmcid"], code),
                (f"LICENSE CC_BY SOURCE_SHA MISSING CLAIM_LINK {row['chunk_id']}", 1, row["pmcid"], code),
                (f"LICENSE UNKNOWN SOURCE_SHA MISSING CLAIM_LINK NONE", 2, row["pmcid"], code),
            ]
        )
        quality.extend(
            [
                (f"ANSWER {claim} CITATION {row['chunk_id']} EVIDENCE {row['text']}", 1.0, row["pmcid"], code),
                (f"ANSWER {wrong} CITATION {cross['chunk_id']} EVIDENCE {row['text']}", 0.0, group, code),
            ]
        )
    for cid, examples, kind, rule in (
        ("CAND-S-002", sufficient, "classification", "exact_span_same_source_sufficient_cross_domain_insufficient"),
        ("CAND-S-003", arbitration, "classification", "exact_supported_side_else_abstain"),
        ("CAND-S-004", refusal, "classification", "exact_source_accept_cross_domain_refuse"),
        ("CAND-S-006", provenance, "classification", "license_hash_claim_link_completeness_tier"),
        ("CAND-S-007", quality, "regression", "exact_cited_support_1_cross_document_unsupported_0"),
    ):
        output.append(classification_rows(root, cid, examples, task_hashes, corpus_sha, split_sha, task_kind=kind, truth_class="MIXED_LITERATURE_AND_CONTROLLED_STRUCTURE_FIXTURE", rule=rule))

    for cid, domain in ENCODER_TASKS.items():
        pairs = list(pair_examples(rows, domain))
        xq = np.asarray([text_features(f"{a['title']} {a['section']}") for a, _, _ in pairs], dtype=np.float32)
        xp = np.asarray([text_features(b["text"]) for _, b, _ in pairs], dtype=np.float32)
        y = np.asarray([label for _, _, label in pairs], dtype=np.int64)
        groups = np.asarray([f"{a['pmcid']}+{b['pmcid']}" for a, b, _ in pairs])
        split = np.asarray([SPLIT_CODE[a["split"]] for a, _, _ in pairs], dtype=np.int8)
        counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
        metadata = base_metadata(cid, task_hashes[cid], "contrastive_embedding", "STRUCTURE_DERIVED_LICENSED_CORPUS", corpus_sha, split_sha, counts, len(pairs), "shared_hashed_query_passage_256_same_document_positive_cross_document_negative", 4)
        metadata["label_derivation_rule"] = "same_document_positive_cross_document_same_split_negative"
        output.append(save_arrays(root, metadata, {"x_query": xq, "x_passage": xp, "y": y, "groups": groups, "split": split}))

    for cid, domain in RERANK_TASKS.items():
        examples = []
        for query, passage, label in pair_examples(rows, domain):
            examples.append((f"QUERY {query['title']} {query['section']} PASSAGE {passage['text']}", label, f"{query['pmcid']}+{passage['pmcid']}", SPLIT_CODE[query["split"]]))
        output.append(classification_rows(root, cid, examples, task_hashes, corpus_sha, split_sha, rule="same_document_relevant_cross_document_same_split_irrelevant"))

    for cid, domain in NLI_TASKS.items():
        selected = [row for row in rows if row["domain"] == domain]
        row_index = {row["chunk_id"]: index for index, row in enumerate(rows)}
        examples = []
        for row in selected:
            cross = choose_cross(rows, row_index[row["chunk_id"]], different_domain=True)
            claim = atomic_sentence(row["text"])
            code = SPLIT_CODE[row["split"]]
            examples.extend(
                [
                    (f"CLAIM {claim} EVIDENCE {row['text']}", 0, row["pmcid"], code),
                    (f"CLAIM {mutated_numeric_claim(claim, row['pmcid'])} EVIDENCE {row['text']}", 1, row["pmcid"], code),
                    (f"CLAIM {claim} EVIDENCE {cross['text']}", 2, f"{row['pmcid']}+{cross['pmcid']}", code),
                ]
            )
        output.append(classification_rows(root, cid, examples, task_hashes, corpus_sha, split_sha, truth_class="MIXED_LITERATURE_AND_CONTROLLED_NUMERIC_MUTATION", rule="exact_span_entails_numeric_or_source_id_mutation_contradicts_cross_document_unknown"))

    output.append(
        classification_rows(
            root,
            "CAND-S-027",
            ((f"QUERY {row['title']} {row['section']} {atomic_sentence(row['text'])}", DOMAIN_IDS[row["domain"]], row["pmcid"], SPLIT_CODE[row["split"]]) for row in rows),
            task_hashes,
            corpus_sha,
            split_sha,
            rule="six_domain_route_from_frozen_namespace",
        )
    )
    ood_by_split = {
        0: ("guitar chord progression and garden irrigation schedule", "football match tactics and pastry recipe"),
        1: ("marine mammal migration and municipal zoning appeal", "language poetry meter and bicycle maintenance"),
        2: ("stellar orbit catalog and restaurant menu planning", "classical piano fingering and bird identification"),
    }
    ood_examples = []
    for row in rows:
        code = SPLIT_CODE[row["split"]]
        ood_examples.append((f"QUERY {row['title']} {row['section']}", 0, row["pmcid"], code))
        for index, text in enumerate(ood_by_split[code]):
            ood_examples.append((f"QUERY {text} fixture_{row['chunk_id']}_{index}", 1, row["pmcid"], code))
    output.append(classification_rows(root, "CAND-S-028", ood_examples, task_hashes, corpus_sha, split_sha, truth_class="LICENSED_IN_DOMAIN_PLUS_CONTROLLED_OOD_FIXTURE", rule="licensed_domain_query_vs_split_distinct_explicit_OOD_templates"))

    shared_pairs = []
    for index, row in enumerate(rows):
        cross = choose_cross(rows, index, different_domain=True)
        shared_pairs.extend(((row, row, 1), (row, cross, 0)))
    xq = np.asarray([text_features(f"{a['title']} {a['section']}") for a, _, _ in shared_pairs], dtype=np.float32)
    xp = np.asarray([text_features(b["text"]) for _, b, _ in shared_pairs], dtype=np.float32)
    y = np.asarray([label for _, _, label in shared_pairs], dtype=np.int64)
    groups = np.asarray([f"{a['pmcid']}+{b['pmcid']}" for a, b, _ in shared_pairs])
    split = np.asarray([SPLIT_CODE[a["split"]] for a, _, _ in shared_pairs], dtype=np.int8)
    counts = {name: int(np.sum(split == code)) for name, code in SPLIT_CODE.items()}
    metadata = base_metadata(
        "CAND-S-029",
        task_hashes["CAND-S-029"],
        "contrastive_embedding",
        "STRUCTURE_DERIVED_LICENSED_CORPUS",
        corpus_sha,
        split_sha,
        counts,
        len(shared_pairs),
        "shared_six_domain_hashed_query_passage_256_same_document_positive_cross_domain_negative",
        4,
    )
    metadata["label_derivation_rule"] = "same_document_positive_cross_domain_same_split_hard_negative"
    output.append(save_arrays(root, metadata, {"x_query": xq, "x_passage": xp, "y": y, "groups": groups, "split": split}))
    output.append(
        classification_rows(
            root,
            "CAND-S-030",
            (
                (
                    f"QUERY {query['title']} {query['section']} DOMAIN {query['domain']} PASSAGE {passage['text']}",
                    label,
                    f"{query['pmcid']}+{passage['pmcid']}",
                    SPLIT_CODE[query["split"]],
                )
                for query, passage, label in shared_pairs
            ),
            task_hashes,
            corpus_sha,
            split_sha,
            rule="shared_relevance_grade_same_document_1_cross_domain_same_split_0",
        )
    )

    si_templates = {
        0: (
            ("energy {v} joule equals power {half} watt times 2 second", 0),
            ("temperature {v} degree_C is convertible to {kelvin} kelvin", 1),
            ("length {v} metre equals temperature {v} kelvin", 2),
        ),
        1: (
            ("pressure {v} pascal equals force {v} newton per square_metre", 0),
            ("length {v} millimetre is convertible to {metre} metre", 1),
            ("electric_current {v} ampere equals mass {v} kilogram", 2),
        ),
        2: (
            ("heat_flux {v} watt_per_square_metre times area 1 square_metre equals power {v} watt", 0),
            ("time {v} millisecond is convertible to {second} second", 1),
            ("thermal_conductivity {v} watt_per_metre_kelvin equals energy {v} joule", 2),
        ),
    }
    si_examples = []
    for split_code, templates in si_templates.items():
        for index in range(40):
            value = float(index + 1)
            fields = {
                "v": f"{value:g}",
                "half": f"{value / 2:g}",
                "kelvin": f"{value + 273.15:g}",
                "metre": f"{value / 1000:g}",
                "second": f"{value / 1000:g}",
            }
            for template_index, (template, label) in enumerate(templates):
                si_examples.append((template.format(**fields), label, f"SI_TEMPLATE_{split_code}_{template_index}", split_code))
    output.append(
        classification_rows(
            root,
            "CAND-S-036",
            si_examples,
            task_hashes,
            corpus_sha,
            split_sha,
            truth_class="CURATED_SI_DIMENSIONAL_FIXTURE",
            rule="SI_dimensional_algebra_with_template_family_split_consistent_convertible_invalid",
        )
    )

    process_nodes = (
        ("sinter", 0, "SINTERING"),
        ("etch", 1, "ETCHING"),
        ("deposition", 2, "DEPOSITION"),
        ("chemical mechanical planar", 3, "CMP"),
        ("underfill", 4, "UNDERFILL"),
        ("solder", 5, "SOLDERING"),
        ("mold", 6, "MOLDING"),
    )
    entity_examples = []
    candidate_nodes = ",".join(name for _, _, name in process_nodes) + ",UNRESOLVED"
    for row in rows:
        haystack = f"{row['title']} {row['section']} {row['text']}".lower()
        matched = next(((term, label, name) for term, label, name in process_nodes if term in haystack), None)
        if matched:
            term, label, name = matched
            mention = term
        else:
            label, name, mention = 7, "UNRESOLVED", atomic_sentence(row["text"]).split(maxsplit=1)[0]
        entity_examples.append(
            (
                f"MENTION {mention} CONTEXT {row['title']} {row['section']} CANDIDATE_NODES {candidate_nodes}",
                label,
                row["pmcid"],
                SPLIT_CODE[row["split"]],
            )
        )
    output.append(
        classification_rows(
            root,
            "CAND-S-044",
            entity_examples,
            task_hashes,
            corpus_sha,
            split_sha,
            truth_class="LICENSED_PROCESS_TEXT_WITH_CONTROLLED_VOCABULARY_LINKS",
            rule="exact_controlled_process_vocabulary_match_else_unresolved",
        )
    )
    return output


def update_queue(root: Path, staged: list[dict[str, Any]]) -> dict[str, Any]:
    queue_path = root / "queue" / "dual_5090_queue.v1.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    status_by_id = {item["candidate_id"]: item for item in staged}
    for shard in ("GPU_A", "GPU_B"):
        for job in queue["jobs"][shard]:
            record = status_by_id.get(job["candidate_id"])
            if record is None:
                continue
            if record["status"] == "PASS":
                job["admission_state"] = "ADMITTED"
                job["staged_dataset"] = record["path"]
                job["staged_dataset_sha256"] = record["sha256"]
                job["staged_metadata"] = f"data/staged/{job['candidate_id']}.metadata.json"
                job["data_binding"] = {
                    "full_data_state": "MATERIALIZED",
                    "source_family": record["source_id"],
                    "truth_class": record["truth_class"],
                    "claim_state": record["claim_state"],
                }
            else:
                job["admission_state"] = "BLOCKED_PRE_GPU"
                job["data_binding"]["local_build_attempt"] = record["status"]
                job.pop("staged_dataset", None)
                job.pop("staged_dataset_sha256", None)
                job.pop("staged_metadata", None)
    jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
    queue["admitted_jobs"] = sum(job["admission_state"] == "ADMITTED" for job in jobs)
    queue["blocked_jobs"] = len(jobs) - queue["admitted_jobs"]
    queue["admitted_by_shard"] = {shard: sum(job["admission_state"] == "ADMITTED" for job in queue["jobs"][shard]) for shard in ("GPU_A", "GPU_B")}
    queue["status"] = "RECOVERABLE_QUEUE_SOURCE_GATED"
    write_json(queue_path, queue)
    for shard, filename in (("GPU_A", "gpu_a.queue.json"), ("GPU_B", "gpu_b.queue.json")):
        write_json(root / "queue" / filename, {"schema": queue["schema"], "shard": shard, "jobs": queue["jobs"][shard]})
    return queue


def update_bindings(root: Path, staged: list[dict[str, Any]], queue: dict[str, Any]) -> None:
    path = root / "data" / "ledgers" / "task_source_bindings.v1.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    records = {item["candidate_id"]: item for item in ledger["records"]}
    for item in staged:
        record = records[item["candidate_id"]]
        record["local_build_status"] = item["status"]
        if item["status"] == "PASS":
            record.update(
                {
                    "full_data_state": "MATERIALIZED",
                    "source_family": item["source_id"],
                    "truth_class": item["truth_class"],
                    "staged_dataset_sha256": item["sha256"],
                    "split_sha256": item["split_sha256"],
                }
            )
    jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
    ledger["gpu_admitted_count"] = queue["admitted_jobs"]
    ledger["gpu_blocked_count"] = queue["blocked_jobs"]
    ledger["materialized_direct_count"] = sum(record["full_data_state"] == "MATERIALIZED" for record in ledger["records"])
    ledger["record_or_corpus_build_required_count"] = sum("REQUIRED" in record["full_data_state"] for record in ledger["records"])
    ledger["status"] = "PASS_WITH_FAIL_CLOSED_TASKS"
    ledger["queue_state_counts"] = dict(Counter(job["admission_state"] for job in jobs))
    write_json(path, ledger)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus-stem", default="ccby_multidomain_v1")
    parser.add_argument("--staged-subdir", default="staged")
    parser.add_argument("--skip-queue-update", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not args.corpus_stem.replace("_", "").isalnum() or not args.staged_subdir.replace("_", "").isalnum():
        raise RuntimeError("UNSAFE_DATA_SUBDIRECTORY")
    global STAGED_SUBDIR, CORPUS_SOURCE_ID
    STAGED_SUBDIR = args.staged_subdir
    CORPUS_SOURCE_ID = f"europe_pmc_{args.corpus_stem}"
    corpus_path = root / "data" / "corpora" / f"{args.corpus_stem}.jsonl"
    ledger_suffix = args.corpus_stem.rsplit("_", 1)[-1]
    corpus_ledger = json.loads((root / "data" / "ledgers" / f"ccby_multidomain_corpus.{ledger_suffix}.json").read_text(encoding="utf-8"))
    if corpus_ledger["status"] != "PASS" or sha256_file(corpus_path) != corpus_ledger["corpus_sha256"]:
        raise RuntimeError("CCBY_CORPUS_GATE")
    rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines() if line]
    if any(row["authority"] != 0 for row in rows):
        raise RuntimeError("CORPUS_AUTHORITY_NONZERO")
    contracts = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    task_hashes = {row["candidate_id"]: hashlib.sha256(canonical_bytes(row)).hexdigest() for row in contracts}
    split_map = sorted({(row["pmcid"], row["split"]) for row in rows})
    split_sha = hashlib.sha256(canonical_bytes(split_map)).hexdigest()
    staged = stage_lm(root, rows, task_hashes, corpus_ledger["corpus_sha256"], split_sha)
    staged.extend(stage_support(root, rows, task_hashes, corpus_ledger["corpus_sha256"], split_sha))
    if args.skip_queue_update:
        pass_count = sum(item["status"] == "PASS" for item in staged)
        queue = {
            "admitted_jobs": pass_count,
            "blocked_jobs": len(staged) - pass_count,
            "admitted_by_shard": {"LOCAL4050": pass_count},
        }
    else:
        queue = update_queue(root, staged)
        update_bindings(root, staged, queue)
    manifest = {
        "schema": "cimc.forge200.rag-staging-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_FAIL_CLOSED_EXPERT_TASKS",
        "source_corpus_sha256": corpus_ledger["corpus_sha256"],
        "split_sha256": split_sha,
        "records": staged,
        "staged_pass": sum(item["status"] == "PASS" for item in staged),
        "staged_blocked": sum(item["status"] != "PASS" for item in staged),
        "teacher_outputs": 0,
        "expert_labels": 0,
        "queue_admitted_jobs": queue["admitted_jobs"],
        "queue_blocked_jobs": queue["blocked_jobs"],
        "content_root_sha256": hashlib.sha256(canonical_bytes(staged)).hexdigest(),
    }
    manifest["corpus_stem"] = args.corpus_stem
    manifest["staged_subdir"] = STAGED_SUBDIR
    manifest["queue_updated"] = not args.skip_queue_update
    write_json(root / "data" / STAGED_SUBDIR / "rag_staging_manifest.v1.json", manifest)
    print(json.dumps({"status": manifest["status"], "pass": manifest["staged_pass"], "blocked": manifest["staged_blocked"], "queue_admitted": queue["admitted_jobs"], "by_shard": queue["admitted_by_shard"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
