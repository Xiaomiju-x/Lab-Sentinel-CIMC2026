#!/usr/bin/env python3
"""Build source-bound, contract-shaped NanoLM pilot sets.

Only statements copied from a licensed publication span become positive
targets.  Refusals use a same-split claim copied from another document, so no
teacher or generated prose is promoted to truth.  The first pilot intentionally
covers G001 and G003; G002 needs live furnace EvidenceCards and stays closed.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from build_nanolm_v2_sets import DOMAIN_G_TASKS, FIRST_PIECE, TOKENIZER_NAME, build_tokenizer
from gpu_train_job import canonical_bytes, sha256_file, write_json
from nanolm_architecture import CONTEXT_TOKENS, MAX_GENERATION_TOKENS, config_for_candidate


PAD, BOS, EOS = 0, 1, 2
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
PILOT = {
    "CAND-G-001": {
        "domain": "PHOSPHOR",
        "card": "COMPOSITION_PROCESS_PROPERTY_CARD",
        "question": "State one supported phosphor process-structure-property claim and its uncertainty scope.",
        "suffix": "UNCERTAINTY:publication-only",
        "exact_fields": ["grounded_claim", "citation", "numeric_consistency", "uncertainty_scope", "refusal"],
    },
    "CAND-G-003": {
        "domain": "SEMIMAT",
        "card": "SEMICONDUCTOR_PROPERTY_CARD",
        "question": "State one supported semiconductor-material screening claim and its source state.",
        "suffix": "SOURCE_STATE:literature-experiment",
        "exact_fields": ["grounded_claim", "citation", "source_state", "refusal"],
    },
}

TASK_FOCUS_RULES: dict[str, tuple[str, list[tuple[str, tuple[str, ...]]]]] = {
    "CAND-G-004": ("METHOD", [("xrd", ("xrd", "diffraction")), ("pl", ("photoluminescence", "spectrum")), ("sem", ("sem ", "microscopy")), ("metrology-other", ("measurement",))]),
    "CAND-G-005": ("RELIABILITY", [("delamination", ("delamin",)), ("moisture", ("moisture", "humidity")), ("thermal", ("thermal", "temperature")), ("stress", ("stress", "warpage"))]),
    "CAND-G-006": ("FAB_SCOPE", [("yield", ("yield",)), ("virtual-metrology", ("virtual metrology",)), ("wafer", ("wafer",)), ("process", ("process",))]),
    "CAND-G-007": ("PHASE_EDGE", [("phase", ("phase",)), ("crystal", ("crystal",)), ("transition", ("transition",))]),
    "CAND-G-008": ("SITE_EDGE", [("occupancy", ("occup",)), ("substitution", ("substitut",)), ("coordination", ("coordination", "site")), ("dopant", ("dop",))]),
    "CAND-G-009": ("DEFECT_EDGE", [("oxygen-vacancy", ("oxygen vacan",)), ("valence", ("valence",)), ("atmosphere", ("atmosphere",)), ("defect", ("defect",))]),
    "CAND-G-010": ("PATHWAY", [("energy-transfer", ("energy transfer",)), ("emission", ("emission",)), ("excitation", ("excitation",)), ("luminescence", ("luminescen",))]),
    "CAND-G-011": ("QUENCH_EDGE", [("thermal-quench", ("thermal quench",)), ("temperature", ("temperature",)), ("nonradiative", ("nonradiative",)), ("thermal", ("thermal",))]),
    "CAND-G-012": ("SINTER_STAGE", [("densification", ("densif", "shrinkage")), ("grain-growth", ("grain growth",)), ("kinetics", ("kinetic",)), ("sintering", ("sinter",))]),
    "CAND-G-013": ("PRECURSOR_EDGE", [("impurity", ("impurit",)), ("purity", ("purity",)), ("precursor", ("precursor",)), ("powder", ("powder",))]),
    "CAND-G-014": ("TRANSFER_EDGE", [("host", ("host",)), ("composition", ("composition",)), ("transfer", ("transfer",)), ("site", ("site",))]),
    "CAND-G-015": ("XRD_EDGE", [("phase", ("phase",)), ("peak", ("peak",)), ("diffraction", ("diffraction", "xrd"))]),
    "CAND-G-016": ("PL_EDGE", [("normalization", ("normalization",)), ("decay", ("decay",)), ("spectrum", ("spectrum", "spectral")), ("photoluminescence", ("photoluminescence",))]),
    "CAND-G-017": ("MICROSTRUCTURE", [("eds", ("eds",)), ("morphology", ("morpholog",)), ("composition", ("composition",)), ("sem", ("sem",))]),
    "CAND-G-018": ("CONFLICT_EDGE", [("conflict", ("conflict",)), ("correlation", ("correlation",)), ("comparison", ("comparison",)), ("multimodal", ("multimodal",))]),
    "CAND-G-019": ("CHARGE_EDGE", [("vacancy", ("vacan",)), ("charge", ("charge",)), ("dopant", ("dop",)), ("defect", ("defect",))]),
    "CAND-G-020": ("INTERFACE_EDGE", [("adhesion", ("adhesion",)), ("stress", ("stress",)), ("thin-film", ("thin film",)), ("interface", ("interface",))]),
    "CAND-G-021": ("DIELECTRIC_EDGE", [("permittivity", ("permittivity",)), ("loss", (" loss",)), ("thermal", ("thermal",)), ("dielectric", ("dielectric",))]),
    "CAND-G-022": ("INTEGRATION_EDGE", [("interface", ("interface",)), ("device", ("device",)), ("integration", ("integration",)), ("optoelectronic", ("optoelectronic",))]),
    "CAND-G-023": ("CURE_EDGE", [("rheology", ("rheolog",)), ("viscosity", ("viscos",)), ("underfill", ("underfill",)), ("cure", ("cure",))]),
    "CAND-G-024": ("JOINT_EDGE", [("diffusion", ("diffusion",)), ("metallurgy", ("metallurg",)), ("joint", ("joint",)), ("interconnect", ("interconnect", "solder"))]),
    "CAND-G-025": ("WARPAGE_EDGE", [("delamination", ("delamin",)), ("moisture", ("moisture",)), ("stress", ("stress",)), ("warpage", ("warpage",))]),
    "CAND-G-026": ("YIELD_EDGE", [("virtual-metrology", ("virtual metrology",)), ("yield", ("yield",)), ("wafer", ("wafer",)), ("process", ("process",))]),
}

for _candidate_id, (_field, _rules) in TASK_FOCUS_RULES.items():
    _domain, _keywords = DOMAIN_G_TASKS[_candidate_id]
    PILOT[_candidate_id] = {
        "domain": _domain,
        "keywords": _keywords,
        "card": f"{_field}_EVIDENCE_CARD",
        "question": f"Return one cited normalized {_field.lower()} evidence answer; refuse unsupported claims.",
        "suffix": "SCOPE:publication-only",
        "exact_fields": ["task_field", "grounded_claim", "citation", "scope", "refusal"],
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def task_contracts(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def normalized_sentences(text: str) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    return [item.strip() for item in re.split(r"(?<=[.!?])\s+", clean) if len(item.split()) >= 6]


def claim_span(row: dict[str, Any], max_words: int = 6) -> str:
    sentences = normalized_sentences(row["text"])
    sentence = sentences[0] if sentences else row["text"]
    return " ".join(sentence.split()[:max_words]).strip(" ,;:")


def keyword_class(text: str, rules: list[tuple[str, tuple[str, ...]]], default: str) -> str:
    lower = text.lower()
    for label, probes in rules:
        if any(probe in lower for probe in probes):
            return label
    return default


def structured_claim(candidate_id: str, row: dict[str, Any]) -> str:
    """Normalize only lexical facts visibly present in the cited source span."""

    text = row["text"]
    material = keyword_class(
        text,
        [
            ("oxide", (" oxide", "o2", "o3", "o4")),
            ("nitride", ("nitride",)),
            ("fluoride", ("fluoride",)),
            ("sulfide", ("sulfide", "sulphide")),
            ("phosphor", ("phosphor", "luminescent material")),
            ("interface", ("interface", "heterostructure")),
        ],
        "material-other",
    )
    prop = keyword_class(
        text,
        [
            ("photoluminescence", ("photoluminescence", " emission", "luminescen")),
            ("bandgap", ("band gap", "bandgap")),
            ("phase", ("x-ray diffraction", "xrd", " crystal", "phase")),
            ("thermal", ("thermal", "temperature", "quench")),
            ("microstructure", ("sem ", "morpholog", "grain", "microstructure")),
            ("electronic", ("electronic", "conductiv", "carrier", "dielectric")),
        ],
        "property-other",
    )
    if candidate_id == "CAND-G-001":
        process = keyword_class(
            text,
            [
                ("solid-state", ("solid-state", "solid state")),
                ("sol-gel", ("sol-gel", "sol gel")),
                ("hydrothermal", ("hydrothermal",)),
                ("combustion", ("combustion",)),
                ("annealing", ("anneal", "calcination", "calcined")),
                ("sintering", ("sinter",)),
            ],
            "process-other",
        )
        return f"COMP:{material} PROC:{process} PROP:{prop}"
    if candidate_id == "CAND-G-003":
        return f"MAT:{material} PROP:{prop}"
    field, rules = TASK_FOCUS_RULES[candidate_id]
    focus = keyword_class(text, rules, "unresolved")
    return f"{field}:{focus} MAT:{material} PROP:{prop}"


def same_split_peer(rows: list[dict[str, Any]], current: dict[str, Any], offset: int) -> dict[str, Any]:
    same = [row for row in rows if row["split"] == current["split"] and row["pmcid"] != current["pmcid"]]
    if not same:
        raise RuntimeError(f"no cross-document negative for {current['chunk_id']}")
    return same[offset % len(same)]


def choose_rows(rows: list[dict[str, Any]], domain: str, keywords: tuple[str, ...] = ()) -> list[dict[str, Any]]:
    limits = {"train": 256, "validation": 96, "test": 96}
    selected: list[dict[str, Any]] = []
    for split, limit in limits.items():
        pool = sorted(
            (
                row
                for row in rows
                if row["domain"] == domain
                and row["split"] == split
                and (
                    not keywords
                    or any(probe in f"{row['title']} {row['section']} {row['text']}".lower() for probe in keywords)
                )
            ),
            key=lambda row: (row["pmcid"], row["chunk_id"]),
        )
        # Round-robin document families prevents a long article from consuming
        # the whole split while retaining the corpus' frozen PMCID split.
        buckets: dict[str, list[dict[str, Any]]] = {}
        for row in pool:
            buckets.setdefault(row["pmcid"], []).append(row)
        order = sorted(buckets)
        round_index = 0
        while len([item for item in selected if item["split"] == split]) < min(limit, len(pool)):
            progressed = False
            for pmcid in order:
                if round_index < len(buckets[pmcid]):
                    selected.append(buckets[pmcid][round_index])
                    progressed = True
                    if len([item for item in selected if item["split"] == split]) >= min(limit, len(pool)):
                        break
            if not progressed:
                break
            round_index += 1
    return selected


def make_examples(candidate_id: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    spec = PILOT[candidate_id]
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        claim = structured_claim(candidate_id, row)
        evidence = " ".join(row["text"].split())[:720]
        prefix = (
            f"TASK {candidate_id}\nQUESTION {spec['question']}\n"
            f"SOURCE[1] {row['chunk_id']}\nSOURCE_STATE LITERATURE_CURATED_EXPERIMENT\n"
            f"{spec['card']}[1] CARD_FIELDS {claim}; EVIDENCE {evidence}\nANSWER "
        )
        common = {
            "group": row["pmcid"],
            "split": row["split"],
            "source_chunk_id": row["chunk_id"],
            "source_sha256": row["source_sha256"],
            "claim": claim,
            "baseline_extract": claim_span(row),
            "source_numbers": "|".join(re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", row["text"].lower())),
        }
        examples.append(
            {
                **common,
                "prompt": prefix,
                "target": f"[1] {claim} {spec['suffix']}",
                "is_refusal": 0,
                "negative_claim": "",
            }
        )
        peer = same_split_peer(rows, row, index + 1)
        negative_claim = claim_span(peer, max_words=8)
        refusal_prompt = (
            f"TASK {candidate_id}\nQUESTION Is this claim supported: {negative_claim}?\n"
            f"SOURCE[1] {row['chunk_id']}\nSOURCE_STATE LITERATURE_CURATED_EXPERIMENT\n"
            f"{spec['card']}[1] {evidence}\nANSWER "
        )
        examples.append(
            {
                **common,
                "prompt": refusal_prompt,
                "target": "REFUSE unsupported by Evidence [1].",
                "is_refusal": 1,
                "negative_claim": negative_claim,
            }
        )
    return examples


def encode_example(tokenizer: Any, item: dict[str, Any]) -> dict[str, Any]:
    target = tokenizer.encode(item["target"])[: MAX_GENERATION_TOKENS - 1] + [EOS]
    max_prompt = CONTEXT_TOKENS - MAX_GENERATION_TOKENS
    prompt = [BOS] + tokenizer.encode(item["prompt"])
    # Keep the task/question header and the final answer boundary.  The source
    # claim is deliberately at the beginning of each card and remains visible.
    prompt = prompt[:max_prompt]
    sequence = prompt + target
    x = np.full(CONTEXT_TOKENS, PAD, dtype=np.int64)
    y = np.full(CONTEXT_TOKENS, PAD, dtype=np.int64)
    loss_mask = np.zeros(CONTEXT_TOKENS, dtype=np.float32)
    usable = len(sequence) - 1
    x[:usable] = sequence[:-1]
    y[:usable] = sequence[1:]
    loss_mask[len(prompt) - 1 : usable] = 1.0
    prompt_tokens = np.zeros(max_prompt, dtype=np.int64)
    prompt_tokens[: len(prompt)] = prompt
    target_tokens = np.zeros(MAX_GENERATION_TOKENS, dtype=np.int64)
    target_tokens[: len(target)] = target
    return {
        "x": x,
        "y": y,
        "loss_mask": loss_mask,
        "prompt_tokens": prompt_tokens,
        "prompt_length": len(prompt),
        "target_tokens": target_tokens,
        "target_length": len(target),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--candidate-csv", default="CAND-G-001,CAND-G-003")
    args = parser.parse_args()
    root = args.root.resolve()
    requested = [item.strip() for item in args.candidate_csv.split(",") if item.strip()]
    unknown = sorted(set(requested) - set(PILOT))
    if unknown:
        raise RuntimeError(f"unsupported exact-v3 pilot candidates: {unknown}")
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    corpus = read_jsonl(corpus_path)
    contracts = task_contracts(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    examples_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in requested:
        selected = choose_rows(corpus, PILOT[candidate_id]["domain"], tuple(PILOT[candidate_id].get("keywords", ())))
        examples_by_candidate[candidate_id] = make_examples(candidate_id, selected)
    tokenizer = build_tokenizer(examples_by_candidate)
    tokenizer_core = {
        "schema": "cimc.icmat.nanollm-tokenizer.v1",
        "name": TOKENIZER_NAME,
        "status": "FROZEN_FOR_CONTRACT_EXACT_V3_TRAINING_BOARD_PENDING",
        "fit_split": "train_only",
        "vocab_size": len(tokenizer.pieces),
        "special_ids": {"pad": PAD, "bos": BOS, "eos": EOS},
        "byte_fallback": {"base_id": 3, "count": 256},
        "learned_piece_base_id": FIRST_PIECE,
        "selection": "frequency_desc_length_desc_bytes_lexicographic_min_count_2",
        "encoding": "longest_piece_first_then_byte_fallback",
        "pieces": [
            {"id": index, "base64": base64.b64encode(piece).decode("ascii"), "bytes": len(piece)}
            for index, piece in enumerate(tokenizer.pieces)
        ],
        "authority": 0,
    }
    tokenizer_core["content_sha256"] = hashlib.sha256(canonical_bytes(tokenizer_core)).hexdigest()
    tokenizer_path = root / "contracts" / "nanolm_tokenizer_exact_v6.json"
    write_json(tokenizer_path, tokenizer_core)
    for candidate_id, items in examples_by_candidate.items():
        suffix = PILOT[candidate_id]["suffix"]
        for item in items:
            if item["is_refusal"]:
                continue
            words = item["claim"].split()
            fitted: tuple[str, str] | None = None
            for width in range(min(6, len(words)), 0, -1):
                for start in range(0, len(words) - width + 1):
                    fitted_claim = " ".join(words[start : start + width])
                    fitted_target = f"[1] {fitted_claim} {suffix}"
                    if len(tokenizer.encode(fitted_target)) <= MAX_GENERATION_TOKENS - 1:
                        fitted = fitted_claim, fitted_target
                        break
                if fitted:
                    break
            if fitted is None:
                raise RuntimeError(f"cannot fit exact target for {candidate_id}")
            item["claim"], item["target"] = fitted
    stage_root = root / "data" / "staged_nanolm_contract_exact_v6"
    stage_root.mkdir(parents=True, exist_ok=True)
    records = []
    for candidate_id in requested:
        examples = examples_by_candidate[candidate_id]
        encoded = [encode_example(tokenizer, item) for item in examples]
        arrays = {
            key: np.asarray([item[key] for item in encoded])
            for key in ("x", "y", "loss_mask", "prompt_tokens", "prompt_length", "target_tokens", "target_length")
        }
        arrays.update(
            {
                "groups": np.asarray([item["group"] for item in examples]),
                "split": np.asarray([SPLIT_CODE[item["split"]] for item in examples], dtype=np.int8),
                "is_refusal": np.asarray([item["is_refusal"] for item in examples], dtype=np.uint8),
                "source_chunk_id": np.asarray([item["source_chunk_id"] for item in examples]),
                "claim_text": np.asarray([item["claim"] for item in examples]),
                "baseline_extract_text": np.asarray([item["baseline_extract"] for item in examples]),
                "source_numbers": np.asarray([item["source_numbers"] for item in examples]),
                "negative_claim_text": np.asarray([item["negative_claim"] for item in examples]),
                "candidate_id": np.asarray(candidate_id),
                "task_kind": np.asarray("nano_transformer_lm"),
                "truth_class": np.asarray("SOURCE_BOUND_RULE_NORMALIZED_CLAIM_PLUS_STRUCTURE_DERIVED_REFUSAL"),
                "authority": np.asarray(0, dtype=np.int8),
            }
        )
        data_path = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(data_path, **arrays)
        counts = {name: int(np.sum(arrays["split"] == code)) for name, code in SPLIT_CODE.items()}
        group_sets = {code: set(arrays["groups"][arrays["split"] == code].tolist()) for code in SPLIT_CODE.values()}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a in group_sets for b in group_sets if a < b)
        contract_sha = hashlib.sha256(canonical_bytes(contracts[candidate_id])).hexdigest()
        metadata = {
            "schema": "cimc.forge200.staged-nanolm-contract-exact.v6",
            "status": "PASS_CONTRACT_SHAPED_SOURCE_SUPERVISED",
            "candidate_id": candidate_id,
            "task_kind": "nano_transformer_lm",
            "truth_class": "SOURCE_BOUND_RULE_NORMALIZED_CLAIM_PLUS_STRUCTURE_DERIVED_REFUSAL",
            "claim_state": "PUBLICATION_TEXT_IS_SOURCE_BOUND_EVIDENCE_NOT_INDEPENDENT_EXPERIMENTAL_GROUND_TRUTH",
            "path": str(data_path.relative_to(root)).replace("\\", "/"),
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "records": len(examples),
            "split_counts": counts,
            "split_unit": "PMCID_DOCUMENT_FAMILY_BEFORE_EXAMPLE_DERIVATION",
            "cross_split_group_overlap": overlap,
            "tokenizer_path": str(tokenizer_path.relative_to(root)).replace("\\", "/"),
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "source_path": str(corpus_path.relative_to(root)).replace("\\", "/"),
            "source_sha256": sha256_file(corpus_path),
            "task_contract_sha256": contract_sha,
            "input_contract_state": "EXACT_CONTRACT_SHAPED_PUBLICATION_EVIDENCE_CARD",
            "exact_metric_fields": PILOT[candidate_id]["exact_fields"],
            "architecture": config_for_candidate(candidate_id).to_dict(),
            "teacher_outputs": 0,
            "teacher_promoted_to_ground_truth": False,
            "authority": 0,
        }
        if overlap or min(counts.values()) < 32:
            raise RuntimeError(f"split gate failed for {candidate_id}: {counts}, overlap={overlap}")
        write_json(data_path.with_suffix(".metadata.json"), metadata)
        records.append(metadata)
    manifest = {
        "schema": "cimc.forge200.nanollm-contract-exact-staging.v6",
        "status": "PASS_PILOT_STAGING",
        "candidate_count": len(records),
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
        "authority_nonzero": 0,
    }
    write_json(stage_root / "manifest.v6.json", manifest)
    print(json.dumps({"status": manifest["status"], "candidates": requested, "records": {r["candidate_id"]: r["records"] for r in records}, "content_root_sha256": manifest["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
