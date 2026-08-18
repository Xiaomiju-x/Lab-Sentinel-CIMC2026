#!/usr/bin/env python3
"""Stage six contract-shaped NLI datasets with explicit controlled mutations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


TASKS = {
    "CAND-S-021": "PHOSPHOR",
    "CAND-S-022": "FURNACE",
    "CAND-S-023": "SEMIMAT",
    "CAND-S-024": "METROLOGY",
    "CAND-S-025": "PACKAGING",
    "CAND-S-026": "FABQUALITY",
}
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
TOKEN_RE = re.compile(r"[a-z][a-z0-9_+.-]*|\d+(?:\.\d+)?|[^\W\d_]", re.I | re.UNICODE)
TEXT_FEATURES = 768
FEATURES = 800


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


def atomic(text: str, words: int = 36) -> str:
    sentence = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return " ".join(sentence.split()[:words])


def vector(text: str, vocabulary: dict[str, int], structured: Iterable[float]) -> np.ndarray:
    value = np.zeros(FEATURES, dtype=np.float32)
    for term in tokens(text):
        index = vocabulary.get(term)
        if index is not None:
            value[index] += 1.0
    norm = float(np.linalg.norm(value[:TEXT_FEATURES]))
    if norm:
        value[:TEXT_FEATURES] /= norm
    fields = list(structured)[: FEATURES - TEXT_FEATURES]
    value[TEXT_FEATURES : TEXT_FEATURES + len(fields)] = fields
    return value


def stable_rows(rows: list[dict[str, Any]], split_name: str, domain: str, limit: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if row["split"] == split_name and row["domain"] == domain]
    return sorted(selected, key=lambda row: hashlib.sha256((row["chunk_id"] + domain).encode()).digest())[:limit]


def examples(candidate_id: str, row: dict[str, Any], peer: dict[str, Any], vocabulary: dict[str, int]) -> list[tuple[np.ndarray, int, int, int]]:
    claim, evidence = atomic(row["text"]), row["text"][:700]
    unknown = atomic(peer["text"])
    # Returned tuple: features, target (entails=0/contradicts=1/unknown=2),
    # frozen baseline prediction, special mismatch flag.
    if candidate_id == "CAND-S-021":
        specs = ((claim, [.96, 1, 1], 0, 0, 0), (claim + " NUMERIC_MUTATED", [.94, 0, 1], 1, 0, 1), (unknown, [.12, 1, 0], 2, 2, 0))
    elif candidate_id == "CAND-S-022":
        specs = ((claim, [.95, .08, 1, 1], 0, 0, 0), (claim + " LOG_VALUE_MUTATED", [.94, .08, 1, 0], 1, 0, 0), (claim, [.95, .92, 0, 1], 2, 2, 1))
    elif candidate_id == "CAND-S-023":
        specs = ((claim + " SOURCE_STATE EXPERIMENTAL", [.95, 1, 1], 0, 0, 0), (claim + " SOURCE_STATE COMPUTED", [.95, 0, 1], 1, 0, 1), (unknown, [.12, 1, 0], 2, 2, 0))
    elif candidate_id == "CAND-S-024":
        specs = ((claim + " UNIT_MATCH METHOD_MATCH", [.95, 1, 1], 0, 0, 0), (claim + " METHOD_MISMATCH", [.93, 1, 0], 1, 0, 1), (unknown + " UNIT_UNKNOWN", [.12, 0, 0], 2, 2, 0))
    elif candidate_id == "CAND-S-025":
        specs = ((claim + " CONDITION_MATCH STACK_MATCH", [.95, 1, 1], 0, 0, 0), (claim + " STACK_ORDER_MISMATCH", [.93, 1, 0], 1, 0, 1), (unknown + " CONDITION_UNKNOWN", [.12, 0, 0], 2, 2, 0))
    elif candidate_id == "CAND-S-026":
        specs = ((claim + " SCOPE_MATCH TOOL_STEP_MATCH", [.95, 1, 1], 0, 0, 0), (claim + " TOOL_STEP_MISMATCH", [.93, 1, 0], 1, 0, 1), (unknown + " SCOPE_UNKNOWN", [.12, 0, 0], 2, 2, 0))
    else:
        raise KeyError(candidate_id)
    return [(vector(f"CLAIM {text} EVIDENCE {evidence}", vocabulary, fields), target, base, reason) for text, fields, target, base, reason in specs]


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1]); args = parser.parse_args()
    root = args.root.resolve(); corpus_path = root / "data" / "corpora" / "ccby_multidomain_v2.jsonl"
    with corpus_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    vocabulary_receipt = json.loads((root / "contracts" / "support_exact_v3_vocabulary.json").read_text(encoding="utf-8"))
    vocabulary = {term: index for index, term in enumerate(vocabulary_receipt["terms"])}
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    stage = root / "data" / "staged_nli_exact_v1"; stage.mkdir(parents=True, exist_ok=True); receipts = []
    for candidate_id, domain in TASKS.items():
        xs: list[np.ndarray] = []; ys: list[int] = []; baseline: list[int] = []; reasons: list[int] = []; groups: list[str] = []; splits: list[int] = []
        for split_name, code in SPLIT_CODE.items():
            selected = stable_rows(rows, split_name, domain, 2500 if code == 0 else 700)
            peer_pool = stable_rows(rows, split_name, domain, len(selected) + 11)
            for index, row in enumerate(selected):
                peer = peer_pool[(index + 11) % len(peer_pool)]
                if peer["pmcid"] == row["pmcid"]:
                    peer = next(item for item in peer_pool if item["pmcid"] != row["pmcid"])
                for features, target, base, reason in examples(candidate_id, row, peer, vocabulary):
                    xs.append(features); ys.append(target); baseline.append(base); reasons.append(reason)
                    groups.append(f"{row['pmcid']}+{peer['pmcid']}"); splits.append(code)
        split_array = np.asarray(splits, dtype=np.int8); group_array = np.asarray(groups)
        group_sets = {code: set(group_array[split_array == code].tolist()) for code in range(3)}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a in range(3) for b in range(a + 1, 3))
        counts = {name: int(np.sum(split_array == code)) for name, code in SPLIT_CODE.items()}
        contract = contracts[candidate_id]
        path = stage / f"{candidate_id}.npz"
        np.savez_compressed(path, x=np.asarray(xs), y=np.asarray(ys), baseline_prediction=np.asarray(baseline), reason_code=np.asarray(reasons, dtype=np.uint8), groups=group_array, split=split_array, candidate_id=np.asarray(candidate_id), task_kind=np.asarray("classification"), truth_class=np.asarray("LICENSED_SOURCE_ANCHOR_PLUS_CONTROLLED_CONTRACT_MUTATIONS"), authority=np.asarray(0, dtype=np.int8))
        metadata = {
            "schema": "cimc.forge200.nli-exact-staged.v1", "status": "PASS", "candidate_id": candidate_id,
            "task_kind": "classification", "truth_class": "LICENSED_SOURCE_ANCHOR_PLUS_CONTROLLED_CONTRACT_MUTATIONS",
            "claim_state": "ENTAILMENT_FROM_EXACT_SOURCE_ANCHOR_CONTRADICTION_OR_UNKNOWN_FROM_EXPLICIT_MUTATION",
            "path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path),
            "records": len(ys), "counts": counts, "features": FEATURES, "cross_split_group_overlap": overlap,
            "split_sha256": hashlib.sha256(canonical_bytes(sorted({(g, int(s)) for g, s in zip(groups, splits)}))).hexdigest(),
            "feature_contract": contract["input_contract"], "label_derivation_rule": "exact_source_entails_explicit_field_mutation_contradicts_cross_source_unknown",
            "vocabulary_sha256": vocabulary_receipt["content_sha256"], "source_sha256": sha256_file(corpus_path),
            "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(), "contract_baseline": contract["baseline"],
            "contract_primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"], "authority": 0,
        }
        if overlap or min(counts.values()) < 16:
            raise RuntimeError(f"{candidate_id} split gate {overlap} {counts}")
        write_json(path.with_suffix(".metadata.json"), metadata); receipts.append(metadata)
    manifest = {"schema": "cimc.forge200.nli-exact-staging.v1", "status": "PASS", "candidate_count": len(receipts), "candidates": list(TASKS), "records": sum(item["records"] for item in receipts), "authority_nonzero": 0, "source_sha256": sha256_file(corpus_path), "content_root_sha256": hashlib.sha256(canonical_bytes(receipts)).hexdigest()}
    write_json(stage / "manifest.v1.json", manifest); print(json.dumps(manifest, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
