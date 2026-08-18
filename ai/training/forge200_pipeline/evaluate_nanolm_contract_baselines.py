#!/usr/bin/env python3
"""Execute frozen NanoLM baselines on the available source-bound test contract.

This closes the executable part of each pre-registered baseline without
pretending that generic citation/refusal examples measure domain-expert fields
such as mechanism accuracy, phase order, or numeric parameter consistency.
Those missing components remain fail-closed in the receipt.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_nanolm_v2_job import ContractTokenizer, evaluation_subset, generation_metrics
from nanolm_architecture import MAX_GENERATION_TOKENS


EOS = 2


AVAILABLE_COMPONENTS = {
    "grounded_claim_F1": "answer_token_f1_source_bound_proxy",
    "citation_precision": "positive_citation_exact",
    "refusal_F1": "refusal_accuracy_balanced_examples",
    "unsupported_rate": "unsupported_negative_rate",
    "unsupported_claim_rate": "unsupported_negative_rate",
    "false_resolution_rate": "unsupported_negative_rate_proxy",
    "source_state_accuracy": "citation_and_source_bound_target_proxy",
    "scope_accuracy": "source_bound_target_proxy",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_contracts(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def evidence_excerpt(prompt: str, words: int = 14) -> str:
    match = re.search(r"(?:EVIDENCE|[A-Z0-9_]+_CARD)\[1\]\s+(.*)", prompt, flags=re.S)
    evidence = match.group(1) if match else prompt
    evidence = evidence.split("\nANSWER", 1)[0]
    sentence = re.split(r"(?<=[.!?])\s+", evidence.strip(), maxsplit=1)[0]
    return " ".join(sentence.split()[:words]).strip(" ,;:")


def baseline_text(name: str, prompt: str, legacy: dict[str, Any]) -> tuple[str, str]:
    lower = name.lower()
    excerpt = evidence_excerpt(prompt)
    cluster = legacy["actual_outputs"]["cluster"]
    if "generic_flagship" in lower:
        return legacy["actual_outputs"]["flagship"][0], "ACTUAL_FROZEN_FLAGSHIP_FIXED_NATIVE_OUTPUT"
    if lower.startswith("e5_"):
        return cluster["E5 BRIEF"], "ACTUAL_FROZEN_E5_FIXED_NATIVE_OUTPUT"
    if lower.startswith("e4_"):
        return cluster["E4 QC"], "ACTUAL_FROZEN_E4_FIXED_NATIVE_OUTPUT"
    if lower.startswith("e6_"):
        return cluster["E6 CHEM"], "ACTUAL_FROZEN_E6_FIXED_NATIVE_OUTPUT"
    if lower.startswith("ai7_"):
        out = legacy["actual_outputs"]
        return (
            f"Thermal band {out['AI7_band']} at {out['AI7_thermal_percent']:.2f} percent.",
            "ACTUAL_FROZEN_AI7_SCALAR_PLUS_STATIC_TEMPLATE",
        )
    if lower.startswith("ai6_ai16_"):
        return "[1] PL scalar outputs require matching measurement conditions.", "FROZEN_SCALAR_STATIC_TEMPLATE"
    if "retrieval" in lower or "top_peak_lookup" in lower or "keyword" in lower:
        return f"[1] {excerpt}", "DETERMINISTIC_TOP_EVIDENCE_EXTRACT_NON_REFUSING"
    readable = re.sub(r"[_+]+", " ", name).strip()
    return f"[1] {readable}: {' '.join(excerpt.split()[:6])}", "DETERMINISTIC_PRE_REGISTERED_RULE_TEMPLATE_NON_REFUSING"


def encode_prediction(tokenizer: ContractTokenizer, text: str) -> np.ndarray:
    tokens = tokenizer.encode(text)[: MAX_GENERATION_TOKENS - 1] + [EOS]
    output = np.zeros(MAX_GENERATION_TOKENS, dtype=np.int64)
    output[: len(tokens)] = tokens
    return output


def component_coverage(metric: str) -> tuple[list[dict[str, str]], list[str]]:
    remaining = metric
    covered = []
    for component, evidence in sorted(AVAILABLE_COMPONENTS.items(), key=lambda item: -len(item[0])):
        if component.lower() in remaining.lower():
            covered.append({"contract_component": component, "available_evidence": evidence})
            remaining = re.sub(re.escape(component), "", remaining, flags=re.I)
    tokens = [item.strip("_") for item in re.split(r"_and_", remaining) if item.strip("_")]
    return covered, tokens


def evaluate_candidate(
    root: Path,
    artifact_root: Path,
    candidate_id: str,
    contract: dict[str, str],
    legacy: dict[str, Any],
) -> dict[str, Any]:
    dataset_path = root / "data" / "staged_nanolm_v2" / f"{candidate_id}.npz"
    metadata_path = dataset_path.with_suffix(".metadata.json")
    data = np.load(dataset_path, allow_pickle=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    tokenizer = ContractTokenizer(root / metadata["tokenizer_path"])
    test = np.flatnonzero(data["split"].astype(np.int8) == 2)
    selected = evaluation_subset(test, data["is_refusal"].astype(np.uint8))
    predictions = []
    execution_modes = set()
    for prompt_tokens in data["prompt_tokens"][selected]:
        prompt = tokenizer.decode(prompt_tokens.tolist())
        text, mode = baseline_text(contract["baseline"], prompt, legacy)
        predictions.append(encode_prediction(tokenizer, text))
        execution_modes.add(mode)
    metrics = generation_metrics(
        tokenizer,
        np.asarray(predictions, dtype=np.int64),
        data["target_tokens"][selected],
        data["target_length"][selected],
        data["is_refusal"][selected],
    )
    evaluation_path = artifact_root / candidate_id / "eval_grouped.json"
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    seed_scores = [item["generation"]["primary_composite"] for item in evaluation["seed_reports"]]
    covered, uncovered = component_coverage(contract["primary_metric"])
    all_seed_pass = all(score > metrics["primary_composite"] for score in seed_scores)
    receipt = {
        "schema": "cimc.forge200.nanollm-contract-baseline-evaluation.v1",
        "status": (
            "PASS_AVAILABLE_SURROGATE_EXACT_CONTRACT_COMPONENTS_PENDING"
            if all_seed_pass
            else "FAIL_AVAILABLE_SURROGATE_BASELINE"
        ),
        "candidate_id": candidate_id,
        "contract_baseline": contract["baseline"],
        "contract_primary_metric": contract["primary_metric"],
        "baseline_execution_modes": sorted(execution_modes),
        "baseline_metrics": metrics,
        "candidate_three_seed_primary_composite": seed_scores,
        "candidate_all_three_seeds_beat_available_baseline": all_seed_pass,
        "covered_contract_components": covered,
        "uncovered_contract_components": uncovered,
        "exact_contract_baseline_pending": bool(uncovered),
        "source_bound_surrogate_is_independent_expert_truth": False,
        "legacy_host_receipt_content_sha256": hashlib.sha256(canonical_bytes(legacy)).hexdigest(),
        "dataset_sha256": sha256_file(dataset_path),
        "test_examples": len(selected),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(artifact_root / candidate_id / "contract_baseline_evaluation.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    artifact_root = args.artifact_root.resolve()
    legacy_path = root / "evidence" / "legacy_nanolm_baseline_host_receipt.v1.json"
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    if legacy["status"] != "PASS_ACTUAL_FROZEN_HOST_BASELINES":
        raise RuntimeError("legacy host baseline gate")
    contracts = read_contracts(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    records = [
        evaluate_candidate(root, artifact_root, candidate_id, contracts[candidate_id], legacy)
        for candidate_id in args.candidate_csv.split(",")
    ]
    report = {
        "schema": "cimc.forge200.nanollm-contract-baseline-shard.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_AVAILABLE_SURROGATE_EXACT_COMPONENTS_PENDING"
        if all(item["candidate_all_three_seeds_beat_available_baseline"] for item in records)
        else "FAIL_AVAILABLE_SURROGATE_BASELINE",
        "candidate_count": len(records),
        "available_baseline_pass": sum(item["candidate_all_three_seeds_beat_available_baseline"] for item in records),
        "exact_contract_fully_covered": sum(not item["exact_contract_baseline_pending"] for item in records),
        "board_accepted": 0,
        "countable_models": 0,
        "authority_nonzero": 0,
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }
    write_json(args.output, report)
    print(json.dumps({key: report[key] for key in ("status", "candidate_count", "available_baseline_pass", "exact_contract_fully_covered", "content_root_sha256")}, sort_keys=True))
    return 0 if report["status"].startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
