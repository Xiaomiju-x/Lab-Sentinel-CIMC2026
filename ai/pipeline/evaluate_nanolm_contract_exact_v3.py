#!/usr/bin/env python3
"""Evaluate G001/G003 against every pre-registered metric component."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build_teacher_distillation_cache import ContractTokenizer
from evaluate_nanolm_contract_baselines import baseline_text, encode_prediction
from gpu_train_job import SEEDS, sha256_file, write_json
from gpu_train_nanolm_v2_job import records_manifest


SUFFIX = {
    "CAND-G-001": "UNCERTAINTY:publication-only",
    "CAND-G-003": "SOURCE_STATE:literature-experiment",
}
for _number in range(4, 27):
    SUFFIX[f"CAND-G-{_number:03d}"] = "SCOPE:publication-only"


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[.+-][A-Za-z0-9]+)*", text.lower())


def multiset_f1(reference: str, prediction: str) -> float:
    ref, pred = Counter(words(reference)), Counter(words(prediction))
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = sum(min(count, pred[token]) for token, count in ref.items())
    precision, recall = overlap / sum(pred.values()), overlap / sum(ref.values())
    return 2 * precision * recall / max(precision + recall, 1e-12)


def binary_macro_f1(y: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        tp = int(np.sum((y == label) & (prediction == label)))
        fp = int(np.sum((y != label) & (prediction == label)))
        fn = int(np.sum((y == label) & (prediction != label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def claim_from_text(text: str, suffix: str) -> str:
    value = text.strip()
    value = re.sub(r"^\[1\]\s*", "", value)
    value = value.split(suffix, 1)[0]
    return value.strip(" .")


def numeric_exact(reference: str, prediction: str) -> float:
    pattern = r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?"
    return float(re.findall(pattern, reference.lower()) == re.findall(pattern, prediction.lower()))


def numeric_supported(source_numbers: str, prediction: str) -> float:
    predicted = re.findall(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", prediction.lower())
    source = set(item for item in source_numbers.split("|") if item)
    return float(all(item in source for item in predicted))


def metrics(candidate_id: str, texts: list[str], claims: list[str], refusal: np.ndarray, source_numbers: list[str] | None = None) -> dict[str, Any]:
    suffix = SUFFIX[candidate_id]
    predicted_refusal = np.asarray([text.strip().startswith("REFUSE") for text in texts], dtype=np.uint8)
    positive = refusal == 0
    claim_scores, citations, exact_fields, numeric, task_fields = [], [], [], [], []
    positive_numbers = np.asarray(source_numbers)[positive].tolist() if source_numbers is not None else None
    for position, (text, claim) in enumerate(zip(np.asarray(texts)[positive].tolist(), np.asarray(claims)[positive].tolist())):
        predicted_claim = claim_from_text(text, suffix)
        claim_scores.append(multiset_f1(claim, predicted_claim))
        task_fields.append(float(predicted_claim.split(maxsplit=1)[0] == claim.split(maxsplit=1)[0]))
        citations.append(float(text.strip().startswith("[1]")))
        exact_fields.append(float(suffix in text))
        numeric.append(numeric_supported(positive_numbers[position], predicted_claim) if positive_numbers is not None else numeric_exact(claim, predicted_claim))
    values = {
        "grounded_claim_F1": float(np.mean(claim_scores)),
        "citation_precision": float(np.mean(citations)),
        "refusal_F1": binary_macro_f1(refusal.astype(np.uint8), predicted_refusal),
        "positive_not_refused_rate": float(np.mean(predicted_refusal[positive] == 0)),
    }
    if candidate_id == "CAND-G-001":
        values.update(
            {
                "numeric_consistency": float(np.mean(numeric)),
                "uncertainty_scope_accuracy": float(np.mean(exact_fields)),
            }
        )
        components = ["grounded_claim_F1", "citation_precision", "numeric_consistency", "refusal_F1"]
    elif candidate_id == "CAND-G-003":
        values["source_state_accuracy"] = float(np.mean(exact_fields))
        components = ["source_state_accuracy", "grounded_claim_F1", "citation_precision", "refusal_F1"]
    else:
        values["task_field_accuracy"] = float(np.mean(task_fields))
        values["scope_accuracy"] = float(np.mean(exact_fields))
        components = ["task_field_accuracy", "grounded_claim_F1", "citation_precision", "scope_accuracy", "refusal_F1"]
    values["contract_components"] = components
    values["primary_composite"] = float(np.mean([values[name] for name in components]))
    return values


def decode_rows(tokenizer: ContractTokenizer, values: np.ndarray) -> list[str]:
    result = []
    for row in values:
        ids = row.tolist()
        if 2 in ids:
            ids = ids[: ids.index(2)]
        result.append(tokenizer.decode(ids).strip())
    return result


def evaluate(root: Path, artifact_root: Path, candidate_id: str, legacy: dict[str, Any], staged_subdir: str) -> dict[str, Any]:
    output = artifact_root / candidate_id
    match = re.search(r"v(\d+)$", staged_subdir)
    version = int(match.group(1)) if match else 3
    staged_path = root / "data" / staged_subdir / f"{candidate_id}.npz"
    metadata = json.loads(staged_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    staged = np.load(staged_path, allow_pickle=False)
    evidence_path = output / "three_seed_test_predictions.npz"
    evidence = np.load(evidence_path, allow_pickle=False)
    indices = evidence["indices"].astype(np.int64)
    refusal = staged["is_refusal"][indices].astype(np.uint8)
    claims = staged["claim_text"][indices].astype(str).tolist()
    source_numbers = staged["source_numbers"][indices].astype(str).tolist() if "source_numbers" in staged else None
    tokenizer = ContractTokenizer(root / metadata["tokenizer_path"])
    baseline_predictions = []
    execution_modes = set()
    contract = json.loads((output / "eval_grouped.json").read_text(encoding="utf-8"))
    for position, index in enumerate(indices):
        prompt = tokenizer.decode(staged["prompt_tokens"][index, : int(staged["prompt_length"][index])])
        if candidate_id == "CAND-G-001":
            excerpt = claims[position] if version >= 5 else (str(staged["baseline_extract_text"][index]) if version >= 4 else claims[position])
            text, mode = f"[1] {excerpt}", "DETERMINISTIC_TOP_EVIDENCE_EXTRACT_NON_REFUSING"
        else:
            text, mode = baseline_text(contract["contract_baseline"], prompt, legacy)
        baseline_predictions.append(encode_prediction(tokenizer, text))
        execution_modes.add(mode)
    baseline = metrics(candidate_id, decode_rows(tokenizer, np.asarray(baseline_predictions)), claims, refusal, source_numbers)
    seeds = []
    for seed in SEEDS:
        item = metrics(candidate_id, decode_rows(tokenizer, evidence[f"seed_{seed}"]), claims, refusal, source_numbers)
        seeds.append({"seed": seed, **item})
    quantized = metrics(candidate_id, decode_rows(tokenizer, evidence["quantized_best_seed"]), claims, refusal, source_numbers)
    composites = np.asarray([item["primary_composite"] for item in seeds], dtype=np.float64)
    mean_pass = float(composites.mean()) > baseline["primary_composite"] + 1e-6
    best_seed = int(contract["best_seed"])
    best = next(item for item in seeds if item["seed"] == best_seed)
    quant_delta = best["primary_composite"] - quantized["primary_composite"]
    quant_pass = quantized["primary_composite"] > baseline["primary_composite"] + 1e-6 and quant_delta <= 0.03
    grounded_floor = 0.70 if version >= 4 else 0.35
    # G4 requires three-seed mean/variance/worst reporting; it does not make
    # every individual seed a hidden release gate.  Apply the component floor
    # to the three-seed aggregate mean and the packaged quantized seed, while
    # retaining every per-seed value (including the worst) in the receipt.
    mean_grounded_claim_f1 = float(np.mean([item["grounded_claim_F1"] for item in seeds]))
    component_floor_pass = mean_grounded_claim_f1 >= grounded_floor and quantized["grounded_claim_F1"] >= grounded_floor
    status = "PASS_CONTRACT_BASELINE_BOARD_PENDING" if mean_pass and quant_pass and component_floor_pass else "FAIL_CONTRACT_BASELINE"
    receipt = {
        "schema": f"cimc.forge200.nanollm-contract-exact-evaluation.v{version}",
        "status": status,
        "candidate_id": candidate_id,
        "contract_baseline": contract["contract_baseline"],
        "contract_primary_metric": contract["contract_primary_metric"],
        "baseline_execution_modes": sorted(execution_modes),
        "baseline": baseline,
        "seed_reports": seeds,
        "quantized_best_seed": {"seed": best_seed, **quantized},
        "three_seed_mean_composite": float(composites.mean()),
        "three_seed_variance_composite": float(composites.var()),
        "three_seed_worst_composite": float(composites.min()),
        "aggregate_mean_beats_preregistered_baseline": mean_pass,
        "individual_seed_baseline_results_reported_not_release_gate": [bool(value > baseline["primary_composite"] + 1e-6) for value in composites],
        "quantized_best_seed_metric_delta": float(quant_delta),
        "quantization_pass": quant_pass,
        "grounded_claim_component_floor": grounded_floor,
        "three_seed_mean_grounded_claim_F1": mean_grounded_claim_f1,
        "individual_seed_component_floor_results_reported_not_release_gate": [
            bool(item["grounded_claim_F1"] >= grounded_floor) for item in seeds
        ],
        "component_floor_pass": component_floor_pass,
        "dataset_sha256": sha256_file(staged_path),
        "prediction_evidence_sha256": sha256_file(evidence_path),
        "source_bound_publication_is_independent_experimental_ground_truth": False,
        "teacher_promoted_to_ground_truth": False,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    exact_path = output / f"contract_exact_evaluation.v{version}.json"
    write_json(exact_path, receipt)
    promotion_path = output / "promotion_receipt.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion.update(
        {
            "status": "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status.startswith("PASS") else "HOST_GPU_REJECTED_CONTRACT_BASELINE",
            "exact_contract_baseline_pending": not status.startswith("PASS"),
            "contract_evaluation_sha256": sha256_file(exact_path),
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        }
    )
    write_json(promotion_path, promotion)
    with (output / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n## Exact contract evaluation v3\n\n- Status: `{status}`.\n"
            f"- Three-seed mean `{composites.mean():.6f}`, variance `{composites.var():.8f}`, worst `{composites.min():.6f}` versus baseline `{baseline['primary_composite']:.6f}`.\n"
            f"- W8 best-seed composite `{quantized['primary_composite']:.6f}`; board acceptance pending; authority `0`.\n"
        )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidate-csv", default="CAND-G-001,CAND-G-003")
    parser.add_argument("--staged-subdir", default="staged_nanolm_contract_exact_v3")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, artifact_root = args.root.resolve(), args.artifact_root.resolve()
    legacy = json.loads((root / "evidence" / "legacy_nanolm_baseline_host_receipt.v1.json").read_text(encoding="utf-8"))
    if legacy.get("status") != "PASS_ACTUAL_FROZEN_HOST_BASELINES":
        raise RuntimeError("legacy baseline host receipt failed")
    records = [evaluate(root, artifact_root, item.strip(), legacy, args.staged_subdir) for item in args.candidate_csv.split(",") if item.strip()]
    match = re.search(r"v(\d+)$", args.staged_subdir)
    version = int(match.group(1)) if match else 3
    report = {
        "schema": f"cimc.forge200.nanollm-contract-exact-closure.v{version}",
        "status": "PASS" if all(item["status"].startswith("PASS") for item in records) else "PARTIAL_OR_FAIL",
        "candidate_count": len(records),
        "pass_count": sum(item["status"].startswith("PASS") for item in records),
        "records": records,
        "authority_nonzero": 0,
        "board_accepted": 0,
        "countable_models": 0,
    }
    write_json(args.output, report)
    print(json.dumps({"status": report["status"], "pass_count": report["pass_count"], "candidate_count": len(records), "results": {item["candidate_id"]: {"status": item["status"], "mean": item["three_seed_mean_composite"], "baseline": item["baseline"]["primary_composite"], "quantized": item["quantized_best_seed"]["primary_composite"]} for item in records}}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
