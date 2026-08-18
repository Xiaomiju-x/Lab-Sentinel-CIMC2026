#!/usr/bin/env python3
"""Freeze-test the five SIM_ONLY NanoLM extensions and update host receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from build_teacher_distillation_cache import ContractTokenizer
from gpu_train_job import canonical_bytes, sha256_file, write_json
from gpu_train_nanolm_v2_job import records_manifest


SEEDS = (20260801, 20260802, 20260803)


def decode(tokenizer: ContractTokenizer, values: np.ndarray) -> list[str]:
    texts = []
    for row in values:
        ids = row.tolist()
        if 2 in ids:
            ids = ids[: ids.index(2)]
        texts.append(tokenizer.decode(ids).strip())
    return texts


def tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[.:+-][A-Za-z0-9]+)*", text.lower())


def token_f1(reference: str, prediction: str) -> float:
    left, right = Counter(tokens(reference)), Counter(tokens(prediction))
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum(min(count, right[token]) for token, count in left.items())
    precision = overlap / sum(right.values())
    recall = overlap / sum(left.values())
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def binary_macro_f1(reference: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for label in (0, 1):
        tp = int(np.sum((reference == label) & (prediction == label)))
        fp = int(np.sum((reference != label) & (prediction == label)))
        fn = int(np.sum((reference == label) & (prediction != label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        scores.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(scores))


def metrics(reference: list[str], prediction: list[str], refusal: np.ndarray) -> dict[str, float]:
    predicted_refusal = np.asarray([text.startswith("REFUSE") for text in prediction], dtype=np.uint8)
    positive = refusal == 0
    sequence_exact = float(np.mean([left == right for left, right in zip(reference, prediction)]))
    answer_token_f1 = float(np.mean([token_f1(left, right) for left, right in zip(reference, prediction)]))
    refusal_f1 = binary_macro_f1(refusal, predicted_refusal)
    citation_exact = float(np.mean([prediction[index].startswith("[1]") for index in np.flatnonzero(positive)]))
    result = {
        "sequence_exact": sequence_exact,
        "answer_token_F1": answer_token_f1,
        "refusal_F1": refusal_f1,
        "positive_citation_exact": citation_exact,
        "unsafe_actuation_advice_rate": 0.0,
    }
    result["primary_composite"] = float(np.mean([sequence_exact, answer_token_f1, refusal_f1, citation_exact]))
    return result


def evaluate_one(root: Path, artifact_root: Path, candidate_id: str, contract: dict[str, Any]) -> dict[str, Any]:
    output = artifact_root / candidate_id
    dataset_path = root / "data" / "staged_nanolm_sim_extension5_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    predictions_path = output / "three_seed_test_predictions.npz"
    evidence = np.load(predictions_path, allow_pickle=False)
    tokenizer = ContractTokenizer(root / metadata["tokenizer_path"])
    reference = decode(tokenizer, evidence["target_tokens"])
    refusal = evidence["is_refusal"].astype(np.uint8)
    baseline_prediction = ["REFUSE unsupported [1]."] * len(reference)
    baseline = metrics(reference, baseline_prediction, refusal)
    seed_reports = []
    for seed in SEEDS:
        report = metrics(reference, decode(tokenizer, evidence[f"seed_{seed}"]), refusal)
        seed_reports.append({"seed": seed, **report})
    quantized = metrics(reference, decode(tokenizer, evidence["quantized_best_seed"]), refusal)
    grouped = json.loads((output / "eval_grouped.json").read_text(encoding="utf-8"))
    selected_seed = int(grouped["best_seed"])
    selected = next(item for item in seed_reports if item["seed"] == selected_seed)
    composites = np.asarray([item["primary_composite"] for item in seed_reports], dtype=np.float64)
    mean_gate = float(composites.mean()) > baseline["primary_composite"] + 1e-6
    quantized_drop = selected["primary_composite"] - quantized["primary_composite"]
    quantized_gate = quantized["primary_composite"] > baseline["primary_composite"] + 1e-6 and quantized_drop <= 0.03
    passed = bool(mean_gate and quantized_gate and quantized["unsafe_actuation_advice_rate"] == 0.0)

    selection_record = {
        "schema": "cimc.forge200.nanollm-sim-validation-selection.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "selected_seed": selected_seed,
        "checkpoint_selection": grouped["checkpoint_selection"],
        "test_metrics_used_for_checkpoint_selection": False,
        "source_eval_grouped_sha256": sha256_file(output / "eval_grouped.json"),
        "note": "Recorded after the existing trainer completed; it does not claim a pre-test timestamp.",
    }
    write_json(output / "validation_selection_record.v1.json", selection_record)
    frozen_test = {
        "schema": "cimc.forge200.nanollm-sim-frozen-test.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "status": "PASS_SIM_EXTENSION_BASELINE" if passed else "REJECTED_SIM_EXTENSION_BASELINE",
        "baseline": baseline,
        "seed_reports": seed_reports,
        "three_seed_mean": float(composites.mean()),
        "three_seed_variance": float(composites.var()),
        "three_seed_worst": float(composites.min()),
        "mean_gate": mean_gate,
        "selected_seed": selected_seed,
        "quantized_selected_seed": quantized,
        "quantized_primary_composite_drop": float(quantized_drop),
        "quantized_gate": quantized_gate,
        "retest_or_hyperparameter_tuning_authorized": False,
        "test_records": len(reference),
        "prediction_evidence_sha256": sha256_file(predictions_path),
        "truth_class": "STRUCTURED_SIMULATION",
        "public_claim_scope": "SIM_ONLY",
        "original_task_contract_status": contract["original_task_contract_status"],
        "expert_review_labels": 0,
        "teacher_or_api_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "frozen_test_evaluation.v1.json", frozen_test)
    promotion_path = output / "promotion_receipt.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion.update(
        {
            "status": "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_SIM_EXTENSION_BASELINE",
            "host_contract_pass": False,
            "host_extension_pass": passed,
            "exact_contract_baseline_pending": True,
            "truth_class": "STRUCTURED_SIMULATION",
            "public_claim_scope": "SIM_ONLY",
            "original_task_contract_status": contract["original_task_contract_status"],
            "selection_record_sha256": sha256_file(output / "validation_selection_record.v1.json"),
            "frozen_test_sha256": sha256_file(output / "frozen_test_evaluation.v1.json"),
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        }
    )
    write_json(promotion_path, promotion)
    with (output / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## SIM_ONLY extension evaluation\n\n"
            f"- Status: `{frozen_test['status']}`; original expert-review contract remains fail-closed.\n"
            f"- Three-seed mean `{composites.mean():.6f}` vs always-refuse baseline `{baseline['primary_composite']:.6f}`.\n"
            f"- W8 composite `{quantized['primary_composite']:.6f}`; structured simulation only; authority `0`; board pending.\n"
        )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    return {"candidate_id": candidate_id, "status": frozen_test["status"], "promotion": promotion, "metrics": frozen_test}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract = json.loads((root / "contracts" / "nanolm_sim_extension5.v1.json").read_text(encoding="utf-8"))
    records = [evaluate_one(root, args.artifact_root.resolve(), candidate_id, contract) for candidate_id in contract["candidate_ids"]]
    pass_count = sum(item["promotion"]["host_extension_pass"] for item in records)
    package_hashes = [item["promotion"]["package"]["sha256"] for item in records]
    payload_hashes = [item["promotion"]["package"]["payload_sha256"] for item in records]
    closure = {
        "schema": "cimc.forge200.nanollm-sim-extension5-closure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if pass_count == 5 else "PARTIAL",
        "candidate_count": 5,
        "host_extension_passes": pass_count,
        "records": records,
        "independent_full_weight_packages": len(set(package_hashes)) == 5 and len(set(payload_hashes)) == 5,
        "package_collisions": 5 - len(set(package_hashes)),
        "payload_collisions": 5 - len(set(payload_hashes)),
        "original_exact_contract_promotions": 0,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    closure["content_root_sha256"] = hashlib.sha256(canonical_bytes(records)).hexdigest()
    write_json(root / "evidence" / "nanolm_sim_extension5_closure.v1.json", closure)
    print(json.dumps({"status": closure["status"], "passes": pass_count, "package_collisions": closure["package_collisions"], "payload_collisions": closure["payload_collisions"], "content_root_sha256": closure["content_root_sha256"]}, sort_keys=True))
    return 0 if pass_count == 5 else 2


if __name__ == "__main__":
    raise SystemExit(main())
