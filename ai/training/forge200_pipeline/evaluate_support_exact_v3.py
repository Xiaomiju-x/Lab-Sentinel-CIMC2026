#!/usr/bin/env python3
"""Evaluate S001-S007/S027/S028 against their frozen contract baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = (20260801, 20260802, 20260803)
CANDIDATES = tuple([f"CAND-S-{index:03d}" for index in range(1, 8)] + ["CAND-S-027", "CAND-S-028"])


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


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0 + 1.0
        cursor = end
    return ranks


def spearman(y: np.ndarray, score: np.ndarray) -> float:
    a, b = rankdata(y), rankdata(score)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def binary_auroc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    positives, negatives = int(y.sum()), int((~y).sum())
    if not positives or not negatives:
        return 0.5
    ranks = rankdata(score)
    return float((ranks[y].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(bool)
    if not np.any(y):
        return 0.0
    order = np.argsort(-score, kind="mergesort")
    ranked = y[order]
    precision = np.cumsum(ranked) / np.arange(1, len(ranked) + 1)
    return float(precision[ranked].mean())


def macro_f1(y: np.ndarray, prediction: np.ndarray, classes: int | None = None) -> float:
    if classes is None:
        classes = int(max(np.max(y), np.max(prediction))) + 1
    values = []
    for label in range(classes):
        tp = int(np.sum((y == label) & (prediction == label)))
        fp = int(np.sum((y != label) & (prediction == label)))
        fn = int(np.sum((y == label) & (prediction != label)))
        values.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(values))


def ece(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    value = 0.0
    for start in np.linspace(0, 1, bins + 1)[:-1]:
        end = start + 1 / bins
        selected = (confidence >= start) & (confidence <= end if end >= 1 else confidence < end)
        if np.any(selected):
            value += float(np.mean(selected)) * abs(float(np.mean(prediction[selected] == y[selected])) - float(np.mean(confidence[selected])))
    return value


def hard_probability(prediction: np.ndarray, classes: int) -> np.ndarray:
    result = np.zeros((len(prediction), classes), dtype=np.float64)
    result[np.arange(len(prediction)), prediction.astype(int)] = 1.0
    return result


def brier_binary(y: np.ndarray, score: np.ndarray) -> float:
    return float(np.mean((score - y) ** 2))


def topk_recall(y: np.ndarray, probability: np.ndarray, k: int) -> float:
    top = np.argpartition(probability, -k, axis=1)[:, -k:]
    return float(np.mean(np.any(top == y[:, None], axis=1)))


def reciprocal_rank(y: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(-probability, axis=1)
    ranks = np.argmax(order == y[:, None], axis=1) + 1
    return float(np.mean(1.0 / ranks))


def span_set(labels: np.ndarray) -> set[tuple[int, int]]:
    spans: set[tuple[int, int]] = set()
    start = None
    for index, label in enumerate(labels.tolist() + [0]):
        if label == 1:
            if start is not None:
                spans.add((start, index))
            start = index
        elif label != 2 and start is not None:
            spans.add((start, index)); start = None
        elif label == 2 and start is None:
            start = index
    return spans


def span_metrics(y: np.ndarray, prediction: np.ndarray, sequence_id: np.ndarray, position: np.ndarray) -> dict[str, float]:
    true_spans: set[tuple[int, int, int]] = set(); pred_spans: set[tuple[int, int, int]] = set()
    boundary_errors = 0; boundaries = 0
    for seq in np.unique(sequence_id):
        idx = np.flatnonzero(sequence_id == seq)
        idx = idx[np.argsort(position[idx])]
        truth = span_set(y[idx]); pred = span_set(prediction[idx])
        true_spans.update((int(seq), a, b) for a, b in truth); pred_spans.update((int(seq), a, b) for a, b in pred)
        true_boundaries = {(a, "S") for a, _ in truth} | {(b, "E") for _, b in truth}
        pred_boundaries = {(a, "S") for a, _ in pred} | {(b, "E") for _, b in pred}
        boundary_errors += len(true_boundaries ^ pred_boundaries); boundaries += max(len(true_boundaries), 1)
    match = len(true_spans & pred_spans)
    precision = match / max(len(pred_spans), 1); recall = match / max(len(true_spans), 1)
    return {
        "span_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "exact_claim_recall": recall,
        "boundary_error": min(boundary_errors / max(boundaries, 1), 1.0),
    }


def metrics(candidate_id: str, y: np.ndarray, output: np.ndarray, evidence: Any, baseline: bool = False) -> dict[str, float]:
    if candidate_id == "CAND-S-007":
        prediction = np.clip(output.reshape(-1), 0, 1)
        bad = evidence["bad_answer"].astype(bool)
        result = {
            "spearman_rho": spearman(y, prediction),
            "mae": float(np.mean(np.abs(y - prediction))),
            "bad_answer_auprc": average_precision(bad, 1.0 - prediction),
        }
        result["composite"] = ((result["spearman_rho"] + 1) / 2 + (1 - result["mae"]) + result["bad_answer_auprc"]) / 3
        return result
    if baseline:
        prediction = output.astype(int)
        classes = int(np.max(y)) + 1
        probability = hard_probability(prediction, classes)
    else:
        probability = output
        prediction = probability.argmax(axis=1)
        classes = probability.shape[1]
    if candidate_id in {"CAND-S-001", "CAND-S-027"}:
        result = {
            "accuracy": float(np.mean(prediction == y)),
            "macro_f1": macro_f1(y, prediction, classes),
            "mrr": reciprocal_rank(y, probability),
            "top2_recall": topk_recall(y, probability, min(2, classes)),
            "ece": ece(y, probability),
        }
        if candidate_id == "CAND-S-001":
            result["composite"] = (result["accuracy"] + result["mrr"] + (1 - result["ece"])) / 3
        else:
            result["composite"] = (result["macro_f1"] + result["top2_recall"] + (1 - result["ece"])) / 3
        return result
    if candidate_id == "CAND-S-002":
        score = probability[:, 1]
        insufficient = y == 0
        result = {
            "auroc": binary_auroc(y, score),
            "brier": brier_binary(y, score),
            "insufficient_recall": float(np.mean(prediction[insufficient] == 0)),
        }
        result["composite"] = (result["auroc"] + (1 - result["brier"]) + result["insufficient_recall"]) / 3
        return result
    if candidate_id == "CAND-S-003":
        abstain = y == 2
        result = {
            "macro_f1": macro_f1(y, prediction, 3),
            "abstain_recall": float(np.mean(prediction[abstain] == 2)),
            "decision_regret": float(np.mean(prediction != y)),
        }
        result["composite"] = (result["macro_f1"] + result["abstain_recall"] + (1 - result["decision_regret"])) / 3
        return result
    if candidate_id == "CAND-S-004":
        score = probability[:, 1]; unsafe = y == 1
        result = {
            "auroc": binary_auroc(y, score),
            "unsafe_answer_fnr": float(np.mean(prediction[unsafe] == 0)),
            "ece": ece(y, probability),
        }
        result["composite"] = (result["auroc"] + (1 - result["unsafe_answer_fnr"]) + (1 - result["ece"])) / 3
        return result
    if candidate_id == "CAND-S-005":
        result = span_metrics(y, prediction, evidence["sequence_id"], evidence["token_position"])
        result["composite"] = (result["span_f1"] + result["exact_claim_recall"] + (1 - result["boundary_error"])) / 3
        return result
    if candidate_id == "CAND-S-006":
        high = y >= 4
        result = {
            "macro_f1": macro_f1(y, prediction, 7),
            "high_risk_recall": float(np.mean(prediction[high] >= 4)),
            "ece": ece(y, probability),
        }
        result["composite"] = (result["macro_f1"] + result["high_risk_recall"] + (1 - result["ece"])) / 3
        return result
    if candidate_id == "CAND-S-028":
        ood = y != 0; score = 1.0 - probability[:, 0]
        positives = np.sort(score[ood]); threshold = positives[max(0, int(np.floor(.05 * len(positives))))]
        result = {
            "auroc": binary_auroc(ood, score),
            "fpr95": float(np.mean(score[~ood] >= threshold)),
            "in_domain_false_reject_rate": float(np.mean(prediction[~ood] != 0)),
            "reason_macro_f1": macro_f1(y, prediction, 4),
        }
        result["composite"] = (result["auroc"] + (1 - result["fpr95"]) + (1 - result["in_domain_false_reject_rate"]) + result["reason_macro_f1"]) / 4
        return result
    raise KeyError(candidate_id)


def rebuild_manifest(output: Path) -> str:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}
    write_json(output / "artifact_manifest.json", receipt)
    return receipt["content_root_sha256"]


def evaluate_one(root: Path, artifact_root: Path, candidate_id: str) -> dict[str, Any]:
    output = artifact_root / candidate_id
    evidence = np.load(output / "three_seed_test_predictions.npz", allow_pickle=False)
    y = evidence["y"]
    baseline = metrics(candidate_id, y, evidence["baseline_prediction"], evidence, baseline=True)
    seeds = []
    for seed in SEEDS:
        seeds.append({"seed": seed, **metrics(candidate_id, y, evidence[f"seed_{seed}"], evidence)})
    composites = np.asarray([item["composite"] for item in seeds], dtype=np.float64)
    evaluation = json.loads((output / "eval_grouped.json").read_text(encoding="utf-8"))
    best_seed = int(evaluation["best_seed"])
    best_report = next(item for item in evaluation["seed_reports"] if int(item["seed"]) == best_seed)
    quant = evaluation["quantized_test"]
    if candidate_id == "CAND-S-007":
        quant_delta = float(quant["mae"] - best_report["test"]["mae"])
        quant_pass = quant_delta <= .02
    else:
        quant_delta = float(best_report["test"]["balanced_accuracy"] - quant["balanced_accuracy"])
        quant_pass = quant_delta <= .02
    aggregate_pass = float(np.mean(composites)) > float(baseline["composite"]) + 1e-6
    status = "PASS_CONTRACT_BASELINE_BOARD_PENDING" if aggregate_pass and quant_pass else "FAIL_CONTRACT_BASELINE"
    receipt = {
        "schema": "cimc.forge200.support-exact-contract-evaluation.v3",
        "status": status,
        "candidate_id": candidate_id,
        "baseline": baseline,
        "seed_reports": seeds,
        "three_seed_mean_composite": float(np.mean(composites)),
        "three_seed_variance_composite": float(np.var(composites)),
        "three_seed_worst_composite": float(np.min(composites)),
        "aggregate_mean_beats_preregistered_baseline": aggregate_pass,
        "individual_seed_baseline_results_reported_not_release_gate": [bool(value > baseline["composite"] + 1e-6) for value in composites],
        "quantized_best_seed_metric_delta": quant_delta,
        "quantization_pass": quant_pass,
        "dataset_sha256": sha256_file(root / "data" / "staged_support_exact_v3" / f"{candidate_id}.npz"),
        "prediction_evidence_sha256": sha256_file(output / "three_seed_test_predictions.npz"),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_evaluation.v3.json", receipt)
    promotion_path = output / "promotion_receipt.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion.update({
        "status": "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status.startswith("PASS") else "HOST_GPU_REJECTED_CONTRACT_BASELINE",
        "exact_contract_baseline_pending": not status.startswith("PASS"),
        "contract_evaluation_sha256": sha256_file(output / "contract_exact_evaluation.v3.json"),
        "board_accepted": False,
        "countable_model": False,
        "authority": 0,
    })
    write_json(promotion_path, promotion)
    with (output / "model_card.md").open("a", encoding="utf-8") as handle:
        handle.write(f"\n- Exact contract evaluation: `{status}`; three-seed mean `{float(np.mean(composites)):.6f}` vs baseline `{float(baseline['composite']):.6f}`.\n")
    root_hash = rebuild_manifest(output)
    receipt["artifact_content_root_sha256"] = root_hash
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, artifact_root = args.root.resolve(), args.artifact_root.resolve()
    records = [evaluate_one(root, artifact_root, candidate_id) for candidate_id in CANDIDATES]
    result = {
        "schema": "cimc.forge200.support-exact-closure.v3",
        "status": "PASS" if all(item["status"].startswith("PASS") for item in records) else "PARTIAL",
        "candidate_count": len(records),
        "contract_pass": sum(item["status"].startswith("PASS") for item in records),
        "contract_fail": sum(not item["status"].startswith("PASS") for item in records),
        "authority_nonzero": 0,
        "board_accepted": 0,
        "countable_models": 0,
        "records": records,
        "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest(),
    }
    write_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("status", "candidate_count", "contract_pass", "contract_fail", "content_root_sha256")}, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
