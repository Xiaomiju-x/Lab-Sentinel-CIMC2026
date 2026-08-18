#!/usr/bin/env python3
"""Audit P067/P068 against their preregistered train-fitted baselines."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SEEDS = [20260801, 20260802, 20260803]


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


def auc(y: np.ndarray, score: np.ndarray) -> float:
    y = y.astype(np.int64)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and score[order[end]] == score[order[position]]:
            end += 1
        ranks[order[position:end]] = (position + 1 + end) / 2.0
        position = end
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1))


def ece(y: np.ndarray, probability: np.ndarray) -> float:
    result = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        mask = (probability >= lower) & (probability < lower + 0.1 if lower < 0.9 else probability <= 1.0)
        if np.any(mask):
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return result


def classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(np.int64)
    f1s = []
    for label in (0, 1):
        tp = int(np.sum((prediction == label) & (y == label)))
        fp = int(np.sum((prediction == label) & (y != label)))
        fn = int(np.sum((prediction != label) & (y == label)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1s.append(2 * precision * recall / max(precision + recall, 1e-12))
    metrics = {"macro_f1": float(np.mean(f1s)), "auroc": auc(y, probability), "ece_10bin": ece(y, probability)}
    metrics["composite_higher_is_better"] = (metrics["macro_f1"] + metrics["auroc"] + 1.0 - metrics["ece_10bin"]) / 3.0
    return metrics


def regression_metrics(y: np.ndarray, prediction: np.ndarray, half_width: float) -> dict[str, float]:
    error = prediction.reshape(-1) - y.reshape(-1)
    coverage = float(np.mean(np.abs(error) <= half_width))
    mae = float(np.mean(np.abs(error)))
    return {
        "mae_eV": mae,
        "rmse_eV": float(np.sqrt(np.mean(error**2))),
        "calibration_interval_half_width_eV": half_width,
        "calibration_interval_coverage": coverage,
        "coverage_abs_error_from_0p90": abs(coverage - 0.90),
        "composite_higher_is_better": 1.0 / (1.0 + mae) - 0.25 * abs(coverage - 0.90),
    }


def load_contract(root: Path, candidate_id: str) -> dict[str, str]:
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["candidate_id"] == candidate_id:
                return row
    raise RuntimeError(f"missing contract: {candidate_id}")


def rebuild_manifest(output: Path) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})


def run_candidate(root: Path, artifact_root: Path, candidate_id: str, torch: Any) -> dict[str, Any]:
    dataset_path = root / "data" / "staged_matbench_experimental_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    data = np.load(dataset_path, allow_pickle=False)
    x_raw, y, split = data["x"].astype(np.float32), data["y"], data["split"].astype(np.int8)
    baseline_pred, baseline_score = data["baseline_pred"], data["baseline_score"]
    prep = json.loads((artifact_root / candidate_id / "preprocessing_train_only.json").read_text(encoding="utf-8"))
    median = np.asarray(prep["median"], dtype=np.float32)
    mean = np.asarray(prep["mean"], dtype=np.float32)
    std = np.asarray(prep["std"], dtype=np.float32)
    x = (np.where(np.isfinite(x_raw), x_raw, median) - mean) / std
    indices = {name: np.flatnonzero(split == code) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    hidden = 96
    output_count = 1 if candidate_id == "CAND-P-067" else 2

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, output_count))

        def forward(self, value: Any) -> Any:
            return self.net(value)

    def predict(state: dict[str, Any], take: np.ndarray) -> np.ndarray:
        model = Model()
        model.load_state_dict(state)
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(x[take])).numpy()
        if output_count == 1:
            return logits.reshape(-1)
        logits -= logits.max(axis=1, keepdims=True)
        probability = np.exp(logits)
        probability /= probability.sum(axis=1, keepdims=True)
        return probability[:, 1]

    seed_results = []
    for seed in SEEDS:
        state = torch.load(artifact_root / candidate_id / f"train_seed_{seed}" / "best.pt", map_location="cpu", weights_only=True)
        validation_prediction = predict(state, indices["validation"])
        test_prediction = predict(state, indices["test"])
        if candidate_id == "CAND-P-067":
            half_width = float(np.quantile(np.abs(validation_prediction - y[indices["validation"]]), 0.90))
            metrics = regression_metrics(y[indices["test"]], test_prediction, half_width)
        else:
            metrics = classification_metrics(y[indices["test"]], test_prediction)
        seed_results.append({"seed": seed, "test": metrics})

    if candidate_id == "CAND-P-067":
        baseline_half_width = float(np.quantile(np.abs(baseline_pred[indices["validation"]] - y[indices["validation"]]), 0.90))
        baseline = regression_metrics(y[indices["test"]], baseline_pred[indices["test"]], baseline_half_width)
    else:
        baseline = classification_metrics(y[indices["test"]], baseline_score[indices["test"]])

    output = artifact_root / candidate_id
    package = (output / "w8_or_w8a8.bin").read_bytes()[256:]
    quant_archive = np.load(io.BytesIO(package), allow_pickle=False)
    quant_state = {}
    for key in quant_archive.files:
        if key.startswith("scale::"):
            continue
        scale = float(quant_archive[f"scale::{key}"])
        quant_state[key] = torch.from_numpy(quant_archive[key].astype(np.float32) * scale)
    quant_prediction = predict(quant_state, indices["test"])
    if candidate_id == "CAND-P-067":
        best_seed = json.loads((output / "promotion_receipt.json").read_text(encoding="utf-8"))["best_seed"]
        best_state = torch.load(output / f"train_seed_{best_seed}" / "best.pt", map_location="cpu", weights_only=True)
        validation_prediction = predict(best_state, indices["validation"])
        half_width = float(np.quantile(np.abs(validation_prediction - y[indices["validation"]]), 0.90))
        quant_metrics = regression_metrics(y[indices["test"]], quant_prediction, half_width)
    else:
        quant_metrics = classification_metrics(y[indices["test"]], quant_prediction)

    scores = np.asarray([item["test"]["composite_higher_is_better"] for item in seed_results], dtype=np.float64)
    aggregate = {"mean": float(scores.mean()), "variance": float(scores.var()), "std": float(scores.std()), "worst": float(scores.min())}
    baseline_score_value = baseline["composite_higher_is_better"]
    metric_gate = aggregate["mean"] > baseline_score_value
    if candidate_id == "CAND-P-067":
        metric_gate = metric_gate and float(np.mean([item["test"]["mae_eV"] for item in seed_results])) < baseline["mae_eV"]
    else:
        metric_gate = metric_gate and float(np.mean([item["test"]["macro_f1"] for item in seed_results])) > baseline["macro_f1"]
    quant_gate = quant_metrics["composite_higher_is_better"] > baseline_score_value
    contract = load_contract(root, candidate_id)
    package_bytes = (output / "w8_or_w8a8.bin").stat().st_size - 256
    parameter_count = (x.shape[1] * hidden + hidden) + (hidden * output_count + output_count)
    cap = 52000 if candidate_id == "CAND-P-067" else 44000
    size_cap = 64 * 1024 if candidate_id == "CAND-P-067" else 56 * 1024
    gates = {
        "G1_source_and_license": metadata["status"] == "PASS" and metadata["truth_class"] == "OPEN_EXPERIMENT",
        "G2_group_split_no_leakage": metadata["cross_split_group_overlap"] == 0,
        "G3_three_seed_mean_beats_preregistered_baseline": metric_gate,
        "G4_mean_variance_worst_reported": len(seed_results) == 3,
        "G5_parameter_and_weight_caps": parameter_count <= cap and package_bytes <= size_cap,
        "G6_quantized_model_beats_baseline": quant_gate,
        "G8_authority_zero_board_pending": True,
    }
    passed = all(gates.values())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_REJECTED_CONTRACT_OR_BASELINE"
    receipt = {
        "schema": "cimc.forge200.matbench-experimental-exact-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
        "candidate_id": candidate_id, "objective_id": contract["objective_id"],
        "truth_class": metadata["truth_class"], "input_contract_state": metadata["input_contract_state"],
        "baseline_contract": contract["baseline"], "primary_metric_contract": contract["primary_metric"],
        "baseline": baseline, "seed_reports": seed_results, "aggregate": aggregate,
        "quantized_test": quant_metrics, "parameter_count": parameter_count,
        "parameter_cap": cap, "quantized_payload_bytes": package_bytes, "quantized_payload_cap_bytes": size_cap,
        "gates": gates, "authority": 0, "board_accepted": False, "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", receipt)
    promotion_path = output / "promotion_receipt.json"
    promotion = json.loads(promotion_path.read_text(encoding="utf-8"))
    promotion["status"] = status
    promotion["contract_exact_audit_sha256"] = sha256_file(output / "contract_exact_audit.json")
    promotion["host_contract_pass"] = passed
    write_json(promotion_path, promotion)
    card_path = output / "model_card.md"
    original = card_path.read_text(encoding="utf-8").split("\n## Exact contract audit", 1)[0].rstrip()
    card_path.write_text(original + f"\n\n## Exact contract audit\n\n- Status: `{status}`\n- Three-seed aggregate composite: mean `{aggregate['mean']:.6f}`, variance `{aggregate['variance']:.8f}`, worst `{aggregate['worst']:.6f}`.\n- Preregistered baseline composite: `{baseline_score_value:.6f}`.\n- Authority remains `0`; board acceptance remains pending.\n", encoding="utf-8")
    rebuild_manifest(output)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    args = parser.parse_args()
    root, artifact_root = args.root.resolve(), args.artifact_root.resolve()
    import torch

    records = [run_candidate(root, artifact_root, candidate_id, torch) for candidate_id in ("CAND-P-067", "CAND-P-068")]
    content = {"records": records, "authority_nonzero": 0, "board_actions": 0}
    receipt = {
        "schema": "cimc.forge200.matbench-experimental-closure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(item["status"].endswith("PASS_BOARD_PENDING") for item in records) else "PARTIAL",
        **content, "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }
    write_json(root / "evidence" / "matbench_experimental_exact_closure.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "candidates": {item["candidate_id"]: item["status"] for item in records}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
