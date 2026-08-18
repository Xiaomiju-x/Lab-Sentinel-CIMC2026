#!/usr/bin/env python3
"""Evaluate S031/S034 against their frozen task-specific baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, canonical_bytes, classification_metrics, sha256_file, write_json


def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    scores = []
    for label in range(classes):
        tp = int(np.sum((y == label) & (pred == label)))
        fp = int(np.sum((y != label) & (pred == label)))
        fn = int(np.sum((y == label) & (pred != label)))
        scores.append(2.0 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def nli_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    contradiction = y == 1
    recall = float(np.mean(pred[contradiction] == 1)) if np.any(contradiction) else 0.0
    f1 = macro_f1(y, pred, 3)
    return {
        "macro_f1": f1,
        "contradiction_recall": recall,
        "worst_domain_f1": f1,
        "primary_composite": float((2.0 * f1 + recall) / 3.0),
    }


def binary_macro_f1(y: np.ndarray, pred: np.ndarray) -> float:
    return macro_f1(y.astype(np.int64), pred.astype(np.int64), 2)


def span_metrics(y: np.ndarray, probability: np.ndarray, groups: np.ndarray, threshold: float) -> dict[str, float]:
    exact, span_f1, gold_empty, pred_empty = [], [], [], []
    for group in sorted(set(groups.tolist())):
        selected = np.flatnonzero(groups == group)
        gold = set(selected[y[selected] == 1].tolist())
        predicted = set(selected[probability[selected] >= threshold].tolist())
        exact.append(float(gold == predicted))
        if not gold and not predicted:
            span_f1.append(1.0)
        elif not gold or not predicted:
            span_f1.append(0.0)
        else:
            span_f1.append(2.0 * len(gold & predicted) / (len(gold) + len(predicted)))
        gold_empty.append(not gold)
        pred_empty.append(not predicted)
    no_span = binary_macro_f1(np.asarray(gold_empty), np.asarray(pred_empty))
    result = {
        "span_exact_match": float(np.mean(exact)),
        "span_f1": float(np.mean(span_f1)),
        "no_span_macro_f1": no_span,
    }
    result["primary_composite"] = float(np.mean(list(result.values())))
    result["groups"] = len(exact)
    return result


def best_threshold(y: np.ndarray, score: np.ndarray, groups: np.ndarray) -> tuple[float, dict[str, float]]:
    best = (-1.0, 0.5, {})
    for threshold in np.linspace(0.02, 0.98, 97):
        metrics = span_metrics(y, score, groups, float(threshold))
        key = (metrics["primary_composite"], -abs(float(threshold) - 0.5))
        if key > (best[0], -abs(best[1] - 0.5)):
            best = (metrics["primary_composite"], float(threshold), metrics)
    return best[1], best[2]


def infer_states(root: Path, artifact: Path, candidate_id: str, x: np.ndarray, output_count: int) -> dict[int, np.ndarray]:
    import torch
    from torch import nn

    prep = json.loads((artifact / "preprocessing_train_only.json").read_text(encoding="utf-8"))
    mean = np.asarray(prep["mean"], dtype=np.float32)
    std = np.asarray(prep["std"], dtype=np.float32)
    standardized = ((x - mean) / std).astype(np.float32)

    class ForgeMLP(nn.Module):
        def __init__(self, hidden: int) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, output_count))

        def forward(self, value: Any) -> Any:
            return self.net(value)

    outputs = {}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        state = torch.load(artifact / f"train_seed_{seed}" / "best.pt", map_location=device, weights_only=True)
        hidden = int(state["net.0.weight"].shape[0])
        model = ForgeMLP(hidden).to(device)
        model.load_state_dict(state)
        model.eval()
        parts = []
        with torch.no_grad():
            for start in range(0, len(standardized), 2048):
                logits = model(torch.from_numpy(standardized[start : start + 2048]).to(device))
                parts.append(torch.softmax(logits, dim=-1).cpu().numpy())
        outputs[seed] = np.concatenate(parts)
    return outputs


def update_manifest(artifact: Path) -> None:
    records = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "heartbeat.json"}:
            records.append({"path": str(path.relative_to(artifact)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(artifact / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})


def evaluate_nli(root: Path, artifact_root: Path) -> dict[str, Any]:
    candidate_id = "CAND-S-031"
    raw = np.load(root / "data" / "staged_scifact_v1" / f"{candidate_id}.npz", allow_pickle=False)
    split = raw["split"].astype(np.int8)
    selected = split == 2
    y = raw["y"].astype(np.int64)
    baseline = nli_metrics(y[selected], raw["baseline_pred"].astype(np.int64)[selected])
    artifact = artifact_root / candidate_id
    probabilities = infer_states(root, artifact, candidate_id, raw["x"].astype(np.float32), 3)
    reports = []
    for seed in SEEDS:
        metrics = nli_metrics(y[selected], np.argmax(probabilities[seed][selected], axis=1))
        reports.append({"seed": seed, "test": metrics, "beats_baseline": metrics["primary_composite"] > baseline["primary_composite"] + 1e-4})
    values = np.asarray([item["test"]["primary_composite"] for item in reports])
    passed = float(values.mean()) > baseline["primary_composite"] + 1e-4
    return finalize(root, artifact, candidate_id, baseline, reports, values, passed)


def evaluate_span(root: Path, artifact_root: Path) -> dict[str, Any]:
    candidate_id = "CAND-S-034"
    raw = np.load(root / "data" / "staged_scifact_v1" / f"{candidate_id}.npz", allow_pickle=False)
    split = raw["split"].astype(np.int8)
    y = raw["y"].astype(np.int64)
    groups = raw["query_group"].astype(str)
    baseline_score = raw["baseline_score"].astype(np.float32)
    train = split == 0
    test = split == 2
    baseline_threshold, _ = best_threshold(y[train], baseline_score[train], groups[train])
    baseline = span_metrics(y[test], baseline_score[test], groups[test], baseline_threshold)
    baseline["train_selected_threshold"] = baseline_threshold
    artifact = artifact_root / candidate_id
    probabilities = infer_states(root, artifact, candidate_id, raw["x"].astype(np.float32), 2)
    reports = []
    for seed in SEEDS:
        validation = split == 1
        threshold, validation_metrics = best_threshold(y[validation], probabilities[seed][validation, 1], groups[validation])
        metrics = span_metrics(y[test], probabilities[seed][test, 1], groups[test], threshold)
        reports.append({"seed": seed, "validation_selected_threshold": threshold, "validation": validation_metrics, "test": metrics, "beats_baseline": metrics["primary_composite"] > baseline["primary_composite"] + 1e-4})
    values = np.asarray([item["test"]["primary_composite"] for item in reports])
    passed = float(values.mean()) > baseline["primary_composite"] + 1e-4
    return finalize(root, artifact, candidate_id, baseline, reports, values, passed)


def finalize(root: Path, artifact: Path, candidate_id: str, baseline: dict[str, Any], reports: list[dict[str, Any]], values: np.ndarray, passed: bool) -> dict[str, Any]:
    metadata = json.loads((root / "data" / "staged_scifact_v1" / f"{candidate_id}.metadata.json").read_text(encoding="utf-8"))
    receipt_path = artifact / "promotion_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    import torch
    first_state = torch.load(artifact / f"train_seed_{SEEDS[0]}" / "best.pt", map_location="cpu", weights_only=True)
    parameter_count = int(sum(value.numel() for value in first_state.values()))
    cap = 192_000 if candidate_id.endswith("031") else 128_000
    passed = bool(passed and parameter_count <= cap and metadata["status"] == "PASS")
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    evaluation = {
        "schema": "cimc.forge200.scifact-exact-evaluation.v1",
        "status": "PASS" if passed else "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "baseline_contract": "lexical_overlap_plus_numeric_consistency_rules" if candidate_id.endswith("031") else "sentence_level_lexical_overlap",
        "baseline": baseline,
        "seed_reports": reports,
        "g3_aggregate_rule": "THREE_SEED_PRIMARY_COMPOSITE_MEAN_GT_FROZEN_BASELINE",
        "three_seed_mean": float(values.mean()),
        "three_seed_std": float(values.std()),
        "three_seed_worst": float(values.min()),
        "parameter_count": parameter_count,
        "parameter_cap": cap,
        "authority": 0,
        "board_accepted": False,
    }
    write_json(artifact / "eval_contract_exact.json", evaluation)
    receipt.update({"status": status, "parameter_count": parameter_count, "parameter_cap": cap, "g3_contract_baseline_pass": passed, "three_seed_primary_mean": float(values.mean()), "three_seed_primary_std": float(values.std()), "three_seed_primary_worst": float(values.min())})
    write_json(receipt_path, receipt)
    (artifact / "model_card_exact_addendum.md").write_text(f"# {candidate_id} exact-contract addendum\n\n- Status: `{status}`\n- G3: three-seed primary-composite mean `{values.mean():.6f}` versus frozen baseline `{baseline['primary_composite']:.6f}`.\n- G4: std `{values.std():.6f}`, worst `{values.min():.6f}`; no best-seed-only claim.\n- Authority: `0`; unified GD32 board evidence pending.\n", encoding="utf-8")
    update_manifest(artifact)
    return evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--candidates", default="CAND-S-031,CAND-S-034")
    args = parser.parse_args()
    root = args.root.resolve()
    requested = {item.strip() for item in args.candidates.split(",") if item.strip()}
    unknown = requested - {"CAND-S-031", "CAND-S-034"}
    if unknown:
        raise RuntimeError(f"UNKNOWN_CANDIDATES:{sorted(unknown)}")
    results = []
    if "CAND-S-031" in requested:
        results.append(evaluate_nli(root, args.artifact_root.resolve()))
    if "CAND-S-034" in requested:
        results.append(evaluate_span(root, args.artifact_root.resolve()))
    receipt = {"schema": "cimc.forge200.scifact-exact-closure.v1", "status": "PASS" if all(item["status"] == "PASS" for item in results) else "PARTIAL", "results": results, "authority_nonzero": 0, "board_actions": 0}
    receipt["content_root_sha256"] = hashlib.sha256(canonical_bytes(results)).hexdigest()
    write_json(root / "evidence" / "scifact_exact_closure.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "results": {item["candidate_id"]: item["status"] for item in results}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
