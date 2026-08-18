#!/usr/bin/env python3
"""Evaluate S015-S020 full top-50 reranking contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluate_support_exact_v3 import canonical_bytes, sha256_file, write_json


CANDIDATES = tuple(f"CAND-S-{index:03d}" for index in range(15, 21))
SEEDS = (20260801, 20260802, 20260803)
SPECIAL = {
    "CAND-S-016": "critical_evidence_recall",
    "CAND-S-017": "experimental_computed_filter_accuracy",
    "CAND-S-018": "method_match_accuracy",
    "CAND-S-019": "condition_match_accuracy",
    "CAND-S-020": "scope_match_accuracy",
}


def ranking_metrics(candidate_id: str, y: np.ndarray, score: np.ndarray, query_id: np.ndarray, special_match: np.ndarray) -> dict[str, float]:
    ndcg, mrr, pairwise, recall, special_accuracy = [], [], [], [], []
    for query in np.unique(query_id):
        selected = np.flatnonzero(query_id == query)
        order = selected[np.argsort(-score[selected], kind="mergesort")]
        relevance = y[order].astype(np.float64); relevant = int(np.sum(y[selected]))
        if relevant <= 0:
            raise RuntimeError("RERANK_QUERY_WITHOUT_RELEVANCE")
        top = relevance[:10]; discounts = 1.0 / np.log2(np.arange(2, len(top) + 2))
        ndcg.append(float(np.sum(top * discounts)) / max(float(np.sum(discounts[: min(relevant, 10)])), 1e-12))
        positions = np.flatnonzero(relevance); mrr.append(1.0 / float(positions[0] + 1)); recall.append(float(np.sum(top) / relevant))
        positive = score[selected][y[selected] == 1]; negative = score[selected][y[selected] == 0]
        pairwise.append(float(np.mean(positive[:, None] > negative[None, :]) + .5 * np.mean(positive[:, None] == negative[None, :])))
        special_accuracy.append(float(special_match[order[0]] == 1))
    result = {"ndcg_at_10": float(np.mean(ndcg)), "mrr_at_10": float(np.mean(mrr)), "pairwise_accuracy": float(np.mean(pairwise)), "recall_at_10": float(np.mean(recall)), "top1_cross_field_match_accuracy": float(np.mean(special_accuracy)), "queries": len(ndcg)}
    third = result["pairwise_accuracy"] if candidate_id == "CAND-S-015" else (result["recall_at_10"] if candidate_id == "CAND-S-016" else result["top1_cross_field_match_accuracy"])
    if candidate_id in SPECIAL:
        result[SPECIAL[candidate_id]] = third
    result["composite"] = (result["ndcg_at_10"] + result["mrr_at_10"] + third) / 3
    return result


def rebuild_manifest(output: Path) -> str:
    records = [{"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json")]
    root_hash = hashlib.sha256(canonical_bytes(records)).hexdigest(); write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": root_hash}); return root_hash


def evaluate_one(root: Path, artifact_root: Path, candidate_id: str) -> dict[str, Any]:
    output = artifact_root / candidate_id; evidence = np.load(output / "three_seed_test_predictions.npz", allow_pickle=False)
    y = evidence["y"].astype(int); query_id = evidence["query_id"].astype(np.int64); special = evidence["special_match"].astype(np.uint8)
    baseline = ranking_metrics(candidate_id, y, evidence["baseline_score"].astype(float), query_id, special)
    seeds = []
    for seed in SEEDS:
        probability = evidence[f"seed_{seed}"]
        seeds.append({"seed": seed, **ranking_metrics(candidate_id, y, probability[:, 1], query_id, special)})
    quantized = ranking_metrics(candidate_id, y, evidence["quantized_best_seed"][:, 1], query_id, special)
    composites = np.asarray([item["composite"] for item in seeds]); mean_pass = float(np.mean(composites)) > float(baseline["composite"]) + 1e-6
    grouped = json.loads((output / "eval_grouped.json").read_text(encoding="utf-8")); best_seed = int(grouped["best_seed"])
    best = next(item for item in seeds if item["seed"] == best_seed); quant_delta = float(best["composite"] - quantized["composite"]); quant_pass = quant_delta <= .02
    status = "PASS_CONTRACT_BASELINE_BOARD_PENDING" if mean_pass and quant_pass else "FAIL_CONTRACT_BASELINE"
    receipt = {"schema": "cimc.forge200.reranker-exact-contract-evaluation.v1", "status": status, "candidate_id": candidate_id, "baseline": baseline, "seed_reports": seeds, "quantized_best_seed": quantized, "three_seed_mean_composite": float(np.mean(composites)), "three_seed_variance_composite": float(np.var(composites)), "three_seed_worst_composite": float(np.min(composites)), "aggregate_mean_beats_preregistered_baseline": mean_pass, "individual_seed_baseline_results_reported_not_release_gate": [bool(value > baseline["composite"] + 1e-6) for value in composites], "quantized_best_seed_metric_delta": quant_delta, "quantization_pass": quant_pass, "dataset_sha256": sha256_file(root / "data" / "staged_reranker_exact_v1" / f"{candidate_id}.npz"), "prediction_evidence_sha256": sha256_file(output / "three_seed_test_predictions.npz"), "authority": 0, "board_accepted": False, "countable_model": False}
    write_json(output / "contract_exact_evaluation.v1.json", receipt); promotion_path = output / "promotion_receipt.json"; promotion = json.loads(promotion_path.read_text(encoding="utf-8")); promotion.update({"status": "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status.startswith("PASS") else "HOST_GPU_REJECTED_CONTRACT_BASELINE", "exact_contract_baseline_pending": not status.startswith("PASS"), "contract_evaluation_sha256": sha256_file(output / "contract_exact_evaluation.v1.json"), "authority": 0, "board_accepted": False, "countable_model": False}); write_json(promotion_path, promotion)
    with (output / "model_card.md").open("a", encoding="utf-8") as handle: handle.write(f"\n- Exact contract evaluation: `{status}`; three-seed mean `{float(np.mean(composites)):.6f}` vs baseline `{float(baseline['composite']):.6f}`.\n")
    receipt["artifact_content_root_sha256"] = rebuild_manifest(output); return receipt


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); parser.add_argument("--artifact-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--candidate-csv", default=",".join(CANDIDATES)); args = parser.parse_args(); candidate_ids = tuple(item.strip() for item in args.candidate_csv.split(",") if item.strip()); records = [evaluate_one(args.root.resolve(), args.artifact_root.resolve(), candidate_id) for candidate_id in candidate_ids]
    report = {"schema": "cimc.forge200.reranker-exact-closure.v1", "status": "PASS" if all(item["status"].startswith("PASS") for item in records) else "PARTIAL", "candidate_count": len(records), "contract_pass": sum(item["status"].startswith("PASS") for item in records), "contract_fail": sum(not item["status"].startswith("PASS") for item in records), "authority_nonzero": 0, "board_accepted": 0, "countable_models": 0, "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}; write_json(args.output, report); print(json.dumps({key: report[key] for key in ("status", "candidate_count", "contract_pass", "contract_fail", "content_root_sha256")}, sort_keys=True)); return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__": raise SystemExit(main())
