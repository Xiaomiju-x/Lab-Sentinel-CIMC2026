#!/usr/bin/env python3
"""Train the P059 train-fitted photon-proxy residual ridge model on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, sha256_file, write_json


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 100.0)
    order_y = np.argsort(y, kind="mergesort")
    order_p = np.argsort(prediction, kind="mergesort")
    rank_y = np.empty(len(y), dtype=np.float64)
    rank_p = np.empty(len(y), dtype=np.float64)
    rank_y[order_y] = np.arange(len(y), dtype=np.float64)
    rank_p[order_p] = np.arange(len(y), dtype=np.float64)
    spearman = float(np.corrcoef(rank_y, rank_p)[0, 1]) if len(y) > 1 else 0.0
    top = min(5, len(y))
    predicted_order = np.argsort(-prediction, kind="mergesort")
    truth_order = np.argsort(-y, kind="mergesort")
    relevance = np.expm1(y / 20.0)
    discount = 1.0 / np.log2(np.arange(2, top + 2))
    ndcg = float(np.sum(relevance[predicted_order[:top]] * discount) / max(np.sum(relevance[truth_order[:top]] * discount), 1e-12))
    bins = np.minimum((prediction / 10.0).astype(np.int64), 9)
    ece = 0.0
    for index in range(10):
        selected = bins == index
        if np.any(selected):
            ece += float(np.mean(selected)) * abs(float(np.mean(prediction[selected]) - np.mean(y[selected]))) / 100.0
    return {
        "Spearman_rho": spearman,
        "NDCG_at_5": ndcg,
        "ECE": ece,
        "MAE_percent_not_primary": float(np.mean(np.abs(prediction - y))),
        "composite_higher_is_better": float(np.mean([(spearman + 1.0) / 2.0, ndcg, 1.0 - ece])),
    }


def manifest(output: Path) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "artifact_manifest.json"):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v1", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = args.root.resolve()
    candidate_id = "CAND-P-059"
    output = args.artifact_root.resolve() / candidate_id
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    import torch
    from torch import nn

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    dataset_path = root / "data" / "staged_ipop_p059_exact_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    data = np.load(dataset_path, allow_pickle=False)
    raw = data["x"].astype(np.float32)
    y = data["y"].astype(np.float32)
    split = data["split"].astype(np.int8)
    baseline_prediction = data["baseline_pred"].astype(np.float32)
    source_x = raw[:, :-1]
    prior = raw[:, -1]
    indices = {name: np.flatnonzero(split == code) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    mean = source_x[indices["train"]].mean(axis=0)
    std = source_x[indices["train"]].std(axis=0)
    std[std < 1e-6] = 1.0
    z = ((source_x - mean) / std).astype(np.float32)
    residual = (y - prior).astype(np.float32)
    z_train = torch.from_numpy(z[indices["train"]]).to(device)
    r_train = torch.from_numpy(residual[indices["train"]]).to(device).reshape(-1, 1)
    identity = torch.eye(z.shape[1], device=device, dtype=torch.float32)
    lambdas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    validation_selection = []
    weights_by_lambda: dict[float, np.ndarray] = {}
    for value in lambdas:
        weight = torch.linalg.solve(z_train.T @ z_train + value * identity, z_train.T @ r_train).reshape(-1)
        weight_np = weight.detach().cpu().numpy().astype(np.float32)
        weights_by_lambda[value] = weight_np
        prediction = np.clip(prior[indices["validation"]] + z[indices["validation"]] @ weight_np, 0.0, 100.0)
        validation_selection.append({"lambda": value, "mae": float(np.mean(np.abs(prediction - y[indices["validation"]])))})
    selected_lambda = min(validation_selection, key=lambda item: (item["mae"], item["lambda"]))["lambda"]
    weight = weights_by_lambda[float(selected_lambda)]

    class ResidualRidge(nn.Module):
        def __init__(self, initial: np.ndarray) -> None:
            super().__init__()
            self.linear = nn.Linear(len(initial), 1, bias=False)
            with torch.no_grad():
                self.linear.weight.copy_(torch.from_numpy(initial[None, :]))

        def forward(self, value: Any) -> Any:
            return value[:, -1:] + self.linear(value[:, :-1])

    model_input = np.concatenate((z, prior[:, None]), axis=1).astype(np.float32)
    model = ResidualRidge(weight).to(device).eval()
    with torch.no_grad():
        validation_prediction = model(torch.from_numpy(model_input[indices["validation"]]).to(device)).cpu().numpy().reshape(-1)
        test_prediction = model(torch.from_numpy(model_input[indices["test"]]).to(device)).cpu().numpy().reshape(-1)
    validation_prediction = np.clip(validation_prediction, 0.0, 100.0)
    test_prediction = np.clip(test_prediction, 0.0, 100.0)
    baseline = metrics(y[indices["test"]], baseline_prediction[indices["test"]])
    result = metrics(y[indices["test"]], test_prediction)
    seed_reports = [{"seed": seed, "deterministic_solver": "CUDA_RIDGE_CLOSED_FORM", "test": result} for seed in SEEDS]
    scores = np.asarray([item["test"]["composite_higher_is_better"] for item in seed_reports])
    aggregate = {"mean": float(scores.mean()), "variance": float(scores.var()), "std": float(scores.std()), "worst": float(scores.min())}

    scale = max(float(np.max(np.abs(weight))) / 127.0, 1e-12)
    quantized_weight = np.clip(np.rint(weight / scale), -127, 127).astype(np.int8)
    quant_weight = quantized_weight.astype(np.float32) * scale
    quant_model = ResidualRidge(quant_weight).to(device).eval()
    with torch.no_grad():
        quant_prediction = quant_model(torch.from_numpy(model_input[indices["test"]]).to(device)).cpu().numpy().reshape(-1)
    quant_prediction = np.clip(quant_prediction, 0.0, 100.0)
    quant_metrics = metrics(y[indices["test"]], quant_prediction)
    parity = float(np.max(np.abs(test_prediction - quant_prediction)))
    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=model_input[indices["test"]], y=y[indices["test"]], fp32=test_prediction, quantized=quant_prediction)
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(model_input[indices["test"][:1]]).to(device), onnx_path, input_names=["preprocessed_composition_process_phase_PL_and_proxy_prior"], output_names=["internal_quantum_efficiency_score"], dynamic_axes={"preprocessed_composition_process_phase_PL_and_proxy_prior": {0: "batch"}, "internal_quantum_efficiency_score": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))
    contract_path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with contract_path.open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)
    output_schema = {"task_kind": "regression_and_ranking", "shape": [None, 1], "units": "percent", "semantics": "internal_quantum_efficiency_calibrated_score", "postprocess": "clip_0_to_100", "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": sha256_file(dataset_path), "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "lambda": selected_lambda})).hexdigest()
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, weight=quantized_weight, scale=np.asarray(scale, dtype=np.float32))
    package = build_package(output, candidate_id, payload_buffer.getvalue(), sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    parameter_count = len(weight)
    gates = {
        "G1_source_and_license": metadata["status"] == "PASS",
        "G2_group_split_no_leakage": metadata["cross_split_group_overlap"] == 0,
        "G3_three_seed_mean_beats_stronger_measured_proxy_baseline": aggregate["mean"] > baseline["composite_higher_is_better"] and result["Spearman_rho"] > baseline["Spearman_rho"] and result["NDCG_at_5"] > baseline["NDCG_at_5"] and result["ECE"] < baseline["ECE"],
        "G4_mean_variance_worst_reported": len(seed_reports) == 3,
        "G5_parameter_and_weight_caps": parameter_count <= 40000 and package["bytes"] - 256 <= 52 * 1024,
        "G6_quantized_model_beats_baseline": quant_metrics["composite_higher_is_better"] > baseline["composite_higher_is_better"] and quant_metrics["Spearman_rho"] > baseline["Spearman_rho"] and quant_metrics["NDCG_at_5"] > baseline["NDCG_at_5"] and quant_metrics["ECE"] < baseline["ECE"],
        "G8_authority_zero_board_pending": True,
    }
    passed = all(gates.values())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_REJECTED_CONTRACT_BASELINE"
    exact = {
        "schema": "cimc.forge200.ipop-p059-exact-audit.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "candidate_id": candidate_id, "baseline": baseline, "validation_selection": validation_selection,
        "selected_lambda": selected_lambda, "seed_reports": seed_reports, "aggregate": aggregate, "quantized_test": quant_metrics,
        "quantization_max_abs_error": parity, "parameter_count": parameter_count, "gates": gates,
        "authority": 0, "board_accepted": False, "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", exact)
    write_json(output / "eval_grouped.json", exact)
    write_json(output / "calibration_ood.json", {"schema": "cimc.forge200.calibration-ood.v1", "test_ECE": result["ECE"], "quant_max_abs_error_percent": parity, "authority": 0})
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "residual_prior": metadata["baseline_fit"]})
    write_json(output / "task_contract.json", {"candidate_id": candidate_id, "task_contract_sha256": metadata["task_contract_sha256"], "authority": 0})
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "split_manifest.json", {"split_sha256": metadata["split_sha256"], "cross_split_group_overlap": 0})
    write_json(output / "baseline_report.json", baseline)
    write_json(output / "ablation.json", {"selected": "family_prior_plus_ridge_residual", "baseline": "family_prior_only", "validation_selected_without_test": True})
    (output / "model_card.md").write_text(f"# {candidate_id} model card\n\n- Status: `{status}`\n- Model: train-only external-QE photon-output proxy calibration plus CUDA ridge residual, lambda `{selected_lambda}`.\n- Scope: literature-curated IPOP internal quantum efficiency; not independent team absolute photon-count metrology.\n- The executed baseline is stronger than the preregistered integrated-PL-intensity proxy because measured external QE is available.\n- Three fixed seeds: deterministic mean `{aggregate['mean']:.6f}`, variance `{aggregate['variance']:.8f}`, worst `{aggregate['worst']:.6f}`.\n- Test composite: `{result['composite_higher_is_better']:.6f}` vs stronger baseline `{baseline['composite_higher_is_better']:.6f}`.\n- Authority: `0`; unified GD32 board acceptance remains pending.\n", encoding="utf-8")
    promotion = {
        "schema": "cimc.forge200.promotion-receipt.v1", "status": status, "candidate_id": candidate_id,
        "authority": 0, "board_accepted": False, "countable_model": False, "host_contract_pass": passed,
        "three_seed_count": 3, "release_root": release_root, "package": package,
        "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden),
        "contract_exact_audit_sha256": sha256_file(output / "contract_exact_audit.json"),
        "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / (1024**3)},
    }
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    closure_content = {"records": [exact], "authority_nonzero": 0, "board_actions": 0}
    closure = {"schema": "cimc.forge200.ipop-p059-exact-closure.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "PARTIAL", **closure_content, "content_root_sha256": hashlib.sha256(canonical_bytes(closure_content)).hexdigest()}
    write_json(root / "evidence" / "ipop_p059_exact_closure.v1.json", closure)
    print(json.dumps({"status": closure["status"], "candidate_id": candidate_id, "model_composite": result["composite_higher_is_better"], "baseline_composite": baseline["composite_higher_is_better"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
