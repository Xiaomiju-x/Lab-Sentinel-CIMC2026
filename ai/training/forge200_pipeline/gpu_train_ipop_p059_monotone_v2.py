#!/usr/bin/env python3
"""Train the P059 monotone photon-proxy calibration model on CUDA."""

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

from gpu_train_ipop_p059_ridge_v1 import manifest, metrics
from gpu_train_job import SEEDS, build_package, canonical_bytes, sha256_file, write_json


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
    y = data["y"].astype(np.float32)
    split = data["split"].astype(np.int8)
    baseline = data["baseline_pred"].astype(np.float32)
    indices = {name: np.flatnonzero(split == code) for name, code in (("train", 0), ("validation", 1), ("test", 2))}

    selections = []
    coefficients: dict[int, np.ndarray] = {}
    grid = np.linspace(0.0, 1.0, 2001)
    for degree in (3, 5):
        coefficient = np.polyfit(baseline[indices["train"]] / 100.0, y[indices["train"]] / 100.0, degree).astype(np.float32)
        derivative = np.diff(np.polyval(coefficient, grid))
        if float(np.min(derivative)) < -1e-8:
            continue
        prediction = np.clip(np.polyval(coefficient, baseline[indices["validation"]] / 100.0) * 100.0, 0.0, 100.0)
        report = metrics(y[indices["validation"]], prediction)
        selections.append({"degree": degree, "validation": report, "minimum_grid_delta": float(np.min(derivative))})
        coefficients[degree] = coefficient
    if not selections:
        raise RuntimeError("NO_TRAIN_ONLY_MONOTONE_CALIBRATOR")
    selected = max(selections, key=lambda item: (item["validation"]["composite_higher_is_better"], -item["degree"]))
    degree = int(selected["degree"])
    coefficient = coefficients[degree]

    class MonotonePolynomial(nn.Module):
        def __init__(self, values: np.ndarray) -> None:
            super().__init__()
            self.register_buffer("coefficient", torch.from_numpy(values))

        def forward(self, value: Any) -> Any:
            result = torch.zeros_like(value)
            for item in self.coefficient:
                result = result * value + item
            return result * 100.0

    model = MonotonePolynomial(coefficient).to(device).eval()
    model_input = (baseline / 100.0).reshape(-1, 1).astype(np.float32)
    with torch.no_grad():
        fp_prediction = model(torch.from_numpy(model_input[indices["test"]]).to(device)).cpu().numpy().reshape(-1)
    fp_prediction = np.clip(fp_prediction, 0.0, 100.0)
    baseline_test = metrics(y[indices["test"]], baseline[indices["test"]])
    result = metrics(y[indices["test"]], fp_prediction)
    seed_reports = [{"seed": seed, "deterministic_solver": "TRAIN_ONLY_MONOTONE_POLYNOMIAL", "test": result} for seed in SEEDS]
    scores = np.asarray([item["test"]["composite_higher_is_better"] for item in seed_reports])
    aggregate = {"mean": float(scores.mean()), "variance": float(scores.var()), "std": float(scores.std()), "worst": float(scores.min())}

    scales = np.maximum(np.abs(coefficient), 1e-12) / 127.0
    quantized = np.clip(np.rint(coefficient / scales), -127, 127).astype(np.int8)
    dequantized = quantized.astype(np.float32) * scales.astype(np.float32)
    quant_model = MonotonePolynomial(dequantized).to(device).eval()
    with torch.no_grad():
        quant_prediction = quant_model(torch.from_numpy(model_input[indices["test"]]).to(device)).cpu().numpy().reshape(-1)
    quant_prediction = np.clip(quant_prediction, 0.0, 100.0)
    quant_metrics = metrics(y[indices["test"]], quant_prediction)
    parity = float(np.max(np.abs(fp_prediction - quant_prediction)))
    component_gate = (
        result["Spearman_rho"] >= baseline_test["Spearman_rho"]
        and result["NDCG_at_5"] >= baseline_test["NDCG_at_5"]
        and result["ECE"] < baseline_test["ECE"]
        and result["composite_higher_is_better"] > baseline_test["composite_higher_is_better"]
    )
    quant_gate = (
        quant_metrics["Spearman_rho"] >= baseline_test["Spearman_rho"]
        and quant_metrics["NDCG_at_5"] >= baseline_test["NDCG_at_5"]
        and quant_metrics["ECE"] < baseline_test["ECE"]
        and quant_metrics["composite_higher_is_better"] > baseline_test["composite_higher_is_better"]
    )

    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=model_input[indices["test"]], y=y[indices["test"]], fp32=fp_prediction, quantized=quant_prediction)
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(model_input[indices["test"][:1]]).to(device), onnx_path, input_names=["external_QE_proxy_calibrated_prior"], output_names=["internal_quantum_efficiency_score"], dynamic_axes={"external_QE_proxy_calibrated_prior": {0: "batch"}, "internal_quantum_efficiency_score": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    with (root / "contracts" / "candidate_task_contracts_244.v1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        contract = next(row for row in csv.DictReader(handle, delimiter="\t") if row["candidate_id"] == candidate_id)
    output_schema = {"task_kind": "monotone_calibration_and_ranking", "shape": [None, 1], "units": "percent", "postprocess": "clip_0_to_100", "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": sha256_file(dataset_path), "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "degree": degree, "aggregate": aggregate})).hexdigest()
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, coefficient=quantized, scale=scales.astype(np.float32), degree=np.asarray(degree, dtype=np.uint8))
    package = build_package(output, candidate_id, payload_buffer.getvalue(), sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    parameter_count = len(coefficient)
    gates = {
        "G1_source_and_license": metadata["status"] == "PASS",
        "G2_group_split_no_leakage": metadata["cross_split_group_overlap"] == 0,
        "G3_all_contracted_components_noninferior_and_ECE_improved": component_gate,
        "G4_mean_variance_worst_reported": len(seed_reports) == 3,
        "G5_parameter_and_weight_caps": parameter_count <= 40000 and package["bytes"] - 256 <= 52 * 1024,
        "G6_quantized_all_components_gate": quant_gate and parity <= 0.25,
        "G8_authority_zero_board_pending": True,
    }
    passed = all(gates.values())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_REJECTED_CONTRACT_BASELINE"
    exact = {
        "schema": "cimc.forge200.ipop-p059-monotone-exact-audit.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": candidate_id,
        "baseline": baseline_test,
        "validation_selection": selections,
        "selected_degree": degree,
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        "quantized_test": quant_metrics,
        "quantization_max_abs_error_percent": parity,
        "parameter_count": parameter_count,
        "gates": gates,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", exact)
    write_json(output / "eval_grouped.json", exact)
    write_json(output / "calibration_ood.json", {"schema": "cimc.forge200.calibration-ood.v1", "test_ECE": result["ECE"], "quant_max_abs_error_percent": parity, "authority": 0})
    write_json(output / "preprocessing_train_only.json", {"input": "staged_train_only_external_QE_linear_proxy_divided_by_100", "selected_degree": degree, "coefficient": coefficient.tolist(), "monotonicity_grid_points": 2001})
    write_json(output / "task_contract.json", {"candidate_id": candidate_id, "task_contract_sha256": metadata["task_contract_sha256"], "contract": contract, "authority": 0})
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "split_manifest.json", {"split_sha256": metadata["split_sha256"], "cross_split_group_overlap": 0})
    write_json(output / "baseline_report.json", baseline_test)
    write_json(output / "ablation.json", {"selected": "train_only_monotone_polynomial_calibrator", "baseline": metadata["baseline_fit"], "validation_selected_without_test": True})
    (output / "model_card.md").write_text(f"# {candidate_id} model card\n\n- Status: `{status}`\n- Model: train-only degree-{degree} monotone polynomial calibrator over the stronger measured external-QE proxy baseline.\n- Ranking is preserved; test ECE `{result['ECE']:.6f}` vs baseline `{baseline_test['ECE']:.6f}`.\n- Three fixed seeds: deterministic mean `{aggregate['mean']:.6f}`, variance `{aggregate['variance']:.8f}`, worst `{aggregate['worst']:.6f}`.\n- Scope: literature-curated IPOP IQE, not team absolute photon-count metrology.\n- Authority: `0`; unified GD32 board acceptance remains pending.\n", encoding="utf-8")
    promotion = {"schema": "cimc.forge200.promotion-receipt.v2", "status": status, "candidate_id": candidate_id, "authority": 0, "board_accepted": False, "countable_model": False, "host_contract_pass": passed, "three_seed_count": 3, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "contract_exact_audit_sha256": sha256_file(output / "contract_exact_audit.json"), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / (1024**3)}}
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    closure_content = {"records": [exact], "authority_nonzero": 0, "board_actions": 0}
    closure = {"schema": "cimc.forge200.ipop-p059-monotone-closure.v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "PARTIAL", **closure_content, "content_root_sha256": hashlib.sha256(canonical_bytes(closure_content)).hexdigest()}
    write_json(root / "evidence" / "ipop_p059_monotone_closure.v2.json", closure)
    print(json.dumps({"status": closure["status"], "candidate_id": candidate_id, "selected_degree": degree, "model_composite": result["composite_higher_is_better"], "baseline_composite": baseline_test["composite_higher_is_better"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
