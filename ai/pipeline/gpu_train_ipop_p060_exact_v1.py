#!/usr/bin/env python3
"""Train, evaluate, W8-quantize, and package the exact P060 CIE model."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json


PARAMETER_CAP = 28_000
WEIGHT_BYTE_CAP = 36 * 1024


def xy_to_lab(xy: np.ndarray) -> np.ndarray:
    x = np.clip(xy[:, 0], 1e-6, 0.999999)
    y = np.clip(xy[:, 1], 1e-6, 0.999999)
    xyz = np.column_stack((x / y, np.ones(len(x)), np.maximum(1.0 - x - y, 0.0) / y))
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float64)
    delta = 6.0 / 29.0
    transformed = np.where(xyz > delta**3, np.cbrt(xyz), xyz / (3.0 * delta**2) + 4.0 / 29.0)
    return np.column_stack((116.0 * transformed[:, 1] - 16.0, 500.0 * (transformed[:, 0] - transformed[:, 1]), 200.0 * (transformed[:, 1] - transformed[:, 2])))


def delta_e_2000(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    l1, a1, b1 = left[:, 0], left[:, 1], left[:, 2]
    l2, a2, b2 = right[:, 0], right[:, 1], right[:, 2]
    c1, c2 = np.hypot(a1, b1), np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7 + 1e-30)))
    ap1, ap2 = (1.0 + g) * a1, (1.0 + g) * a2
    cp1, cp2 = np.hypot(ap1, b1), np.hypot(ap2, b2)
    hp1 = np.mod(np.degrees(np.arctan2(b1, ap1)), 360.0)
    hp2 = np.mod(np.degrees(np.arctan2(b2, ap2)), 360.0)
    dl = l2 - l1
    dc = cp2 - cp1
    dh_angle = hp2 - hp1
    dh_angle = np.where(np.abs(dh_angle) <= 180.0, dh_angle, np.where(dh_angle > 180.0, dh_angle - 360.0, dh_angle + 360.0))
    dh_angle = np.where((cp1 * cp2) == 0.0, 0.0, dh_angle)
    dh = 2.0 * np.sqrt(cp1 * cp2) * np.sin(np.radians(dh_angle) / 2.0)
    l_bar = (l1 + l2) / 2.0
    cp_bar = (cp1 + cp2) / 2.0
    hp_sum = hp1 + hp2
    hp_bar = np.where(
        (cp1 * cp2) == 0.0,
        hp_sum,
        np.where(np.abs(hp1 - hp2) <= 180.0, hp_sum / 2.0, np.where(hp_sum < 360.0, (hp_sum + 360.0) / 2.0, (hp_sum - 360.0) / 2.0)),
    )
    t = 1.0 - 0.17 * np.cos(np.radians(hp_bar - 30.0)) + 0.24 * np.cos(np.radians(2.0 * hp_bar)) + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0)) - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cp_bar
    sh = 1.0 + 0.015 * cp_bar * t
    delta_theta = 30.0 * np.exp(-((hp_bar - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt(cp_bar**7 / (cp_bar**7 + 25.0**7 + 1e-30))
    rt = -rc * np.sin(np.radians(2.0 * delta_theta))
    return np.sqrt((dl / sl) ** 2 + (dc / sc) ** 2 + (dh / sh) ** 2 + rt * (dc / sc) * (dh / sh))


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0.0, 1.0)
    xy_mae = float(np.mean(np.abs(prediction - y)))
    delta_e = float(np.mean(delta_e_2000(xy_to_lab(y.astype(np.float64)), xy_to_lab(prediction.astype(np.float64)))))
    result = {
        "xy_MAE": xy_mae,
        "DeltaE2000_D65_normalized_Y": delta_e,
        "valid_xy_rate": float(np.mean((prediction[:, 0] > 0.0) & (prediction[:, 1] > 0.0) & (prediction.sum(axis=1) <= 1.0))),
    }
    result["primary_composite"] = 0.5 / (1.0 + xy_mae / 0.05) + 0.5 / (1.0 + delta_e / 10.0)
    return result


def quantize_state(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quantized, scales = {}, {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim == 2:
            scale = np.maximum(np.max(np.abs(array), axis=1, keepdims=True), 1e-12) / 127.0
        else:
            scale = np.asarray(max(float(np.max(np.abs(array))), 1e-12) / 127.0, dtype=np.float32)
        quantized[name] = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        scales[name] = np.asarray(scale, dtype=np.float32)
    return quantized, scales


def dequantized_state(torch: Any, quantized: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: torch.from_numpy(value.astype(np.float32) * scales[name]) for name, value in quantized.items()}


def manifest(output: Path) -> None:
    records = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name not in {"artifact_manifest.json", "heartbeat.json"}):
        records.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()})


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    root = args.root.resolve()
    candidate_id = "CAND-P-060"
    dataset = root / "data" / "staged_ipop_p060_exact_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if metadata["status"] != "PASS" or metadata["cross_split_host_overlap"] != 0 or metadata["cross_split_doi_overlap"] != 0 or sha256_file(dataset) != metadata["sha256"]:
        raise RuntimeError("DATA_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = args.artifact_root.resolve() / candidate_id
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    started = time.perf_counter()

    raw = np.load(dataset, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    split = raw["split"].astype(np.int8)
    baseline_prediction = raw["baseline_pred"].astype(np.float32)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    mean, std = x_raw[train].mean(axis=0), x_raw[train].std(axis=0)
    std[std < 1e-6] = 1.0
    x = np.clip((x_raw - mean) / std, -12.0, 12.0).astype(np.float32)
    model_input = np.concatenate((x, baseline_prediction), axis=1).astype(np.float32)
    baseline_validation = metrics(y[validation], baseline_prediction[validation])
    baseline_test = metrics(y[test], baseline_prediction[test])

    class CIEResidualMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(nn.Linear(x.shape[1], 88), nn.GELU(), nn.Linear(88, 32), nn.GELU(), nn.Linear(32, 2))

        def forward(self, value: Any) -> Any:
            return value[:, -2:] + self.net(value[:, :-2]) * 0.25

    parameter_count = sum(parameter.numel() for parameter in CIEResidualMLP().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")
    loader = TensorDataset(torch.from_numpy(model_input[train]), torch.from_numpy(y[train]))
    reports, states = [], {}
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = CIEResidualMLP().to(device)
        with torch.no_grad():
            model.net[-1].weight.zero_()
            model.net[-1].bias.zero_()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=3e-4)
        batches = DataLoader(loader, batch_size=min(args.batch_size, len(train)), shuffle=True, generator=torch.Generator().manual_seed(seed))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_score, patience = -float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            for batch_x, batch_y in batches:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x.to(device))
                loss = nn.functional.smooth_l1_loss(prediction, batch_y.to(device), beta=0.03)
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = model(torch.from_numpy(model_input[validation]).to(device)).cpu().numpy()
            score = metrics(y[validation], validation_prediction)["primary_composite"]
            if score > best_score + 1e-5:
                best_score, patience = score, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_IPOP_CIE_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy()
        report = metrics(y[test], prediction)
        reports.append({"seed": seed, "epochs": epoch + 1, "validation_primary_composite": best_score, "test": report, "beats_baseline": report["primary_composite"] > baseline_test["primary_composite"] + 1e-4})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    composites = np.asarray([report["test"]["primary_composite"] for report in reports])
    aggregate = {"mean": float(composites.mean()), "variance": float(composites.var()), "std": float(composites.std()), "worst": float(composites.min())}
    aggregate_pass = aggregate["mean"] > baseline_test["primary_composite"] + 1e-4
    best_report = max(reports, key=lambda report: report["validation_primary_composite"])
    best_seed = int(best_report["seed"])
    model = CIEResidualMLP().to(device)
    model.load_state_dict(states[best_seed])
    model.eval()
    with torch.no_grad():
        fp_prediction = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy()
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    model.eval()
    with torch.no_grad():
        quant_prediction = model(torch.from_numpy(model_input[test]).to(device)).cpu().numpy()
    quant_metrics = metrics(y[test], quant_prediction)
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.02
    passed = aggregate_pass and quant_pass

    golden = output / "golden_vectors.npz"
    np.savez_compressed(golden, x=model_input[test[:64]], y=y[test[:64]], fp32=np.clip(fp_prediction[:64], 0.0, 1.0), quantized=np.clip(quant_prediction[:64], 0.0, 1.0))
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(model, torch.from_numpy(model_input[test[:1]]).to(device), onnx_path, input_names=["preprocessed_phosphor_features_and_centroid_baseline"], output_names=["cie_xy"], dynamic_axes={"preprocessed_phosphor_features_and_centroid_baseline": {0: "batch"}, "cie_xy": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"task_kind": "two_output_regression", "shape": [None, 2], "units": "CIE_1931_xy", "semantics": ["CIE_x", "CIE_y"], "postprocess": "clip_each_0_to_1", "authority": 0}
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "dataset_sha256": metadata["sha256"], "task_contract_sha256": metadata["task_contract_sha256"], "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "three_seed_mean": aggregate["mean"]})).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest())
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if passed else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    audit = {
        "schema": "cimc.forge200.ipop-p060-exact-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": candidate_id,
        "baseline": {"validation": baseline_validation, "test": baseline_test, "kind": metadata["baseline_fit"]},
        "seed_reports": reports,
        "aggregate": aggregate,
        "g3_aggregate_mean_gate": aggregate_pass,
        "quantized_best_seed": {"seed": best_seed, "test": quant_metrics, "metric_delta": quant_delta, "gate": quant_pass},
        "parameter_count": parameter_count,
        "w8_payload_bytes": len(payload),
        "evaluation_convention": "DeltaE2000 after xy-to-XYZ with normalized Y=1 and D65 white",
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", audit)
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "baseline_report.json", audit["baseline"])
    write_json(output / "preprocessing_train_only.json", {"mean": mean.tolist(), "std": std.tolist(), "baseline_coefficient": metadata["baseline_coefficient"]})
    write_json(output / "output_schema.json", output_schema)
    write_json(output / "quantization_parity.json", {"primary_composite_delta": quant_delta, "gate": quant_pass})
    (output / "model_card.md").write_text(
        f"# {candidate_id} exact IPOP CIE model\n\n"
        f"- Status: `{status}`.\n"
        f"- Three-seed mean composite: `{aggregate['mean']:.6f}`; spectral-centroid linear baseline: `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Best validation-selected test xy MAE: `{best_report['test']['xy_MAE']:.6f}`; normalized-Y D65 DeltaE2000: `{best_report['test']['DeltaE2000_D65_normalized_Y']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        f"- Full emission spectrum and measurement geometry are unavailable and explicitly masked; the model is scoped to IPOP peak/context inputs.\n"
        f"- Authority: `0`; unified GD32 board acceptance remains pending.\n",
        encoding="utf-8",
    )
    promotion = {"schema": "cimc.forge200.promotion-receipt.v3", "status": status, "candidate_id": candidate_id, "authority": 0, "board_accepted": False, "countable_model": False, "host_contract_pass": passed, "three_seed_count": 3, "release_root": release_root, "package": package, "onnx_sha256": sha256_file(onnx_path), "golden_sha256": sha256_file(golden), "contract_exact_audit_sha256": sha256_file(output / "contract_exact_audit.json"), "runtime_seconds": time.perf_counter() - started, "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3}}
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    closure = {"schema": "cimc.forge200.ipop-p060-exact-closure.v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS" if passed else "PARTIAL", "record": audit, "authority_nonzero": 0, "board_actions": 0}
    write_json(root / "evidence" / "ipop_p060_exact_closure.v1.json", closure)
    print(json.dumps({"status": closure["status"], "candidate_id": candidate_id, "mean_composite": aggregate["mean"], "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": promotion["runtime_seconds"]}, sort_keys=True))
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=30)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=7e-4)
    args = parser.parse_args()
    try:
        receipt = run(args)
        return 0 if receipt["host_contract_pass"] else 2
    except Exception as exc:
        output = args.artifact_root.resolve() / "CAND-P-060"
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": "CAND-P-060", "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
