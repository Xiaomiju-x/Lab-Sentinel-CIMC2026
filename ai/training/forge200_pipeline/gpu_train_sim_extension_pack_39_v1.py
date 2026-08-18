#!/usr/bin/env python3
"""Train, freeze-test, quantize, export, and package 39 SIM_ONLY extensions."""

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

from gpu_train_job import build_package, canonical_bytes, heartbeat, sha256_file, write_json
from gpu_train_tematdb_p141_p144_exact_v1 import dequantized_state, manifest, quantize_state


def metrics(y: np.ndarray, prediction: np.ndarray, groups: np.ndarray, train_scale: np.ndarray) -> dict[str, float]:
    normalized = (prediction - y) / train_scale
    rmse = float(np.sqrt(np.mean(normalized**2)))
    mae = float(np.mean(np.abs(normalized)))
    group_rmse = []
    for group in np.unique(groups):
        selected = groups == group
        group_rmse.append(float(np.sqrt(np.mean(normalized[selected] ** 2))))
    worst = max(group_rmse) if group_rmse else rmse
    result = {
        "normalized_RMSE": rmse,
        "normalized_MAE": mae,
        "worst_family_normalized_RMSE": worst,
        "RMSE_skill": 1.0 - rmse,
        "MAE_score": 1.0 / (1.0 + mae),
        "worst_family_skill": 1.0 - worst,
        "evaluated_families": len(group_rmse),
    }
    result["primary_composite"] = float(
        np.mean([result["RMSE_skill"], result["MAE_score"], result["worst_family_skill"]])
    )
    return result


def train_one(root: Path, artifact_root: Path, task: dict[str, Any], config: dict[str, Any], device_name: str) -> dict[str, Any]:
    import onnx
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    candidate_id = task["candidate_id"]
    dataset = root / "data" / "staged_sim_extension_pack_39_v1" / f"{candidate_id}.npz"
    metadata = json.loads(dataset.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    if (
        metadata["status"] != "PASS_SIM_EXTENSION_DATA_FROZEN"
        or metadata["public_claim_scope"] != "SIM_ONLY"
        or metadata["original_task_contract_status"] != "UNCHANGED_FAIL_CLOSED"
        or metadata["cross_split_family_overlap"] != 0
        or sha256_file(dataset) != metadata["sha256"]
    ):
        raise RuntimeError(f"{candidate_id}:SIM_EXTENSION_DATA_GATE")

    output = artifact_root / candidate_id
    prior = output / "promotion_receipt.json"
    frozen_test = output / "frozen_test_evaluation.v1.json"
    if prior.is_file() and frozen_test.is_file():
        receipt = json.loads(prior.read_text(encoding="utf-8"))
        receipt["resume_action"] = "SKIPPED_ALREADY_FROZEN"
        return receipt

    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    started = time.perf_counter()
    raw = np.load(dataset, allow_pickle=False)
    x_raw = raw["x"].astype(np.float32)
    y_raw = raw["y"].astype(np.float32)
    baseline = raw["baseline"].astype(np.float32)
    groups = raw["group"].astype(str)
    split = raw["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))

    x_mean = x_raw[train].mean(axis=0)
    x_std = x_raw[train].std(axis=0)
    x_std[x_std < 1e-7] = 1.0
    x = np.clip((x_raw - x_mean) / x_std, -12.0, 12.0).astype(np.float32)
    y_mean = y_raw[train].mean(axis=0)
    y_std = y_raw[train].std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    y = ((y_raw - y_mean) / y_std).astype(np.float32)
    device = torch.device(device_name)

    class SimExtensionMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(x.shape[1], 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, y.shape[1]),
            )

        def forward(self, value: Any) -> Any:
            return self.net(value)

    model_config = config["model"]
    parameter_count = sum(parameter.numel() for parameter in SimExtensionMLP().parameters())
    if parameter_count > 40_000:
        raise RuntimeError(f"{candidate_id}:PARAMETER_CAP:{parameter_count}")

    train_dataset = TensorDataset(torch.from_numpy(x[train]), torch.from_numpy(y[train]))
    validation_metrics: list[dict[str, Any]] = []
    states: dict[int, dict[str, Any]] = {}
    for seed in model_config["seeds"]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        model = SimExtensionMLP().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"], weight_decay=3e-4)
        loader = DataLoader(
            train_dataset,
            batch_size=model_config["batch_size"],
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best = -float("inf")
        patience = 0
        for epoch in range(model_config["max_epochs"]):
            model.train()
            for batch_x, batch_y in loader:
                optimizer.zero_grad(set_to_none=True)
                prediction = model(batch_x.to(device))
                loss = nn.functional.smooth_l1_loss(prediction, batch_y.to(device), beta=0.1)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_prediction = (
                    model(torch.from_numpy(x[validation]).to(device)).cpu().numpy() * y_std + y_mean
                )
            current = metrics(y_raw[validation], validation_prediction, groups[validation], y_std)
            if current["primary_composite"] > best + 1e-5:
                best = current["primary_composite"]
                patience = 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_SIM_EXTENSION", seed, epoch)
            if epoch + 1 >= model_config["min_epochs"] and patience >= model_config["early_stop_patience"]:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(torch.from_numpy(x[validation]).to(device)).cpu().numpy() * y_std + y_mean
        validation_report = metrics(y_raw[validation], validation_prediction, groups[validation], y_std)
        validation_metrics.append({"seed": seed, "epochs": epoch + 1, "validation": validation_report})
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    selected_seed = int(max(validation_metrics, key=lambda item: item["validation"]["primary_composite"])["seed"])
    baseline_validation = metrics(y_raw[validation], baseline[validation], groups[validation], y_std)
    frozen_selection = {
        "schema": "cimc.forge200.sim-extension-selection-freeze.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "selection_rule": model_config["selection"],
        "selected_seed": selected_seed,
        "validation_reports": validation_metrics,
        "baseline_validation": baseline_validation,
        "test_metrics_observed": False,
        "contract_sha256": sha256_file(root / "contracts" / "sim_extension_pack_39.v1.json"),
    }
    write_json(output / "baseline_selection_frozen_before_test.json", frozen_selection)

    baseline_test = metrics(y_raw[test], baseline[test], groups[test], y_std)
    seed_reports: list[dict[str, Any]] = []
    predictions: dict[int, np.ndarray] = {}
    for seed in model_config["seeds"]:
        model = SimExtensionMLP().to(device)
        model.load_state_dict(states[seed])
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy() * y_std + y_mean
        predictions[seed] = prediction
        report = metrics(y_raw[test], prediction, groups[test], y_std)
        seed_reports.append({"seed": seed, "test": report})

    composites = np.asarray([item["test"]["primary_composite"] for item in seed_reports])
    aggregate = {
        "mean": float(composites.mean()),
        "variance": float(composites.var()),
        "std": float(composites.std()),
        "worst": float(composites.min()),
    }
    aggregate_gate = aggregate["mean"] > baseline_test["primary_composite"] + 1e-4
    frozen_test_receipt = {
        "schema": "cimc.forge200.sim-extension-frozen-test.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": candidate_id,
        "baseline": baseline_test,
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        "aggregate_gate": aggregate_gate,
        "retest_or_hyperparameter_tuning_authorized": False,
    }
    write_json(frozen_test, frozen_test_receipt)

    model = SimExtensionMLP().to(device)
    model.load_state_dict(states[selected_seed])
    model.eval()
    selected_fp32 = predictions[selected_seed]
    quantized, scales = quantize_state(states[selected_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(
        payload_buffer,
        **quantized,
        **{f"scale::{name}": value for name, value in scales.items()},
    )
    payload = payload_buffer.getvalue()
    if len(payload) > 48 * 1024:
        raise RuntimeError(f"{candidate_id}:W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    with torch.no_grad():
        quantized_prediction = model(torch.from_numpy(x[test]).to(device)).cpu().numpy() * y_std + y_mean
    quantized_metrics = metrics(y_raw[test], quantized_prediction, groups[test], y_std)
    selected_metrics = next(item["test"] for item in seed_reports if item["seed"] == selected_seed)
    quantized_delta = selected_metrics["primary_composite"] - quantized_metrics["primary_composite"]
    quantized_gate = (
        quantized_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4
        and quantized_delta <= model_config["quantized_primary_composite_max_drop"]
    )
    passed = bool(aggregate_gate and quantized_gate)

    golden = output / "golden_vectors.npz"
    np.savez_compressed(
        golden,
        x=x[test[:64]],
        y=y_raw[test[:64]],
        fp32=selected_fp32[:64],
        quantized=quantized_prediction[:64],
    )
    model.load_state_dict(states[selected_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(x[test[:1]]).to(device),
        onnx_path,
        input_names=["normalized_sim_extension_features"],
        output_names=["normalized_sim_extension_outputs"],
        dynamic_axes={"normalized_sim_extension_features": {0: "batch"}, "normalized_sim_extension_outputs": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    schema = {
        "task_kind": "sim_extension_multitarget_regression",
        "shape": [None, y_raw.shape[1]],
        "semantics": metadata["output_semantics"],
        "postprocess": "multiply_train_only_output_std_add_train_only_output_mean",
        "truth_class": "PHYSICS_SIM",
        "public_claim_scope": "SIM_ONLY",
        "original_task_contract_status": "UNCHANGED_FAIL_CLOSED",
        "authority": 0,
    }
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": candidate_id,
                "dataset": metadata["sha256"],
                "onnx": sha256_file(onnx_path),
                "golden": sha256_file(golden),
                "aggregate": aggregate,
            }
        )
    ).hexdigest()
    package = build_package(
        output,
        candidate_id,
        payload,
        sha256_file(golden),
        release_root,
        hashlib.sha256(canonical_bytes(schema)).hexdigest(),
    )
    status = (
        "HOST_GPU_TRAINED_SIM_EXTENSION_BASELINE_PASS_BOARD_PENDING"
        if passed
        else "HOST_GPU_REJECTED_SIM_EXTENSION_BASELINE"
    )
    audit = {
        "schema": "cimc.forge200.sim-extension-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_id": candidate_id,
        "truth_class": "PHYSICS_SIM",
        "public_claim_scope": "SIM_ONLY",
        "original_task_contract_status": "UNCHANGED_FAIL_CLOSED",
        "extension_baseline": {"kind": metadata["baseline_execution"], "validation": baseline_validation, "test": baseline_test},
        "seed_reports": seed_reports,
        "aggregate": aggregate,
        "aggregate_gate": aggregate_gate,
        "selected_seed_by_validation": selected_seed,
        "quantized_selected_seed": {
            "test": quantized_metrics,
            "primary_composite_drop": quantized_delta,
            "gate": quantized_gate,
        },
        "parameter_count": parameter_count,
        "w8_payload_bytes": len(payload),
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output / "contract_exact_audit.json", audit)
    write_json(output / "source_manifest.json", metadata)
    write_json(
        output / "preprocessing_train_only.json",
        {"x_mean": x_mean.tolist(), "x_std": x_std.tolist(), "y_mean": y_mean.tolist(), "y_std": y_std.tolist()},
    )
    write_json(output / "output_schema.json", schema)
    write_json(output / "quantization_parity.json", {"primary_composite_drop": quantized_delta, "gate": quantized_gate})
    (output / "model_card.md").write_text(
        f"# {candidate_id} SIM_ONLY extension\n\n"
        f"- Status: `{status}`.\n"
        "- Scope: numerical simulation/host ABI benchmark only; the frozen experimental task contract remains fail-closed.\n"
        f"- Three-seed mean composite: `{aggregate['mean']:.6f}`; reduced simulator baseline: `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        "- Authority: `0`; board pending; not publicly countable before unified board acceptance.\n",
        encoding="utf-8",
    )
    props = torch.cuda.get_device_properties(device)
    promotion = {
        "schema": "cimc.forge200.promotion-receipt.v4",
        "status": status,
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "host_contract_pass": False,
        "host_extension_pass": passed,
        "truth_class": "PHYSICS_SIM",
        "public_claim_scope": "SIM_ONLY",
        "original_task_contract_status": "UNCHANGED_FAIL_CLOSED",
        "three_seed_count": 3,
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3},
    }
    write_json(output / "promotion_receipt.json", promotion)
    manifest(output)
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    return promotion


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--candidate-id")
    args = parser.parse_args()
    root = args.root.resolve()
    config = json.loads((root / "contracts" / "sim_extension_pack_39.v1.json").read_text(encoding="utf-8"))
    tasks = config["tasks"]
    if args.candidate_id:
        tasks = [task for task in tasks if task["candidate_id"] == args.candidate_id]
        if not tasks:
            raise RuntimeError("UNKNOWN_CANDIDATE_ID")
    if not __import__("torch").cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")

    results = []
    for index, task in enumerate(tasks, 1):
        print(json.dumps({"event": "TASK_START", "index": index, "total": len(tasks), "candidate_id": task["candidate_id"]}), flush=True)
        receipt = train_one(root, args.artifact_root.resolve(), task, config, args.device)
        results.append(receipt)
        print(
            json.dumps(
                {
                    "event": "TASK_DONE",
                    "candidate_id": task["candidate_id"],
                    "status": receipt["status"],
                    "runtime_seconds": receipt.get("runtime_seconds"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    passed = sum(bool(item.get("host_extension_pass")) for item in results)
    closure = {
        "schema": "cimc.forge200.sim-extension-pack-closure.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed == len(results) else "PARTIAL",
        "requested_tasks": len(results),
        "host_extension_passes": passed,
        "rejections": len(results) - passed,
        "records": results,
        "original_exact_contract_promotions": 0,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "sim_extension_pack_39_closure.v1.json", closure)
    print(json.dumps({"status": closure["status"], "passed": passed, "rejected": len(results) - passed}, sort_keys=True))
    return 0 if passed == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
