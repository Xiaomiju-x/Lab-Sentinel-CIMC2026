#!/usr/bin/env python3
"""CPU fixture implementation of the Forge200 train-to-package contract.

The fixture path proves orchestration, metrics, calibration, quantization,
export, golden parity, package ABI and model-card generation.  Its outputs are
explicitly non-promotable and never count as model quality or board evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import struct
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Keep the local dry-run toolchain self-contained and avoid changing the
# workspace Python installation.  The directory is populated from pinned
# wheels and is hashed by the GPU-readiness receipt.
LOCAL_TOOLING = Path(__file__).resolve().parents[1] / ".tooling" / "python"
if LOCAL_TOOLING.is_dir():
    sys.path.insert(0, str(LOCAL_TOOLING))

import numpy as np


SEEDS = [20260801, 20260802, 20260803]
HEADER_BYTES = 256


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def stable_split(groups: np.ndarray) -> np.ndarray:
    result = []
    for group in groups.astype(str):
        bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 100
        result.append(0 if bucket < 70 else 1 if bucket < 85 else 2)
    return np.asarray(result, dtype=np.int8)


def make_fixtures(root: Path) -> list[Path]:
    fixture_dir = root / "data" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("CAND-P-001", "regression", 240, 8, 1),
        ("CAND-G-001", "token_classification", 288, 16, 16),
        ("CAND-S-001", "classification", 240, 10, 3),
    ]
    outputs: list[Path] = []
    for candidate_id, kind, rows, features, outputs_count in specs:
        seed = int(hashlib.sha256(candidate_id.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        groups = np.asarray([f"{candidate_id}:group:{index // 8:03d}" for index in range(rows)])
        entities = np.asarray([f"{candidate_id}:entity:{index // 4:03d}" for index in range(rows)])
        split = stable_split(groups)
        as_of_ms = np.arange(rows, dtype=np.int64) * 1000
        label_available_ms = as_of_ms + 5000
        if kind == "token_classification":
            tokens = rng.integers(0, features, size=rows)
            x = np.eye(features, dtype=np.float32)[tokens]
            y = ((tokens * 5 + 3) % outputs_count).astype(np.int64)
        else:
            x = rng.normal(size=(rows, features)).astype(np.float32)
            if kind == "regression":
                weights = np.linspace(-0.8, 0.9, features, dtype=np.float32)
                y = (x @ weights + 0.15 * np.sin(x[:, 0]) + rng.normal(0, 0.03, rows)).astype(np.float32)
            else:
                logits = np.stack((x[:, 0] - x[:, 1], x[:, 2] + 0.4 * x[:, 3], -x[:, 0] + x[:, 4]), axis=1)
                y = np.argmax(logits + rng.normal(0, 0.08, logits.shape), axis=1).astype(np.int64)
        path = fixture_dir / f"{candidate_id}.npz"
        np.savez_compressed(
            path,
            x=x,
            y=y,
            groups=groups,
            entities=entities,
            split=split,
            as_of_monotonic_ms=as_of_ms,
            label_available_monotonic_ms=label_available_ms,
            candidate_id=np.asarray(candidate_id),
            task_kind=np.asarray(kind),
            truth_class=np.asarray("CONTROLLED_FIXTURE"),
            authority=np.asarray(0, dtype=np.int8),
        )
        outputs.append(path)
    manifest = {
        "schema": "cimc.forge200.micro-fixture.v1",
        "status": "CONTROLLED_FIXTURE_PIPELINE_ONLY",
        "authority": 0,
        "files": [
            {"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in outputs
        ],
        "forbidden_claims": ["MODEL_QUALITY", "REAL_PROCESS_ACCURACY", "BOARD_PERFORMANCE", "CONTROL_AUTHORITY"],
    }
    write_json(fixture_dir / "manifest.v1.json", manifest)
    return outputs


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def train_classifier(x: np.ndarray, y: np.ndarray, classes: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 0.02, size=(x.shape[1], classes)).astype(np.float32)
    b = np.zeros(classes, dtype=np.float32)
    y_onehot = np.eye(classes, dtype=np.float32)[y]
    losses = []
    for step in range(240):
        order = rng.permutation(len(x))
        xb = x[order]
        yb = y_onehot[order]
        prob = softmax(xb @ w + b)
        grad = (prob - yb) / len(xb)
        lr = 0.25 / math.sqrt(1.0 + step / 40.0)
        w -= lr * (xb.T @ grad + 1e-4 * w)
        b -= lr * grad.sum(axis=0)
        if step % 20 == 0 or step == 239:
            losses.append(float(-np.mean(np.log(np.maximum(prob[np.arange(len(yb)), np.argmax(yb, axis=1)], 1e-8)))))
    return w, b, losses


def train_regressor(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rng = np.random.default_rng(seed)
    # Seed-specific tiny jitter is part of the declared fixture stability test.
    xj = x + rng.normal(0, 1e-4, size=x.shape).astype(np.float32)
    augmented = np.concatenate([xj, np.ones((len(xj), 1), dtype=np.float32)], axis=1)
    ridge = np.eye(augmented.shape[1], dtype=np.float32) * 1e-3
    params = np.linalg.solve(augmented.T @ augmented + ridge, augmented.T @ y)
    pred = augmented @ params
    loss = float(np.mean((pred - y) ** 2))
    return params[:-1, None].astype(np.float32), np.asarray([params[-1]], dtype=np.float32), [loss]


def predict(x: np.ndarray, w: np.ndarray, b: np.ndarray, kind: str) -> np.ndarray:
    raw = x @ w + b
    if kind == "regression":
        return raw.reshape(-1)
    return softmax(raw)


def macro_f1(y: np.ndarray, pred: np.ndarray, classes: int) -> float:
    values = []
    for cls in range(classes):
        tp = int(np.sum((y == cls) & (pred == cls)))
        fp = int(np.sum((y != cls) & (pred == cls)))
        fn = int(np.sum((y == cls) & (pred != cls)))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return float(np.mean(values))


def ece_score(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    confidence = prob.max(axis=1)
    correct = (prob.argmax(axis=1) == y).astype(np.float32)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (confidence >= low) & (confidence < high if index < bins - 1 else confidence <= high)
        if np.any(mask):
            value += float(np.mean(mask)) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return value


def metrics(y: np.ndarray, output: np.ndarray, kind: str) -> dict[str, float]:
    if kind == "regression":
        return {
            "mae": float(np.mean(np.abs(output - y))),
            "rmse": float(np.sqrt(np.mean((output - y) ** 2))),
        }
    pred = output.argmax(axis=1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": macro_f1(y, pred, output.shape[1]),
        "ece": ece_score(y, output),
    }


def baseline_metrics(y_train: np.ndarray, y_test: np.ndarray, kind: str) -> dict[str, float]:
    if kind == "regression":
        pred = np.full_like(y_test, np.mean(y_train), dtype=np.float32)
        return metrics(y_test, pred, kind)
    classes = int(max(np.max(y_train), np.max(y_test))) + 1
    majority = int(np.bincount(y_train.astype(np.int64), minlength=classes).argmax())
    prob = np.zeros((len(y_test), classes), dtype=np.float32)
    prob[:, majority] = 1.0
    return metrics(y_test, prob, kind)


def quantize(w: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    w_scale = float(max(np.max(np.abs(w)) / 127.0, 1e-8))
    b_scale = float(max(np.max(np.abs(b)) / 127.0, 1e-8))
    wq = np.clip(np.rint(w / w_scale), -127, 127).astype(np.int8)
    bq = np.clip(np.rint(b / b_scale), -127, 127).astype(np.int8)
    return wq, bq, w_scale, b_scale


def dequant_predict(x: np.ndarray, wq: np.ndarray, bq: np.ndarray, w_scale: float, b_scale: float, kind: str) -> np.ndarray:
    return predict(x, wq.astype(np.float32) * w_scale, bq.astype(np.float32) * b_scale, kind)


def maybe_export_onnx(path: Path, w: np.ndarray, b: np.ndarray, kind: str) -> dict[str, Any]:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        return {"status": "SKIPPED_LOCAL_ONNX_DEPENDENCY_NOT_INSTALLED", "path": None}
    nodes = [helper.make_node("MatMul", ["input", "W"], ["linear"]), helper.make_node("Add", ["linear", "B"], ["logits"])]
    output_name = "logits"
    if kind != "regression":
        nodes.append(helper.make_node("Softmax", ["logits"], ["output"], axis=1))
        output_name = "output"
    graph = helper.make_graph(
        nodes,
        "cimc_forge200_fixture",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, w.shape[0]])],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [None, w.shape[1]])],
        [numpy_helper.from_array(w.astype(np.float32), "W"), numpy_helper.from_array(b.astype(np.float32), "B")],
    )
    model = helper.make_model(graph, producer_name="cimc-forge200", opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())
    return {"status": "ONNX_CHECKER_PASS", "path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}


def build_package(
    path: Path,
    candidate_id: str,
    wq: np.ndarray,
    bq: np.ndarray,
    w_scale: float,
    b_scale: float,
    golden_sha: str,
    release_root: str,
    output_schema_sha: str,
) -> dict[str, Any]:
    tensor_meta = canonical_bytes(
        {
            "weights_shape": list(wq.shape),
            "bias_shape": list(bq.shape),
            "weights_scale": w_scale,
            "bias_scale": b_scale,
            "quantization": "W8A8_FIXTURE",
        }
    )
    payload = struct.pack("<I", len(tensor_meta)) + tensor_meta + wq.tobytes(order="C") + bq.tobytes(order="C")
    payload_sha = sha256_bytes(payload)
    header = bytearray(HEADER_BYTES)
    header[0:4] = b"ICMF"
    struct.pack_into("<HHHHBBHQQIII", header, 4, 1, HEADER_BYTES, 240, 1, 0, 1, 2, 1, len(payload), 0, 0, 0)
    model_id = candidate_id.encode("utf-8")[:31]
    header[44 : 44 + len(model_id)] = model_id
    header[76:108] = bytes.fromhex(payload_sha)
    header[108:140] = bytes.fromhex(golden_sha)
    header[140:172] = bytes.fromhex(release_root)
    header[172:204] = bytes.fromhex(output_schema_sha)
    path.write_bytes(header + payload)
    parsed = path.read_bytes()
    if parsed[:4] != b"ICMF" or len(parsed) != HEADER_BYTES + len(payload):
        raise RuntimeError("package ABI self-check failed")
    if sha256_bytes(parsed[HEADER_BYTES:]) != payload_sha:
        raise RuntimeError("package payload SHA self-check failed")
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "payload_sha256": payload_sha,
        "magic": "ICMF",
        "schema_version": 1,
        "engine_id": 240,
        "authority": 0,
    }


@dataclass
class RunResult:
    candidate_id: str
    output_dir: Path
    receipt: dict[str, Any]


def run_one(root: Path, fixture: Path, output_root: Path) -> RunResult:
    data = np.load(fixture, allow_pickle=False)
    candidate_id = str(data["candidate_id"])
    kind = str(data["task_kind"])
    if str(data["truth_class"]) != "CONTROLLED_FIXTURE" or int(data["authority"]) != 0:
        raise RuntimeError("fixture truth/authority gate failed")
    if np.any(data["as_of_monotonic_ms"] >= data["label_available_monotonic_ms"]):
        raise RuntimeError("fixture temporal cutoff gate failed")
    split = data["split"]
    group_sets = [set(data["groups"][split == value].tolist()) for value in (0, 1, 2)]
    if group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2]:
        raise RuntimeError("group leakage detected")
    x = data["x"].astype(np.float32)
    y = data["y"]
    x_train, y_train = x[split == 0], y[split == 0]
    x_validation, y_validation = x[split == 1], y[split == 1]
    x_test, y_test = x[split == 2], y[split == 2]
    classes = 1 if kind == "regression" else int(np.max(y)) + 1
    output_dir = output_root / candidate_id
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_reports = []
    models = []
    started = time.perf_counter()
    for seed in SEEDS:
        if kind == "regression":
            w, b, losses = train_regressor(x_train, y_train.astype(np.float32), seed)
        else:
            w, b, losses = train_classifier(x_train, y_train.astype(np.int64), classes, seed)
        validation_output = predict(x_validation, w, b, kind)
        test_output = predict(x_test, w, b, kind)
        seed_dir = output_dir / f"train_seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        np.savez(seed_dir / "checkpoint.npz", weights=w, bias=b)
        seed_report = {
            "seed": seed,
            "loss_trace": losses,
            "validation": metrics(y_validation, validation_output, kind),
            "test": metrics(y_test, test_output, kind),
            "checkpoint_sha256": sha256_file(seed_dir / "checkpoint.npz"),
        }
        write_json(seed_dir / "metrics.json", seed_report)
        seed_reports.append(seed_report)
        models.append((w, b))
    primary_key = "mae" if kind == "regression" else "macro_f1"
    best_index = min(range(3), key=lambda idx: seed_reports[idx]["validation"][primary_key]) if kind == "regression" else max(range(3), key=lambda idx: seed_reports[idx]["validation"][primary_key])
    w, b = models[best_index]
    float_output = predict(x_test, w, b, kind)
    baseline = baseline_metrics(y_train, y_test, kind)
    wq, bq, w_scale, b_scale = quantize(w, b)
    quant_output = dequant_predict(x_test, wq, bq, w_scale, b_scale, kind)
    quant_metrics = metrics(y_test, quant_output, kind)
    parity_error = float(np.max(np.abs(float_output - quant_output)))
    np.savez(output_dir / "fp32.npz", weights=w, bias=b)
    golden_path = output_dir / "golden_vectors.npz"
    np.savez_compressed(golden_path, input=x_test[:16], fp32_output=float_output[:16], quant_output=quant_output[:16])
    golden_sha = sha256_file(golden_path)
    output_schema = {"kind": kind, "shape": [None, 1 if kind == "regression" else classes], "dtype": "float32"}
    output_schema_sha = sha256_bytes(canonical_bytes(output_schema))
    release_inputs = {
        "candidate_id": candidate_id,
        "fixture_sha256": sha256_file(fixture),
        "pipeline_sha256": sha256_file(Path(__file__)),
        "seeds": SEEDS,
        "truth_class": "CONTROLLED_FIXTURE",
        "authority": 0,
    }
    release_root = sha256_bytes(canonical_bytes(release_inputs))
    package = build_package(output_dir / "w8_or_w8a8.bin", candidate_id, wq, bq, w_scale, b_scale, golden_sha, release_root, output_schema_sha)
    model_ir = {
        "schema": "cimc.forge200.model-ir.v1",
        "candidate_id": candidate_id,
        "kind": kind,
        "engine_id": 240,
        "input_shape": [None, int(x.shape[1])],
        "output_schema": output_schema,
        "operator_whitelist": ["MatMul", "Add"] + ([] if kind == "regression" else ["Softmax"]),
        "quantization": {"weights": "int8_symmetric", "activations": "float_fixture_only", "w_scale": w_scale, "b_scale": b_scale},
        "authority": 0,
    }
    write_json(output_dir / "model_ir.json", model_ir)
    onnx_receipt = maybe_export_onnx(output_dir / "fp32.onnx", w, b, kind)
    write_json(output_dir / "onnx_export_status.json", onnx_receipt)
    grouped = {
        "schema": "cimc.forge200.fixture-eval.v1",
        "candidate_id": candidate_id,
        "truth_class": "CONTROLLED_FIXTURE",
        "split_counts": {"train": len(x_train), "validation": len(x_validation), "test": len(x_test)},
        "baseline": baseline,
        "seeds": seed_reports,
        "quantized_test": quant_metrics,
        "quant_max_abs_parity_error": parity_error,
        "group_overlap": 0,
    }
    write_json(output_dir / "eval_grouped.json", grouped)
    calibration = {
        "schema": "cimc.forge200.calibration-ood.v1",
        "candidate_id": candidate_id,
        "ece": None if kind == "regression" else quant_metrics["ece"],
        "ood_stress": "FIXTURE_FEATURE_SCALE_X1P5",
        "ood_output_finite": bool(np.all(np.isfinite(dequant_predict(x_test * 1.5, wq, bq, w_scale, b_scale, kind)))),
        "quality_claim_allowed": False,
    }
    write_json(output_dir / "calibration_ood.json", calibration)
    ablation = {
        "schema": "cimc.forge200.ablation.v1",
        "candidate_id": candidate_id,
        "constant_or_majority_baseline": baseline,
        "quantized_candidate": quant_metrics,
        "interpretation": "TOOLCHAIN_FIXTURE_ONLY",
    }
    write_json(output_dir / "ablation.json", ablation)
    model_card = f"""# {candidate_id} fixture model card

Status: `FIXTURE_TOOLCHAIN_PASS_NOT_MODEL_QUALITY`  
Authority: `0`  
Truth class: `CONTROLLED_FIXTURE`

This artifact proves the Forge200 dataset -> three-seed train -> grouped evaluate -> calibration/OOD -> W8A8 quantize -> export -> golden -> ABI package path. It is not a real-process model, is not board accepted, and cannot enter the public model count.

- Fixture SHA-256: `{sha256_file(fixture)}`
- Pipeline SHA-256: `{sha256_file(Path(__file__))}`
- Release root: `{release_root}`
- Package SHA-256: `{package['sha256']}`
- Golden SHA-256: `{golden_sha}`
- Max FP32/quantized output error: `{parity_error:.8f}`
- ONNX status: `{onnx_receipt['status']}`

Failure of this model or package must return refusal/offline diagnostics and cannot affect deterministic control.
"""
    (output_dir / "model_card.md").write_text(model_card, encoding="utf-8")
    elapsed = time.perf_counter() - started
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v1",
        "status": "FIXTURE_TOOLCHAIN_PASS_NOT_MODEL_QUALITY",
        "candidate_id": candidate_id,
        "authority": 0,
        "truth_class": "CONTROLLED_FIXTURE",
        "three_seed_count": len(seed_reports),
        "package": package,
        "golden_sha256": golden_sha,
        "release_root": release_root,
        "onnx": onnx_receipt,
        "quant_max_abs_parity_error": parity_error,
        "runtime_seconds": elapsed,
        "board_accepted": False,
        "countable_model": False,
    }
    write_json(output_dir / "promotion_receipt.json", receipt)
    manifest_records = []
    for path in sorted(
        item
        for item in output_dir.rglob("*")
        if item.is_file()
        and item.name not in {"artifact_manifest.json", "transfer_manifest.json"}
        and not item.name.startswith("worker_attempt_")
    ):
        manifest_records.append({"path": str(path.relative_to(output_dir)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_json(output_dir / "artifact_manifest.json", {"schema": "cimc.forge200.artifact-manifest.v1", "records": manifest_records, "content_root_sha256": sha256_bytes(canonical_bytes(manifest_records))})
    return RunResult(candidate_id, output_dir, receipt)


def dry_run(root: Path, output_root: Path) -> dict[str, Any]:
    fixtures = make_fixtures(root)
    output_root.mkdir(parents=True, exist_ok=True)
    results = [run_one(root, fixture, output_root) for fixture in fixtures]
    receipt = {
        "schema": "cimc.forge200.fixture-dry-run.v1",
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "dataset->train_3_seeds->evaluate->calibration_ood->quantize->export->golden->package->model_card",
        "tasks": [result.receipt for result in results],
        "authority_nonzero": sum(result.receipt["authority"] != 0 for result in results),
        "board_actions": 0,
        "gpu_actions": 0,
        "countable_models": 0,
    }
    write_json(output_root / "dry_run_receipt.v1.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare-fixtures", "dry-run"])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare-fixtures":
        paths = make_fixtures(root)
        print(json.dumps({"status": "PASS", "fixtures": len(paths)}, sort_keys=True))
        return 0
    output = (args.output or root / "artifacts" / "fixture_dry_run").resolve()
    receipt = dry_run(root, output)
    print(json.dumps({"status": receipt["status"], "tasks": len(receipt["tasks"]), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
