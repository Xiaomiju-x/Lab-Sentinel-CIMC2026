#!/usr/bin/env python3
"""CPU-only fixture proof for new GPU-B data, ONNX, golden, and ICMF paths."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

LOCAL_TOOLING = Path(__file__).resolve().parents[1] / ".tooling" / "python"
if LOCAL_TOOLING.is_dir():
    sys.path.insert(0, str(LOCAL_TOOLING))

import onnx
from onnx import TensorProto, helper, numpy_helper

from gpu_train_job import build_package, canonical_bytes, sha256_file, write_json


def load_admitted(root: Path, candidate_id: str) -> tuple[dict[str, Any], Any]:
    queue = json.loads((root / "queue" / "dual_5090_queue.v1.json").read_text(encoding="utf-8"))
    jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
    job = next(item for item in jobs if item["candidate_id"] == candidate_id)
    if job["admission_state"] != "ADMITTED" or job["authority"] != 0:
        raise RuntimeError(f"{candidate_id}: admission")
    dataset = root / job["staged_dataset"]
    if sha256_file(dataset) != job["staged_dataset_sha256"]:
        raise RuntimeError(f"{candidate_id}: dataset hash")
    metadata = json.loads((root / job["staged_metadata"]).read_text(encoding="utf-8"))
    if metadata["cross_split_group_overlap"] != 0 or metadata["authority"] != 0:
        raise RuntimeError(f"{candidate_id}: metadata")
    return metadata, np.load(dataset, allow_pickle=False)


def fixture_lm(root: Path, output: Path) -> dict[str, Any]:
    candidate_id = "CAND-G-001"
    metadata, data = load_admitted(root, candidate_id)
    x, y, mask, split = data["x"], data["y"], data["loss_mask"], data["split"]
    train = split == 0
    counts = np.ones((259, 259), dtype=np.float64)
    for source, target in zip(x[train][mask[train] > 0], y[train][mask[train] > 0]):
        counts[int(source), int(target)] += 1.0
    logits = np.log(counts / counts.sum(axis=1, keepdims=True)).astype(np.float32)
    scale = max(float(np.max(np.abs(logits))) / 127.0, 1e-12)
    quantized = np.clip(np.rint(logits / scale), -127, 127).astype(np.int8)
    restored = quantized.astype(np.float32) * scale
    test_x = x[split == 2][:4].astype(np.int64)
    fp32 = logits[test_x]
    w8 = restored[test_x]
    artifact = output / candidate_id
    artifact.mkdir(parents=True, exist_ok=True)
    golden = artifact / "golden_vectors.npz"
    np.savez_compressed(golden, x=test_x, fp32=fp32, quantized=w8)
    graph = helper.make_graph(
        [helper.make_node("Gather", ["transition_logits", "tokens"], ["logits"], axis=0)],
        "forge200_fixture_byte_lm",
        [helper.make_tensor_value_info("tokens", TensorProto.INT64, [None, None])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, None, 259])],
        [numpy_helper.from_array(logits, "transition_logits")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="CIMC-Forge200-fixture")
    model.ir_version = 10
    onnx_path = artifact / "fixture.onnx"
    onnx.save(model, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    payload = io.BytesIO()
    np.savez_compressed(payload, transition_logits=quantized, scale=np.asarray(scale, dtype=np.float32))
    schema_sha = hashlib.sha256(canonical_bytes({"task_kind": "token_lm", "shape": [None, None, 259], "authority": 0})).hexdigest()
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "fixture": True, "dataset": metadata["sha256"], "onnx": sha256_file(onnx_path)})).hexdigest()
    package = build_package(artifact, candidate_id, payload.getvalue(), sha256_file(golden), release_root, schema_sha, engine_id=5)
    return {
        "candidate_id": candidate_id,
        "task_kind": "token_lm",
        "status": "FIXTURE_DRY_RUN_PASS_NOT_TRAINED",
        "onnx_checker": "PASS",
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden),
        "quant_max_abs_error": float(np.max(np.abs(fp32 - w8))),
        "package": package,
        "authority": 0,
        "countable_model": False,
    }


def fixture_encoder(root: Path, output: Path) -> dict[str, Any]:
    candidate_id = "CAND-S-009"
    metadata, data = load_admitted(root, candidate_id)
    x, split = data["x_query"].astype(np.float32), data["split"]
    generator = np.random.default_rng(20260801)
    projection = generator.normal(0.0, 0.08, size=(x.shape[1], 32)).astype(np.float32)
    scale = max(float(np.max(np.abs(projection))) / 127.0, 1e-12)
    quantized = np.clip(np.rint(projection / scale), -127, 127).astype(np.int8)
    restored = quantized.astype(np.float32) * scale

    def encode(values: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        embedded = np.tanh(values @ matrix)
        return embedded / np.maximum(np.linalg.norm(embedded, axis=1, keepdims=True), 1e-12)

    test_x = x[split == 2][:32]
    fp32 = encode(test_x, projection)
    w8 = encode(test_x, restored)
    artifact = output / candidate_id
    artifact.mkdir(parents=True, exist_ok=True)
    golden = artifact / "golden_vectors.npz"
    np.savez_compressed(golden, x=test_x, fp32=fp32, quantized=w8)
    nodes = [
        helper.make_node("MatMul", ["input", "projection"], ["projected"]),
        helper.make_node("Tanh", ["projected"], ["activated"]),
        helper.make_node("ReduceL2", ["activated"], ["norm"], axes=[1], keepdims=1),
        helper.make_node("Max", ["norm", "epsilon"], ["safe_norm"]),
        helper.make_node("Div", ["activated", "safe_norm"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes,
        "forge200_fixture_query_encoder",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, x.shape[1]])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [None, 32])],
        [numpy_helper.from_array(projection, "projection"), numpy_helper.from_array(np.asarray([1e-12], dtype=np.float32), "epsilon")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="CIMC-Forge200-fixture")
    model.ir_version = 10
    onnx_path = artifact / "fixture.onnx"
    onnx.save(model, onnx_path)
    onnx.checker.check_model(onnx.load(onnx_path))
    payload = io.BytesIO()
    np.savez_compressed(payload, projection=quantized, scale=np.asarray(scale, dtype=np.float32))
    schema_sha = hashlib.sha256(canonical_bytes({"task_kind": "contrastive_embedding", "shape": [None, 32], "authority": 0})).hexdigest()
    release_root = hashlib.sha256(canonical_bytes({"candidate_id": candidate_id, "fixture": True, "dataset": metadata["sha256"], "onnx": sha256_file(onnx_path)})).hexdigest()
    package = build_package(artifact, candidate_id, payload.getvalue(), sha256_file(golden), release_root, schema_sha, engine_id=4)
    return {
        "candidate_id": candidate_id,
        "task_kind": "contrastive_embedding",
        "status": "FIXTURE_DRY_RUN_PASS_NOT_TRAINED",
        "onnx_checker": "PASS",
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden),
        "quant_max_abs_error": float(np.max(np.abs(fp32 - w8))),
        "package": package,
        "authority": 0,
        "countable_model": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("evidence/rag_fixture_dry_run_v1"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = [fixture_lm(root, output), fixture_encoder(root, output)]
    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "receipt.json"):
        files.append({"path": str(path.relative_to(output)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    receipt = {
        "schema": "cimc.forge200.rag-fixture-dry-run.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "records": records,
        "files": files,
        "file_count": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "content_root_sha256": hashlib.sha256(canonical_bytes(files)).hexdigest(),
        "gpu_used": False,
        "teacher_outputs": 0,
        "authority_nonzero": 0,
        "countable_models": 0,
    }
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "records": len(records), "files": receipt["file_count"], "bytes": receipt["bytes"], "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
