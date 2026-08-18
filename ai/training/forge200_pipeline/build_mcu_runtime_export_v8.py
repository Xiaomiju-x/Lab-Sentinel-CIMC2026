#!/usr/bin/env python3
"""Re-export v7 weights into an uncompressed, MCU-parseable Forge runtime payload.

The v7 packages intentionally preserved the training NPZ payloads.  Those are
content-verifiable on a host but are not an embedded tensor ABI.  This exporter
keeps the frozen W8 values, adds an explicit tensor table, emits one compact
binary golden case per model, and changes no training or test selection.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


HEADER_BYTES = 256
RUNTIME_HEADER_BYTES = 64
TENSOR_ENTRY_BYTES = 48
ALIGNMENT = 32
DTYPE_INT8 = 1
DTYPE_FLOAT32 = 2
DTYPE_UINT8 = 3
DTYPE_UINT16 = 4

KIND_SEQUENCE = 1
KIND_RESIDUAL = 2
KIND_RIDGE_PRIOR = 3
KIND_POLYNOMIAL = 4
KIND_CIE_RESIDUAL = 5
KIND_INPUT_PRIOR_RESIDUAL = 6
KIND_SKIP = 7
KIND_MULTIHEAD = 8
KIND_CONV_SEQUENCE = 9
KIND_NANOLM = 10

ACT_LINEAR = 0
ACT_RELU = 1
ACT_GELU = 2

POST_RAW = 0
POST_LAST_SIGMOID = 1
POST_FIRST_SIGMOID_REST_SOFTMAX = 2
POST_FIRST_RAW_LAST_SIGMOID = 3
POST_SOFTPLUS = 4


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def align(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def gelu(value: np.ndarray) -> np.ndarray:
    vectorized = np.vectorize(math.erf, otypes=[np.float32])
    return (0.5 * value * (1.0 + vectorized(value / np.float32(math.sqrt(2.0))))).astype(np.float32)


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-value))).astype(np.float32)


def softmax(value: np.ndarray) -> np.ndarray:
    shifted = value - np.max(value)
    exponential = np.exp(shifted)
    return (exponential / np.sum(exponential)).astype(np.float32)


def softplus(value: np.ndarray) -> np.ndarray:
    return (np.log1p(np.exp(-np.abs(value))) + np.maximum(value, 0.0)).astype(np.float32)


def dequantize(archive: Any, name: str) -> np.ndarray:
    quantized = archive[name].astype(np.float32)
    scale_name = f"scale::{name}"
    if scale_name in archive.files:
        scale = archive[scale_name].astype(np.float32)
    elif name == "weight" and "scale" in archive.files:
        scale = archive["scale"].astype(np.float32)
    elif name == "coefficient" and "scale" in archive.files:
        scale = archive["scale"].astype(np.float32)
    else:
        raise RuntimeError(f"SCALE_MISSING:{name}")
    return (quantized * scale).astype(np.float32)


def dense(value: np.ndarray, archive: Any, weight: str, bias: str) -> np.ndarray:
    return (dequantize(archive, weight) @ value + dequantize(archive, bias)).astype(np.float32)


def numbered_layers(archive: Any, prefix: str) -> list[tuple[str, str]]:
    layers = []
    for name in archive.files:
        if name.startswith(prefix + ".") and name.endswith(".weight"):
            middle = name[len(prefix) + 1 : -len(".weight")]
            if middle.isdigit():
                layers.append((int(middle), name, name[:-6] + "bias"))
    return [(weight, bias) for _, weight, bias in sorted(layers)]


def run_layers(value: np.ndarray, archive: Any, layers: list[tuple[str, str]], activation: int) -> np.ndarray:
    current = value.astype(np.float32)
    for index, (weight, bias) in enumerate(layers):
        current = dense(current, archive, weight, bias)
        if index + 1 != len(layers):
            current = gelu(current) if activation == ACT_GELU else np.maximum(current, 0.0).astype(np.float32)
    return current


def infer_predictive(candidate_id: str, archive: Any, x: np.ndarray, has_relu: bool) -> tuple[np.ndarray, dict[str, int], list[tuple[str, str]]]:
    activation = ACT_RELU if has_relu else ACT_GELU
    if candidate_id in {"CAND-P-045", "CAND-P-070", "CAND-P-138", "CAND-P-142", "CAND-P-145"}:
        layers = numbered_layers(archive, "net")
        baseline = dense(x, archive, "baseline_weight", "baseline_bias")
        residual = run_layers(x, archive, layers, ACT_GELU)
        scale = float(dequantize(archive, "residual_scale").reshape(-1)[0])
        return baseline + residual * scale, {"kind": KIND_RESIDUAL, "activation": ACT_GELU, "post": POST_RAW}, layers
    if candidate_id in {"CAND-P-057", "CAND-P-058"}:
        weight = dequantize(archive, "weight")
        return np.asarray([x[-1] + float(x[:-1] @ weight)], dtype=np.float32), {"kind": KIND_RIDGE_PRIOR, "activation": ACT_LINEAR, "post": POST_RAW}, []
    if candidate_id == "CAND-P-059":
        result = np.float32(0.0)
        for coefficient in dequantize(archive, "coefficient"):
            result = result * x[0] + coefficient
        return np.asarray([result * 100.0], dtype=np.float32), {"kind": KIND_POLYNOMIAL, "activation": ACT_LINEAR, "post": POST_RAW}, []
    if candidate_id == "CAND-P-060":
        layers = numbered_layers(archive, "net")
        return x[-2:] + run_layers(x[:-2], archive, layers, ACT_GELU) * 0.25, {"kind": KIND_CIE_RESIDUAL, "activation": ACT_GELU, "post": POST_RAW}, layers
    if candidate_id == "CAND-P-087":
        hidden = numbered_layers(archive, "hidden")
        output = dense(x, archive, "skip.weight", "skip.bias") + run_layers(x, archive, hidden, ACT_GELU)
        return output, {"kind": KIND_SKIP, "activation": ACT_GELU, "post": POST_RAW}, hidden
    if candidate_id == "CAND-P-096":
        import torch
        import torch.nn.functional as functional

        current = torch.from_numpy(x[None].astype(np.float32))
        layers = numbered_layers(archive, "net")
        with torch.no_grad():
            for index, (weight, bias) in enumerate(layers):
                w = torch.from_numpy(dequantize(archive, weight))
                b = torch.from_numpy(dequantize(archive, bias))
                padding = 1 if w.shape[-1] == 3 else 0
                current = functional.conv2d(current, w, b, stride=1, padding=padding)
                if index + 1 != len(layers):
                    current = functional.gelu(current, approximate="none")
        return current.numpy().reshape(-1).astype(np.float32), {"kind": KIND_CONV_SEQUENCE, "activation": ACT_GELU, "post": POST_RAW}, layers
    if candidate_id == "CAND-P-104":
        body = numbered_layers(archive, "body")
        hidden = run_layers(x, archive, body, ACT_GELU)
        alpha = sigmoid(dense(hidden, archive, "alpha.weight", "alpha.bias"))
        classes = softmax(dense(hidden, archive, "cls.weight", "cls.bias"))
        return np.concatenate((alpha, classes)), {"kind": KIND_MULTIHEAD, "activation": ACT_GELU, "post": POST_FIRST_SIGMOID_REST_SOFTMAX}, body
    if candidate_id == "CAND-P-141":
        layers = numbered_layers(archive, "net")
        return x[-1:] + run_layers(x[:-1], archive, layers, ACT_GELU), {"kind": KIND_INPUT_PRIOR_RESIDUAL, "activation": ACT_GELU, "post": POST_RAW}, layers

    if any(name.startswith("body.") and name.endswith(".weight") for name in archive.files):
        layers = numbered_layers(archive, "body") + [("head.weight", "head.bias")]
    elif "linear.weight" in archive.files:
        layers = [("linear.weight", "linear.bias")]
        activation = ACT_LINEAR
    else:
        layers = numbered_layers(archive, "net")
    output = run_layers(x, archive, layers, activation)
    post = POST_RAW
    if candidate_id == "CAND-P-102":
        output[-1:] = sigmoid(output[-1:]); post = POST_LAST_SIGMOID
    elif candidate_id == "CAND-P-108":
        output = np.concatenate((sigmoid(output[:1]), softmax(output[1:]))); post = POST_FIRST_SIGMOID_REST_SOFTMAX
    elif candidate_id == "CAND-P-111":
        output[-1:] = sigmoid(output[-1:]); post = POST_FIRST_RAW_LAST_SIGMOID
    elif candidate_id == "CAND-P-112":
        output = softplus(output); post = POST_SOFTPLUS
    return output.astype(np.float32), {"kind": KIND_SEQUENCE, "activation": activation, "post": post}, layers


def tensor_pair(archive: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    quantized = np.ascontiguousarray(archive[name])
    scale_name = f"scale::{name}"
    if scale_name in archive.files:
        scale = np.ascontiguousarray(archive[scale_name].astype(np.float32))
    elif name in {"weight", "coefficient"} and "scale" in archive.files:
        scale = np.ascontiguousarray(archive["scale"].astype(np.float32))
    else:
        raise RuntimeError(f"SCALE_MISSING:{name}")
    return quantized, scale


def predictive_tensor_names(candidate_id: str, archive: Any, layers: list[tuple[str, str]]) -> list[str]:
    if candidate_id in {"CAND-P-045", "CAND-P-070", "CAND-P-138", "CAND-P-142", "CAND-P-145"}:
        return ["baseline_weight", "baseline_bias", "residual_scale", *(name for layer in layers for name in layer)]
    if candidate_id in {"CAND-P-057", "CAND-P-058"}:
        return ["weight"]
    if candidate_id == "CAND-P-059":
        return ["coefficient"]
    if candidate_id == "CAND-P-087":
        return ["skip.weight", "skip.bias", *(name for layer in layers for name in layer)]
    if candidate_id == "CAND-P-104":
        return [*(name for layer in layers for name in layer), "alpha.weight", "alpha.bias", "cls.weight", "cls.bias"]
    return [name for layer in layers for name in layer]


def nanolm_tensor_names(archive: Any, layers: int) -> list[str]:
    names = ["token_embedding.weight", "position_embedding.weight"]
    for layer in range(layers):
        prefix = f"blocks.{layer}."
        names.extend(prefix + suffix for suffix in (
            "norm1.weight", "norm1.bias", "qkv.weight", "qkv.bias", "proj.weight", "proj.bias",
            "norm2.weight", "norm2.bias", "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias",
        ))
    names.extend(("final_norm.weight", "final_norm.bias"))
    missing = [name for name in names if name not in archive.files]
    if missing:
        raise RuntimeError(f"NANOLM_TENSOR_MISSING:{missing[:2]}")
    return names


def build_runtime_payload(kind: int, activation: int, post: int, layers: int, input_elems: int, output_elems: int,
                          workspace_elems: int, aux: tuple[int, int, int, int], tensors: list[tuple[np.ndarray, np.ndarray]]) -> bytes:
    table_end = RUNTIME_HEADER_BYTES + len(tensors) * TENSOR_ENTRY_BYTES
    cursor = align(table_end)
    entries = []
    chunks: list[tuple[int, bytes]] = []
    for role, (array, scale) in enumerate(tensors, 1):
        array = np.ascontiguousarray(array)
        scale = np.ascontiguousarray(scale.astype(np.float32))
        dtype = {np.dtype("int8"): DTYPE_INT8, np.dtype("float32"): DTYPE_FLOAT32, np.dtype("uint8"): DTYPE_UINT8, np.dtype("uint16"): DTYPE_UINT16}.get(array.dtype)
        if dtype is None or array.ndim > 4:
            raise RuntimeError(f"TENSOR_DTYPE_OR_RANK:{array.dtype}:{array.shape}")
        data_offset = cursor
        data = array.tobytes(order="C")
        chunks.append((data_offset, data)); cursor = align(cursor + len(data))
        scale_offset = cursor
        scale_data = scale.tobytes(order="C")
        chunks.append((scale_offset, scale_data)); cursor = align(cursor + len(scale_data))
        dims = list(array.shape) + [1] * (4 - array.ndim)
        entries.append(struct.pack("<12I", role, dtype, array.ndim, 0, *dims, data_offset, len(data), scale_offset, scale.size))
    total = cursor
    header = struct.pack(
        "<4s15I", b"F2RT", 1, RUNTIME_HEADER_BYTES, kind, activation, layers, len(tensors), input_elems,
        output_elems, workspace_elems, post, aux[0], aux[1], aux[2], aux[3], total,
    )
    payload = bytearray(total)
    payload[:RUNTIME_HEADER_BYTES] = header
    for index, entry in enumerate(entries):
        start = RUNTIME_HEADER_BYTES + index * TENSOR_ENTRY_BYTES
        payload[start:start + TENSOR_ENTRY_BYTES] = entry
    for offset, data in chunks:
        payload[offset:offset + len(data)] = data
    return bytes(payload)


def build_golden(engine: int, kind: int, input_dtype: int, output_dtype: int, input_values: np.ndarray,
                 output_values: np.ndarray, prompt_length: int = 0, tolerance: float = 2e-3) -> bytes:
    input_values = np.ascontiguousarray(input_values)
    output_values = np.ascontiguousarray(output_values)
    tolerance_bits = struct.unpack("<I", struct.pack("<f", tolerance))[0]
    header = struct.pack(
        "<4s15I", b"F2GV", 1, 64, engine, kind, input_dtype, output_dtype, input_values.size,
        output_values.size, prompt_length, tolerance_bits, 0, 0, 0, 0, 0,
    )
    return header + input_values.tobytes(order="C") + output_values.tobytes(order="C")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    releases = (root / "releases").resolve()
    if releases not in output.parents or output == releases:
        raise RuntimeError("OUTPUT_SCOPE_GATE")
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")

    bank = root / "releases/forge200-host-modelbank-v7-20260803"
    catalog = json.loads((bank / "catalog_A.json").read_text(encoding="utf-8"))
    verification = json.loads((root / "evidence/host_artifact_verification.v7.json").read_text(encoding="utf-8"))
    onnx_by_id = {record["candidate_id"]: record["onnx"] for record in verification["records"]}
    runtime_contract = {
        "schema": "cimc.forge200.mcu-runtime-payload.v1",
        "payload_magic": "F2RT",
        "payload_header_bytes": 64,
        "tensor_entry_bytes": 48,
        "alignment": 32,
        "engines": [1, 2, 5],
        "kinds": list(range(1, 11)),
        "authority": 0,
    }
    runtime_root = sha256_bytes(canonical(runtime_contract))
    write_json(output / "MCU_RUNTIME_CONTRACT.v1.json", {**runtime_contract, "content_root_sha256": runtime_root})

    records = []
    for source in catalog["models"]:
        candidate_id = source["candidate_id"]
        source_package = bank / source["files"]["model.icmf"]["path"]
        raw = source_package.read_bytes()
        archive = np.load(io.BytesIO(raw[HEADER_BYTES:]), allow_pickle=False)
        destination = output / "packages" / candidate_id
        destination.mkdir(parents=True, exist_ok=True)
        engine = int(source["engine_id"])
        if engine == 5:
            number = int(candidate_id[-3:])
            if number <= 6:
                d_model, heads, layers, d_ff = 160, 5, 6, 320
            elif number <= 26:
                d_model, heads, layers, d_ff = 128, 4, 4, 256
            else:
                d_model, heads, layers, d_ff = 112, 4, 4, 224
            names = nanolm_tensor_names(archive, layers)
            tensors = [tensor_pair(archive, name) for name in names]
            runtime_workspace_elems = 2 * layers * 192 * d_model + 3 * d_model + d_ff + 2048
            payload = build_runtime_payload(
                KIND_NANOLM, ACT_RELU, POST_RAW, layers, 192, 24,
                runtime_workspace_elems,
                (d_model, heads, d_ff, 2048), tensors,
            )
            source_golden = np.load(bank / source["files"]["golden_vectors.npz"]["path"], allow_pickle=False)
            prompt = source_golden["prompt_tokens"][0].astype(np.uint16)
            nonzero = np.flatnonzero(prompt)
            prompt_length = int(nonzero[-1] + 1) if len(nonzero) else 1
            expected = source_golden["w8_generated"][0].astype(np.uint16)
            golden = build_golden(engine, KIND_NANOLM, DTYPE_UINT16, DTYPE_UINT16, prompt[:prompt_length], expected, prompt_length, 0.0)
            kind_info = {"kind": KIND_NANOLM, "activation": ACT_RELU, "post": POST_RAW}
            input_elems, output_elems = prompt_length, 24
        else:
            import onnx

            model = onnx.load(root / onnx_by_id[candidate_id]["path"], load_external_data=False)
            has_relu = any(node.op_type == "Relu" for node in model.graph.node)
            source_golden = np.load(bank / source["files"]["golden_vectors.npz"]["path"], allow_pickle=False)
            sample = source_golden["x"][0].astype(np.float32)
            expected, kind_info, layers_spec = infer_predictive(candidate_id, archive, sample, has_relu)
            names = predictive_tensor_names(candidate_id, archive, layers_spec)
            tensors = [tensor_pair(archive, name) for name in names]
            aux = (0, 0, 0, 0)
            if kind_info["kind"] == KIND_CONV_SEQUENCE:
                aux = (2, 64, 64, 1)
            runtime_workspace_elems = max(sample.size, expected.size, 2 * 24 * 64 * 64)
            payload = build_runtime_payload(
                kind_info["kind"], kind_info["activation"], kind_info["post"], len(layers_spec),
                sample.size, expected.size, runtime_workspace_elems, aux, tensors,
            )
            golden = build_golden(engine, kind_info["kind"], DTYPE_FLOAT32, DTYPE_FLOAT32, sample, expected, 0, 3e-3)
            input_elems, output_elems = sample.size, expected.size

        golden_path = destination / "golden.f2gv"
        golden_path.write_bytes(golden)
        golden_sha = sha256_file(golden_path)
        payload_sha = sha256_bytes(payload)
        output_schema_source = bank / source["files"]["output_schema.json"]["path"] if "output_schema.json" in source["files"] else None
        output_schema_sha = raw[172:204].hex()
        release_root = sha256_bytes(canonical({
            "candidate_id": candidate_id,
            "source_package_sha256": source["files"]["model.icmf"]["sha256"],
            "runtime_contract_sha256": runtime_root,
            "payload_sha256": payload_sha,
            "golden_sha256": golden_sha,
        }))
        outer = bytearray(HEADER_BYTES)
        kv_bytes = 2 * layers * 192 * d_model * 4 if kind_info["kind"] == KIND_NANOLM else 0
        arena_bytes = runtime_workspace_elems * 4 if engine != 5 else 512 * 1024
        struct.pack_into(
            "<4sHHHHBBHQQIII", outer, 0, b"ICMF", 1, HEADER_BYTES, engine, 1, 0, 1,
            len(tensors), 2, len(payload), 256 * 1024 if engine == 5 else 32 * 1024, arena_bytes, kv_bytes,
        )
        outer[44:76] = candidate_id.encode("utf-8")[:31].ljust(32, b"\0")
        outer[76:108] = bytes.fromhex(payload_sha)
        outer[108:140] = bytes.fromhex(golden_sha)
        outer[140:172] = bytes.fromhex(release_root)
        outer[172:204] = bytes.fromhex(output_schema_sha)
        package_path = destination / "model.icmf"
        package_path.write_bytes(bytes(outer) + payload)
        if output_schema_source:
            shutil.copy2(output_schema_source, destination / "output_schema.json")
        model_card_source = bank / source["files"]["model_card.md"]["path"] if "model_card.md" in source["files"] else None
        if model_card_source:
            shutil.copy2(model_card_source, destination / "model_card.md")
        records.append({
            "candidate_id": candidate_id,
            "category": source["category"],
            "tier": source["tier"],
            "engine_id": engine,
            "runtime_kind": kind_info["kind"],
            "input_elems": input_elems,
            "output_elems": output_elems,
            "tensor_count": len(tensors),
            "source_v7_package_sha256": source["files"]["model.icmf"]["sha256"],
            "package": {"path": package_path.relative_to(output).as_posix(), "bytes": package_path.stat().st_size, "sha256": sha256_file(package_path), "payload_sha256": payload_sha},
            "golden": {"path": golden_path.relative_to(output).as_posix(), "bytes": golden_path.stat().st_size, "sha256": golden_sha},
            "release_root": release_root,
            "runtime_contract_sha256": runtime_root,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        })

    records.sort(key=lambda item: item["candidate_id"])
    package_hashes = [item["package"]["sha256"] for item in records]
    payload_hashes = [item["package"]["payload_sha256"] for item in records]
    if len(records) != 170 or len(set(package_hashes)) != 170 or len(set(payload_hashes)) != 170:
        raise RuntimeError("MCU_EXPORT_UNIQUENESS_GATE")
    manifest = {
        "schema": "cimc.forge200.mcu-runtime-export.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "MCU_PARSEABLE_EXPORT_170_BUILT_C_RUNTIME_GOLDEN_PENDING",
        "source_modelbank": "releases/forge200-host-modelbank-v7-20260803",
        "runtime_contract_sha256": runtime_root,
        "model_count": 170,
        "exact_count": sum(item["tier"] == "EXACT_CONTRACT" for item in records),
        "sim_only_count": sum(item["tier"] == "SIM_ONLY_EXTENSION" for item in records),
        "by_category": {category: sum(item["category"] == category for item in records) for category in ("P", "G", "S")},
        "package_collisions": 0,
        "payload_collisions": 0,
        "records": records,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    manifest["content_root_sha256"] = sha256_bytes(canonical(records))
    write_json(output / "MANIFEST.v8.json", manifest)
    write_json(root / "evidence/mcu_runtime_export.v8.json", {
        **manifest,
        "manifest": {"path": (output / "MANIFEST.v8.json").relative_to(root).as_posix(), "sha256": sha256_file(output / "MANIFEST.v8.json")},
    })
    print(json.dumps({
        "status": manifest["status"], "models": 170, "by_category": manifest["by_category"],
        "exact": manifest["exact_count"], "sim_only": manifest["sim_only_count"],
        "content_root_sha256": manifest["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
