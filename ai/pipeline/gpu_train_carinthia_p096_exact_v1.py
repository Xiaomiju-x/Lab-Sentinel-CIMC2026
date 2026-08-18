#!/usr/bin/env python3
"""Train the exact Carinthia-S pixel-mask candidate P096 on a local GPU."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from gpu_train_job import SEEDS, build_package, canonical_bytes, heartbeat, sha256_file, write_json


PARAMETER_CAP = 88_000
WEIGHT_BYTE_CAP = 104 * 1024
IMAGE_SIZE = 64


def quantize_state(state: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    quantized, scales = {}, {}
    for name, tensor in state.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        if array.ndim >= 2:
            axes = tuple(range(1, array.ndim))
            scale = np.maximum(np.max(np.abs(array), axis=axes, keepdims=True), 1e-12) / 127.0
        else:
            scale = np.asarray(max(float(np.max(np.abs(array))), 1e-12) / 127.0, dtype=np.float32)
        quantized[name] = np.clip(np.rint(array / scale), -127, 127).astype(np.int8)
        scales[name] = np.asarray(scale, dtype=np.float32)
    return quantized, scales


def dequantized_state(torch: Any, quantized: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> dict[str, Any]:
    return {name: torch.from_numpy(value.astype(np.float32) * scales[name]) for name, value in quantized.items()}


def records_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "heartbeat.json"}:
            records.append({"path": str(path.relative_to(root)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"schema": "cimc.forge200.artifact-manifest.v2", "records": records, "content_root_sha256": hashlib.sha256(canonical_bytes(records)).hexdigest()}


def build_cache(root: Path, staged: Any, metadata: dict[str, Any], cache_path: Path) -> dict[str, Any]:
    receipt_path = cache_path.with_suffix(".receipt.json")
    if cache_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if receipt.get("staged_sha256") == metadata["sha256"] and sha256_file(cache_path) == receipt.get("sha256"):
            return receipt
    images = np.empty((len(staged["image_path"]), IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    masks = np.empty_like(images)
    for index, (image_value, mask_value) in enumerate(zip(staged["image_path"].astype(str), staged["mask_path"].astype(str), strict=True)):
        with Image.open(root / image_value) as image:
            images[index] = np.asarray(image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
        with Image.open(root / mask_value) as mask:
            masks[index] = np.asarray(mask.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.uint8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, image=images, mask=(masks >= 128).astype(np.uint8))
    receipt = {
        "schema": "cimc.forge200.carinthia-resize-cache.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "staged_sha256": metadata["sha256"],
        "records": len(images),
        "shape": [IMAGE_SIZE, IMAGE_SIZE],
        "interpolation": {"image": "PIL_BILINEAR", "mask": "PIL_NEAREST_THEN_GE_128"},
        "path": str(cache_path.relative_to(root)).replace("\\", "/"),
        "bytes": cache_path.stat().st_size,
        "sha256": sha256_file(cache_path),
        "authority": 0,
    }
    write_json(receipt_path, receipt)
    return receipt


def segmentation_metrics(truth: np.ndarray, prediction: np.ndarray, small_area_threshold: int) -> dict[str, float | int]:
    truth = truth.astype(bool)
    prediction = prediction.astype(bool)
    intersection = np.sum(truth & prediction, axis=(1, 2))
    union = np.sum(truth | prediction, axis=(1, 2))
    iou = np.where(union > 0, intersection / np.maximum(union, 1), 1.0)
    boundary_f1 = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for gt, pred in zip(truth, prediction, strict=True):
        gt_u8, pred_u8 = gt.astype(np.uint8), pred.astype(np.uint8)
        gt_boundary = gt_u8 ^ cv2.erode(gt_u8, kernel, iterations=1)
        pred_boundary = pred_u8 ^ cv2.erode(pred_u8, kernel, iterations=1)
        gt_count, pred_count = int(gt_boundary.sum()), int(pred_boundary.sum())
        if gt_count == 0 and pred_count == 0:
            boundary_f1.append(1.0)
            continue
        gt_tolerance = cv2.dilate(gt_boundary, kernel, iterations=1)
        pred_tolerance = cv2.dilate(pred_boundary, kernel, iterations=1)
        precision = float(np.sum((pred_boundary > 0) & (gt_tolerance > 0))) / max(pred_count, 1)
        recall = float(np.sum((gt_boundary > 0) & (pred_tolerance > 0))) / max(gt_count, 1)
        boundary_f1.append(2.0 * precision * recall / max(precision + recall, 1e-12))
    area = np.sum(truth, axis=(1, 2))
    small = (area > 0) & (area <= small_area_threshold)
    recall_per_image = intersection[small] / np.maximum(area[small], 1)
    small_recall = float(np.mean(recall_per_image)) if np.any(small) else 0.0
    result = {
        "mIoU": float(np.mean(iou)),
        "boundary_F1_tolerance_1px": float(np.mean(boundary_f1)),
        "small_defect_recall": small_recall,
        "small_defect_area_threshold_pixels_train_q25": int(small_area_threshold),
        "small_defect_records": int(np.sum(small)),
    }
    result["primary_composite"] = float(np.mean([result["mIoU"], result["boundary_F1_tolerance_1px"], result["small_defect_recall"]]))
    return result


def adaptive_watershed(image: np.ndarray, inverse: bool, block_size: int, constant: int) -> np.ndarray:
    threshold_type = cv2.THRESH_BINARY_INV if inverse else cv2.THRESH_BINARY
    binary = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, constant)
    kernel = np.ones((3, 3), dtype=np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    distance = cv2.distanceTransform(opened, cv2.DIST_L2, 3)
    if float(distance.max()) <= 0:
        return np.zeros_like(image, dtype=np.uint8)
    _, sure_foreground = cv2.threshold(distance, 0.35 * float(distance.max()), 255, 0)
    sure_foreground = sure_foreground.astype(np.uint8)
    sure_background = cv2.dilate(opened, kernel, iterations=1)
    unknown = cv2.subtract(sure_background, sure_foreground)
    _, markers = cv2.connectedComponents(sure_foreground)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), markers)
    return (markers > 1).astype(np.uint8)


def select_and_run_baseline(images: np.ndarray, masks: np.ndarray, split: np.ndarray, small_threshold: int) -> tuple[np.ndarray, dict[str, Any]]:
    train_indices = np.flatnonzero(split == 0)
    # Freeze parameters using a deterministic train-only calibration subset.
    calibration = train_indices[
        np.argsort([hashlib.sha256(str(int(index)).encode()).hexdigest() for index in train_indices])[:400]
    ]
    best = None
    for inverse in (False, True):
        for block in (7, 11, 15):
            for constant in (-3, 2, 7):
                predictions = np.asarray([adaptive_watershed(images[index], inverse, block, constant) for index in calibration])
                score = segmentation_metrics(masks[calibration], predictions, small_threshold)["primary_composite"]
                item = (float(score), inverse, block, constant)
                if best is None or item[0] > best[0]:
                    best = item
    assert best is not None
    prediction = np.asarray([adaptive_watershed(image, best[1], best[2], best[3]) for image in images])
    return prediction, {
        "kind": "adaptive_threshold_plus_watershed",
        "fit_split": "train_only_deterministic_400_record_calibration_subset",
        "inverse": best[1],
        "block_size": best[2],
        "constant": best[3],
        "calibration_composite": best[0],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from torch import nn

    root = args.root.resolve()
    candidate_id = "CAND-P-096"
    dataset = root / "data" / "staged_carinthia_exact_v1" / f"{candidate_id}.npz"
    metadata_path = dataset.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("status") != "PASS" or metadata.get("authority") != 0 or metadata.get("cross_split_group_overlap") != 0:
        raise RuntimeError("DATA_GATE")
    if sha256_file(dataset) != metadata["sha256"]:
        raise RuntimeError("DATA_HASH_GATE")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED")
    device = torch.device(args.device)
    props = torch.cuda.get_device_properties(device)
    output = (args.artifact_root / candidate_id).resolve()
    output.mkdir(parents=True, exist_ok=True)
    heartbeat_path = output / "heartbeat.json"
    staged = np.load(dataset, allow_pickle=False)
    cache_path = root / "data" / "staged_carinthia_exact_v1" / f"cache_{IMAGE_SIZE}x{IMAGE_SIZE}.npz"
    cache_receipt = build_cache(root, staged, metadata, cache_path)
    cache = np.load(cache_path, allow_pickle=False)
    images_u8 = cache["image"].astype(np.uint8)
    masks = cache["mask"].astype(np.uint8)
    split = staged["split"].astype(np.int8)
    train, validation, test = (np.flatnonzero(split == code) for code in (0, 1, 2))
    positive_train_areas = masks[train].sum(axis=(1, 2))
    small_threshold = max(int(np.quantile(positive_train_areas[positive_train_areas > 0], 0.25)), 1)
    baseline_prediction, baseline_info = select_and_run_baseline(images_u8, masks, split, small_threshold)
    baseline_validation = segmentation_metrics(masks[validation], baseline_prediction[validation], small_threshold)
    baseline_test = segmentation_metrics(masks[test], baseline_prediction[test], small_threshold)

    train_mean = float(images_u8[train].mean() / 255.0)
    train_std = max(float(images_u8[train].std() / 255.0), 1e-3)
    image_float = ((images_u8.astype(np.float32) / 255.0 - train_mean) / train_std).astype(np.float32)
    scale_channel = np.zeros_like(image_float, dtype=np.float32)
    x = np.stack((image_float, scale_channel), axis=1)
    y = masks[:, None].astype(np.float32)

    class TinySegNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers = [nn.Conv2d(2, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 24, 3, padding=1), nn.GELU()]
            for _ in range(4):
                layers.extend((nn.Conv2d(24, 24, 3, padding=1), nn.GELU()))
            layers.extend((nn.Conv2d(24, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 1, 1)))
            self.net = nn.Sequential(*layers)

        def forward(self, value: Any) -> Any:
            return self.net(value)

    parameter_count = sum(parameter.numel() for parameter in TinySegNet().parameters())
    if parameter_count > PARAMETER_CAP:
        raise RuntimeError(f"PARAMETER_CAP:{parameter_count}")

    def logits_for(model: Any, selected: np.ndarray) -> np.ndarray:
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(selected), args.eval_batch_size):
                batch = torch.from_numpy(x[selected[start : start + args.eval_batch_size]]).to(device)
                values.append(model(batch).cpu().numpy()[:, 0])
        return np.concatenate(values)

    reports, states, thresholds = [], {}, {}
    started = time.perf_counter()
    positive = float(y[train].sum())
    negative = float(y[train].size - positive)
    pos_weight = min(max((negative / max(positive, 1.0)) ** 0.5, 1.0), 12.0)
    for seed in SEEDS:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        model = TinySegNet().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=2e-4)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        checkpoint = output / f"train_seed_{seed}" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        best_loss, patience = float("inf"), 0
        for epoch in range(args.max_epochs):
            model.train()
            order = train[torch.randperm(len(train), generator=generator).numpy()]
            for start in range(0, len(order), args.batch_size):
                selected = order[start : start + args.batch_size]
                batch_x = torch.from_numpy(x[selected]).to(device)
                batch_y = torch.from_numpy(y[selected]).to(device)
                if torch.rand((), generator=generator).item() < 0.5:
                    batch_x, batch_y = batch_x.flip(-1), batch_y.flip(-1)
                if torch.rand((), generator=generator).item() < 0.5:
                    batch_x, batch_y = batch_x.flip(-2), batch_y.flip(-2)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_x)
                probability = torch.sigmoid(logits)
                intersection = (probability * batch_y).sum(dim=(1, 2, 3))
                dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (probability.sum(dim=(1, 2, 3)) + batch_y.sum(dim=(1, 2, 3)) + 1.0)).mean()
                loss = 0.55 * bce(logits, batch_y) + 0.45 * dice_loss
                if not torch.isfinite(loss):
                    raise RuntimeError("NAN_LOSS")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            validation_logits = logits_for(model, validation)
            validation_probability = 1.0 / (1.0 + np.exp(-np.clip(validation_logits, -30, 30)))
            val_intersection = np.sum(validation_probability * masks[validation], axis=(1, 2))
            val_soft_dice_loss = float(1.0 - np.mean((2.0 * val_intersection + 1.0) / (np.sum(validation_probability, axis=(1, 2)) + np.sum(masks[validation], axis=(1, 2)) + 1.0)))
            if val_soft_dice_loss < best_loss - 1e-5:
                best_loss, patience = val_soft_dice_loss, 0
                torch.save(model.state_dict(), checkpoint)
            else:
                patience += 1
            heartbeat(heartbeat_path, candidate_id, "TRAIN_CARINTHIA_EXACT", seed, epoch)
            if epoch + 1 >= args.min_epochs and patience >= args.early_stop_patience:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        validation_probability = 1.0 / (1.0 + np.exp(-np.clip(logits_for(model, validation), -30, 30)))
        threshold_results = []
        for threshold in np.linspace(0.20, 0.75, 12):
            item = segmentation_metrics(masks[validation], validation_probability >= threshold, small_threshold)
            threshold_results.append((item["primary_composite"], float(threshold), item))
        selected_threshold = max(threshold_results, key=lambda item: item[0])[1]
        test_probability = 1.0 / (1.0 + np.exp(-np.clip(logits_for(model, test), -30, 30)))
        test_metrics = segmentation_metrics(masks[test], test_probability >= selected_threshold, small_threshold)
        reports.append(
            {
                "seed": seed,
                "epochs": epoch + 1,
                "validation_soft_dice_loss": best_loss,
                "threshold_selected_on_validation": selected_threshold,
                "test": test_metrics,
                "beats_baseline": test_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4,
            }
        )
        states[seed] = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        thresholds[seed] = selected_threshold

    composites = np.asarray([item["test"]["primary_composite"] for item in reports], dtype=np.float64)
    mean_composite = float(composites.mean())
    aggregate_pass = mean_composite > float(baseline_test["primary_composite"]) + 1e-4
    best_report = min(reports, key=lambda item: item["validation_soft_dice_loss"])
    best_seed = int(best_report["seed"])
    selected_threshold = thresholds[best_seed]
    model = TinySegNet().to(device)
    model.load_state_dict(states[best_seed])
    fp_probability = 1.0 / (1.0 + np.exp(-np.clip(logits_for(model, test), -30, 30)))
    quantized, scales = quantize_state(states[best_seed])
    payload_buffer = io.BytesIO()
    np.savez_compressed(payload_buffer, **quantized, **{f"scale::{key}": value for key, value in scales.items()})
    payload = payload_buffer.getvalue()
    if len(payload) > WEIGHT_BYTE_CAP:
        raise RuntimeError(f"W8_PAYLOAD_CAP:{len(payload)}")
    model.load_state_dict(dequantized_state(torch, quantized, scales))
    quant_probability = 1.0 / (1.0 + np.exp(-np.clip(logits_for(model, test), -30, 30)))
    quant_metrics = segmentation_metrics(masks[test], quant_probability >= selected_threshold, small_threshold)
    quant_delta = float(best_report["test"]["primary_composite"] - quant_metrics["primary_composite"])
    quant_pass = quant_metrics["primary_composite"] > baseline_test["primary_composite"] + 1e-4 and quant_delta <= 0.03
    status_pass = aggregate_pass and quant_pass

    golden_path = output / "golden_vectors.npz"
    np.savez_compressed(
        golden_path,
        x=x[test[:16]],
        y=masks[test[:16]],
        fp32_probability=fp_probability[:16],
        quantized_probability=quant_probability[:16],
        threshold=np.asarray(selected_threshold, dtype=np.float32),
    )
    model.load_state_dict(states[best_seed])
    model.eval()
    onnx_path = output / "fp32.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(x[test[:1]]).to(device),
        onnx_path,
        input_names=["sem_image_and_scale_availability"],
        output_names=["defect_mask_logits"],
        dynamic_axes={"sem_image_and_scale_availability": {0: "batch"}, "defect_mask_logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(onnx_path))
    output_schema = {"shape": [None, 1, IMAGE_SIZE, IMAGE_SIZE], "activation": "sigmoid", "threshold": selected_threshold, "authority": 0}
    release_root = hashlib.sha256(
        canonical_bytes(
            {
                "candidate_id": candidate_id,
                "dataset_sha256": metadata["sha256"],
                "cache_sha256": cache_receipt["sha256"],
                "task_contract_sha256": metadata["task_contract_sha256"],
                "onnx_sha256": sha256_file(onnx_path),
                "golden_sha256": sha256_file(golden_path),
                "three_seed_mean_composite": mean_composite,
            }
        )
    ).hexdigest()
    package = build_package(output, candidate_id, payload, sha256_file(golden_path), release_root, hashlib.sha256(canonical_bytes(output_schema)).hexdigest(), engine_id=2)
    evaluation = {
        "schema": "cimc.forge200.carinthia-p096-contract-exact-evaluation.v1",
        "status": "PASS" if status_pass else "FAIL_CLOSED",
        "candidate_id": candidate_id,
        "baseline_contract": metadata["baseline"],
        "primary_metric_contract": metadata["primary_metric"],
        "baseline": {**baseline_info, "validation": baseline_validation, "test": baseline_test},
        "seed_reports": reports,
        "g3_aggregate_mean_gate": aggregate_pass,
        "g4": {"mean_composite": mean_composite, "variance_composite": float(composites.var()), "worst_composite": float(composites.min())},
        "quantized_best_seed": {"seed": best_seed, "threshold": selected_threshold, "test": quant_metrics, "metric_delta": quant_delta, "gate": quant_pass},
        "authority": 0,
        "board_accepted": False,
    }
    write_json(output / "eval_contract_exact.json", evaluation)
    write_json(output / "eval_grouped.json", evaluation)
    write_json(output / "baseline_report.json", evaluation["baseline"])
    write_json(output / "source_manifest.json", metadata)
    write_json(output / "preprocessing_train_only.json", {"image_mean": train_mean, "image_std": train_std, "physical_scale_available_channel": 0})
    write_json(output / "quantization_parity.json", {"scheme": "W8_per_output_channel", "primary_composite_delta": quant_delta, "gate": quant_pass})
    status = "HOST_GPU_TRAINED_CONTRACT_BASELINE_PASS_BOARD_PENDING" if status_pass else "HOST_GPU_REJECTED_CONTRACT_BASELINE"
    receipt = {
        "schema": "cimc.forge200.promotion-receipt.v3",
        "status": status,
        "candidate_id": candidate_id,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "three_seed_count": 3,
        "g3_aggregate_mean_gate": aggregate_pass,
        "parameter_count": parameter_count,
        "parameter_cap": PARAMETER_CAP,
        "w8_payload_bytes": len(payload),
        "w8_payload_byte_cap": WEIGHT_BYTE_CAP,
        "best_seed_by_validation": best_seed,
        "release_root": release_root,
        "package": package,
        "onnx_sha256": sha256_file(onnx_path),
        "golden_sha256": sha256_file(golden_path),
        "runtime_seconds": time.perf_counter() - started,
        "gpu": {"name": props.name, "vram_gib": props.total_memory / 1024**3},
    }
    write_json(output / "promotion_receipt.json", receipt)
    (output / "model_card.md").write_text(
        f"# {candidate_id} Carinthia-S segmentation model card\n\n"
        f"- Status: `{status}`\n"
        f"- Three-seed mean composite: `{mean_composite:.6f}`; baseline: `{baseline_test['primary_composite']:.6f}`.\n"
        f"- Parameters: `{parameter_count}`; W8 payload: `{len(payload)}` bytes.\n"
        "- Physical scale is absent upstream and represented by an explicit zero availability channel; no scale was fabricated.\n"
        "- Authority: `0`; unified GD32 board evidence remains pending.\n",
        encoding="utf-8",
    )
    write_json(output / "artifact_manifest.json", records_manifest(output))
    heartbeat(heartbeat_path, candidate_id, "COMPLETE")
    print(json.dumps({"candidate_id": candidate_id, "status": status, "mean_composite": mean_composite, "baseline_composite": baseline_test["primary_composite"], "runtime_seconds": receipt["runtime_seconds"]}, sort_keys=True))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=16)
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        output = args.artifact_root.resolve() / "CAND-P-096"
        write_json(output / "failure.json", {"schema": "cimc.forge200.job-failure.v3", "status": "FAIL_CLOSED", "candidate_id": "CAND-P-096", "authority": 0, "error_type": type(exc).__name__, "error": str(exc), "utc": datetime.now(timezone.utc).isoformat()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
