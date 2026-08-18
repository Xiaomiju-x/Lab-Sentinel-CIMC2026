#!/usr/bin/env python3
"""Independently verify every artifact admitted to host closure v4.

This is deliberately separate from the ModelBank builder.  It validates the
source exports and their recorded provenance; it does not make a board claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_root(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("content_root_sha256", None)
    encoded = json.dumps(
        clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest(directory: Path) -> tuple[int, int, int, int, bool]:
    path = directory / "artifact_manifest.json"
    if not path.is_file():
        return 0, 0, 0, 0, False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    checked_files = 0
    checked_bytes = 0
    omitted_cache_files = 0
    omitted_cache_bytes = 0
    for record in manifest.get("records", []):
        target = directory / record["path"]
        if not target.is_file():
            normalized = record["path"].replace("\\", "/")
            if normalized.startswith("train_seed_") and normalized.endswith(
                ("/best.pt", "/last.pt")
            ):
                omitted_cache_files += 1
                omitted_cache_bytes += int(record["bytes"])
                continue
            raise RuntimeError(f"DEPLOYMENT_MANIFEST_FILE_MISSING:{target}")
        size = target.stat().st_size
        if size != int(record["bytes"]):
            raise RuntimeError(f"MANIFEST_SIZE_MISMATCH:{target}")
        if sha256(target) != record["sha256"]:
            raise RuntimeError(f"MANIFEST_SHA_MISMATCH:{target}")
        checked_files += 1
        checked_bytes += size
    if checked_files == 0:
        raise RuntimeError(f"EMPTY_ARTIFACT_MANIFEST:{path}")
    if "bytes" in manifest and int(manifest["bytes"]) != (
        checked_bytes + omitted_cache_bytes
    ):
        raise RuntimeError(f"MANIFEST_TOTAL_BYTES_MISMATCH:{path}")
    expected_root = hashlib.sha256(
        json.dumps(
            manifest["records"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if manifest.get("content_root_sha256") != expected_root:
        raise RuntimeError(f"MANIFEST_CONTENT_ROOT_MISMATCH:{path}")
    return checked_files, checked_bytes, omitted_cache_files, omitted_cache_bytes, True


def verify_golden(path: Path) -> list[dict]:
    arrays: list[dict] = []
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise RuntimeError(f"EMPTY_GOLDEN:{path}")
        for name in sorted(archive.files):
            value = archive[name]
            if value.size == 0:
                raise RuntimeError(f"EMPTY_GOLDEN_ARRAY:{path}:{name}")
            if value.dtype.kind in "fc" and not np.isfinite(value).all():
                raise RuntimeError(f"NONFINITE_GOLDEN_ARRAY:{path}:{name}")
            arrays.append(
                {"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
            )
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--closure", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    closure_path = args.closure.resolve()
    closure = json.loads(closure_path.read_text(encoding="utf-8"))
    records = list(closure["exact_contract"]["records"]) + list(
        closure["sim_only_extensions"]["records"]
    )

    verified: list[dict] = []
    manifest_files = 0
    manifest_bytes = 0
    omitted_training_cache_files = 0
    omitted_training_cache_bytes = 0
    artifact_manifests_present = 0
    artifact_manifests_not_emitted = 0
    onnx_bytes = 0
    golden_arrays = 0
    for record in records:
        candidate_id = record["candidate_id"]
        receipt_path = root / record["promotion_receipt"]
        if sha256(receipt_path) != record["promotion_receipt_sha256"]:
            raise RuntimeError(f"RECEIPT_SHA_MISMATCH:{candidate_id}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        directory = receipt_path.parent

        onnx_path = directory / "fp32.onnx"
        if not onnx_path.is_file():
            raise RuntimeError(f"ONNX_MISSING:{candidate_id}")
        onnx_hash = sha256(onnx_path)
        if onnx_hash != receipt.get("onnx_sha256"):
            raise RuntimeError(f"ONNX_SHA_MISMATCH:{candidate_id}")
        onnx.checker.check_model(str(onnx_path), full_check=True)

        golden_path = root / record["golden"]["path"]
        if sha256(golden_path) != record["golden"]["sha256"]:
            raise RuntimeError(f"GOLDEN_SHA_MISMATCH:{candidate_id}")
        arrays = verify_golden(golden_path)

        files, byte_count, omitted_files, omitted_bytes, manifest_present = (
            verify_manifest(directory)
        )
        manifest_files += files
        manifest_bytes += byte_count
        omitted_training_cache_files += omitted_files
        omitted_training_cache_bytes += omitted_bytes
        artifact_manifests_present += int(manifest_present)
        artifact_manifests_not_emitted += int(not manifest_present)
        onnx_bytes += onnx_path.stat().st_size
        golden_arrays += len(arrays)
        verified.append(
            {
                "candidate_id": candidate_id,
                "onnx": {
                    "path": str(onnx_path.relative_to(root)).replace("\\", "/"),
                    "bytes": onnx_path.stat().st_size,
                    "sha256": onnx_hash,
                    "checker": "PASS_FULL_CHECK",
                },
                "golden": {"arrays": arrays, "status": "PASS"},
                "artifact_manifest": {
                    "files": files,
                    "bytes": byte_count,
                    "declared_training_cache_files_not_migrated": omitted_files,
                    "declared_training_cache_bytes_not_migrated": omitted_bytes,
                    "status": "PASS" if manifest_present else "NOT_EMITTED_LEGACY",
                },
                "authority": 0,
                "board_accepted": False,
            }
        )

    closure_version = closure.get("schema", "").rsplit(".", 1)[-1]
    result = {
        "schema": f"cimc.forge200.host-artifact-verification.{closure_version}",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_HOST_ONLY_BOARD_PENDING",
        "closure": {
            "path": str(closure_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256(closure_path),
            "content_root_sha256": closure["content_root_sha256"],
        },
        "model_count": len(verified),
        "onnx_full_check_pass": len(verified),
        "onnx_bytes_checked": onnx_bytes,
        "golden_archives_pass": len(verified),
        "golden_arrays_checked": golden_arrays,
        "manifest_files_checked": manifest_files,
        "manifest_bytes_checked": manifest_bytes,
        "artifact_manifests_present": artifact_manifests_present,
        "artifact_manifests_not_emitted_legacy": artifact_manifests_not_emitted,
        "declared_training_cache_files_not_migrated": omitted_training_cache_files,
        "declared_training_cache_bytes_not_migrated": omitted_training_cache_bytes,
        "authority_nonzero": 0,
        "board_accepted": 0,
        "records": verified,
        "claim_boundary": (
            "Host integrity/export evidence only; no GD32 latency, memory, "
            "shared-SPI electrical, or physical-process claim."
        ),
    }
    result["content_root_sha256"] = canonical_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_count": result["model_count"],
                "onnx_full_check_pass": result["onnx_full_check_pass"],
                "manifest_files_checked": manifest_files,
                "manifest_bytes_checked": manifest_bytes,
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
