#!/usr/bin/env python3
"""Verify the Mendeley SEM source and retain a fail-closed P095 disposition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def matlab_v5_inflated_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if not data.startswith(b"MATLAB 5.0 MAT-file"):
        raise ValueError("unexpected MATLAB file header")
    position = 128
    inflated: list[bytes] = []
    while position + 8 <= len(data):
        element_type, length = struct.unpack_from("<II", data, position)
        if element_type != 15 or position + 8 + length > len(data):
            raise ValueError(f"unexpected top-level MATLAB element at {position}")
        inflated.append(zlib.decompress(data[position + 8 : position + 8 + length]))
        # MATLAB's miCOMPRESSED top-level elements in this file are contiguous,
        # without the optional 8-byte padding accepted for ordinary elements.
        position += 8 + length
    if position != len(data):
        raise ValueError("MATLAB file has unparsed trailing bytes")
    return b"".join(inflated)


def printable_strings(payload: bytes) -> list[str]:
    ascii_strings = [item.decode("ascii") for item in re.findall(rb"[ -~]{4,}", payload)]
    utf16_strings = [item.decode("utf-16le") for item in re.findall(rb"(?:[ -~]\x00){4,}", payload)]
    return ascii_strings + utf16_strings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    binding_path = root / "contracts/mendeley_p095_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["candidate_id"] != "CAND-P-095" or binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("invalid P095 identity or authority")

    verified_artifacts = {}
    for name, item in binding["artifacts"].items():
        path = root / item["path"]
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"]:
            raise ValueError(f"{name} SHA-256 mismatch: {actual_sha}")
        if path.stat().st_size != item["bytes"]:
            raise ValueError(f"{name} byte count mismatch")
        verified_artifacts[name] = {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "verified": True,
        }

    page = (root / binding["artifacts"]["landing_page"]["path"]).read_text(encoding="utf-8")
    for required in (binding["source"]["doi"], binding["source"]["license"], "78 SEM microstructure images"):
        if required not in page:
            raise ValueError(f"landing-page evidence missing: {required}")

    api_files = json.loads((root / binding["artifacts"]["public_api_file_manifest"]["path"]).read_text(encoding="utf-8"))
    filenames = [item["filename"] for item in api_files]
    raster_suffixes = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    raster_files = [name for name in filenames if Path(name).suffix.lower() in raster_suffixes]
    if len(api_files) != 3 or raster_files:
        raise ValueError("official API file inventory changed; re-audit required")
    for item in api_files:
        local_path = root / "data/raw/mendeley_sem_gtcb8j5gcb_v1" / item["filename"]
        if sha256_file(local_path) != item["content_details"]["sha256_hash"]:
            raise ValueError(f"official manifest hash mismatch: {item['filename']}")

    mat_path = root / binding["artifacts"]["matlab_labeling_session"]["path"]
    inflated = matlab_v5_inflated_payload(mat_path)
    strings = printable_strings(inflated)
    tiff_paths = sorted({value for value in strings if re.match(r"^[A-Za-z]:\\.*\.tiff?$", value, re.IGNORECASE)})
    image_paths = [value for value in tiff_paths if PureWindowsPath(value).name.lower() != "imagelabelingsessionsem.mat"]
    observed_labels = sorted({
        label
        for label in binding["observed_source"]["source_defect_taxonomy"]
        if label in strings or label.replace("_", " ") in strings
    })
    expected_labels = sorted(binding["observed_source"]["source_defect_taxonomy"])
    if len(image_paths) != binding["observed_source"]["matlab_session_external_tiff_paths"]:
        raise ValueError(f"unexpected external TIFF path count: {len(image_paths)}")
    if observed_labels != expected_labels:
        raise ValueError(f"unexpected MATLAB label taxonomy: {observed_labels}")
    if "ImageFilenames" not in strings:
        raise ValueError("MATLAB session no longer exposes ImageFilenames")

    inventory = json.loads((root / binding["artifacts"]["detection_workbook_inventory"]["path"]).read_text(encoding="utf-8"))
    ground_truth = next(sheet for sheet in inventory["sheets"] if sheet["name"] == "Ground truth table")
    if ground_truth["rowCount"] != binding["observed_source"]["ground_truth_workbook_rows_including_headers"]:
        raise ValueError("unexpected ground-truth workbook row count")

    receipt = {
        "schema": "cimc.forge200.mendeley-p095-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "CAND-P-095",
        "status": "SOURCE_LICENSE_AND_BOX_LABELS_VERIFIED_EXACT_INPUT_AND_TAXONOMY_REJECTED",
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "source": binding["source"],
        "verified_artifacts": verified_artifacts,
        "observed": {
            **binding["observed_source"],
            "public_api_filenames": filenames,
            "external_tiff_path_count_verified": len(image_paths),
            "external_tiff_filename_examples": [PureWindowsPath(path).name for path in image_paths[:5]],
            "source_defect_taxonomy_verified": expected_labels,
            "downloaded_raster_image_count": 0,
            "leakage_safe_split_materialized": False,
        },
        "exact_rejection": binding["exact_rejection"],
        "training_actions": 0,
        "test_evaluation_actions": 0,
        "host_promoted": False,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Seek a licensed source that includes immutable SEM crop pixels, physical scale/acquisition metadata, the exact frozen six-class taxonomy, and group identifiers; do not infer a class mapping from these seven labels.",
    }
    receipt["content_root_sha256"] = content_root(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "api_files": len(api_files),
        "raster_files": len(raster_files),
        "external_tiff_paths": len(image_paths),
        "content_root_sha256": receipt["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
