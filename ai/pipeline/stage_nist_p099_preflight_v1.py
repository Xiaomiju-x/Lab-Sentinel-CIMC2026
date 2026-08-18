#!/usr/bin/env python3
"""Freeze and verify the source/split preflight for CAND-P-099 without touching test labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SHARED_FIELDS = (
    "Noise_level",
    "Contrast_level",
    "Mean_intensity",
    "Std_intensity",
    "Variance_intensity",
    "Michelson_contrast",
    "RMS_contrast",
    "Edge_density",
)


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


def canonical_number(value: str) -> str:
    numeric = float(value)
    if not numeric.is_integer():
        raise ValueError(f"expected integer-valued source parameter, got {value!r}")
    return str(int(numeric))


def load_table(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 567:
        raise ValueError(f"{path}: expected 567 records, got {len(rows)}")
    result = {row["IMAGE-NAME"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path}: duplicate IMAGE-NAME")
    return result


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-output", type=Path, required=True)
    args = parser.parse_args()

    root = args.root.resolve()
    binding_path = root / "contracts/nist_p099_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["candidate_id"] != "CAND-P-099" or binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("invalid P099 binding identity or authority")

    upstream = root / binding["upstream_code"]["local_path"]
    actual_commit = git_head(upstream)
    if actual_commit != binding["upstream_code"]["commit"]:
        raise ValueError(f"upstream commit mismatch: {actual_commit}")

    tables: dict[str, dict[str, dict[str, str]]] = {}
    table_audit = []
    for item in binding["official_merged_tables"]:
        path = root / item["path"]
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"] or item["records"] != 567:
            raise ValueError(f"source table mismatch: {item['observer']}")
        tables[item["observer"]] = load_table(path)
        table_audit.append({
            "observer": item["observer"],
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "records": 567,
        })

    observers = sorted(tables)
    image_names = set(tables[observers[0]])
    if any(set(tables[name]) != image_names for name in observers[1:]):
        raise ValueError("observer IMAGE-NAME sets differ")

    source_rows = tables[observers[0]]
    for image_name in image_names:
        reference = source_rows[image_name]
        for observer in observers[1:]:
            candidate = tables[observer][image_name]
            for field in SHARED_FIELDS:
                if candidate[field] != reference[field]:
                    raise ValueError(f"shared field mismatch: {image_name} {observer} {field}")

    noise_values = sorted({canonical_number(row["Noise_level"]) for row in source_rows.values()}, key=int)
    contrast_values = sorted({canonical_number(row["Contrast_level"]) for row in source_rows.values()}, key=int)
    if len(noise_values) != 27 or len(contrast_values) != 21:
        raise ValueError("unexpected NIST noise/contrast grid")
    if len(image_names) != len(noise_values) * len(contrast_values):
        raise ValueError("NIST source grid is not a complete 27x21 product")

    ranked_noise = sorted(
        noise_values,
        key=lambda value: hashlib.sha256(
            f"forge200-p099-nist-v1|noise={value}".encode("ascii")
        ).hexdigest(),
    )
    group_split = {
        **{value: "train" for value in ranked_noise[:19]},
        **{value: "validation" for value in ranked_noise[19:23]},
        **{value: "test" for value in ranked_noise[23:]},
    }
    records = []
    for image_name in sorted(image_names):
        row = source_rows[image_name]
        noise = canonical_number(row["Noise_level"])
        records.append({
            "record_id": image_name,
            "noise_group": noise,
            "contrast_level": canonical_number(row["Contrast_level"]),
            "split": group_split[noise],
        })
    record_counts = Counter(row["split"] for row in records)
    group_counts = Counter(group_split.values())
    expected_records = binding["split_binding"]["expected_record_counts"]
    expected_groups = binding["split_binding"]["expected_group_counts"]
    if dict(record_counts) != expected_records or dict(group_counts) != expected_groups:
        raise ValueError(f"split count mismatch: records={record_counts}, groups={group_counts}")

    split_document = {
        "schema": "cimc.forge200.nist-p099-split-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "CAND-P-099",
        "status": "FROZEN_BEFORE_TRAINING_TEST_LABELS_NOT_EVALUATED",
        "group_key": "Noise_level",
        "assignment_salt": "forge200-p099-nist-v1",
        "groups": {
            split: sorted([value for value, assigned in group_split.items() if assigned == split], key=int)
            for split in ("train", "validation", "test")
        },
        "group_counts": dict(group_counts),
        "record_counts": dict(record_counts),
        "cross_split_group_overlap": 0,
        "records": records,
        "test_labels_read_for_split_selection": False,
    }
    split_document["content_root_sha256"] = content_root(split_document)

    raw_dir = root / "data/sources/nist_sem_detection_limits_v1/raw"
    downloads = []
    required_ready = True
    for item in binding["download_contract"]:
        path = raw_dir / item["name"]
        present = path.is_file()
        actual_sha = sha256_file(path) if present else None
        hash_match = present and actual_sha == item["sha256"]
        if item["required_for_training"] and not hash_match:
            required_ready = False
        downloads.append({
            "name": item["name"],
            "url": item["url"],
            "expected_sha256": item["sha256"],
            "required_for_training": item["required_for_training"],
            "present": present,
            "bytes": path.stat().st_size if present else 0,
            "actual_sha256": actual_sha,
            "hash_match": hash_match,
        })

    receipt = {
        "schema": "cimc.forge200.nist-p099-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_id": "CAND-P-099",
        "status": "PREFLIGHT_PASS_REQUIRED_RAW_IMAGE_DOWNLOAD_PENDING" if not required_ready else "SOURCE_AND_SPLIT_READY_FOR_TRAINING",
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "upstream_code": {
            "path": binding["upstream_code"]["local_path"],
            "commit": actual_commit,
        },
        "official_tables": table_audit,
        "source_grid": {
            "records": 567,
            "noise_groups": 27,
            "contrast_levels": 21,
            "complete_cartesian_grid": True,
            "observer_image_set_overlap": 567,
            "shared_input_field_mismatches": 0,
        },
        "split_manifest": {
            "path": args.split_output.resolve().relative_to(root).as_posix(),
            "content_root_sha256": split_document["content_root_sha256"],
            "test_labels_evaluated": False,
        },
        "downloads": downloads,
        "required_training_payloads_ready": required_ready,
        "training_actions": 0,
        "test_evaluation_actions": 0,
        "host_promoted": False,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Verify intensity_sets.zip and mask_sets.zip hashes, extract no-reference features, select architecture and Fourier threshold on validation only, then evaluate the frozen test once.",
    }
    receipt["content_root_sha256"] = content_root(receipt)

    args.split_output.parent.mkdir(parents=True, exist_ok=True)
    args.split_output.write_text(json.dumps(split_document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "records": 567,
        "record_counts": dict(record_counts),
        "required_training_payloads_ready": required_ready,
        "content_root_sha256": receipt["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
