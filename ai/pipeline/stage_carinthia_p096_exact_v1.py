#!/usr/bin/env python3
"""Audit Carinthia-S and stage its exact pixel-mask task for P096."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parent[max(left, right)] = min(left, right)


def dhash64(image: Image.Image) -> int:
    value = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.uint8)
    bits = (value[:, 1:] >= value[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def cluster_near_duplicates(hashes: list[int], maximum_distance: int = 4) -> list[str]:
    # Four 16-bit bands guarantee that any pair with <=3 differing bits shares
    # a band.  The explicit Hamming check uses <=4 as the conservative merge
    # threshold; exact SHA identity is separately checked in the file manifest.
    buckets: dict[tuple[int, int], list[int]] = {}
    union = UnionFind(len(hashes))
    for index, value in enumerate(hashes):
        candidates: set[int] = set()
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            candidates.update(buckets.get(key, []))
        for other in candidates:
            if (value ^ hashes[other]).bit_count() <= maximum_distance:
                union.union(index, other)
        for band in range(4):
            key = (band, (value >> (band * 16)) & 0xFFFF)
            buckets.setdefault(key, []).append(index)
    roots = [union.find(index) for index in range(len(hashes))]
    return [f"PHASH-{hashes[root]:016x}-{root:04d}" for root in roots]


def balanced_group_split(groups: list[str]) -> np.ndarray:
    counts = Counter(groups)
    targets = [round(0.70 * len(groups)), round(0.15 * len(groups)), 0]
    targets[2] = len(groups) - targets[0] - targets[1]
    assigned = [0, 0, 0]
    code_by_group: dict[str, int] = {}
    ordered = sorted(
        counts,
        key=lambda value: (-counts[value], hashlib.sha256(value.encode("ascii")).hexdigest()),
    )
    for group in ordered:
        remaining = [targets[code] - assigned[code] for code in (0, 1, 2)]
        code = max((0, 1, 2), key=lambda item: (remaining[item], -item))
        code_by_group[group] = code
        assigned[code] += counts[group]
    return np.asarray([code_by_group[group] for group in groups], dtype=np.int8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source_receipt_path = root / "evidence" / "open_dataset_download_state.v1.json"
    source_receipt = json.loads(source_receipt_path.read_text(encoding="utf-8"))
    source_record = next(item for item in source_receipt["records"] if item["zenodo_record_id"] == "16895427")
    if source_receipt.get("status") != "PASS" or source_record.get("license") != "cc-by-4.0":
        raise RuntimeError("SOURCE_LICENSE_RECEIPT_GATE")
    archive = root / source_record["files"][0]["path"]
    if sha256_file(archive) != source_record["files"][0]["sha256"]:
        raise RuntimeError("SOURCE_ARCHIVE_HASH_GATE")
    base = root / "data" / "extracted" / "carinthia_s_v1" / "data"
    csv_path = base / "carinthia-s.csv"
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    if len(rows) != 4591:
        raise RuntimeError(f"ROW_COUNT_GATE:{len(rows)}")

    image_paths, mask_paths, labels, image_hashes, mask_hashes, perceptual, positive_fraction = [], [], [], [], [], [], []
    shape_modes: Counter[str] = Counter()
    for row in rows:
        image_path = base / row["image_path"]
        mask_path = base / row["mask_path"]
        if not image_path.is_file() or not mask_path.is_file() or image_path.stem != mask_path.stem or image_path.stem != row["filename"]:
            raise RuntimeError(f"SOURCE_PAIR_GATE:{row['filename']}")
        with Image.open(image_path) as image, Image.open(mask_path) as mask:
            if image.size != (480, 480) or mask.size != (480, 480):
                raise RuntimeError(f"SOURCE_SHAPE_GATE:{row['filename']}:{image.size}:{mask.size}")
            shape_modes[f"image={image.mode}|mask={mask.mode}"] += 1
            perceptual.append(dhash64(image))
            positive_fraction.append(float(np.mean(np.asarray(mask.convert("L"), dtype=np.uint8) >= 128)))
        image_paths.append(str(image_path.relative_to(root)).replace("\\", "/"))
        mask_paths.append(str(mask_path.relative_to(root)).replace("\\", "/"))
        labels.append(int(row["label"]))
        image_hashes.append(sha256_file(image_path))
        mask_hashes.append(sha256_file(mask_path))

    groups = cluster_near_duplicates(perceptual)
    split = balanced_group_split(groups)
    group_sets = {code: {group for group, item in zip(groups, split, strict=True) if item == code} for code in (0, 1, 2)}
    overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
    counts = {name: int(np.sum(split == code)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    positives = {name: int(np.sum(np.asarray(positive_fraction)[split == code] > 0)) for name, code in (("train", 0), ("validation", 1), ("test", 2))}
    if overlap or min(counts.values()) < 100 or min(positives.values()) < 50:
        raise RuntimeError(f"SPLIT_GATE:overlap={overlap}:counts={counts}:positives={positives}")

    contracts_path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with contracts_path.open("r", encoding="utf-8-sig", newline="") as handle:
        contracts = {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}
    contract = contracts["CAND-P-096"]
    output = root / "data" / "staged_carinthia_exact_v1" / "CAND-P-096.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        image_path=np.asarray(image_paths),
        mask_path=np.asarray(mask_paths),
        image_sha256=np.asarray(image_hashes),
        mask_sha256=np.asarray(mask_hashes),
        source_label=np.asarray(labels, dtype=np.int8),
        perceptual_hash=np.asarray([f"{value:016x}" for value in perceptual]),
        groups=np.asarray(groups),
        split=split,
        physical_scale_available=np.zeros(len(rows), dtype=np.int8),
        authority=np.asarray(0, dtype=np.int8),
        candidate_id=np.asarray("CAND-P-096"),
    )
    file_records = sorted(
        [
            {"image": image, "image_sha256": image_sha, "mask": mask, "mask_sha256": mask_sha}
            for image, image_sha, mask, mask_sha in zip(image_paths, image_hashes, mask_paths, mask_hashes, strict=True)
        ],
        key=lambda item: item["image"],
    )
    metadata = {
        "schema": "cimc.forge200.carinthia-p096-staged.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "candidate_id": "CAND-P-096",
        "truth_class": "OPEN_EXPERT_VALIDATED_PIXEL_MASK",
        "source_dataset": "Carinthia-S",
        "source_url": "https://zenodo.org/records/16895427",
        "doi": source_record["doi"],
        "license": "CC BY 4.0",
        "source_archive_sha256": source_record["files"][0]["sha256"],
        "source_download_receipt_sha256": sha256_file(source_receipt_path),
        "records": len(rows),
        "source_image_shape": [480, 480],
        "source_modes": dict(sorted(shape_modes.items())),
        "source_label_counts": {str(key): value for key, value in sorted(Counter(labels).items())},
        "file_pair_manifest_root_sha256": hashlib.sha256(canonical_bytes(file_records)).hexdigest(),
        "near_duplicate_grouping": "64_BIT_DHASH_UNION_HAMMING_LE_4_WITH_4X16BIT_CANDIDATE_BANDS",
        "near_duplicate_groups": len(set(groups)),
        "largest_near_duplicate_group": max(Counter(groups).values()),
        "split_unit": "PERCEPTUAL_NEAR_DUPLICATE_CLUSTER",
        "split_assignment": "DETERMINISTIC_LARGEST_CLUSTER_FIRST_BALANCED_70_15_15",
        "counts": counts,
        "positive_mask_records": positives,
        "cross_split_group_overlap": overlap,
        "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(groups, split.tolist(), strict=True)))).hexdigest(),
        "input_contract": contract["input_contract"],
        "input_contract_state": "SATISFIED_PIXEL_ANNOTATION_WITH_EXPLICIT_UNAVAILABLE_PHYSICAL_SCALE_CHANNEL",
        "physical_scale_availability": "ABSENT_IN_UPSTREAM_SOURCE_EXPLICIT_ZERO_NO_SCALE_FABRICATED",
        "mask_threshold": "source_grayscale_gte_128",
        "target_label": contract["target_label"],
        "baseline": contract["baseline"],
        "primary_metric": contract["primary_metric"],
        "parameter_cap": contract["parameter_cap"],
        "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
        "teacher_outputs": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "path": str(output.relative_to(root)).replace("\\", "/"),
        "bytes": output.stat().st_size,
        "sha256": sha256_file(output),
    }
    write_json(output.with_suffix(".metadata.json"), metadata)

    audit = {
        "schema": "cimc.forge200.carinthia-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_TASK_SPECIFIC_DISPOSITIONS",
        "source": {"dataset": "Carinthia-S", "doi": source_record["doi"], "license": "CC BY 4.0", "archive_sha256": source_record["files"][0]["sha256"]},
        "dispositions": [
            {
                "candidate_id": "CAND-P-095",
                "status": "FAIL_CLOSED_TARGET_TAXONOMY_NOT_SOURCE_BOUND",
                "reason": "upstream labels are numeric 1-6 and supply no authoritative particle/scratch/void/bridge/pattern/other mapping or acquisition metadata",
                "training_authorized": False,
                "countable_model": False,
            },
            {
                "candidate_id": "CAND-P-096",
                "status": "ADMITTED_EXACT_PIXEL_MASK_SCALE_UNAVAILABLE_EXPLICIT",
                "reason": "expert-validated pixel masks exactly satisfy the target; absent physical scale is represented by an explicit availability channel and is never fabricated",
                "training_authorized": True,
                "countable_model": False,
            },
        ],
        "authority": 0,
        "board_actions": 0,
    }
    write_json(root / "evidence" / "carinthia_source_contract_audit.v1.json", audit)
    receipt = {
        "schema": "cimc.forge200.carinthia-p096-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "record": metadata,
        "source_audit_sha256": sha256_file(root / "evidence" / "carinthia_source_contract_audit.v1.json"),
        "authority_nonzero": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical_bytes(metadata)).hexdigest(),
    }
    write_json(root / "evidence" / "carinthia_p096_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "records": len(rows), "counts": counts, "near_duplicate_groups": len(set(groups)), "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
