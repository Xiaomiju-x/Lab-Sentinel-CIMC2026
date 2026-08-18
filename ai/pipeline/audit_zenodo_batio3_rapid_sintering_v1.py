#!/usr/bin/env python3
"""Verify Zenodo 18233071 and retain fail-closed Forge200 task dispositions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path


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


def semicolon_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.reader(stream, delimiter=";"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    binding_path = root / "contracts/zenodo_batio3_rapid_sintering_contract_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if binding["source_id"] != "ZENODO-18233071" or binding["promotion_boundary"]["authority"] != 0:
        raise ValueError("invalid source identity or authority")

    verified_artifacts: dict[str, dict] = {}
    for name, item in binding["artifacts"].items():
        path = root / item["path"]
        actual_sha = sha256_file(path)
        if actual_sha != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"artifact mismatch: {name}")
        verified_artifacts[name] = {
            "path": item["path"],
            "bytes": path.stat().st_size,
            "sha256": actual_sha,
            "verified": True,
        }

    metadata = json.loads((root / binding["artifacts"]["record_metadata"]["path"]).read_text(encoding="utf-8"))
    if metadata["id"] != 18233071 or metadata["metadata"]["license"]["id"] != "cc-by-4.0":
        raise ValueError("Zenodo identity or license changed")
    if metadata["doi"] != binding["source"]["doi"]:
        raise ValueError("Zenodo DOI changed")

    archive_path = root / binding["artifacts"]["archive"]["path"]
    extracted_root = root / "data/raw/zenodo_batio3_rapid_sintering_v1/extracted"
    with zipfile.ZipFile(archive_path) as archive:
        archive_files = [item for item in archive.infolist() if not item.is_dir()]
        archive_file_entries = len(archive_files)
        archive_uncompressed_bytes = sum(item.file_size for item in archive_files)
        bad_members = archive.testzip()
    extracted_files = [path for path in extracted_root.rglob("*") if path.is_file()]
    extracted_bytes = sum(path.stat().st_size for path in extracted_files)
    expected = binding["expected_inventory"]
    if bad_members is not None:
        raise ValueError(f"ZIP CRC failure: {bad_members}")
    if (archive_file_entries, archive_uncompressed_bytes) != (
        expected["archive_file_entries"], expected["archive_uncompressed_bytes"]
    ):
        raise ValueError("archive inventory changed")
    if (len(extracted_files), extracted_bytes) != (expected["extracted_files"], expected["extracted_bytes"]):
        raise ValueError("extracted inventory changed")

    xrd = semicolon_rows(root / binding["artifacts"]["xrd_csv"]["path"])
    xrd_patterns = len(xrd[0]) - 1
    density = semicolon_rows(root / binding["artifacts"]["relative_density_csv"]["path"])
    density_names = [row[0] for row in density[1:] if row and row[0].strip()]
    grain = semicolon_rows(root / binding["artifacts"]["grain_measurements_csv"]["path"])
    grain_families = sorted({name.removesuffix("_A") for name in grain[0]})
    dilatometry = semicolon_rows(root / binding["artifacts"]["dilatometry_csv"]["path"])
    temperature_values: list[float] = []
    for column, name in enumerate(dilatometry[0]):
        if re.fullmatch(r"T\d+", name):
            temperature_values.extend(
                float(row[column].replace(",", "."))
                for row in dilatometry[1:]
                if column < len(row) and row[column].strip()
            )
    sem_tiffs = list((extracted_root / "3_SEM micrographs").rglob("*.tif"))
    domain_images = [
        path for path in (extracted_root / "5_domain width measurements").rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff", ".jpg", ".jpeg", ".png"}
    ]
    observed = {
        "xrd_patterns": xrd_patterns,
        "xrd_points_per_pattern": len(xrd) - 1,
        "relative_density_named_samples_including_green_body": len(density_names),
        "relative_density_sample_names": density_names,
        "sintered_sample_families": len(grain_families),
        "grain_population_columns": len(grain[0]),
        "grain_measurement_values": sum(
            1 for row in grain[1:] for value in row if value.strip()
        ),
        "sem_tiff_files": len(sem_tiffs),
        "domain_image_files": len(domain_images),
        "dilatometry_temperature_min_c": min(temperature_values),
        "dilatometry_temperature_max_c": max(temperature_values),
        "dilatometry_temperature_values": len(temperature_values),
    }
    for key in (
        "xrd_patterns",
        "relative_density_named_samples_including_green_body",
        "sintered_sample_families",
        "grain_population_columns",
        "sem_tiff_files",
        "domain_image_files",
        "dilatometry_temperature_min_c",
        "dilatometry_temperature_max_c",
    ):
        if observed[key] != expected[key]:
            raise ValueError(f"observed inventory changed: {key}={observed[key]}")

    description = (root / binding["artifacts"]["dilatometry_description"]["path"]).read_text(
        encoding="utf-8", errors="replace"
    )
    for phrase in ("followed by", "cooling", "sample shrinkage"):
        if phrase not in description:
            raise ValueError(f"dilatometry description changed: {phrase}")
    if observed["dilatometry_temperature_max_c"] >= binding["article_facts_visually_verified"]["sintering_temperature_range_c"][0]:
        raise ValueError("dilatometry unexpectedly reaches the sintering temperature range; re-audit required")

    dispositions = binding["candidate_dispositions"]
    expected_ids = {"CAND-P-049", "CAND-P-050", "CAND-P-051", "CAND-P-066", "CAND-P-085"}
    if {item["candidate_id"] for item in dispositions} != expected_ids:
        raise ValueError("candidate disposition set changed")
    if any(item["status"] != "EXACT_REJECTED" for item in dispositions):
        raise ValueError("this audit cannot promote candidates")

    receipt = {
        "schema": "cimc.forge200.zenodo-batio3-rapid-sintering-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": binding["source_id"],
        "status": "SOURCE_AND_LICENSE_VERIFIED_FIVE_EXACT_CONTRACTS_REJECTED",
        "contract_binding": {
            "path": binding_path.relative_to(root).as_posix(),
            "sha256": sha256_file(binding_path),
        },
        "source": binding["source"],
        "verified_artifacts": verified_artifacts,
        "archive_inventory": {
            "file_entries": archive_file_entries,
            "uncompressed_bytes": archive_uncompressed_bytes,
            "crc_verified": True,
            "extracted_files": len(extracted_files),
            "extracted_bytes": extracted_bytes,
        },
        "observed": observed,
        "article_facts_visually_verified": binding["article_facts_visually_verified"],
        "candidate_dispositions": dispositions,
        "training_actions": 0,
        "host_promotions": 0,
        "leakage_safe_splits_materialized": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "next_action": "Retain this source as licensed evidence and seek datasets with the exact missing trajectory, exponent, transfer-risk, or uncertainty labels; do not reinterpret post-sintered thermal expansion as sintering shrinkage.",
    }
    receipt["content_root_sha256"] = content_root(receipt)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "xrd_patterns": xrd_patterns,
        "sem_tiffs": len(sem_tiffs),
        "rejected_exact_contracts": len(dispositions),
        "content_root_sha256": receipt["content_root_sha256"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
