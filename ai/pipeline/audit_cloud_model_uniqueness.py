#!/usr/bin/env python3
"""Audit trained Forge200 artifacts for genuine, content-addressed uniqueness.

The ICMF header contains the candidate identifier, so hashing the whole package
cannot reveal duplicated weights.  This audit hashes the payload after the
256-byte ABI header and compares it with the payload hash sealed in each
promotion receipt.  It never promotes host artifacts to board-accepted models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


HEADER_BYTES = 256


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def audit(artifact_root: Path) -> dict[str, Any]:
    errors: list[str] = []
    records: list[dict[str, Any]] = []
    for receipt_path in sorted(artifact_root.glob("CAND-*/promotion_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        candidate_id = receipt_path.parent.name
        if receipt.get("candidate_id") != candidate_id:
            errors.append(f"{candidate_id}:receipt_identity")
        package = receipt.get("package", {})
        package_path = receipt_path.parent / str(package.get("path", ""))
        if not package_path.is_file():
            errors.append(f"{candidate_id}:missing_package")
            continue
        raw = package_path.read_bytes()
        if len(raw) <= HEADER_BYTES or raw[:4] != b"ICMF":
            errors.append(f"{candidate_id}:invalid_icmf")
            continue
        package_sha = sha256_bytes(raw)
        payload_sha = sha256_bytes(raw[HEADER_BYTES:])
        if package_sha != package.get("sha256"):
            errors.append(f"{candidate_id}:package_sha")
        if payload_sha != package.get("payload_sha256"):
            errors.append(f"{candidate_id}:payload_sha")
        source_path = receipt_path.parent / "source_manifest.json"
        source = (
            json.loads(source_path.read_text(encoding="utf-8"))
            if source_path.is_file()
            else {}
        )
        records.append(
            {
                "candidate_id": candidate_id,
                "payload_sha256": payload_sha,
                "package_sha256": package_sha,
                "package_bytes": len(raw),
                "dataset_sha256": source.get("dataset_sha256"),
                "parameter_count": receipt.get("parameter_count"),
                "three_seed_count": receipt.get("three_seed_count"),
                "status": receipt.get("status"),
                "authority": receipt.get("authority"),
                "board_accepted": receipt.get("board_accepted"),
                "countable_model": receipt.get("countable_model"),
            }
        )
        if receipt.get("authority") != 0:
            errors.append(f"{candidate_id}:authority_nonzero")
        if receipt.get("board_accepted") is not False:
            errors.append(f"{candidate_id}:board_state")

    by_payload: dict[str, list[str]] = defaultdict(list)
    by_dataset: dict[str, list[str]] = defaultdict(list)
    for record in records:
        by_payload[record["payload_sha256"]].append(record["candidate_id"])
        if record["dataset_sha256"]:
            by_dataset[record["dataset_sha256"]].append(record["candidate_id"])
    payload_collisions = [
        {"payload_sha256": digest, "candidate_ids": ids}
        for digest, ids in sorted(by_payload.items())
        if len(ids) > 1
    ]
    dataset_reuse = [
        {"dataset_sha256": digest, "candidate_ids": ids}
        for digest, ids in sorted(by_dataset.items())
        if len(ids) > 1
    ]
    if payload_collisions:
        errors.append("duplicate_weight_payloads")
    content = {
        "records": records,
        "payload_collisions": payload_collisions,
        "dataset_reuse": dataset_reuse,
    }
    return {
        "schema": "cimc.forge200.cloud-model-uniqueness-audit.v1",
        "status": "PASS" if not errors else "FAIL_CLOSED",
        "artifact_root": str(artifact_root),
        "candidate_count": len(records),
        "unique_weight_payloads": len(by_payload),
        "payload_collision_groups": len(payload_collisions),
        "dataset_reuse_groups": len(dataset_reuse),
        "authority_nonzero": sum(record["authority"] != 0 for record in records),
        "board_accepted": sum(record["board_accepted"] is True for record in records),
        "countable_models": sum(record["countable_model"] is True for record in records),
        **content,
        "content_root_sha256": sha256_bytes(canonical_bytes(content)),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact_root.resolve())
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
