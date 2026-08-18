#!/usr/bin/env python3
"""Build an immutable ModelBank v5 evidence refresh without duplicating weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
        return "HARDLINK"
    except OSError:
        shutil.copy2(source, target)
        return "COPY_FALLBACK"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source_bank = args.source_bank.resolve()
    output = args.output.resolve()
    releases = (root / "releases").resolve()
    if releases not in output.parents or output == releases:
        raise RuntimeError("OUTPUT_SCOPE_GATE")
    if output.exists():
        raise RuntimeError(f"OUTPUT_ALREADY_EXISTS:{output}")

    source_manifest_path = source_bank / "MANIFEST.v4.json"
    source_catalog_a_path = source_bank / "catalog_A.json"
    source_catalog_b_path = source_bank / "catalog_B.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_a = json.loads(source_catalog_a_path.read_text(encoding="utf-8"))
    source_b = json.loads(source_catalog_b_path.read_text(encoding="utf-8"))
    if source_a["models"] != source_b["models"]:
        raise RuntimeError("SOURCE_AB_MODEL_MISMATCH")
    if source_manifest["model_count"] != len(source_a["models"]):
        raise RuntimeError("SOURCE_MODEL_COUNT_MISMATCH")

    linkage_counts = {"HARDLINK": 0, "COPY_FALLBACK": 0}
    package_hashes: set[str] = set()
    for record in source_a["models"]:
        for spec in record["files"].values():
            source = source_bank / spec["path"]
            if source.stat().st_size != spec["bytes"] or sha256(source) != spec["sha256"]:
                raise RuntimeError(f"SOURCE_FILE_GATE:{record['candidate_id']}:{spec['path']}")
            target = output / spec["path"]
            mode = link_or_copy(source, target)
            linkage_counts[mode] += 1
            if target.stat().st_size != spec["bytes"] or sha256(target) != spec["sha256"]:
                raise RuntimeError(f"TARGET_FILE_GATE:{record['candidate_id']}:{spec['path']}")
        package_hash = record["files"]["model.icmf"]["sha256"]
        if package_hash in package_hashes:
            raise RuntimeError(f"PACKAGE_HASH_COLLISION:{record['candidate_id']}")
        package_hashes.add(package_hash)

    audit_names = sorted(
        path.name
        for path in (root / "evidence").glob("*.json")
        if "audit" in path.name or "quarantine" in path.name
    )
    required_evidence = [
        "host_closure.v4.json",
        "modelbank_host_dry_run.v4.json",
        "host_artifact_verification.v4.json",
        "interface_freeze_verification.v2.json",
        "firmware_adapter_host_compile.v4.json",
        "unified_staging.v4.json",
        "release_gap_audit.v4.json",
        "exact_data_intake.v4.json",
        *audit_names,
    ]
    evidence_records = []
    for name in dict.fromkeys(required_evidence):
        path = root / "evidence" / name
        if not path.is_file():
            raise RuntimeError(f"EVIDENCE_MISSING:{name}")
        document = json.loads(path.read_text(encoding="utf-8"))
        evidence_records.append({
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "status": document.get("status", "NO_STATUS"),
        })
    evidence_index = {
        "schema": "cimc.forge200.modelbank-evidence-index.v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_EVIDENCE_INDEXED_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "source_bank": {
            "path": source_bank.relative_to(root).as_posix(),
            "manifest_sha256": sha256(source_manifest_path),
            "catalog_A_sha256": sha256(source_catalog_a_path),
            "catalog_B_sha256": sha256(source_catalog_b_path),
        },
        "evidence_count": len(evidence_records),
        "records": evidence_records,
        "authority": 0,
        "board_accepted": False,
        "countable_models": 0,
    }
    evidence_index["content_root_sha256"] = hashlib.sha256(canonical(evidence_records)).hexdigest()
    evidence_index_path = output / "EVIDENCE_INDEX.v5.json"
    write_json(evidence_index_path, evidence_index)

    common = {
        "schema": "cimc.forge200.modelbank-catalog.v5",
        "status": "HOST_STAGING_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "abi": source_a["abi"],
        "closure": source_a["closure"],
        "source_bank_manifest_sha256": sha256(source_manifest_path),
        "evidence_index_sha256": sha256(evidence_index_path),
        "model_count": len(source_a["models"]),
        "exact_count": source_a["exact_count"],
        "sim_only_extension_count": source_a["sim_only_extension_count"],
        "models": source_a["models"],
        "authority": 0,
        "board_accepted": False,
        "countable_models": 0,
    }
    catalogs = []
    for slot in ("A", "B"):
        catalog = {
            **common,
            "slot": slot,
            "generation_counter": 0,
            "commit_state": "PREPARED_NOT_BOARD_COMMITTED",
        }
        catalog["content_root_sha256"] = hashlib.sha256(canonical({
            "slot": slot,
            "generation_counter": 0,
            "models": common["models"],
            "evidence_index_sha256": common["evidence_index_sha256"],
        })).hexdigest()
        path = output / f"catalog_{slot}.json"
        write_json(path, catalog)
        catalogs.append({
            "slot": slot,
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "content_root_sha256": catalog["content_root_sha256"],
        })

    restore = {
        "schema": "cimc.forge200.modelbank-restore.v2",
        "failure_action": "REFUSE_CANDIDATE_AND_RETURN_TO_INITIAL_30_ASSET_BASELINE",
        "catalog_selection": "highest fully valid committed generation; equal uncommitted A/B catalogs are staging only",
        "verification_order": ["SCHEMA", "BOUNDS", "ENGINE_OPSET", "PAYLOAD_SHA256", "GOLDEN", "OUTPUT_SCHEMA"],
        "source_v4_is_immutable": True,
        "production_files_modified": 0,
        "board_actions": 0,
    }
    write_json(output / "RESTORE.json", restore)

    files = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        files.append({
            "path": path.relative_to(output).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    manifest = {
        "schema": "cimc.forge200.modelbank-build.v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_MODELBANK_V5_BUILT_RELEASE_FLOOR_BLOCKED_BOARD_PENDING",
        "root": str(output),
        "model_count": len(source_a["models"]),
        "exact_count": source_a["exact_count"],
        "sim_only_extension_count": source_a["sim_only_extension_count"],
        "package_sha256_collision_count": 0,
        "linkage_counts": linkage_counts,
        "catalogs": catalogs,
        "files": files,
        "file_count": len(files),
        "bytes": sum(item["bytes"] for item in files),
        "authority_nonzero": 0,
        "production_files_modified": 0,
        "board_actions": 0,
    }
    manifest["content_root_sha256"] = hashlib.sha256(canonical(files)).hexdigest()
    manifest_path = output / "MANIFEST.v5.json"
    write_json(manifest_path, manifest)
    receipt = {
        **manifest,
        "manifest_path": manifest_path.relative_to(root).as_posix(),
        "manifest_sha256": sha256(manifest_path),
    }
    write_json(root / "evidence" / "modelbank_build.v5.json", receipt)
    print(json.dumps({
        "status": manifest["status"],
        "model_count": manifest["model_count"],
        "exact_count": manifest["exact_count"],
        "sim_only_extension_count": manifest["sim_only_extension_count"],
        "file_count": manifest["file_count"],
        "bytes": manifest["bytes"],
        "linkage_counts": linkage_counts,
        "content_root_sha256": manifest["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
