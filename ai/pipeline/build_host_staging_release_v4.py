#!/usr/bin/env python3
"""Create a deterministic, cache-free Forge200 host staging handoff archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


FIXED_ZIP_TIME = (2026, 8, 3, 0, 0, 0)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_path(files: dict[str, Path], root: Path, relative: str) -> None:
    path = root / relative
    if path.is_file():
        files[path.relative_to(root).as_posix()] = path
    elif path.is_dir():
        for item in sorted(path.rglob("*")):
            if item.is_file():
                files[item.relative_to(root).as_posix()] = item
    else:
        raise FileNotFoundError(path)


def write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compresslevel=9)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    files: dict[str, Path] = {}

    include = [
        "releases/forge200-host-modelbank-v4-20260803",
        "firmware_integration/modelbank_v4",
        "contracts/schemas",
        "contracts/model_package_abi.v1.json",
        "contracts/model_roster_200.v1.tsv",
        "contracts/model_task_contracts_200.v1.tsv",
        "contracts/candidate_pool_244.v1.tsv",
        "contracts/candidate_task_contracts_244.v1.tsv",
        "docs/CIMC_ICMAT_FORGE200_FINAL_MASTER_PLAN.md",
        "docs/EXECUTION_LEDGER_20260801.md",
        "docs/HOST_STAGING_V4_HANDOFF_20260803.md",
        "tests/test_forge200_local.py",
        "pipeline/build_modelbank_v4.py",
        "pipeline/host_modelbank_dry_run_v4.py",
        "pipeline/verify_host_artifacts_v4.py",
        "pipeline/verify_interface_freeze_v2.py",
        "pipeline/verify_firmware_adapter_v4.py",
        "pipeline/build_unified_staging_v4.py",
        "pipeline/build_release_gap_audit_v4.py",
        "pipeline/build_exact_data_intake_v4.py",
        "pipeline/run_local_acceptance_v4.py",
        "data/ledgers/cimc_existing_asset_audit.v1.json",
        "evidence/local4050_progress.v3.json",
        "evidence/host_closure.v4.json",
        "evidence/modelbank_build.v4.json",
        "evidence/modelbank_host_dry_run.v4.json",
        "evidence/host_artifact_verification.v4.json",
        "evidence/interface_freeze_verification.v2.json",
        "evidence/firmware_adapter_host_compile.v4.json",
        "evidence/unified_staging.v4.json",
        "evidence/release_gap_audit.v4.json",
        "evidence/exact_data_intake.v4.json",
        "evidence/local_host_acceptance.v4.json",
        "evidence/legacy_nanolm_baseline_host_receipt.v1.json",
    ]
    for relative in include:
        add_path(files, root, relative)

    forbidden = ("checkpoint", "pip_cache", ".pt", ".pth", ".safetensors", "heartbeat", "__pycache__")
    bad = [name for name in files if any(token in name.lower() for token in forbidden)]
    if bad:
        raise SystemExit(f"cache/checkpoint material selected: {bad[:3]}")

    entries = []
    for name, path in sorted(files.items()):
        entries.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    restore = (
        "Forge200 host staging v4\n\n"
        "This archive is host-only and board-pending. It does not authorize publication counts.\n"
        "Extract to an isolated D-drive directory. Do not overwrite the frozen production project.\n"
        "Verify every entry in STAGING_MANIFEST.v4.json before use.\n"
        "After compliant exact data closes the release floor, follow docs/HOST_STAGING_V4_HANDOFF_20260803.md.\n"
        "Do not copy files into the Keil production target or burn GD32/microSD before unified acceptance.\n"
    ).encode("utf-8")
    entries.append({"path": "RESTORE.md", "bytes": len(restore), "sha256": sha256_bytes(restore)})
    entries.sort(key=lambda item: item["path"])
    content_root = sha256_bytes(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode())
    manifest = {
        "schema": "cimc.forge200.host-staging-release.v4",
        "status": "HOST_STAGING_BOARD_PENDING_RELEASE_FLOOR_BLOCKED",
        "content_root_sha256": content_root,
        "file_count_excluding_manifest": len(entries),
        "payload_bytes_excluding_manifest": sum(item["bytes"] for item in entries),
        "authority": 0,
        "new_models_board_accepted": 0,
        "new_models_countable_publicly": 0,
        "training_cache_or_checkpoint_files": 0,
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    archive_name = f"forge200-host-staging-v4-{content_root[:16]}.zip"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / archive_name
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for name, path in sorted(files.items()):
            write_member(archive, name, path.read_bytes())
        write_member(archive, "RESTORE.md", restore)
        write_member(archive, "STAGING_MANIFEST.v4.json", manifest_bytes)

    receipt = {
        "schema": "cimc.forge200.host-staging-release-receipt.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HOST_STAGING_ARCHIVE_VERIFIED_BOARD_PENDING_RELEASE_FLOOR_BLOCKED",
        "archive": output.relative_to(root).as_posix(),
        "archive_bytes": output.stat().st_size,
        "archive_sha256": sha256_file(output),
        "content_root_sha256": content_root,
        "file_count_in_archive": len(entries) + 1,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "host_models": 85,
        "exact_source_bound": 78,
        "sim_only": 7,
        "new_board_accepted": 0,
        "new_countable_publicly": 0,
        "production_files_modified": 0,
        "cache_or_checkpoint_files": 0,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
