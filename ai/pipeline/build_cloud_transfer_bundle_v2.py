#!/usr/bin/env python3
"""Build a hash-manifested transfer bundle without checkpoints or teacher caches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


WORK = Path("/root/autodl-tmp/cimc-forge200-20260802/work")
CORRECTIVE = Path("/root/autodl-tmp/cimc-forge200-corrective-20260803")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def wanted(path: Path) -> bool:
    name = path.name
    if name.endswith((".pt", ".safetensors", ".log", ".tmp", ".partial")):
        return False
    if name == "heartbeat.json" or "__pycache__" in path.parts:
        return False
    return path.is_file() and not path.is_symlink()


def walk(source: Path, prefix: str) -> Iterable[tuple[Path, str]]:
    if source.is_file():
        if wanted(source):
            yield source, f"{prefix}/{source.name}"
        return
    if not source.exists():
        return
    for path in sorted(source.rglob("*")):
        if wanted(path):
            yield path, f"{prefix}/{path.relative_to(source).as_posix()}"


def sources(node: str) -> list[tuple[Path, str, str]]:
    common = [
        (WORK / "contracts", "project/contracts", "contracts"),
        (WORK / "queue", "project/queue", "queue_state"),
        (WORK / "evidence", "project/cloud_evidence", "cloud_evidence"),
        (WORK / "artifacts" / "cloud5090", "first_pass/artifacts", "first_pass_retained"),
        (CORRECTIVE / "nanolm_train_v2_metricselect_g32", "corrective/nanolm_selected", "nanolm_selected_no_checkpoints"),
    ]
    if node == "A":
        return common + [
            (CORRECTIVE / "material_contract_v2", "corrective/material_exact", "contract_baseline_pass"),
            (CORRECTIVE / "router_contract_v2", "corrective/router_rejected", "contract_baseline_rejected"),
            (CORRECTIVE / "rag_encoder_contract_v2", "corrective/rag_encoder_rejected", "contract_baseline_rejected"),
            (WORK / "data" / "corpora" / "ccby_multidomain_v2.jsonl", "data/corpus", "source_bound_corpus"),
            (WORK / "data" / "ledgers" / "ccby_multidomain_corpus.v2.json", "data/ledger", "license_and_split_ledger"),
            (WORK / "data" / "raw" / "ccby_multidomain_v2", "data/raw_ccby", "metadata_and_jats_license_evidence"),
            (WORK / "data" / "staged_rag_contract_v2", "data/staged_rag", "rejected_exact_benchmark_data"),
            (WORK / "data" / "staged_router_contract_v2", "data/staged_router", "rejected_exact_benchmark_data"),
        ]
    return common + [
        (CORRECTIVE / "material_contract_v2", "corrective/material_exact", "contract_baseline_pass"),
        (CORRECTIVE / "material_contract_v2_residual", "corrective/material_p087_selected", "contract_baseline_pass"),
        (CORRECTIVE / "postgpu_support", "corrective/postgpu_support", "pass_and_rejected"),
        (CORRECTIVE / "open_data_corrective", "corrective/open_data_rejected", "contract_baseline_rejected"),
        (CORRECTIVE / "support_corrective", "corrective/support_dataset_correction", "baseline_unclosed"),
        (WORK / "data" / "staged_contract_v2", "data/staged_material", "exact_material_data"),
        (WORK / "data" / "staged_postgpu", "data/staged_postgpu", "postgpu_support_data"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=("A", "B"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/cimc-forge200-transfer-20260803"))
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    name = f"cimc-forge200-node-{args.node.lower()}-transfer-20260803"
    archive = output_root / f"{name}.tar.gz"
    manifest_path = output_root / f"{name}.manifest.json"
    restore_path = output_root / f"{name}.RESTORE.md"

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    file_map: list[tuple[Path, str]] = []
    for source, prefix, classification in sources(args.node):
        for path, archive_path in walk(source, prefix):
            if archive_path in seen:
                raise RuntimeError(f"DUPLICATE_ARCHIVE_PATH:{archive_path}")
            seen.add(archive_path)
            stat = path.stat()
            entries.append({
                "archive_path": archive_path,
                "source_path": str(path),
                "bytes": stat.st_size,
                "sha256": sha256_file(path),
                "classification": classification,
            })
            file_map.append((path, archive_path))
    entries.sort(key=lambda item: item["archive_path"])
    file_map.sort(key=lambda item: item[1])
    manifest_core = {
        "schema": "cimc.forge200.cloud-transfer-manifest.v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "node": args.node,
        "status": "PASS",
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
        "exclusions": ["training checkpoints (*.pt)", "teacher weights (*.safetensors)", "logs", "heartbeats", "pilot snapshot duplicates"],
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "files": entries,
    }
    manifest_core["content_root_sha256"] = hashlib.sha256(canonical_bytes(entries)).hexdigest()
    manifest_path.write_text(json.dumps(manifest_core, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest_sha = sha256_file(manifest_path)
    restore = f"""# CIMC Forge200 node {args.node} transfer restore

1. Verify the archive SHA-256 against the `.sha256` sidecar.
2. Extract into a new empty directory; never extract over production firmware.
3. Verify `{manifest_path.name}` SHA-256 is `{manifest_sha}`.
4. For every manifest entry, verify byte count and SHA-256 before use.
5. All model assets remain `authority=0`, `board_accepted=false`, and `countable_model=false` until the unified GD32 ModelBank/loader/microSD/MAX31856 board acceptance.

Excluded on purpose: restartable training checkpoints, teacher caches, logs, pilot snapshot duplicates, installers, and unrelated media.
"""
    restore_path.write_text(restore, encoding="utf-8")

    with archive.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, compresslevel=1, mtime=0) as gzip_handle:
            with tarfile.open(mode="w", fileobj=gzip_handle, format=tarfile.PAX_FORMAT) as tar:
                for path, archive_path in file_map:
                    info = tar.gettarinfo(str(path), arcname=archive_path)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with path.open("rb") as handle:
                        tar.addfile(info, handle)
                for sidecar, archive_path in ((manifest_path, f"bundle/{manifest_path.name}"), (restore_path, f"bundle/{restore_path.name}")):
                    info = tar.gettarinfo(str(sidecar), arcname=archive_path)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    with sidecar.open("rb") as handle:
                        tar.addfile(info, handle)
    archive_sha = sha256_file(archive)
    sidecar = output_root / f"{archive.name}.sha256"
    sidecar.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    result = {
        "status": "PASS",
        "node": args.node,
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "file_count": len(entries),
        "uncompressed_bytes": manifest_core["total_uncompressed_bytes"],
        "content_root_sha256": manifest_core["content_root_sha256"],
    }
    (output_root / f"{name}.receipt.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
