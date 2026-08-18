#!/usr/bin/env python3
"""Build the immutable unified ModelBank + RAG + TraceLedger SD staging tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--rag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-root", type=Path, required=True,
                        help="All input and output paths must stay under this root.")
    args = parser.parse_args()
    root = args.root.resolve()
    base = args.base.resolve()
    rag = args.rag.resolve()
    output = args.output.resolve()
    d_scope = args.scope_root.resolve()
    for path in (root, base, rag, output):
        try:
            path.relative_to(d_scope)
        except ValueError as exc:
            raise RuntimeError(f"D_SCOPE_GATE:{path}") from exc
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    base_manifest = json.loads((base / "MANIFEST.JSON").read_text(encoding="utf-8"))
    rag_manifest = json.loads((rag / "MANIFEST.v9.json").read_text(encoding="utf-8"))
    if base_manifest.get("model_count") != 170 or base_manifest.get("authority_nonzero") != 0:
        raise RuntimeError("BASE_MANIFEST_GATE")
    if rag_manifest.get("content_root_sha256") != "e8a9e983451551bd22f82f4e59b10cbd1c4fb92a45b0763d700284779b77d956":
        raise RuntimeError("RAG_MANIFEST_GATE")

    output.mkdir(parents=True)
    methods = {"hardlink": 0, "copy": 0}
    for source in sorted((base / "F200").rglob("*")):
        if not source.is_file():
            continue
        destination = output / "F200" / source.relative_to(base / "F200")
        methods[link_or_copy(source, destination)] += 1

    rag_root = output / "F200/RAG"
    for domain in range(6):
        for suffix, subdir in (("F2S", "support"), ("RIX", "workload")):
            source = rag / subdir / f"D{domain}.{suffix}"
            methods[link_or_copy(source, rag_root / source.name)] += 1
    lm_map = {
        "CAND-G-001": "G001", "CAND-G-012": "G012",
        "CAND-G-003": "G003", "CAND-G-004": "G004",
        "CAND-G-005": "G005", "CAND-G-006": "G006",
    }
    for candidate_id, short_name in lm_map.items():
        for suffix in ("ICM", "GLD"):
            source = rag / "lm" / f"{candidate_id}.{suffix}"
            methods[link_or_copy(source, rag_root / f"{short_name}.{suffix}")] += 1
    trace = output / "F200/TRACE"
    trace.mkdir(parents=True, exist_ok=True)
    (trace / "README.TXT").write_text(
        "VERIPROCESS V9 TRACELEDGER DIRECTORY\n"
        "VPA.BIN, VPB.BIN AND VPWAL.BIN ARE CREATED BY THE GD32 ACCEPTANCE IMAGE.\n",
        encoding="ascii",
    )

    records = []
    for path in sorted((output / "F200").rglob("*")):
        if path.is_file():
            records.append({
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    file_csv = output / "FILES.v9.csv"
    with file_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256"))
        writer.writeheader()
        writer.writerows(records)
    content_root = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema": "cimc.forge200.sd-staging.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_FOR_SINGLE_PHYSICAL_SD_COPY_AND_GD32_ACCEPTANCE",
        "base_modelbank": {
            "path": base.relative_to(root).as_posix(),
            "manifest_sha256": sha256(base / "MANIFEST.JSON"),
            "model_count": 170,
            "exact_count": base_manifest["exact_count"],
            "sim_only_count": base_manifest["sim_only_count"],
        },
        "rag_runtime": {
            "path": rag.relative_to(root).as_posix(),
            "manifest_sha256": sha256(rag / "MANIFEST.v9.json"),
            "content_root_sha256": rag_manifest["content_root_sha256"],
            "domains": 6,
            "workload_queries": 120,
            "lm_packages": 6,
            "support_models_per_domain": 13,
        },
        "traceledger": {
            "directory": "F200/TRACE",
            "files_created_on_board": ["VPA.BIN", "VPB.BIN", "VPWAL.BIN", "VPDRILL.OK"],
        },
        "file_count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "files_csv_sha256": sha256(file_csv),
        "content_root_sha256": content_root,
        "storage_methods": methods,
        "authority_nonzero": 0,
        "board_actions": 0,
        "board_accepted": False,
    }
    (output / "MANIFEST.v9.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "README.TXT").write_text(
        "FORGE200 UNIFIED SD STAGING V9\n"
        "COPY THE F200 DIRECTORY TO THE ROOT OF THE ALREADY-QUALIFIED FAT32 CARD.\n"
        "VERIFY WITH pipeline/verify_sd_card_copy_v9.py BEFORE GD32 POWER-ON.\n",
        encoding="ascii",
    )
    print(json.dumps({
        "status": manifest["status"], "files": len(records),
        "bytes": manifest["bytes"], "storage_methods": methods,
        "content_root_sha256": content_root,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
