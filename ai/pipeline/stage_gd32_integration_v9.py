#!/usr/bin/env python3
"""Stage the reviewed RAG/VeriProcess v9 sources and Keil entries by hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--rollback-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-root", type=Path, required=True,
                        help="All input and output paths must stay under this root.")
    args = parser.parse_args()
    root = args.root.resolve()
    production = args.production_root.resolve()
    rollback = args.rollback_root.resolve()
    output = args.output.resolve()

    expected_root = args.scope_root.resolve()
    for path in (root, production, rollback, output):
        try:
            path.relative_to(expected_root)
        except ValueError as exc:
            raise RuntimeError(f"D_SCOPE_GATE:{path}") from exc
    rollback_manifest = rollback / "DELTA_SHA256SUMS.csv"
    if not rollback_manifest.is_file():
        raise RuntimeError("ROLLBACK_GATE")

    mappings = [
        ("firmware_integration/rag_v9/forge200_rag_v9.h", "forge200_rag_v9.h"),
        ("firmware_integration/rag_v9/forge200_rag_v9.c", "forge200_rag_v9.c"),
        ("firmware_integration/rag_v9/forge200_rag_board_v9.h", "forge200_rag_board_v9.h"),
        ("firmware_integration/rag_v9/forge200_rag_board_v9.c", "forge200_rag_board_v9.c"),
        ("firmware_integration/veriprocess_v9/veriprocess_v9.h", "veriprocess_v9.h"),
        ("firmware_integration/veriprocess_v9/veriprocess_v9.c", "veriprocess_v9.c"),
        ("firmware_integration/veriprocess_v9/veriprocess_board_v9.h", "veriprocess_board_v9.h"),
        ("firmware_integration/veriprocess_v9/veriprocess_board_v9.c", "veriprocess_board_v9.c"),
    ]
    destination_root = production / "firmware/keil_proj/HardWare/Lab_Sentinel"
    records = []
    for source_relative, destination_name in mappings:
        source = root / source_relative
        destination = destination_root / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = sha256(source)
        if destination.exists() and sha256(destination) != source_sha:
            raise RuntimeError(f"DESTINATION_MISMATCH:{destination_name}")
        if not destination.exists():
            shutil.copy2(source, destination)
        if sha256(destination) != source_sha:
            raise RuntimeError(f"COPY_HASH_GATE:{destination_name}")
        records.append({
            "source": source.relative_to(root).as_posix(),
            "destination": destination.relative_to(production).as_posix(),
            "bytes": destination.stat().st_size,
            "sha256": source_sha,
        })

    project = production / "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx"
    raw = project.read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    indent = b"            "
    new_names = (
        b"forge200_rag_v9.c",
        b"forge200_rag_board_v9.c",
        b"veriprocess_v9.c",
        b"veriprocess_board_v9.c",
    )
    existing_counts = {name.decode(): raw.count(b"<FileName>" + name + b"</FileName>") for name in new_names}
    if all(count == 0 for count in existing_counts.values()):
        marker = (
            indent + b"<File>" + newline
            + indent + b"  <FileName>forge200_board_port.c</FileName>" + newline
        )
        if raw.count(marker) != 3:
            raise RuntimeError(f"PROJECT_MARKER_GATE:{raw.count(marker)}")
        block = b""
        for name in new_names:
            block += (
                indent + b"<File>" + newline
                + indent + b"  <FileName>" + name + b"</FileName>" + newline
                + indent + b"  <FileType>1</FileType>" + newline
                + indent + b"  <FilePath>..\\HardWare\\Lab_Sentinel\\" + name + b"</FilePath>" + newline
                + indent + b"</File>" + newline
            )
        raw = raw.replace(marker, block + marker)
        project.write_bytes(raw)
    final_counts = {
        name.decode(): project.read_bytes().count(b"<FileName>" + name + b"</FileName>")
        for name in new_names
    }
    if any(count != 3 for count in final_counts.values()):
        raise RuntimeError(f"PROJECT_SOURCE_GATE:{final_counts}")

    receipt = {
        "schema": "cimc.forge200.gd32-source-staging.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HASH_VERIFIED_RAG_VERIPROCESS_SOURCES_STAGED",
        "production_root": str(production),
        "rollback_root": str(rollback),
        "rollback_manifest_sha256": sha256(rollback_manifest),
        "file_count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "records": records,
        "project_sha256": sha256(project),
        "project_source_counts": final_counts,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    receipt["content_root_sha256"] = hashlib.sha256(canonical({
        "records": records,
        "project_sha256": receipt["project_sha256"],
        "rollback_manifest_sha256": receipt["rollback_manifest_sha256"],
    })).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "files": receipt["file_count"],
        "project_source_counts": final_counts,
        "content_root_sha256": receipt["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
