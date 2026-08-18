#!/usr/bin/env python3
"""Hash-verified mechanical staging of reviewed Forge200 C sources."""

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
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="Accept an existing destination only when its SHA-256 equals the source.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    production_root = args.production_root.resolve()
    output = args.output.resolve()
    mappings = [
        ("firmware_integration/modelbank_v4/forge200_modelbank.h", "forge200_modelbank.h"),
        ("firmware_integration/modelbank_v4/forge200_modelbank.c", "forge200_modelbank.c"),
        ("firmware_integration/modelbank_v4/forge200_shared_spi.h", "forge200_shared_spi.h"),
        ("firmware_integration/modelbank_v4/forge200_shared_spi.c", "forge200_shared_spi.c"),
        ("firmware_integration/modelbank_v8/forge200_runtime_v8.h", "forge200_runtime_v8.h"),
        ("firmware_integration/modelbank_v8/forge200_runtime_v8.c", "forge200_runtime_v8.c"),
        ("firmware_integration/modelbank_v8_gd32/forge200_bus_guard.h", "forge200_bus_guard.h"),
        ("firmware_integration/modelbank_v8_gd32/forge200_bus_guard.c", "forge200_bus_guard.c"),
        ("firmware_integration/modelbank_v8_gd32/forge200_board_port.h", "forge200_board_port.h"),
        ("firmware_integration/modelbank_v8_gd32/forge200_board_port.c", "forge200_board_port.c"),
    ]
    destination_root = (
        production_root
        / "firmware"
        / "keil_proj"
        / "HardWare"
        / "Lab_Sentinel"
    )
    records = []
    for source_relative, destination_name in mappings:
        source = root / source_relative
        destination = destination_root / destination_name
        if not source.is_file():
            raise FileNotFoundError(source)
        source_sha = sha256(source)
        if destination.exists():
            if not args.verify_existing:
                raise FileExistsError(
                    f"refusing to overwrite production source: {destination}"
                )
            if sha256(destination) != source_sha:
                raise RuntimeError(f"EXISTING_DESTINATION_MISMATCH:{destination_name}")
        else:
            shutil.copy2(source, destination)
        destination_sha = sha256(destination)
        if source_sha != destination_sha:
            raise RuntimeError(f"STAGING_HASH_GATE:{destination_name}")
        records.append(
            {
                "source": source.relative_to(root).as_posix(),
                "source_sha256": source_sha,
                "destination": destination.relative_to(production_root).as_posix(),
                "destination_sha256": destination_sha,
                "bytes": destination.stat().st_size,
            }
        )
    receipt = {
        "schema": "cimc.forge200.gd32-source-staging.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "HASH_VERIFIED_SOURCES_PRESENT_IN_PRODUCTION_TREE",
        "production_root": str(production_root),
        "file_count": len(records),
        "bytes": sum(item["bytes"] for item in records),
        "records": records,
        "authority_nonzero": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical(records)).hexdigest(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "files": receipt["file_count"],
                "bytes": receipt["bytes"],
                "content_root_sha256": receipt["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
