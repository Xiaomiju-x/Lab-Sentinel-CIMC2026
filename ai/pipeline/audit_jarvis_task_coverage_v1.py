#!/usr/bin/env python3
"""Inventory source-bound JARVIS-DFT fields before opening more task contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_SHA256 = "f9e0a3309f0000d5de1ec9e49c93963109ea45e63f451103f8f6595d2eabf7f5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def populated(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, (list, dict)):
        return bool(value)
    if isinstance(value, str) and value.lower() in {"na", "nan", "none", "null", "not available"}:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    source_sha256 = sha256_file(source)
    if source_sha256 != EXPECTED_SHA256:
        raise RuntimeError(f"SOURCE_HASH_GATE:{source_sha256}")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != 1 or ".." in Path(names[0]).parts:
            raise RuntimeError("ARCHIVE_LAYOUT_GATE")
        rows = json.loads(archive.read(names[0]))
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("RECORD_LIST_GATE")

    keys = sorted({key for row in rows for key in row})
    inventory: dict[str, dict[str, Any]] = {}
    for key in keys:
        values = [row.get(key) for row in rows]
        present = [value for value in values if populated(value)]
        types = Counter(type(value).__name__ for value in present)
        examples = []
        for value in present:
            if isinstance(value, (str, int, float, bool)):
                rendered = value
            elif isinstance(value, list):
                rendered = {"container": "list", "length": len(value)}
            elif isinstance(value, dict):
                rendered = {"container": "dict", "keys": sorted(value)[:20]}
            else:
                rendered = repr(value)[:160]
            if rendered not in examples:
                examples.append(rendered)
            if len(examples) == 3:
                break
        inventory[key] = {
            "populated_records": len(present),
            "missing_records": len(rows) - len(present),
            "types": dict(sorted(types.items())),
            "examples": examples,
        }
        numeric_values = sorted(
            float(value)
            for value in present
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
        )
        if numeric_values:
            def quantile(fraction: float) -> float:
                return numeric_values[round(fraction * (len(numeric_values) - 1))]

            inventory[key]["numeric_summary"] = {
                "min": numeric_values[0],
                "p01": quantile(0.01),
                "p10": quantile(0.10),
                "median": quantile(0.50),
                "p90": quantile(0.90),
                "p99": quantile(0.99),
                "max": numeric_values[-1],
                "zero_count": sum(value == 0.0 for value in numeric_values),
            }

    receipt = {
        "schema": "cimc.forge200.jarvis-field-coverage-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "purpose": "source-bound field inventory only; this receipt does not admit or count any model",
        "source_path": str(source.relative_to(root)).replace("\\", "/"),
        "source_sha256": source_sha256,
        "source_pid": "10.6084/m9.figshare.6815699.v11",
        "license": "CC BY 4.0",
        "records": len(rows),
        "field_count": len(keys),
        "fields": inventory,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    output = root / "evidence" / "jarvis_field_coverage_audit.v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "records": len(rows), "field_count": len(keys), "output": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
