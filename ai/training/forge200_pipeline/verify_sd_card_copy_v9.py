#!/usr/bin/env python3
"""Read-only SHA-256 verification of the unified v9 F200 physical card copy."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(root: Path) -> list[dict[str, object]]:
    return [{
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in sorted(item for item in root.rglob("*") if item.is_file())]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sd-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-board-trace", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    target = args.sd_root.resolve() / "F200"
    manifest = json.loads((source / "MANIFEST.v9.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "cimc.forge200.sd-staging.v9":
        raise RuntimeError("SOURCE_MANIFEST_GATE")
    expected = inventory(source / "F200")
    actual = inventory(target) if target.is_dir() else []
    expected_manifest_records = [{**item, "path": f"F200/{item['path']}"} for item in expected]
    expected_content_root = hashlib.sha256(
        json.dumps(expected_manifest_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if expected_content_root != manifest["content_root_sha256"]:
        raise RuntimeError("SOURCE_CONTENT_ROOT_GATE")
    expected_by_path = {str(item["path"]): item for item in expected}
    actual_by_path = {str(item["path"]): item for item in actual}
    allowed_trace = {
        "TRACE/VPA.BIN", "TRACE/VPB.BIN", "TRACE/VPWAL.BIN", "TRACE/VPDRILL.OK"
    } if args.allow_board_trace else set()
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    extra = sorted(set(actual_by_path) - set(expected_by_path) - allowed_trace)
    mismatched = sorted(path for path in set(expected_by_path) & set(actual_by_path)
                        if expected_by_path[path] != actual_by_path[path])
    trace_present = sorted(set(actual_by_path) & allowed_trace)
    accepted = not missing and not extra and not mismatched
    result = {
        "schema": "cimc.forge200.physical-sd-copy.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHYSICAL_SD_COPY_PASS" if accepted else "PHYSICAL_SD_COPY_REJECTED",
        "accepted": accepted,
        "source": str(source),
        "source_content_root_sha256": manifest["content_root_sha256"],
        "sd_root": str(args.sd_root.resolve()),
        "files": len(actual),
        "bytes": sum(int(item["bytes"]) for item in actual),
        "expected_files": len(expected),
        "expected_bytes": sum(int(item["bytes"]) for item in expected),
        "missing": missing,
        "extra": extra,
        "mismatched": mismatched,
        "board_trace_allowed": args.allow_board_trace,
        "board_trace_present": trace_present,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    actual_manifest_records = [{**item, "path": f"F200/{item['path']}"} for item in actual]
    result["actual_content_root_sha256"] = hashlib.sha256(
        json.dumps(actual_manifest_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "files": len(actual), "bytes": result["bytes"],
        "mismatches": len(missing) + len(extra) + len(mismatched),
        "actual_content_root_sha256": result["actual_content_root_sha256"],
    }, sort_keys=True))
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
