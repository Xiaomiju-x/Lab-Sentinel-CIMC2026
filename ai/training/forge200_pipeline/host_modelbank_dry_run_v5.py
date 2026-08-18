#!/usr/bin/env python3
"""Run the proven v4 loader/fault kernel against ModelBank v5 and attest it."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--modelbank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    implementation = root / "pipeline" / "host_modelbank_dry_run_v4.py"
    temporary = args.output.with_suffix(".v4-kernel.tmp.json")
    completed = subprocess.run([
        sys.executable,
        str(implementation),
        "--root", str(root),
        "--modelbank", str(args.modelbank.resolve()),
        "--output", str(temporary),
    ], check=True, text=True, capture_output=True)
    kernel = json.loads(temporary.read_text(encoding="utf-8"))
    temporary.unlink()
    if kernel["status"] != "PASS" or kernel["successful_swaps"] != 1000:
        raise RuntimeError("V4_KERNEL_GATE_FAILED")
    result = {
        **kernel,
        "schema": "cimc.forge200.modelbank-host-dry-run.v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "modelbank_v5_manifest_sha256": sha256(args.modelbank.resolve() / "MANIFEST.v5.json"),
        "verification_implementation": {
            "path": implementation.relative_to(root).as_posix(),
            "sha256": sha256(implementation),
            "stdout": completed.stdout.strip(),
        },
        "v4_kernel_content_root_sha256": kernel["content_root_sha256"],
        "production_files_modified": 0,
    }
    result["content_root_sha256"] = canonical_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "model_count": result["model_count"],
        "successful_swaps": result["successful_swaps"],
        "faults": len(result["fault_injections"]),
        "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
