#!/usr/bin/env python3
"""Execute every MCU v8 binary golden through the portable C runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runtime_root = args.runtime_root.resolve()
    runner = args.runner.resolve()
    manifest_path = runtime_root / "MANIFEST.v8.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    started = time.perf_counter()
    for record in manifest["records"]:
        package = runtime_root / record["package"]["path"]
        golden = runtime_root / record["golden"]["path"]
        if sha256(package) != record["package"]["sha256"] or sha256(golden) != record["golden"]["sha256"]:
            raise RuntimeError(f"FILE_HASH_GATE:{record['candidate_id']}")
        raw = package.read_bytes()
        payload_bytes = struct.unpack_from("<Q", raw, 24)[0]
        if len(raw) != 256 + payload_bytes or hashlib.sha256(raw[256:]).hexdigest() != record["package"]["payload_sha256"]:
            raise RuntimeError(f"PAYLOAD_HASH_GATE:{record['candidate_id']}")
        if raw[108:140].hex() != record["golden"]["sha256"]:
            raise RuntimeError(f"GOLDEN_BINDING_GATE:{record['candidate_id']}")
        completed = subprocess.run(
            [str(runner), str(package), str(golden)],
            text=True,
            capture_output=True,
            timeout=120,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"C_GOLDEN_GATE:{record['candidate_id']}:{completed.returncode}:{completed.stderr.strip()}"
            )
        report = json.loads(completed.stdout.strip())
        if report["status"] != "PASS":
            raise RuntimeError(f"C_GOLDEN_STATUS:{record['candidate_id']}")
        records.append({
            "candidate_id": record["candidate_id"],
            "engine_id": record["engine_id"],
            "runtime_kind": record["runtime_kind"],
            "output_elems": record["output_elems"],
            "c_result": report,
            "authority": 0,
            "board_accepted": False,
        })
    source_paths = [
        root / "firmware_integration/modelbank_v8/forge200_runtime_v8.h",
        root / "firmware_integration/modelbank_v8/forge200_runtime_v8.c",
        root / "firmware_integration/modelbank_v8/forge200_runtime_host_runner.c",
    ]
    result = {
        "schema": "cimc.forge200.mcu-runtime-c-verification.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_170_PORTABLE_C_GOLDENS_CORTEX_M7_EXECUTION_PENDING",
        "manifest": {"path": manifest_path.relative_to(root).as_posix(), "sha256": sha256(manifest_path)},
        "runner": {"path": runner.relative_to(root).as_posix(), "sha256": sha256(runner)},
        "sources": [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path)} for path in source_paths],
        "model_count": len(records),
        "by_engine": {str(engine): sum(item["engine_id"] == engine for item in records) for engine in (1, 2, 5)},
        "by_category": manifest["by_category"],
        "exact_count": manifest["exact_count"],
        "sim_only_count": manifest["sim_only_count"],
        "records": records,
        "runtime_seconds": time.perf_counter() - started,
        "authority_nonzero": 0,
        "board_actions": 0,
        "claim_boundary": "The portable C source reproduced binary goldens on x86. Cortex-M7 timing, memory, FatFs, cache, MPU, and physical shared-SPI behavior remain board pending.",
    }
    result["content_root_sha256"] = hashlib.sha256(canonical(records)).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "models": len(records), "by_engine": result["by_engine"],
        "runtime_seconds": result["runtime_seconds"], "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
