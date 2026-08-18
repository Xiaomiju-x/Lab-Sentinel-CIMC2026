#!/usr/bin/env python3
"""Compile the v4 loader/SPI guard plus v8 inference runtime for Cortex-M7."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    compiler = args.compiler.resolve()
    build = args.build_dir.resolve()
    runtime_root = args.runtime_root.resolve()
    build.mkdir(parents=True, exist_ok=True)

    sources = [
        root / "firmware_integration/modelbank_v4/forge200_modelbank.c",
        root / "firmware_integration/modelbank_v4/forge200_shared_spi.c",
        root / "firmware_integration/modelbank_v8/forge200_runtime_v8.c",
    ]
    includes = [
        root / "firmware_integration/modelbank_v4",
        root / "firmware_integration/modelbank_v8",
    ]
    flags = ["--target=arm-arm-none-eabi", "-mcpu=cortex-m7", "-mthumb", "-std=c11", "-Wall", "-Wextra", "-Werror"]
    for include in includes:
        flags.extend(("-I", str(include)))
    objects = []
    forbidden = ("gpio_bit_set", "gpio_bit_reset", "heater", "fan_enable", "relay", "alarm", "actuator", "pwm")
    for source in sources:
        text = source.read_text(encoding="utf-8").lower()
        matched = [token for token in forbidden if token in text]
        if matched:
            raise RuntimeError(f"CONTROL_SYMBOL_GATE:{source.name}:{matched}")
        target = build / (source.stem + ".o")
        completed = subprocess.run([str(compiler), *flags, "-c", str(source), "-o", str(target)], text=True, capture_output=True)
        if completed.returncode != 0 or completed.stdout.strip() or completed.stderr.strip():
            raise RuntimeError(f"ARMCLANG_GATE:{source.name}:{completed.returncode}:{completed.stdout}:{completed.stderr}")
        objects.append({
            "source": source.relative_to(root).as_posix(),
            "source_sha256": sha256(source),
            "object": target.relative_to(root).as_posix(),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "warnings": 0,
            "errors": 0,
        })

    manifest = json.loads((runtime_root / "MANIFEST.v8.json").read_text(encoding="utf-8"))
    memory_records = []
    for record in manifest["records"]:
        package = (runtime_root / record["package"]["path"]).read_bytes()
        scratch, arena, kv = struct.unpack_from("<III", package, 32)
        runtime_workspace_elems = struct.unpack_from("<I", package, 256 + 36)[0]
        memory_records.append({
            "candidate_id": record["candidate_id"],
            "package_bytes": len(package),
            "scratch_bytes": scratch,
            "arena_bytes": arena,
            "kv_bytes": kv,
            "arena_plus_kv_bytes": arena + kv,
            "runtime_workspace_elems": runtime_workspace_elems,
        })
        if len(package) > 7_340_032 or arena + kv > 2_621_440:
            raise RuntimeError(f"ABI_MEMORY_GATE:{record['candidate_id']}")
        if record["engine_id"] == 5 and len(package) > 2_097_152:
            raise RuntimeError(f"NANOLM_PACKAGE_GATE:{record['candidate_id']}")
    version = subprocess.run([str(compiler), "--version"], text=True, capture_output=True, check=True)
    result = {
        "schema": "cimc.forge200.firmware-runtime-host-compile.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ARMCLANG_CORTEX_M7_THREE_OBJECTS_RUNTIME_LINK_AND_BOARD_PENDING",
        "compiler": {"path": str(compiler), "version": (version.stdout + version.stderr).strip().splitlines(), "flags": flags},
        "objects": objects,
        "runtime_manifest": {"path": (runtime_root / "MANIFEST.v8.json").relative_to(root).as_posix(), "sha256": sha256(runtime_root / "MANIFEST.v8.json")},
        "models_checked": len(memory_records),
        "memory": {
            "max_package_bytes": max(item["package_bytes"] for item in memory_records),
            "max_arena_plus_kv_bytes": max(item["arena_plus_kv_bytes"] for item in memory_records),
            "max_runtime_workspace_elems": max(item["runtime_workspace_elems"] for item in memory_records),
            "package_limit_violations": 0,
            "arena_plus_kv_limit_violations": 0,
            "nanolm_package_limit_violations": 0,
        },
        "authority_nonzero": 0,
        "production_files_modified": 0,
        "board_actions": 0,
        "claim_boundary": "Three Cortex-M7 objects compile with strict warnings-as-errors. Final Keil link, SDRAM placement, FatFs callbacks, timing, and physical execution remain pending.",
    }
    result["content_root_sha256"] = hashlib.sha256(
        json.dumps({"objects": objects, "memory": result["memory"]}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "objects": len(objects), "models": len(memory_records),
        "memory": result["memory"], "content_root_sha256": result["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
