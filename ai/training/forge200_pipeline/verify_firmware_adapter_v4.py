#!/usr/bin/env python3
"""Compile and audit the isolated Cortex-M7 ModelBank adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_root(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("content_root_sha256", None)
    raw = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    compiler = args.compiler.resolve()
    source_dir = root / "firmware_integration" / "modelbank_v4"
    args.build_dir.mkdir(parents=True, exist_ok=True)

    source_names = [
        "forge200_modelbank.h", "forge200_modelbank.c",
        "forge200_shared_spi.h", "forge200_shared_spi.c", "README.md",
    ]
    source_records = []
    forbidden_control_symbols = [
        "gpio_bit_set", "gpio_bit_reset", "heater", "fan_enable",
        "motor", "relay", "alarm", "actuator", "pwm",
    ]
    for name in source_names:
        path = source_dir / name
        if not path.is_file():
            raise RuntimeError(f"SOURCE_MISSING:{path}")
        source_records.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
        if path.suffix in {".c", ".h"}:
            lowered = path.read_text(encoding="utf-8").lower()
            for token in forbidden_control_symbols:
                if token in lowered:
                    raise RuntimeError(f"CONTROL_SYMBOL_FORBIDDEN:{name}:{token}")

    compile_flags = [
        "--target=arm-arm-none-eabi", "-mcpu=cortex-m7", "-mthumb",
        "-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(source_dir),
    ]
    objects = []
    for source_name in ["forge200_modelbank.c", "forge200_shared_spi.c"]:
        source = source_dir / source_name
        target = args.build_dir / (source.stem + ".o")
        command = [str(compiler), *compile_flags, "-c", str(source), "-o", str(target)]
        completed = run(command)
        objects.append({
            "source": source_name,
            "object": str(target.relative_to(root)).replace("\\", "/"),
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
            "warnings": 0,
            "errors": 0,
        })
    version = run([str(compiler), "--version"])

    modelbank_text = (source_dir / "forge200_modelbank.c").read_text(encoding="utf-8")
    spi_text = (source_dir / "forge200_shared_spi.c").read_text(encoding="utf-8")
    invariants = {
        "authority_zero_enforced": "raw[12] != 0U" in modelbank_text,
        "reserved_zero_enforced": "i = 204U" in modelbank_text,
        "payload_sha_before_generation": modelbank_text.index("FORGE200_EVENT_SHA256_VERIFIED") < modelbank_text.index("FORGE200_EVENT_GENERATION_VERIFIED"),
        "generation_before_golden": modelbank_text.index("FORGE200_EVENT_GENERATION_VERIFIED") < modelbank_text.index("FORGE200_EVENT_GOLDEN_VERIFIED"),
        "golden_before_commit": modelbank_text.index("FORGE200_EVENT_GOLDEN_VERIFIED") < modelbank_text.index("FORGE200_EVENT_COMMIT, status"),
        "inactive_slot_load": "bank->active_slot ^ 1U" in modelbank_text,
        "rollback_refuse_event": "FORGE200_EVENT_ROLLBACK_REFUSE" in modelbank_text,
        "sd_cs_deasserted": "deassert_sd_cs_pc5" in spi_text,
        "max31856_cs_deasserted": "deassert_max31856_cs_pg3" in spi_text,
        "spi_contention_refused": "collision_refusals" in spi_text,
        "no_control_symbols": True,
    }
    if not all(invariants.values()):
        raise RuntimeError("ADAPTER_INVARIANT_FAILED")

    result = {
        "schema": "cimc.forge200.firmware-adapter-host-compile.v4",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_ARMCLANG_CORTEX_M7_BOARD_PENDING_NOT_IN_PRODUCTION_TARGET",
        "compiler": {
            "path": str(compiler),
            "version": (version.stdout + version.stderr).strip().splitlines(),
            "flags": compile_flags,
        },
        "sources": source_records,
        "objects": objects,
        "invariants": invariants,
        "authority": 0,
        "production_files_modified": 0,
        "board_actions": 0,
        "claim_boundary": "Cortex-M7 compilation and host static audit only; Keil target link and GD32 execution remain pending.",
    }
    result["content_root_sha256"] = canonical_root(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "objects": len(objects), "content_root_sha256": result["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
