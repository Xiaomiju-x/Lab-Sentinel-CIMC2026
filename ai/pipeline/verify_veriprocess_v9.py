#!/usr/bin/env python3
"""Strict host/Cortex-M7 verification for the executable VeriProcess v9 core."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--cimc", type=Path, required=True)
    parser.add_argument("--zig", type=Path, required=True)
    parser.add_argument("--armclang", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    cimc = args.cimc.resolve()
    source = root / "firmware_integration/veriprocess_v9"
    lab = cimc / "firmware/keil_proj/HardWare/Lab_Sentinel"
    build = root / ".tmp/veriprocess_v9_acceptance"
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "veriprocess_host_test_v9.exe"
    arm_object = build / "veriprocess_v9_arm.o"
    sources = [source / "veriprocess_v9.c", source / "veriprocess_host_test_v9.c"]
    forbidden = ("gpio_bit_set", "gpio_bit_reset", "heater", "fan_enable", "relay", "pwm", "actuator_command")
    matched = {path.name: [token for token in forbidden if token in path.read_text(encoding="utf-8").lower()] for path in sources}
    matched = {name: tokens for name, tokens in matched.items() if tokens}
    if matched:
        raise RuntimeError(f"CONTROL_SYMBOL_GATE:{matched}")

    common = ["-std=c11", "-Wall", "-Wextra", "-Werror", "-I", str(source), "-I", str(lab)]
    host_command = [
        str(args.zig.resolve()), "cc", "-O2", *common,
        str(sources[0]), str(sources[1]), str(lab / "sha256.c"), "-lm", "-o", str(executable),
    ]
    host_compile = subprocess.run(host_command, text=True, capture_output=True, timeout=180, check=False)
    if host_compile.returncode or host_compile.stdout.strip() or host_compile.stderr.strip():
        raise RuntimeError(f"HOST_COMPILE:{host_compile.returncode}:{host_compile.stdout}:{host_compile.stderr}")
    host_run = subprocess.run([str(executable)], text=True, capture_output=True, timeout=60, check=False)
    if host_run.returncode or host_run.stderr.strip():
        raise RuntimeError(f"HOST_TEST:{host_run.returncode}:{host_run.stdout}:{host_run.stderr}")
    host_result = json.loads(host_run.stdout)
    if host_result != {"status": "PASS", "passed": 69, "failed": 0, "authority": 0, "board_accepted": False}:
        raise RuntimeError(f"HOST_RESULT:{host_result}")

    arm_flags = [
        "--target=arm-arm-none-eabi", "-mcpu=cortex-m7", "-mthumb",
        *common, "-c", str(sources[0]), "-o", str(arm_object),
    ]
    arm_compile = subprocess.run([str(args.armclang.resolve()), *arm_flags], text=True, capture_output=True, timeout=180, check=False)
    if arm_compile.returncode or arm_compile.stdout.strip() or arm_compile.stderr.strip():
        raise RuntimeError(f"ARM_COMPILE:{arm_compile.returncode}:{arm_compile.stdout}:{arm_compile.stderr}")

    schemas = root / "contracts/schemas"
    records = {
        "evidence_card_v2": sha256(schemas / "evidence_card_v2.schema.json"),
        "sintergraph_psp_r1": sha256(schemas / "sintergraph_psp_r1.schema.json"),
        "chronospec_r4_events": sha256(schemas / "chronospec_r4.events.v1.json"),
        "source": sha256(sources[0]),
        "header": sha256(source / "veriprocess_v9.h"),
        "host_test": sha256(sources[1]),
        "host_executable": sha256(executable),
        "arm_object": sha256(arm_object),
    }
    receipt = {
        "schema": "cimc.forge200.veriprocess-host-acceptance.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_HOST_AND_ARMCLANG_BOARD_STORAGE_PENDING",
        "implementation": {
            "evidence_card_v2_canonical_sha256": True,
            "sintergraph_preburn_cutoff_and_same_run_postburn_rejection": True,
            "twinguard_three_fulfillment_horizons": True,
            "root_independent_family_gate": True,
            "chronospec_r4_order_and_deadline_state_machine": True,
            "proofpass_r3_payload_binding": True,
            "traceledger_ab_headers": True,
            "traceledger_wal_prepare_sync_recovery": True,
            "traceledger_segment_root": True,
            "ds3231_and_monotonic_regression_rejection": True,
        },
        "host_cases": host_result,
        "compile": {
            "host_werror": True,
            "armclang_cortex_m7_werror": True,
            "armclang_flags": arm_flags,
            "warnings": 0,
            "errors": 0,
        },
        "hashes": records,
        "control_symbol_matches": matched,
        "authority_nonzero": 0,
        "board_accepted": False,
        "board_pending": [
            "FATFS_AB_HEADER_PHYSICAL_SYNC",
            "POWER_CUT_BETWEEN_WAL_AND_HEADER_FLIP",
            "DS3231_LIVE_TIME",
            "CHRONOSPEC_R4_LIVE_EVENT_LATENCY",
            "24H_TRACELEDGER_CONCURRENCY",
        ],
    }
    receipt["content_root_sha256"] = hashlib.sha256(json.dumps(
        {"implementation": receipt["implementation"], "hashes": records, "host": host_result},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output = root / "evidence/veriprocess_host_acceptance.v9.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"], "host_cases": host_result["passed"],
        "arm_object_bytes": arm_object.stat().st_size,
        "content_root_sha256": receipt["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
