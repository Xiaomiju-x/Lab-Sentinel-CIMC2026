#!/usr/bin/env python3
"""Fail-closed receipt for the physical WAL-before-header power-cut drill."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--armed-log", type=Path, required=True)
    parser.add_argument("--recovered-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    armed_raw = args.armed_log.resolve().read_bytes()
    recovered_raw = args.recovered_log.resolve().read_bytes()
    armed = armed_raw.decode("utf-8", errors="replace")
    recovered = recovered_raw.decode("utf-8", errors="replace")
    errors = []
    if "event=POWER_CUT_ARMED|component=VERIPROCESS|wal_synced=1|header_flipped=0" not in armed:
        errors.append("DURABLE_WAL_ARM_EVENT_MISSING")
    if "event=STOP" in armed:
        errors.append("UNEXPECTED_STOP_IN_ARM_LOG")
    match = re.search(
        r"event=VERIPROCESS\|ledger_generation=(\d+)\|ledger_records=(\d+)\|"
        r"chrono_events=11\|independent_families=(\d+)\|ds3231=1\|"
        r"wal_recovered=1\|sintergraph_frozen=1\|authority=0",
        recovered,
    )
    if match is None or int(match.group(1)) < 1 or int(match.group(2)) < 2 or int(match.group(3)) < 2:
        errors.append("RECOVERED_LEDGER_EVENT_MISSING")
    recovery_offset = match.start() if match is not None else -1
    progress = re.search(r"event=SWAP_PROGRESS\|swap_loads=200\|", recovered)
    expected_stop = re.search(r"event=STOP\|reason=SWAP1000_LOAD(?:\||\r?$)", recovered, re.MULTILINE)
    stop_events = list(re.finditer(r"event=STOP\|reason=([^|\r\n]+)", recovered))
    if progress is None:
        errors.append("SD_REMOVAL_ARM_PROGRESS_MISSING")
    if expected_stop is None:
        errors.append("SD_REMOVAL_FAIL_CLOSED_STOP_MISSING")
    if (match is not None and progress is not None and expected_stop is not None and
            not (recovery_offset < progress.start() < expected_stop.start())):
        errors.append("RECOVERY_SD_REMOVAL_EVENT_ORDER")
    if any(item.group(1) != "SWAP1000_LOAD" for item in stop_events):
        errors.append("UNEXPECTED_STOP_IN_RECOVERED_LOG")
    if "event=PASS" in recovered:
        errors.append("PASS_AFTER_SD_REMOVAL")
    result = {
        "schema": "cimc.forge200.veriprocess-physical-powercut.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PHYSICAL_POWER_CUT_RECOVERY_ACCEPTED" if not errors else "PHYSICAL_POWER_CUT_RECOVERY_REJECTED",
        "accepted": not errors,
        "armed_log": {"path": str(args.armed_log.resolve()), "bytes": len(armed_raw), "sha256": sha256(args.armed_log.resolve())},
        "recovered_log": {"path": str(args.recovered_log.resolve()), "bytes": len(recovered_raw), "sha256": sha256(args.recovered_log.resolve())},
        "wal_synced_before_power_cut": "DURABLE_WAL_ARM_EVENT_MISSING" not in errors,
        "header_not_flipped_before_power_cut": "DURABLE_WAL_ARM_EVENT_MISSING" not in errors,
        "wal_recovered_after_power_restore": "RECOVERED_LEDGER_EVENT_MISSING" not in errors,
        "sd_removed_after_swap_loads_200": "SD_REMOVAL_ARM_PROGRESS_MISSING" not in errors,
        "sd_removal_failed_closed": "SD_REMOVAL_FAIL_CLOSED_STOP_MISSING" not in errors,
        "errors": errors,
        "authority_nonzero": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(json.dumps(
        {"armed": result["armed_log"], "recovered": result["recovered_log"], "errors": errors},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "errors": len(errors), "content_root_sha256": result["content_root_sha256"]}, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
