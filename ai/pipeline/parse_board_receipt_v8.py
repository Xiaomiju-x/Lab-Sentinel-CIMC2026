#!/usr/bin/env python3
"""Parse the final unified GD32 UART log into a fail-closed board receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
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


def as_int(fields: dict[str, str], name: str, default: int = -1) -> int:
    try:
        return int(fields.get(name, str(default)))
    except ValueError:
        return default


def parse_f2_lines(text: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        marker = line.find("@F2BOARD|")
        if marker < 0:
            continue
        payload = line[marker:].strip()
        fields: dict[str, str] = {}
        for item in payload.split("|")[1:]:
            if "=" in item:
                key, value = item.split("=", 1)
                fields[key] = value
        records.append(
            {"line": line_number, "event": fields.get("event", ""), "fields": fields}
        )
    return records


def one(events: dict[str, list[dict[str, object]]], name: str) -> dict[str, str] | None:
    rows = events.get(name, [])
    if len(rows) != 1:
        return None
    return rows[0]["fields"]  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    log_path = args.log.resolve()
    output = args.output.resolve()
    raw = log_path.read_bytes()
    text = raw.decode("utf-8", errors="replace")
    records = parse_f2_lines(text)
    events: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        events[str(record["event"])].append(record)
    errors: list[str] = []

    if not re.search(
        r"\[REG\]\s+mode=ROOT_EGUARD_R2\s+init=PASS\s+models=30\s+authority=0",
        text,
    ):
        errors.append("INITIAL_30_REGISTRY_PASS_MISSING")
    required_legacy_patterns = {
        "INITIAL_AI_CORE": r"\[AI\] selftest AI1=1 .* ALL=PASS",
        "INITIAL_AI_CAM_INT8": r"\[AI\] selftest CAM=1 INT8=1 ACONF=1\s+ALL=PASS",
        "INITIAL_NANOLM_ONLINE": r"\[AI\] selftest NLM=1 .* OL=1 ",
        "INITIAL_CLUSTER7": r"\[AI\] selftest CLUSTER=1 \(7 experts",
        "INITIAL_BANK2": r"\[AI\] selftest BANK=1 \(2 sizes",
    }
    for name, pattern in required_legacy_patterns.items():
        if re.search(pattern, text) is None:
            errors.append(f"{name}_MISSING")

    if events.get("STOP"):
        errors.append("FORGE200_STOP_PRESENT")
    for record in records:
        fields = record["fields"]
        if fields.get("v") != "1" or fields.get("authority") != "0":
            errors.append(f"ENVELOPE_GATE_LINE_{record['line']}")
        if fields.get("control") != "unchanged":
            errors.append(f"CONTROL_BOUNDARY_GATE_LINE_{record['line']}")

    begin = one(events, "BEGIN")
    if begin is None or any(
        as_int(begin, key) != value
        for key, value in {
            "models": 170,
            "exact": 78,
            "sim_only": 92,
            "target_swaps": 1000,
            "soak_hours": 24,
        }.items()
    ):
        errors.append("BEGIN_GATE")

    timing = one(events, "CONTROL_TIMING")
    baseline_p99 = as_int(timing or {}, "control_p99_ms")
    if (
        timing is None
        or timing.get("phase") != "baseline"
        or as_int(timing, "samples") < 50
        or baseline_p99 <= 0
    ):
        errors.append("CONTROL_BASELINE_GATE")

    resource = one(events, "RESOURCE")
    if (
        resource is None
        or as_int(resource, "heap_min") < 16384
        or as_int(resource, "critical_stack_min") < 1536
        or as_int(resource, "heap_gate") != 16384
        or as_int(resource, "stack_gate") != 1536
    ):
        errors.append("RESOURCE_BASELINE_GATE")

    sd_ready = one(events, "SD_READY")
    if (
        sd_ready is None
        or as_int(sd_ready, "capacity_mb") <= 0
        or sd_ready.get("fs") not in {"FAT32", "EXFAT"}
        or as_int(sd_ready, "cid") != 1
        or as_int(sd_ready, "csd") != 1
    ):
        errors.append("SD_READY_GATE")

    catalog = one(events, "CATALOG")
    if (
        catalog is None
        or as_int(catalog, "a") != 1
        or as_int(catalog, "b") != 1
        or catalog.get("selected") not in {"A", "B"}
        or as_int(catalog, "generation") < 1
        or as_int(catalog, "entries") != 170
    ):
        errors.append("CATALOG_AB_GATE")

    sd_bench = one(events, "SD_BENCH")
    if (
        sd_bench is None
        or as_int(sd_bench, "sequential_kib_s") < 512
        or as_int(sd_bench, "random_4k_pages_s") <= 0
        or as_int(sd_bench, "min_kib_s") != 512
    ):
        errors.append("SD_BENCH_GATE")

    model_rows = events.get("MODEL", [])
    model_ids: set[str] = set()
    categories: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    model_load_ms: list[int] = []
    for record in model_rows:
        fields = record["fields"]
        model_id = str(fields.get("model", ""))
        if (
            not re.fullmatch(r"CAND-[PGS]-\d{3}", model_id)
            or model_id in model_ids
            or as_int(fields, "status") != 0
            or as_int(fields, "dwt") <= 0
            or as_int(fields, "bytes") <= 256
            or fields.get("slot") not in {"0", "1"}
        ):
            errors.append(f"MODEL_ROW_GATE_LINE_{record['line']}")
            continue
        model_ids.add(model_id)
        categories[str(fields.get("cat", ""))] += 1
        tiers[str(fields.get("tier", ""))] += 1
        model_load_ms.append(as_int(fields, "load_ms"))
    if len(model_ids) != 170:
        errors.append("MODEL_UNIQUE_170_GATE")
    if dict(categories) != {"G": 30, "P": 112, "S": 28}:
        errors.append("MODEL_CATEGORY_GATE")
    if dict(tiers) != {"EXACT": 78, "SIM_ONLY": 92}:
        errors.append("MODEL_TIER_GATE")

    batch = one(events, "BATCH170")
    if (
        batch is None
        or as_int(batch, "passed") != 170
        or as_int(batch, "exact") != 78
        or as_int(batch, "sim_only") != 92
        or as_int(batch, "package_commits") != 170
    ):
        errors.append("BATCH170_GATE")

    faults = one(events, "FAULTS")
    for name in ("bad_magic", "bad_authority", "payload", "rollback", "engine", "golden"):
        if faults is None or faults.get(name) != "REFUSED":
            errors.append(f"FAULT_{name.upper()}_GATE")

    swap = one(events, "SWAP1000")
    if (
        swap is None
        or as_int(swap, "swap_loads") != 1000
        or as_int(swap, "min_swap_per_model") < 4
        or as_int(swap, "total_loads") < 1170
        or as_int(swap, "load_p95_ms") < 0
        or as_int(swap, "load_p99_ms") < as_int(swap, "load_p95_ms")
        or as_int(swap, "load_max_ms") < as_int(swap, "load_p99_ms")
        or swap.get("canary") != "PASS"
        or as_int(swap, "generation") < 1
        or as_int(swap, "max31856_acq_delta") <= 0
        or as_int(swap, "timeouts") != 0
        or as_int(swap, "collisions") != 0
        or as_int(swap, "control_p99_baseline_ms") != baseline_p99
        or as_int(swap, "control_p99_active_ms") * 100 > baseline_p99 * 105
    ):
        errors.append("SWAP1000_GATE")

    soak_begin = one(events, "SOAK_BEGIN")
    if (
        soak_begin is None
        or as_int(soak_begin, "hours") != 24
        or as_int(soak_begin, "period_s") != 300
        or as_int(soak_begin, "loads_per_period") != 2
        or as_int(soak_begin, "rag_golden_per_hour") != 12
    ):
        errors.append("SOAK_BEGIN_GATE")
    soak_hours = {
        as_int(record["fields"], "hour") for record in events.get("SOAK_HOUR", [])
    }
    if soak_hours != set(range(1, 25)):
        errors.append("SOAK_24_HOUR_GATE")
    soak_fault_hours = {
        as_int(record["fields"], "hour") for record in events.get("SOAK_FAULT", [])
    }
    if soak_fault_hours != set(range(2, 25, 2)):
        errors.append("SOAK_FAULT_EVERY_2H_GATE")

    final = one(events, "PASS")
    if (
        final is None
        or as_int(final, "models") != 170
        or as_int(final, "initial_exact") != 78
        or as_int(final, "initial_sim_only") != 92
        or as_int(final, "swap_loads") != 1000
        or as_int(final, "min_swap_per_model") < 4
        or as_int(final, "failures") != 0
        or as_int(final, "sd_kib_s") < 512
        or as_int(final, "timeouts") != 0
        or as_int(final, "collisions") != 0
        or as_int(final, "control_p99_ms") * 100 > baseline_p99 * 105
        or as_int(final, "heap_min") < 16384
        or as_int(final, "critical_stack_min") < 1536
    ):
        errors.append("FINAL_PASS_GATE")

    result = {
        "schema": "cimc.forge200.gd32-board-acceptance.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "GD32_UNIFIED_BOARD_ACCEPTED"
            if not errors
            else "GD32_UNIFIED_BOARD_REJECTED"
        ),
        "accepted": not errors,
        "uart_log": {
            "path": str(log_path),
            "bytes": len(raw),
            "sha256": sha256(log_path),
            "f2_records": len(records),
        },
        "initial_board_baseline": {
            "assets": 30,
            "logical_models": 28,
            "registry_pass": "INITIAL_30_REGISTRY_PASS_MISSING" not in errors,
        },
        "forge200": {
            "models": len(model_ids),
            "by_category": dict(sorted(categories.items())),
            "by_tier": dict(sorted(tiers.items())),
            "model_load_ms_max": max(model_load_ms, default=-1),
            "swap_loads": as_int(swap or {}, "swap_loads"),
            "soak_hours_seen": sorted(soak_hours),
            "soak_fault_hours_seen": sorted(soak_fault_hours),
            "authority_nonzero": 0,
        },
        "public_count_if_accepted": 200 if not errors else 30,
        "public_exact_source_bound_new": 78 if not errors else 0,
        "public_sim_only_new": 92 if not errors else 0,
        "logical_generative_if_accepted": 38 if not errors else 8,
        "errors": sorted(set(errors)),
        "claim_boundary": (
            "Board acceptance does not erase evidence tiers: 78 new assets are "
            "EXACT source/label/split-bound and 92 remain explicitly SIM_ONLY. "
            "All new assets have authority=0 and do not control actuators."
        ),
    }
    result["content_root_sha256"] = hashlib.sha256(
        canonical(
            {
                "uart_log": result["uart_log"],
                "initial_board_baseline": result["initial_board_baseline"],
                "forge200": result["forge200"],
                "errors": result["errors"],
            }
        )
    ).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "models": len(model_ids),
                "errors": len(result["errors"]),
                "content_root_sha256": result["content_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
