#!/usr/bin/env python3
"""Positive and fail-closed mutation tests for the v9 UART parsers."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(command: list[str], expected: int) -> None:
    result = subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)
    if result.returncode != expected:
        raise RuntimeError(f"PARSER_RESULT:{result.returncode}:{result.stdout}:{result.stderr}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    temporary = root / ".tmp/parser_v9"
    temporary.mkdir(parents=True, exist_ok=True)
    text = (root / ".tmp/synthetic_board_pass_v8.log").read_text(encoding="utf-8")
    sd_line = next(line for line in text.splitlines() if "event=SD_READY" in line)
    additions = "\n".join((
        "@F2BOARD|v=1|event=VERIPROCESS|ledger_generation=2|ledger_records=2|chrono_events=11|independent_families=2|ds3231=1|wal_recovered=1|sintergraph_frozen=1|authority=0|control=unchanged",
        "@F2BOARD|v=1|event=RAG120|queries=120|safe=120|source_bound=5|refused=115|negative_refused=60|cold_p95_ms=15000|cold_p99_ms=18000|warm_p95_ms=6000|warm_p99_ms=7000|warm_hits=114|warm_hit_percent=95|authority=0|control=unchanged",
        "@F2BOARD|v=1|event=RAG_STAGE|load_support_p95_ms=1000|route_retrieve_p95_ms=1000|load_lm_p95_ms=1000|generate_p95_ms=1000|unload_p95_ms=10|nli_p95_ms=100|commit_p95_ms=10|zeroize_p95_ms=10|authority=0|control=unchanged",
    ))
    text = text.replace(sd_line, sd_line + "\n" + additions)
    text = text.replace("event=SWAP1000|swap_loads=1000|", "event=SWAP1000|swap_loads=1000|same_model_reload=10|mid_cancel=4|")
    text = text.replace("event=PASS|models=170|initial_exact=78|initial_sim_only=92|", "event=PASS|models=170|initial_exact=78|initial_sim_only=92|rag120_safe=120|rag120_source_bound=5|soak_rag_queries=288|")
    positive = temporary / "synthetic_board_pass_v9.log"
    positive.write_text(text, encoding="utf-8")
    output = temporary / "positive.json"
    command = [sys.executable, str(root / "pipeline/parse_board_receipt_v9.py"), "--log", str(positive), "--output", str(output)]
    run(command, 0)
    if not json.loads(output.read_text(encoding="utf-8"))["accepted"]:
        raise RuntimeError("POSITIVE_GATE")
    negative = temporary / "synthetic_board_reject_v9.log"
    negative.write_text(text.replace("negative_refused=60", "negative_refused=59"), encoding="utf-8")
    run([*command[:3], str(negative), *command[4:-1], str(temporary / "negative.json")], 2)

    armed = temporary / "armed.log"
    recovered = temporary / "recovered.log"
    armed.write_text("@F2BOARD|v=1|event=POWER_CUT_ARMED|component=VERIPROCESS|wal_synced=1|header_flipped=0|instruction=REMOVE_BOARD_POWER_NOW|authority=0|control=unchanged\n", encoding="utf-8")
    recovered.write_text(
        "@F2BOARD|v=1|event=VERIPROCESS|ledger_generation=2|ledger_records=2|chrono_events=11|independent_families=2|ds3231=1|wal_recovered=1|sintergraph_frozen=1|authority=0|control=unchanged\n"
        "@F2BOARD|v=1|event=SWAP_PROGRESS|swap_loads=200|total_loads=370|min_swap_per_model=1|active_slot=1|failures=0|authority=0|control=unchanged\n"
        "@F2BOARD|v=1|event=STOP|reason=SWAP1000_LOAD|authority=0|control=unchanged\n",
        encoding="utf-8",
    )
    recovery_output = temporary / "recovery.json"
    recovery_command = [sys.executable, str(root / "pipeline/parse_powercut_recovery_v9.py"), "--armed-log", str(armed), "--recovered-log", str(recovered), "--output", str(recovery_output)]
    run(recovery_command, 0)
    recovered.write_text(recovered.read_text(encoding="utf-8").replace("wal_recovered=1", "wal_recovered=0"), encoding="utf-8")
    run([*recovery_command[:-1], str(temporary / "recovery_negative.json")], 2)
    recovered.write_text(
        recovered.read_text(encoding="utf-8")
        .replace("wal_recovered=0", "wal_recovered=1")
        .replace("event=STOP|reason=SWAP1000_LOAD", "event=STOP|reason=FINAL_RESOURCE_OR_CONTROL_P99"),
        encoding="utf-8",
    )
    run([*recovery_command[:-1], str(temporary / "recovery_wrong_stop.json")], 2)
    result = {
        "schema": "cimc.forge200.board-parser-tests.v9",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "cases": 5,
        "positive": 2,
        "fail_closed_mutations": 3,
        "authority_nonzero": 0,
        "board_actions": 0,
    }
    result["content_root_sha256"] = hashlib.sha256(json.dumps(
        {"cases": result["cases"], "positive": result["positive"],
         "fail_closed_mutations": result["fail_closed_mutations"]},
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    output_path = (args.output or root / "evidence/board_parser_tests.v9.json").resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
