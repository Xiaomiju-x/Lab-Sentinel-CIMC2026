#!/usr/bin/env python3
"""Fail if the compact public evidence no longer supports the README ledgers."""

from __future__ import annotations

from _common import load_json


def main() -> int:
    gap = load_json("evidence/public/release_gap_audit.v7.json")
    closure = load_json("evidence/public/host_closure.v7.json")
    board = load_json(
        "evidence/public/forge200_correct32gb_sharedbus_hardware_retest_20260806_141457_receipt.v9.json"
    )
    sd = load_json("evidence/public/forge200_new_sdmodule_three_retest_comparison_20260806.json")
    checks = {
        "board_baseline": gap["initial_board_baseline"] == {"assets": 30, "logical_models": 28, "logical_generative_models": 8},
        "host_total": gap["host_total_including_sim_only"] == 170,
        "host_category": gap["host_total_by_category"] == {"P": 112, "G": 30, "S": 28},
        "exact": gap["host_exact_source_bound"]["total"] == 78,
        "sim_only": gap["host_sim_only_extensions"]["total"] == 92,
        "authority_zero": closure["authority_nonzero"] == 0,
        "hash_unique": closure["integrity"]["package_collisions"] == 0 and closure["integrity"]["payload_collisions"] == 0,
        "board_rejected": board["accepted"] is False and board["status"] == "GD32_UNIFIED_BOARD_REJECTED",
        "new_board_zero": board["forge200"]["models"] == 0 and board["public_count_if_accepted"] == 30,
        "three_crc_failures": sd["summary"]["attempts"] == 3 and sd["summary"]["crc_fail"] == 3,
    }
    failed = [name for name, passed in checks.items() if not passed]
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if failed:
        raise SystemExit("Evidence verification failed: " + ", ".join(failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

