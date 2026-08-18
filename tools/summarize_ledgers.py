#!/usr/bin/env python3
"""Print the public BOARD/HOST ledgers from machine-readable evidence."""

from __future__ import annotations

from _common import load_json


def main() -> int:
    gap = load_json("evidence/public/release_gap_audit.v7.json")
    board = load_json(
        "evidence/public/forge200_correct32gb_sharedbus_hardware_retest_20260806_141457_receipt.v9.json"
    )
    baseline = gap["initial_board_baseline"]
    total = gap["host_total_by_category"]
    exact = gap["host_exact_source_bound"]
    sim = gap["host_sim_only_extensions"]
    print(f"BOARD: {baseline['assets']} runtime assets / {baseline['logical_models']} logical models")
    print(f"HOST: {gap['host_total_including_sim_only']} = P{total['P']} + G{total['G']} + S{total['S']}")
    print(f"HOST-EXACT: {exact['total']} = P{exact['by_category']['P']} + G{exact['by_category']['G']} + S{exact['by_category']['S']}")
    print(f"HOST-SIM_ONLY: {sim['total']} = P{sim['by_category']['P']} + G{sim['by_category']['G']} + S{sim['by_category']['S']}")
    print(f"NEW BOARD EXECUTION: {board['forge200']['models']}")
    print(f"BOARD RECEIPT: {board['status']} (accepted={board['accepted']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

