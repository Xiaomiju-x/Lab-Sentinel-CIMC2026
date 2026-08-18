from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicLedgerTests(unittest.TestCase):
    def load(self, name: str) -> dict:
        return json.loads((ROOT / "evidence/public" / name).read_text(encoding="utf-8"))

    def test_three_ledgers(self) -> None:
        gap = self.load("release_gap_audit.v7.json")
        self.assertEqual(gap["initial_board_baseline"]["assets"], 30)
        self.assertEqual(gap["initial_board_baseline"]["logical_models"], 28)
        self.assertEqual(gap["host_exact_source_bound"]["total"], 78)
        self.assertEqual(gap["host_sim_only_extensions"]["total"], 92)
        self.assertEqual(gap["host_total_including_sim_only"], 170)
        self.assertEqual(gap["host_total_by_category"], {"P": 112, "G": 30, "S": 28})

    def test_board_rejection_is_preserved(self) -> None:
        receipt = self.load("forge200_correct32gb_sharedbus_hardware_retest_20260806_141457_receipt.v9.json")
        self.assertFalse(receipt["accepted"])
        self.assertEqual(receipt["status"], "GD32_UNIFIED_BOARD_REJECTED")
        self.assertEqual(receipt["forge200"]["models"], 0)
        self.assertEqual(receipt["public_count_if_accepted"], 30)

    def test_authority_is_zero(self) -> None:
        closure = self.load("host_closure.v7.json")
        self.assertEqual(closure["authority_nonzero"], 0)
        self.assertEqual(closure["integrity"]["package_collisions"], 0)
        self.assertEqual(closure["integrity"]["payload_collisions"], 0)


if __name__ == "__main__":
    unittest.main()

