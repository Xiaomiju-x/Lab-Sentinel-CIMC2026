from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class DocumentationContractTests(unittest.TestCase):
    def test_readme_truth_phrases(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        for phrase in (
            "30 个运行资产 / 28 个逻辑模型",
            "HOST–EXACT",
            "HOST–SIM_ONLY",
            "authority=0",
            "新增板端执行数为 **0**",
            "全国初赛特等奖",
            "全国决赛国家二等奖",
        ):
            self.assertIn(phrase, text)

    def test_no_stage_story(self) -> None:
        for name in ("README.md", "README_EN.md"):
            text = (ROOT / name).read_text(encoding="utf-8")
            self.assertNotIn("评委不懂", text)
            self.assertNotIn("全球第一", text)
            # Allow an explicit negation such as “并非 200 个板上模型”, while
            # rejecting the misleading deployment claims that this repository
            # is designed to prevent.
            self.assertNotIn("200 个模型已上板", text)
            self.assertNotIn("200 个板上模型同时运行", text)
            self.assertNotIn("200 board-deployed models", text.lower())


if __name__ == "__main__":
    unittest.main()
