from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CRuntimeBoundsTests(unittest.TestCase):
    def test_forged_payloads_fail_closed_in_native_runtime(self) -> None:
        compiler = next(
            (path for name in ("cc", "gcc", "clang") if (path := shutil.which(name))),
            None,
        )
        if compiler is None:
            self.skipTest("native C compiler unavailable; CI runs this harness with cc")
        runtime = ROOT / "ai/firmware_integration/modelbank_v8"
        harness = ROOT / "tests/c/test_forge200_runtime_bounds.c"
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / ("bounds.exe" if os.name == "nt" else "bounds")
            subprocess.run(
                [
                    compiler,
                    "-std=c11",
                    "-Wall",
                    "-Wextra",
                    "-Werror",
                    "-Werror=type-limits",
                    f"-I{runtime}",
                    str(harness),
                    str(runtime / "forge200_runtime_v8.c"),
                    "-lm",
                    "-o",
                    str(executable),
                ],
                check=True,
                cwd=ROOT,
            )
            completed = subprocess.run(
                [str(executable)],
                check=True,
                cwd=ROOT,
                text=True,
                capture_output=True,
            )
        self.assertEqual(completed.stdout.strip(), "PASS")


if __name__ == "__main__":
    unittest.main()
