#!/usr/bin/env python3
"""Capture actual frozen CIMC host outputs used by NanoLM baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(executable: Path, cwd: Path) -> str:
    result = subprocess.run(
        [str(executable)], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"{executable.name}:exit={result.returncode}:{result.stdout}:{result.stderr}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--cimc-root", type=Path, required=True)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    cimc_root = args.cimc_root.resolve()
    host = cimc_root / "model" / "host_test"
    outputs = {
        "new_models": run(host / "new_models_test.exe", host),
        "flagship": run(host / "nanolm_test.exe", host),
        "cluster": run(host / "cluster_test.exe", host),
    }
    if "NEW_MODELS ALL_PASS=1" not in outputs["new_models"]:
        raise RuntimeError("legacy discriminative host baseline failed")
    if "RESULT: ALL_PASS=1" not in outputs["flagship"]:
        raise RuntimeError("legacy flagship host baseline failed")
    if "RESULT: ALL_PASS=1" not in outputs["cluster"]:
        raise RuntimeError("legacy cluster host baseline failed")
    # Parse without relying on locale-dependent spacing in role labels.
    cluster_lines = [line.strip() for line in outputs["cluster"].splitlines() if line.strip().startswith("[")]
    cluster_generations = {
        re.search(r"\[(e\d+\s+[^]]+)\]", line, re.I).group(1).upper(): line.split("  ", 1)[-1].split("  ", 1)[-1].strip()
        for line in cluster_lines
    }
    flagship_generations = [
        line.split(":", 1)[1].strip()
        for line in outputs["flagship"].splitlines()
        if "diagnosis:" in line
    ]
    ai7 = re.search(r"AI-7\s+thermal=([0-9.]+)%.*band=(\d+)", outputs["new_models"])
    if not ai7 or len(flagship_generations) != 3 or len(cluster_generations) != 7:
        raise RuntimeError("legacy output parse failure")
    artifacts = [
        host / "new_models_test.exe",
        host / "nanolm_test.exe",
        host / "cluster_test.exe",
        cimc_root / "firmware" / "ai_models_c" / "nanolm_weights.h",
        cimc_root / "firmware" / "ai_models_c" / "cluster_image.bin",
    ]
    receipt = {
        "schema": "cimc.forge200.legacy-nanolm-baseline-host-receipt.v1",
        "status": "PASS_ACTUAL_FROZEN_HOST_BASELINES",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            {
                "path": str(path.relative_to(cimc_root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in artifacts
        ],
        "actual_outputs": {
            "AI7_thermal_percent": float(ai7.group(1)),
            "AI7_band": int(ai7.group(2)),
            "flagship": flagship_generations,
            "cluster": cluster_generations,
        },
        "host_only_not_board_evidence": True,
        "authority": 0,
        "board_accepted": False,
    }
    output = candidate_root / "evidence" / "legacy_nanolm_baseline_host_receipt.v1.json"
    output.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output": str(output), "sha256": sha256_file(output)}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
