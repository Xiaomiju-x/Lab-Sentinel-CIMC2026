"""Shared, dependency-free helpers for the public release tools."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def iter_public_files() -> list[Path]:
    excluded_roots = {".git", ".pytest_cache", "__pycache__"}
    excluded_files = {"PUBLIC_RELEASE_MANIFEST.json", "SBOM.spdx.json"}
    result: list[Path] = []
    candidates: list[Path]
    if (ROOT / ".git").exists():
        process = subprocess.run(
            [
                "git", "-C", str(ROOT), "ls-files", "-z", "--cached",
                "--others", "--exclude-standard",
            ],
            check=False,
            capture_output=True,
        )
        if process.returncode != 0:
            raise RuntimeError("git ls-files failed while enumerating the public payload")
        candidates = [
            ROOT / item.decode("utf-8")
            for item in process.stdout.split(b"\0")
            if item
        ]
    else:
        candidates = list(ROOT.rglob("*"))
    for path in candidates:
        if not path.is_file() or path.name in excluded_files:
            continue
        relative = path.relative_to(ROOT)
        if any(part in excluded_roots for part in relative.parts):
            continue
        result.append(path)
    return sorted(result, key=lambda value: value.relative_to(ROOT).as_posix())
