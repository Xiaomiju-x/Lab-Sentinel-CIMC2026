#!/usr/bin/env python3
"""Fail when Markdown/HTML documentation points at a missing local file."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HTML_ASSET = re.compile(r"(?:src|href)=[\"']([^\"']+)[\"']", re.IGNORECASE)
VENDORED_DOC_ROOTS = (
    ROOT / "firmware/lvgl_ui/lvgl-8.3.11",
    ROOT / "third_party",
)


def local_target(raw: str, source: Path) -> Path | None:
    target = raw.strip().split(maxsplit=1)[0].strip("<>\"'")
    if not target or target.startswith(("#", "http://", "https://", "mailto:", "data:")):
        return None
    target = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not target:
        return None
    if target.startswith("/"):
        return ROOT / target.lstrip("/")
    return source.parent / target


def main() -> int:
    missing: list[str] = []
    for source in sorted(ROOT.rglob("*.md")):
        if ".git" in source.parts:
            continue
        if any(source.is_relative_to(vendor) for vendor in VENDORED_DOC_ROOTS):
            continue
        text = source.read_text(encoding="utf-8")
        links = MARKDOWN_LINK.findall(text) + HTML_ASSET.findall(text)
        for raw in links:
            target = local_target(raw, source)
            if target is not None and not target.exists():
                try:
                    shown = target.relative_to(ROOT)
                except ValueError:
                    shown = target
                missing.append(f"{source.relative_to(ROOT).as_posix()}: {raw} -> {shown}")
    if missing:
        for item in missing:
            print("MISSING", item)
        return 1
    print("PASS local documentation links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
