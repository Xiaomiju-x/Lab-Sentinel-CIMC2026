#!/usr/bin/env python3
"""Dependency-free public-release, privacy and truth-ledger verifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from pathlib import PureWindowsPath

from _common import ROOT, iter_public_files


REQUIRED = [
    "README.md", "README_EN.md", "LICENSE", "NOTICE",
    "THIRD_PARTY_NOTICES.md", "DATA_LICENSES.md", "MODEL_LICENSES.md",
    "MEDIA_LICENSE.md", "SECURITY.md", "CONTRIBUTING.md", "CITATION.cff",
    "PUBLIC_RELEASE_MANIFEST.json", "SBOM.spdx.json",
    "release-manifests/v1.0.1/SHA256SUMS.txt.sha256",
    "release-manifests/v1.0.2/SHA256SUMS.txt.sha256",
    "release-manifests/v1.0.2/RELEASE_ASSETS.json",
    "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx",
    "hardware/design/lab-sentinel-hardware.epro2",
    "evidence/public/release_gap_audit.v7.json",
    "evidence/public/host_closure.v7.json",
]

TEXT_SUFFIXES = {
    ".c", ".h", ".py", ".md", ".txt", ".json", ".yaml", ".yml",
    ".toml", ".csv", ".tsv", ".xml", ".uvprojx", ".cff", ".svg", ".epru",
}

PATTERNS = {
    "private_windows_path": re.compile(r"(?i)[A-Z]:[\\\\/]+(?:Users|WorkData|xrd_backup)[\\\\/]+"),
    "deepseek_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "hardcoded_secret": re.compile(r"(?i)(?:password|passwd|api[_-]?key|token)\s*[=:]\s*[\"'][^\"'\n]{8,}[\"']"),
    # Reject common registration-number and repeated-short-secret shapes without
    # publishing the private values that motivated these release checks.
    "registration_number": re.compile(r"\b20\d{8}\b"),
    "repeated_short_secret": re.compile(r"(?i)\b([a-z0-9]{3})\1{2}\b"),
    "local_username_shape": re.compile(r"(?i)\b[a-z]{6,16}2026\b"),
}

SCAN_EXCEPTIONS: set[str] = set()

ARCHIVE_SUFFIXES = {".zip", ".epro2"}
NESTED_ARCHIVE_SUFFIXES = {
    ".zip", ".epro2", ".7z", ".rar", ".tar", ".gz", ".tgz", ".xz",
}
FORBIDDEN_SUFFIXES = {
    ".axf", ".hex", ".map", ".uvoptx", ".pem", ".pfx", ".p12",
    ".key", ".snk", ".ttf", ".otf", ".woff", ".woff2",
}
KNOWN_FORBIDDEN_SHA256 = {
    # Microsoft/Monotype Arial shipped in an LVGL example; redistribution is prohibited.
    "82afb35eda3a52edb10106bcc04af93646384421ded538d38792c1444d816022",
    # LVGL 8.3.11 optional LRU files derive from C-LRU-Cache, whose upstream
    # project did not define a redistribution license. They are not needed by
    # the selected build and must stay outside the public tree.
    "abeca26f2878a913183b3b48b6af8d18f4837332da64418dc29c3d32188823cc",
    "126ae1ce3c2784f948bc1126a1705bfac16e875cd9642d68a8039b8108b3d7fc",
    # Private CryptoAPI strong-name key found in an unused lwIP contrib tool.
    "0421beb05de86fc121b4e64eb3d0e6f698299bd7e80ea4d5f6fc0c630b61b7f6",
}
ALLOWED_ZERO_BYTE = {
    "firmware/keil_proj/lwip/test/fuzz/config.h",
    "firmware/lvgl_ui/lvgl-8.3.11/src/extra/others/fragment/README.md",
}
PRIVATE_DENYLIST_ENV = "LAB_SENTINEL_PRIVATE_DENYLIST"


def load_private_denylist() -> tuple[tuple[str, ...], list[str]]:
    """Load optional operator-only values without committing them to Git."""
    configured = os.environ.get(PRIVATE_DENYLIST_ENV, "").strip()
    if not configured:
        return (), []
    path = Path(configured)
    if not path.is_file():
        return (), [f"private denylist is missing: {path.name}"]
    try:
        terms = tuple(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    except (OSError, UnicodeDecodeError) as exc:
        return (), [f"cannot read private denylist: {exc}"]
    if not terms:
        return (), ["private denylist is empty"]
    if any(len(term) < 6 for term in terms):
        return (), ["private denylist contains a value shorter than 6 characters"]
    return terms, []


def scan_text() -> list[str]:
    private_terms, findings = load_private_denylist()
    for path in iter_public_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() not in TEXT_SUFFIXES or relative in SCAN_EXCEPTIONS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{name}: {relative}:{line}")
        folded = text.casefold()
        if any(term.casefold() in folded for term in private_terms):
            findings.append(f"private denylist match: {relative}")
    return findings


def _scan_zip(
    path: Path,
    relative: str,
    private_terms: tuple[str, ...] = (),
) -> list[str]:
    """Inspect every ZIP member without trusting its name or outer extension."""
    findings: list[str] = []
    if not zipfile.is_zipfile(path):
        return [f"invalid ZIP container: {relative}"]
    try:
        with zipfile.ZipFile(path) as archive:
            bad_crc = archive.testzip()
            if bad_crc:
                findings.append(f"ZIP CRC failure: {relative}!{bad_crc}")
            for info in archive.infolist():
                member = info.filename.replace("\\", "/")
                parts = [part for part in member.split("/") if part not in ("", ".")]
                if member.startswith(("/", "\\")) or any(part == ".." for part in parts):
                    findings.append(f"unsafe archive path: {relative}!{member}")
                    continue
                if info.is_dir():
                    continue
                suffix = Path(member).suffix.lower()
                if (
                    path.suffix.lower() == ".zip"
                    and relative.startswith("hardware/design/")
                    and suffix == ".txt"
                ):
                    findings.append(
                        f"unexpected exporter instruction in hardware archive: "
                        f"{relative}!{member}"
                    )
                if suffix in FORBIDDEN_SUFFIXES:
                    findings.append(f"forbidden archive member: {relative}!{member}")
                if suffix in NESTED_ARCHIVE_SUFFIXES:
                    findings.append(f"nested archive member: {relative}!{member}")
                if info.file_size > 128 * 1024 * 1024:
                    findings.append(f"oversized archive member: {relative}!{member}")
                    continue
                digest = hashlib.sha256()
                prefix = bytearray()
                text_data = bytearray() if suffix in TEXT_SUFFIXES else None
                with archive.open(info) as stream:
                    while True:
                        chunk = stream.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        if len(prefix) < 16:
                            prefix.extend(chunk[: 16 - len(prefix)])
                        if text_data is not None:
                            if len(text_data) + len(chunk) > 16 * 1024 * 1024:
                                findings.append(
                                    f"oversized text archive member: {relative}!{member}"
                                )
                                text_data = None
                            else:
                                text_data.extend(chunk)
                if digest.hexdigest() in KNOWN_FORBIDDEN_SHA256:
                    findings.append(
                        f"known forbidden archive member: {relative}!{member}"
                    )
                if (
                    len(prefix) >= 12
                    and prefix[0:2] == b"\x07\x02"
                    and prefix[8:12] == b"RSA2"
                ):
                    findings.append(
                        f"private CryptoAPI key blob in archive: {relative}!{member}"
                    )
                if text_data is None:
                    continue
                try:
                    text = text_data.decode("utf-8")
                except UnicodeDecodeError:
                    continue
                for name, pattern in PATTERNS.items():
                    if pattern.search(text):
                        findings.append(f"{name} in archive: {relative}!{member}")
                folded = text.casefold()
                if any(term.casefold() in folded for term in private_terms):
                    findings.append(
                        f"private denylist match in archive: {relative}!{member}"
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        findings.append(f"cannot inspect archive {relative}: {exc}")
    return findings


def scan_archives() -> list[str]:
    """Scan ZIP/EasyEDA containers, including binary members and hashes."""
    private_terms, _ = load_private_denylist()
    findings: list[str] = []
    for path in iter_public_files():
        if path.suffix.lower() in ARCHIVE_SUFFIXES:
            findings.extend(
                _scan_zip(path, path.relative_to(ROOT).as_posix(), private_terms)
            )
    return findings


def validate_file_policy() -> list[str]:
    errors: list[str] = []
    for path in iter_public_files():
        relative = path.relative_to(ROOT).as_posix()
        if path.stat().st_size == 0 and relative not in ALLOWED_ZERO_BYTE:
            errors.append(f"unexpected zero-byte file: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden public artifact: {relative}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in KNOWN_FORBIDDEN_SHA256:
            errors.append(f"known redistribution-prohibited artifact: {relative}")
    return errors


def validate_webp_metadata() -> list[str]:
    """Reject EXIF/XMP chunks from public WebP derivatives."""
    errors: list[str] = []
    for path in (ROOT / "assets").rglob("*.webp"):
        data = path.read_bytes()
        relative = path.relative_to(ROOT).as_posix()
        if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
            errors.append(f"invalid WebP: {relative}")
            continue
        offset = 12
        chunks: set[bytes] = set()
        while offset + 8 <= len(data):
            fourcc = data[offset:offset + 4]
            size = int.from_bytes(data[offset + 4:offset + 8], "little")
            chunks.add(fourcc)
            offset += 8 + size + (size & 1)
        for forbidden in (b"EXIF", b"XMP "):
            if forbidden in chunks:
                errors.append(f"metadata chunk {forbidden.decode().strip()} in {relative}")
    return errors


def validate_json() -> list[str]:
    errors = []
    for path in ROOT.rglob("*.json"):
        if ".git" in path.parts:
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 - report exact invalid artifact
            errors.append(f"invalid JSON {path.relative_to(ROOT)}: {exc}")
    return errors


def validate_keil_project() -> list[str]:
    """Ensure the published Keil target does not reference missing sources."""
    relative = "firmware/keil_proj/project/CIMC_GD32_Template.uvprojx"
    project = ROOT / relative
    if not project.is_file():
        return [f"missing Keil project: {relative}"]
    try:
        root = ET.parse(project).getroot()
    except Exception as exc:  # noqa: BLE001 - report exact invalid artifact
        return [f"invalid Keil project XML: {exc}"]
    errors: list[str] = []
    targets = [node.text for node in root.iter("TargetName") if node.text]
    if targets != ["CIMC_GD32_Template"]:
        errors.append(f"unexpected Keil targets: {targets}")
    for node in root.iter("FilePath"):
        if not node.text:
            continue
        source = project.parent.joinpath(*PureWindowsPath(node.text).parts).resolve()
        if not source.is_file():
            errors.append(f"missing Keil source reference: {node.text}")
    return errors


def validate_sbom() -> list[str]:
    path = ROOT / "SBOM.spdx.json"
    if not path.is_file():
        return []
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # handled again by validate_json, but keep context
        return [f"invalid SBOM: {exc}"]
    packages = document.get("packages", [])
    ids = {package.get("SPDXID") for package in packages}
    required = {
        "SPDXRef-Package-Software", "SPDXRef-Package-Hardware",
        "SPDXRef-Package-GD32-SPL", "SPDXRef-Package-Docs-Media",
        "SPDXRef-Package-Competition-Media", "SPDXRef-Package-License-Metadata",
        "SPDXRef-Package-CMSIS", "SPDXRef-Package-FreeRTOS",
        "SPDXRef-Package-lwIP", "SPDXRef-Package-LVGL",
        "SPDXRef-Package-LVGL-Fonts",
        "SPDXRef-Package-Arm2D", "SPDXRef-Package-LodePNG",
        "SPDXRef-Package-TJpgDec", "SPDXRef-Package-TLSF",
        "SPDXRef-Package-mpaland-printf",
        "SPDXRef-Package-NXP-LVGL-Adapters",
    }
    errors = [f"SBOM missing package: {item}" for item in sorted(required - ids)]
    declarations = {
        package.get("SPDXID"): package.get("licenseDeclared") for package in packages
    }
    if declarations.get("SPDXRef-Package-LVGL") != "MIT":
        errors.append("SBOM LVGL runtime license must be MIT")
    if declarations.get("SPDXRef-Package-LVGL-Fonts") != "OFL-1.1 AND CC-BY-4.0":
        errors.append("SBOM LVGL font subset license expression is incorrect")
    expected_nested = {
        "SPDXRef-Package-Arm2D": "Apache-2.0",
        "SPDXRef-Package-LodePNG": "Zlib",
        "SPDXRef-Package-TJpgDec": "LicenseRef-TJpgDec",
        "SPDXRef-Package-TLSF": "BSD-3-Clause",
        "SPDXRef-Package-mpaland-printf": "MIT",
        "SPDXRef-Package-NXP-LVGL-Adapters": "MIT",
    }
    for package_id, expected_license in expected_nested.items():
        if declarations.get(package_id) != expected_license:
            errors.append(
                f"SBOM {package_id} license must be {expected_license}"
            )
    extracted = {
        item.get("licenseId")
        for item in document.get("hasExtractedLicensingInfos", [])
    }
    if "LicenseRef-TJpgDec" not in extracted:
        errors.append("SBOM must include extracted TJpgDec license text")
    file_licenses = {
        str(item.get("fileName", "")).removeprefix("./"): item.get("licenseConcluded")
        for item in document.get("files", [])
    }
    expected_file_licenses = {
        "README.md": "CC-BY-4.0",
        "LICENSE": "NOASSERTION",
        "LICENSES/Apache-2.0.txt": "NOASSERTION",
        "hardware/README.md": "CC-BY-4.0",
        "hardware/design/README.md": "CC-BY-4.0",
        "firmware/keil_proj/lwip/PUBLIC_SUBSET.md": "CC-BY-4.0",
        "firmware/keil_proj/CMSIS/gd32h7xx_libopt.h": "BSD-3-Clause",
        "firmware/keil_proj/User/gd32h7xx_it.c": "BSD-3-Clause",
        "firmware/keil_proj/User/systick.c": "BSD-3-Clause",
    }
    for relative, expected_license in expected_file_licenses.items():
        if file_licenses.get(relative) != expected_license:
            errors.append(f"SBOM file license mismatch: {relative} must be {expected_license}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in iter_public_files():
        size = path.stat().st_size
        relative = path.relative_to(ROOT).as_posix()
        if size >= 100 * 1024 * 1024:
            errors.append(f"GitHub-blocked file >=100 MiB: {relative} ({size})")
        elif size >= 50 * 1024 * 1024:
            warnings.append(f"large Git file >=50 MiB: {relative} ({size})")

    errors.extend(scan_text())
    errors.extend(scan_archives())
    errors.extend(validate_file_policy())
    errors.extend(validate_webp_metadata())
    errors.extend(validate_json())
    errors.extend(validate_keil_project())
    errors.extend(validate_sbom())

    easyeda = subprocess.run(
        [sys.executable, str(ROOT / "tools/sanitize_easyeda_pro.py"), "--check"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if easyeda.returncode != 0:
        errors.append("EasyEDA Pro privacy/integrity check failed:\n" + easyeda.stdout + easyeda.stderr)

    evidence = subprocess.run(
        [sys.executable, str(ROOT / "tools/verify_evidence.py")],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if evidence.returncode != 0:
        errors.append("evidence verifier failed:\n" + evidence.stdout + evidence.stderr)

    manifest = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_public_manifest.py"), "--check"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if manifest.returncode != 0:
        errors.append("public manifest check failed:\n" + manifest.stdout + manifest.stderr)

    assets = subprocess.run(
        [sys.executable, str(ROOT / "tools/build_asset_checksums.py"), "--check"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    if assets.returncode != 0:
        errors.append("asset checksum check failed:\n" + assets.stdout + assets.stderr)

    for item in warnings:
        print("WARN", item)
    for item in errors:
        print("FAIL", item)
    if errors or (args.strict and warnings):
        return 1
    print(f"PASS public release: {len(iter_public_files())} files; no blocking findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
