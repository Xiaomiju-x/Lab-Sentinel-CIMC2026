#!/usr/bin/env python3
"""Build the immutable 8.3-name microSD staging tree for Forge200 v8."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path


CATALOG_HEADER_BYTES = 128
CATALOG_ENTRY_BYTES = 160


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def fixed_ascii(value: str, size: int) -> bytes:
    encoded = value.encode("ascii")
    if len(encoded) >= size:
        raise ValueError(f"fixed ASCII field overflow: {value!r} >= {size}")
    return encoded.ljust(size, b"\0")


def link_or_copy(source: Path, destination: Path) -> str:
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def build_catalog(records: list[dict], generation: int, source_root: str) -> bytes:
    body = bytearray()
    for record in records:
        category = record["category"]
        tier = 1 if record["tier"] == "EXACT_CONTRACT" else 2
        entry = bytearray(CATALOG_ENTRY_BYTES)
        entry[0:32] = fixed_ascii(record["candidate_id"], 32)
        entry[32:56] = fixed_ascii(record["package_path"], 24)
        entry[56:80] = fixed_ascii(record["golden_path"], 24)
        struct.pack_into(
            "<BBHHHQ",
            entry,
            80,
            ord(category),
            tier,
            record["engine_id"],
            record["opset"],
            0,
            record["package_bytes"],
        )
        entry[96:128] = bytes.fromhex(record["package_sha256"])
        entry[128:160] = bytes.fromhex(record["golden_sha256"])
        body.extend(entry)
    body_sha = hashlib.sha256(body).digest()
    header = bytearray(CATALOG_HEADER_BYTES)
    struct.pack_into(
        "<4sHHQIIQ",
        header,
        0,
        b"F2CT",
        1,
        CATALOG_HEADER_BYTES,
        generation,
        len(records),
        CATALOG_ENTRY_BYTES,
        len(body),
    )
    header[32:64] = body_sha
    header[64:96] = bytes.fromhex(source_root)
    return bytes(header + body)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    runtime_root = args.runtime_root.resolve()
    output = args.output.resolve()
    receipt_path = args.receipt.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")

    source_manifest_path = runtime_root / "MANIFEST.v8.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("model_count") != 170:
        raise RuntimeError("SOURCE_MODEL_COUNT_GATE")
    source_records = source_manifest["records"]
    if len(source_records) != 170:
        raise RuntimeError("SOURCE_RECORD_COUNT_GATE")

    f200 = output / "F200"
    f200.mkdir(parents=True)
    records: list[dict] = []
    transfer_modes = {"hardlink": 0, "copy": 0}
    for source in sorted(source_records, key=lambda item: item["candidate_id"]):
        candidate_id = source["candidate_id"]
        category, ordinal_text = candidate_id.removeprefix("CAND-").split("-")
        ordinal = int(ordinal_text)
        short_base = f"{category}{ordinal:03d}"
        package_name = f"{short_base}.ICM"
        golden_name = f"{short_base}.GLD"
        package_source = runtime_root / source["package"]["path"]
        golden_source = runtime_root / source["golden"]["path"]
        if sha256_file(package_source) != source["package"]["sha256"]:
            raise RuntimeError(f"PACKAGE_HASH_GATE:{candidate_id}")
        if sha256_file(golden_source) != source["golden"]["sha256"]:
            raise RuntimeError(f"GOLDEN_HASH_GATE:{candidate_id}")
        raw_header = package_source.read_bytes()[:256]
        if len(raw_header) != 256 or raw_header[:4] != b"ICMF":
            raise RuntimeError(f"PACKAGE_HEADER_GATE:{candidate_id}")
        engine_id, opset = struct.unpack_from("<HH", raw_header, 8)
        if engine_id != source["engine_id"] or raw_header[12] != 0:
            raise RuntimeError(f"PACKAGE_AUTHORITY_ENGINE_GATE:{candidate_id}")
        transfer_modes[link_or_copy(package_source, f200 / package_name)] += 1
        transfer_modes[link_or_copy(golden_source, f200 / golden_name)] += 1
        records.append(
            {
                "candidate_id": candidate_id,
                "category": category,
                "tier": source["tier"],
                "engine_id": engine_id,
                "opset": opset,
                "package_path": f"0:/F200/{package_name}",
                "golden_path": f"0:/F200/{golden_name}",
                "package_bytes": package_source.stat().st_size,
                "package_sha256": source["package"]["sha256"],
                "payload_sha256": source["package"]["payload_sha256"],
                "golden_bytes": golden_source.stat().st_size,
                "golden_sha256": source["golden"]["sha256"],
                "release_root": source["release_root"],
                "authority": 0,
                "board_accepted": False,
                "countable_model": False,
            }
        )

    if len({item["package_path"] for item in records}) != 170:
        raise RuntimeError("SHORT_PACKAGE_PATH_UNIQUENESS_GATE")
    if len({item["golden_path"] for item in records}) != 170:
        raise RuntimeError("SHORT_GOLDEN_PATH_UNIQUENESS_GATE")
    catalog_a = build_catalog(records, 1, source_manifest["content_root_sha256"])
    catalog_b = build_catalog(records, 1, source_manifest["content_root_sha256"])
    (f200 / "CATALOGA.BIN").write_bytes(catalog_a)
    (f200 / "CATALOGB.BIN").write_bytes(catalog_b)

    # Read-only board fault fixtures. They are deliberately outside both
    # catalogs, are never promotable models and exist only to prove refusal.
    fault_dir = f200 / "FAULT"
    fault_dir.mkdir()
    base_package = bytearray((f200 / "P001.ICM").read_bytes())
    base_golden = bytearray((f200 / "P001.GLD").read_bytes())
    variants: dict[str, bytes] = {}
    value = bytearray(base_package)
    value[0] ^= 0x01
    variants["BADMAG.ICM"] = bytes(value)
    value = bytearray(base_package)
    value[12] = 1
    variants["BADAUT.ICM"] = bytes(value)
    value = bytearray(base_package)
    value[256] ^= 0x01
    variants["BADPAY.ICM"] = bytes(value)
    value = bytearray(base_package)
    struct.pack_into("<Q", value, 16, 0)
    variants["BADGEN.ICM"] = bytes(value)
    value = bytearray(base_package)
    struct.pack_into("<H", value, 8, 99)
    variants["BADENG.ICM"] = bytes(value)
    variants["BADGLD.ICM"] = bytes(base_package)
    for name, value in variants.items():
        (fault_dir / name).write_bytes(value)
    base_golden[-1] ^= 0x01
    (fault_dir / "BADGLD.GLD").write_bytes(base_golden)
    (fault_dir / "README.TXT").write_text(
        "Non-promotable refusal fixtures: bad magic, authority, payload, "
        "generation, engine and golden.\r\n",
        encoding="ascii",
    )

    files_csv = output / "FILES.CSV"
    with files_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    readme = (
        "CIMC Forge200 MCU runtime v8r1 microSD staging\r\n"
        "Copy the F200 directory to the FAT32 card root.\r\n"
        "CATALOGA.BIN and CATALOGB.BIN are identical generation-1 A/B catalogs.\r\n"
        "All 170 new assets are authority=0 and board-pending until unified tests pass.\r\n"
        "Do not overwrite or rename the preliminary-round firmware assets.\r\n"
    )
    (output / "README.TXT").write_text(readme, encoding="ascii")

    manifest = {
        "schema": "cimc.forge200.sd-staging.v8",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SD_STAGING_170_HOST_VERIFIED_UNIFIED_BOARD_PENDING",
        "source_manifest": {
            "path": source_manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(source_manifest_path),
            "content_root_sha256": source_manifest["content_root_sha256"],
        },
        "model_count": 170,
        "by_category": {
            category: sum(item["category"] == category for item in records)
            for category in ("P", "G", "S")
        },
        "exact_count": sum(item["tier"] == "EXACT_CONTRACT" for item in records),
        "sim_only_count": sum(item["tier"] == "SIM_ONLY_EXTENSION" for item in records),
        "catalog": {
            "entry_bytes": CATALOG_ENTRY_BYTES,
            "header_bytes": CATALOG_HEADER_BYTES,
            "generation": 1,
            "a_sha256": sha256_file(f200 / "CATALOGA.BIN"),
            "b_sha256": sha256_file(f200 / "CATALOGB.BIN"),
        },
        "max_package_bytes": max(item["package_bytes"] for item in records),
        "max_golden_bytes": max(item["golden_bytes"] for item in records),
        "transfer_modes": transfer_modes,
        "fault_fixture_count": len(variants) + 1,
        "records": records,
        "authority_nonzero": 0,
        "board_actions": 0,
        "claim_boundary": (
            "This is a host-verified FAT32 staging tree. Physical SD throughput, "
            "MAX31856 coexistence, Cortex-M7 latency, swap and long-run evidence remain pending."
        ),
    }
    manifest["content_root_sha256"] = hashlib.sha256(canonical(records)).hexdigest()
    write_json(output / "MANIFEST.JSON", manifest)

    files = sorted(path for path in output.rglob("*") if path.is_file())
    verification = []
    for path in files:
        verification.append(
            {
                "path": path.relative_to(output).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    receipt = {
        **{key: value for key, value in manifest.items() if key != "records"},
        "output": output.relative_to(root).as_posix(),
        "files": len(verification),
        "bytes": sum(item["bytes"] for item in verification),
        "verification": verification,
        "manifest_sha256": sha256_file(output / "MANIFEST.JSON"),
    }
    receipt["verification_root_sha256"] = hashlib.sha256(canonical(verification)).hexdigest()
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "models": receipt["model_count"],
                "files": receipt["files"],
                "bytes": receipt["bytes"],
                "max_package_bytes": receipt["max_package_bytes"],
                "max_golden_bytes": receipt["max_golden_bytes"],
                "content_root_sha256": receipt["content_root_sha256"],
                "verification_root_sha256": receipt["verification_root_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
