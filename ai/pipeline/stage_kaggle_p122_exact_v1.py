#!/usr/bin/env python3
"""Stage source-bound P122 F1 lifetime records with package-family splits."""

from __future__ import annotations

import argparse
import binascii
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PATTERN = re.compile(
    r"TTA_(?P<family>.+?)_(?P<paste>SAC[^_]+)_(?P<panel>\d+)_(?P<led>\d+)_(?P<cycle>\d{4})TSC\.(?P<suffix>dat|err)$"
)
STATIC_FEATURES = [
    "die_area_mm2", "package_x_mm", "package_y_mm", "pad_count",
    "pad_area_sum_mm2", "pad_gap_mm", "pad_ratio", "nominal_current_A",
    "datasheet_Rth_JC_K_per_W", "submount_lead_frame", "submount_thick_film_AlN",
    "submount_thin_film_AlN", "alloy_Ag_wt_pct", "alloy_Cu_wt_pct", "alloy_Sb_wt_pct",
    "alloy_Bi_wt_pct", "alloy_Ni_wt_pct", "alloy_In_wt_pct",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_root(document: dict) -> str:
    payload = dict(document)
    payload.pop("created_at_utc", None)
    payload.pop("content_root_sha256", None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def static_vector(binding: dict, family: str, paste: str) -> list[float]:
    geometry = binding["geometry"][family]
    alloy = binding["alloy_weight_percent"][paste]
    submount = geometry["submount"]
    return [
        float(geometry[name]) for name in STATIC_FEATURES[:9]
    ] + [
        float(submount == "lead_frame"),
        float(submount == "thick_film_AlN"),
        float(submount == "thin_film_AlN"),
        float(alloy["Ag"]), float(alloy["Cu"]), float(alloy["Sb"]),
        float(alloy["Bi"]), float(alloy["Ni"]), float(alloy["In"]),
    ]


def parse_curve(payload: bytes) -> tuple[float, float, float, int]:
    _, values = payload.split(b"\n", 1)
    array = np.fromstring(values.decode("ascii"), sep=" ", dtype=np.float64)
    if array.size != 256 * 5:
        raise ValueError(f"unexpected processed TTA shape: {array.size}")
    array = array.reshape(256, 5)
    if not np.isfinite(array).all():
        raise ValueError("non-finite processed TTA value")
    return float(array[-1, 3]), float(np.max(array[:, 4])), float(array[-1, 2]), len(array)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    binding_path = root / "contracts/kaggle_p122_solder_fatigue_binding.v1.json"
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    for item in binding["verified_files"].values():
        path = root / item["path"]
        if path.stat().st_size != item["bytes"] or sha256_file(path) != item["sha256"]:
            raise ValueError(f"source hash gate failed: {item['path']}")

    metadata = json.loads((root / binding["verified_files"]["api_metadata"]["path"]).read_text(encoding="utf-8"))
    if metadata.get("licenseName") != "CC BY-NC-SA 4.0" or int(metadata.get("totalBytes")) != 5590741008:
        raise ValueError("dataset license or byte inventory changed")
    inventory = json.loads((root / binding["verified_files"]["api_inventory"]["path"]).read_text(encoding="utf-8"))
    if inventory["files"] != 75064 or inventory["listed_bytes"] != 5590741008:
        raise ValueError("Kaggle inventory gate failed")

    archive_path = root / binding["verified_files"]["archive"]["path"]
    extracted = root / "data/raw/kaggle_led_reliability_extracted_v1"
    observations: dict[tuple[str, str, int, int], dict[int, dict[str, float]]] = defaultdict(dict)
    verified_files = 0
    verified_bytes = 0
    dat_files = 0
    err_files = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = {
            item.filename: item for item in archive.infolist()
            if item.filename.startswith("TTA/") or item.filename.startswith("CrackVoid Ratios/")
        }
        paths = sorted(path for path in extracted.rglob("*") if path.is_file())
        if len(paths) != 21608 or sum(path.stat().st_size for path in paths) != 594810204:
            raise ValueError("selective extraction inventory changed")
        for path in paths:
            relative = path.relative_to(extracted).as_posix()
            member = members.get(relative)
            if member is None:
                raise ValueError(f"extracted file absent from archive: {relative}")
            payload = path.read_bytes()
            if len(payload) != member.file_size or (binascii.crc32(payload) & 0xFFFFFFFF) != member.CRC:
                raise ValueError(f"ZIP CRC gate failed: {relative}")
            verified_files += 1
            verified_bytes += len(payload)
            if not relative.startswith("TTA/"):
                continue
            match = PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected TTA filename: {path.name}")
            if match["suffix"] == "err":
                err_files += 1
                if payload not in {b"#t\tz\tVf\tZth\tB", b"#t\tz\tVf\tZth\tB\n"}:
                    raise ValueError(f"unexpected .err payload: {path.name}")
                continue
            dat_files += 1
            key = (match["family"], match["paste"], int(match["panel"]), int(match["led"]))
            cycle = int(match["cycle"])
            rth, bmax, vf_last, rows = parse_curve(payload)
            observations[key][cycle] = {"rth": rth, "bmax": bmax, "vf_last": vf_last, "rows": rows}

    target = binding["target_binding"]
    required = tuple(int(value) for value in target["required_pre_landmark_cycles"])
    landmark = int(target["landmark_cycle"])
    family_to_split = {
        **{family: 0 for family in binding["split"]["train_families"]},
        **{family: 1 for family in binding["split"]["validation_families"]},
        **{family: 2 for family in binding["split"]["test_families"]},
    }
    records = []
    excluded = Counter()
    for key, history in sorted(observations.items()):
        family, paste, panel, led = key
        if 0 not in history:
            excluded["missing_initial"] += 1
            continue
        initial = history[0]["rth"]
        event_cycle = next(
            (cycle for cycle in sorted(history) if cycle > 0 and history[cycle]["rth"] / initial - 1.0 >= 0.20),
            None,
        )
        if event_cycle is not None and event_cycle <= landmark:
            excluded["failed_on_or_before_landmark"] += 1
            continue
        if any(cycle not in history for cycle in required):
            excluded["incomplete_pre_landmark_history"] += 1
            continue
        censor_cycle = max(history)
        if event_cycle is None and censor_cycle <= landmark:
            excluded["no_post_landmark_followup"] += 1
            continue
        split = family_to_split.get(family)
        if split is None:
            excluded["family_not_in_frozen_split"] += 1
            continue
        rth = np.asarray([history[cycle]["rth"] for cycle in required], dtype=np.float64)
        relative = rth / rth[0] - 1.0
        unit_id = f"{family}|{paste}|P{panel}|L{led:02d}"
        records.append({
            "unit_id": unit_id,
            "family": family,
            "paste": paste,
            "panel": panel,
            "led": led,
            "split": split,
            "static": static_vector(binding, family, paste),
            "rth_history": rth.tolist(),
            "relative_rth_history": relative.tolist(),
            "event_observed": int(event_cycle is not None),
            "event_or_censor_cycle": int(event_cycle if event_cycle is not None else censor_cycle),
            "rul_cycles": float(event_cycle - landmark) if event_cycle is not None else float("nan"),
        })

    if not records:
        raise ValueError("no P122 records survived staging")
    split_families = {
        code: {record["family"] for record in records if record["split"] == code}
        for code in (0, 1, 2)
    }
    overlap = (split_families[0] & split_families[1]) | (split_families[0] & split_families[2]) | (split_families[1] & split_families[2])
    if overlap:
        raise ValueError(f"package-family leakage: {sorted(overlap)}")

    out_dir = root / "data/staged_kaggle_p122_exact_v1"
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "CAND-P-122.npz"
    np.savez_compressed(
        dataset_path,
        unit_id=np.asarray([r["unit_id"] for r in records]),
        family=np.asarray([r["family"] for r in records]),
        paste=np.asarray([r["paste"] for r in records]),
        split=np.asarray([r["split"] for r in records], dtype=np.int8),
        static=np.asarray([r["static"] for r in records], dtype=np.float32),
        rth_history=np.asarray([r["rth_history"] for r in records], dtype=np.float32),
        relative_rth_history=np.asarray([r["relative_rth_history"] for r in records], dtype=np.float32),
        event_observed=np.asarray([r["event_observed"] for r in records], dtype=np.int8),
        event_or_censor_cycle=np.asarray([r["event_or_censor_cycle"] for r in records], dtype=np.float32),
        rul_cycles=np.asarray([r["rul_cycles"] for r in records], dtype=np.float32),
        history_cycles=np.asarray(required, dtype=np.float32),
        landmark_cycle=np.asarray(landmark, dtype=np.float32),
    )
    task_contract_path = root / "contracts/candidate_task_contracts_244.v1.tsv"
    task_row = next(line for line in task_contract_path.read_text(encoding="utf-8").splitlines() if line.startswith("CAND-P-122\t"))
    split_counts = Counter(record["split"] for record in records)
    split_events = Counter(record["split"] for record in records if record["event_observed"])
    receipt = {
        "schema": "cimc.forge200.kaggle-p122-exact-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_EXACT_SOURCE_LABEL_SPLIT_TRAINING_AUTHORIZED",
        "candidate_id": "CAND-P-122",
        "binding": {"path": binding_path.relative_to(root).as_posix(), "sha256": sha256_file(binding_path)},
        "task_contract": {
            "path": task_contract_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(task_contract_path),
            "row_sha256": hashlib.sha256((task_row + "\n").encode("utf-8")).hexdigest(),
        },
        "source_verification": {
            "archive_sha256": sha256_file(archive_path),
            "selectively_extracted_files_crc_verified": verified_files,
            "selectively_extracted_bytes_crc_verified": verified_bytes,
            "processed_dat_files": dat_files,
            "processed_err_files": err_files,
            "curve_rows_each": 256,
        },
        "target_binding": target,
        "dataset": {
            "path": dataset_path.relative_to(root).as_posix(),
            "bytes": dataset_path.stat().st_size,
            "sha256": sha256_file(dataset_path),
            "records": len(records),
            "independent_units": len(records),
            "events": sum(record["event_observed"] for record in records),
            "right_censored": sum(not record["event_observed"] for record in records),
            "split_counts": {"train": split_counts[0], "validation": split_counts[1], "test": split_counts[2]},
            "split_events": {"train": split_events[0], "validation": split_events[1], "test": split_events[2]},
            "excluded": dict(sorted(excluded.items())),
        },
        "features": {"static": STATIC_FEATURES, "history_cycles": list(required), "history": "steady_state_Rth_and_relative_Rth_only_as_of_landmark"},
        "split": {
            **binding["split"],
            "cross_split_family_overlap": len(overlap),
            "cross_split_unit_overlap": 0,
        },
        "future_history_in_inputs": False,
        "paper_group_mean_as_record_truth": False,
        "teacher_or_fixture_labels": 0,
        "training_authorized": True,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
    }
    receipt["content_root_sha256"] = content_root(receipt)
    receipt_path = root / "evidence/kaggle_p122_exact_staging.v1.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": receipt["status"],
        "records": receipt["dataset"]["records"],
        "events": receipt["dataset"]["events"],
        "split_counts": receipt["dataset"]["split_counts"],
        "content_root_sha256": receipt["content_root_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
