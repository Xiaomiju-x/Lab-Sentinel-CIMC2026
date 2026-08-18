#!/usr/bin/env python3
"""Freeze the shared Kaggle/PMC source decision for P107 and P122."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def verified(root: Path, relative: str) -> dict[str, Any]:
    path = root / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    source_dir = root / "data" / "sources" / "p107_solder_xray_pmc11126598_v1"
    repo = load_json(source_dir / "github_repo.json")
    tree = load_json(source_dir / "github_tree.json")
    inventory = load_json(source_dir / "kaggle_file_inventory.v1.json")
    metadata = load_json(source_dir / "kaggle_view.json")
    binding = load_json(root / "contracts" / "kaggle_p122_solder_fatigue_binding.v1.json")
    staging = load_json(root / "evidence" / "kaggle_p122_exact_staging.v1.json")

    artifacts = {
        "github_repo": verified(root, "data/sources/p107_solder_xray_pmc11126598_v1/github_repo.json"),
        "github_tree": verified(root, "data/sources/p107_solder_xray_pmc11126598_v1/github_tree.json"),
        "pmc_jats": verified(root, "data/sources/p107_solder_xray_pmc11126598_v1/PMC11126598.xml"),
        "kaggle_metadata": verified(root, "data/sources/p107_solder_xray_pmc11126598_v1/kaggle_view.json"),
        "kaggle_inventory": verified(root, "data/sources/p107_solder_xray_pmc11126598_v1/kaggle_file_inventory.v1.json"),
        "archive": verified(root, "data/raw/kaggle_led_reliability_archive_v1/archive.zip"),
        "p122_paper": verified(root, "data/sources/p122_led_solder_reliability_v1/schmid_2023_part2_reliability.pdf"),
        "p122_binding": verified(root, "contracts/kaggle_p122_solder_fatigue_binding.v1.json"),
        "p122_staging": verified(root, "evidence/kaggle_p122_exact_staging.v1.json"),
        "p122_dataset": verified(root, "data/staged_kaggle_p122_exact_v1/CAND-P-122.npz"),
    }
    for name in ("archive", "api_inventory", "api_metadata", "paper_pdf"):
        expected = binding["verified_files"][name]
        actual_name = {"api_inventory": "kaggle_inventory", "api_metadata": "kaggle_metadata", "paper_pdf": "p122_paper"}.get(name, name)
        actual = artifacts[actual_name]
        if actual["bytes"] != expected["bytes"] or actual["sha256"] != expected["sha256"]:
            raise RuntimeError(f"P122_BINDING_HASH_GATE:{name}")

    tree_paths = [record["path"] for record in tree["tree"]]
    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    repo_images = [path for path in tree_paths if Path(path).suffix.lower() in image_suffixes]
    repo_masks = [path for path in tree_paths if "mask" in path.lower()]
    inventory_mask_like = int(inventory["keyword_counts"].get("mask", 0))
    inventory_label_like = int(inventory["keyword_counts"].get("label", 0))
    p107_reject = (
        repo.get("license") is None
        and not repo_images
        and not repo_masks
        and inventory_mask_like == 0
        and inventory_label_like == 0
    )
    p122_pass = (
        metadata.get("licenseNameNullable") == "CC BY-NC-SA 4.0"
        and inventory["files"] == 75064
        and inventory["listed_bytes"] == 5590741008
        and staging["status"] == "PASS_EXACT_SOURCE_LABEL_SPLIT_TRAINING_AUTHORIZED"
        and staging["dataset"]["records"] == 1531
        and staging["split"]["cross_split_family_overlap"] == 0
        and staging["split"]["cross_split_unit_overlap"] == 0
        and not staging["future_history_in_inputs"]
        and not staging["paper_group_mean_as_record_truth"]
        and staging["authority"] == 0
    )
    if not p107_reject or not p122_pass:
        raise RuntimeError(f"SOURCE_DECISION_GATE:P107={p107_reject}:P122={p122_pass}")

    receipt = {
        "schema": "cimc.forge200.kaggle-p107-p122-source-contract-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "SOURCE_LICENSE_AND_HASH_PASS_P107_EXACT_REJECTED_P122_EXACT_ADMITTED",
        "shared_source": {
            "dataset": "andreaszippelius/hellastudy-of-leds2",
            "license": metadata["licenseNameNullable"],
            "listed_files": inventory["files"],
            "listed_bytes": inventory["listed_bytes"],
            "xray_images": inventory["top_level_counts"]["XRay"],
            "processed_tta_dat_files": inventory["extension_counts"][".dat"],
        },
        "p107": {
            "candidate_id": "CAND-P-107",
            "status": "EXACT_REJECTED_MISSING_PIXEL_MASKS_PIXEL_SCALE_AND_REUSABLE_REPO_LICENSE",
            "official_repository_entries": len(tree_paths),
            "official_repository_paths": tree_paths,
            "official_repository_license": repo.get("license"),
            "official_repository_raster_images": len(repo_images),
            "official_repository_mask_files": len(repo_masks),
            "kaggle_mask_filename_hits": inventory_mask_like,
            "kaggle_label_filename_hits": inventory_label_like,
            "physical_pixel_scale_bound": False,
            "training_actions": 0,
            "host_promotions": 0,
        },
        "p122": {
            "candidate_id": "CAND-P-122",
            "status": "EXACT_ADMITTED_TRAINING_AUTHORIZED",
            "truth_class": binding["source"]["truth_class"],
            "event_definition": binding["target_binding"],
            "records": staging["dataset"]["records"],
            "events": staging["dataset"]["events"],
            "right_censored": staging["dataset"]["right_censored"],
            "split_counts": staging["dataset"]["split_counts"],
            "cross_split_family_overlap": staging["split"]["cross_split_family_overlap"],
            "cross_split_unit_overlap": staging["split"]["cross_split_unit_overlap"],
            "training_authorized": True,
            "test_evaluation_actions_at_audit_time": 0,
            "host_promotions_at_audit_time": 0,
        },
        "artifacts": artifacts,
        "teacher_or_fixture_labels": 0,
        "authority": 0,
        "board_accepted": False,
        "countable_model": False,
        "board_actions": 0,
    }
    output = root / "evidence" / "kaggle_p107_p122_source_contract_audit.v1.json"
    write_json(output, receipt)
    print(json.dumps({"status": receipt["status"], "p107": receipt["p107"]["status"], "p122_records": receipt["p122"]["records"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
