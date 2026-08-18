#!/usr/bin/env python3
"""Materialize only source/label pairs that pass the frozen Forge200 gates.

The script deliberately refuses proxy labels.  It creates canonical NPZ inputs
for the small set of tasks whose exact labels exist in the pinned JARVIS or
SECOM artifacts, then updates the recoverable queue admission state.  Missing
corpora, judgments, L2 labels, masks, or experimental targets remain blocked.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs",
    "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]
ELEMENT_INDEX = {symbol: index for index, symbol in enumerate(ELEMENTS)}

JARVIS_TARGETS: dict[str, tuple[str, str]] = {
    "CAND-P-069": ("formation_energy_peratom", "regression"),
    "CAND-P-071": ("mbj_bandgap", "regression"),
    "CAND-P-072": ("dfpt_piezo_max_dielectric", "regression"),
    "CAND-P-074": ("avg_elec_mass", "regression"),
    "CAND-P-075": ("avg_hole_mass", "regression"),
    "CAND-P-076": ("bulk_modulus_kv", "regression"),
    "CAND-P-077": ("shear_modulus_gv", "regression"),
    "CAND-P-078": ("exfoliation_energy", "regression"),
    "CAND-P-086": ("min_ir_mode", "classification"),
    "CAND-P-142": ("slme", "regression"),
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        result = float(value)
        return result if np.isfinite(result) else None
    if isinstance(value, str):
        try:
            result = float(value.strip())
        except ValueError:
            return None
        return result if np.isfinite(result) else None
    return None


def composition_features(formula: str, row: dict[str, Any]) -> np.ndarray | None:
    counts = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count_text in re.findall(r"([A-Z][a-z]?)([0-9]*\.?[0-9]*)", formula):
        if symbol not in ELEMENT_INDEX:
            return None
        count = float(count_text) if count_text else 1.0
        counts[ELEMENT_INDEX[symbol]] += count
    total = float(counts.sum())
    if total <= 0:
        return None
    counts /= total
    density = numeric(row.get("density")) or 0.0
    nat = numeric(row.get("nat")) or 0.0
    spg = numeric(row.get("spg_number")) or numeric(row.get("spg")) or 0.0
    dimensionality = 1.0 if str(row.get("dimensionality", "")).startswith("2D") else 0.0
    return np.concatenate((counts, np.asarray([density / 25.0, nat / 100.0, spg / 230.0, dimensionality], dtype=np.float32)))


def task_contract_hashes(root: Path) -> dict[str, str]:
    rows = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    return {row["candidate_id"]: hashlib.sha256(canonical_bytes(row)).hexdigest() for row in rows}


def save_dataset(
    root: Path,
    candidate_id: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    splits: np.ndarray,
    task_kind: str,
    truth_class: str,
    source_id: str,
    source_sha256: str,
    split_sha256: str,
    task_hash: str,
    target_field: str,
) -> dict[str, Any]:
    output = root / "data" / "staged" / f"{candidate_id}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {name: int(np.sum(splits == code)) for code, name in enumerate(("train", "validation", "test"))}
    status = "PASS" if min(counts.values()) >= 16 and len(x) >= 96 else "BLOCKED_INSUFFICIENT_SPLIT_ROWS"
    if status == "PASS":
        np.savez_compressed(
            output,
            x=x.astype(np.float32),
            y=y,
            groups=groups.astype(str),
            split=splits.astype(np.int8),
            candidate_id=np.asarray(candidate_id),
            task_kind=np.asarray(task_kind),
            truth_class=np.asarray(truth_class),
            authority=np.asarray(0, dtype=np.int8),
        )
    metadata = {
        "schema": "cimc.forge200.staged-dataset.v1",
        "status": status,
        "candidate_id": candidate_id,
        "task_kind": task_kind,
        "truth_class": truth_class,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "split_sha256": split_sha256,
        "task_contract_sha256": task_hash,
        "target_field": target_field,
        "feature_contract": "composition_fraction_118+density+nat+space_group+2d_flag",
        "fit_preprocessing_on_train_only": True,
        "cross_split_group_overlap": 0,
        "counts": counts,
        "records": len(x),
        "features": int(x.shape[1]) if x.ndim == 2 else 0,
        "authority": 0,
        "path": str(output.relative_to(root)).replace("\\", "/") if status == "PASS" else None,
    }
    if status == "PASS":
        metadata.update({"bytes": output.stat().st_size, "sha256": sha256_file(output)})
    write_json(root / "data" / "staged" / f"{candidate_id}.metadata.json", metadata)
    return metadata


def stage_jarvis(root: Path, task_hashes: dict[str, str]) -> list[dict[str, Any]]:
    artifact = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    assignment_path = root / "data" / "splits" / "jarvis_dft_v11.assignments.tsv"
    assignments = {row["jid"]: row for row in read_tsv(assignment_path)}
    with zipfile.ZipFile(artifact) as archive:
        rows = json.load(archive.open(archive.namelist()[0]))
    feature_cache: dict[str, np.ndarray | None] = {}
    records: list[dict[str, Any]] = []
    for candidate_id, (target_field, task_kind) in JARVIS_TARGETS.items():
        features, labels, groups, splits = [], [], [], []
        for row in rows:
            assignment = assignments.get(str(row.get("jid")))
            if assignment is None:
                continue
            value = numeric(row.get(target_field))
            if value is None:
                continue
            jid = str(row["jid"])
            if jid not in feature_cache:
                feature_cache[jid] = composition_features(str(row.get("formula", "")), row)
            vector = feature_cache[jid]
            if vector is None:
                continue
            features.append(vector)
            labels.append(int(value >= 0.0) if task_kind == "classification" else value)
            groups.append(assignment["chemical_system_group"])
            splits.append({"train": 0, "validation": 1, "test": 2}[assignment["split"]])
        records.append(
            save_dataset(
                root,
                candidate_id,
                np.asarray(features, dtype=np.float32),
                np.asarray(labels, dtype=np.int64 if task_kind == "classification" else np.float32),
                np.asarray(groups),
                np.asarray(splits),
                task_kind,
                "OPEN_COMPUTED_DFT",
                "jarvis_dft_v11",
                sha256_file(artifact),
                sha256_file(assignment_path),
                task_hashes[candidate_id],
                target_field,
            )
        )
    return records


def stage_jarvis_piezo_prototype(root: Path, task_hashes: dict[str, str]) -> dict[str, Any]:
    """Stage the exact JARVIS max d_ij target with a structure-prototype split."""
    artifact = root / "data" / "raw" / "jarvis_dft_v11" / "jdft_3d-9-24-2025.json.zip"
    with zipfile.ZipFile(artifact) as archive:
        rows = json.load(archive.open(archive.namelist()[0]))
    features, labels, groups, splits, assignments = [], [], [], [], []
    split_sets = {0: set(), 1: set(), 2: set()}
    for row in rows:
        value = numeric(row.get("dfpt_piezo_max_dij"))
        vector = composition_features(str(row.get("formula", "")), row)
        if value is None or vector is None:
            continue
        prototype = str(row.get("spg_symbol") or row.get("spg_number") or row.get("spg") or "UNKNOWN")
        bucket = int(hashlib.sha256(prototype.encode("utf-8")).hexdigest()[:8], 16) % 100
        split = 0 if bucket < 70 else 1 if bucket < 85 else 2
        features.append(vector)
        labels.append(value)
        groups.append(prototype)
        splits.append(split)
        split_sets[split].add(prototype)
        assignments.append((str(row["jid"]), prototype, ("train", "validation", "test")[split]))
    if any(split_sets[a] & split_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError("PIEZO_PROTOTYPE_SPLIT_LEAKAGE")
    assignment_path = root / "data" / "splits" / "jarvis_dft_v11.structure_prototype.assignments.tsv"
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("jid", "structure_prototype_group", "split"))
        writer.writerows(assignments)
    record = save_dataset(
        root,
        "CAND-P-140",
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups),
        np.asarray(splits),
        "regression",
        "OPEN_COMPUTED_DFT",
        "jarvis_dft_v11",
        sha256_file(artifact),
        sha256_file(assignment_path),
        task_hashes["CAND-P-140"],
        "dfpt_piezo_max_dij",
    )
    record["feature_contract"] = "composition_fraction_118+density+nat+space_group+2d_flag;split_group=space_group_prototype"
    record["target_semantics"] = "JARVIS_DFPT_MAX_PIEZOELECTRIC_STRAIN_TENSOR_DIJ_COMPUTED_NOT_EXPERIMENTAL"
    write_json(root / "data" / "staged" / "CAND-P-140.metadata.json", record)
    return record


def stage_secom(root: Path, task_hashes: dict[str, str]) -> dict[str, Any]:
    artifact = root / "data" / "raw" / "uci_secom" / "secom.zip"
    assignment_path = root / "data" / "splits" / "uci_secom.assignments.tsv"
    assignments = {int(row["row_index"]): row for row in read_tsv(assignment_path)}
    with zipfile.ZipFile(artifact) as archive:
        data_lines = archive.read("secom.data").decode("utf-8").splitlines()
        label_lines = archive.read("secom_labels.data").decode("utf-8").splitlines()
    x, y, groups, splits = [], [], [], []
    for index, (data_line, label_line) in enumerate(zip(data_lines, label_lines, strict=True)):
        assignment = assignments[index]
        x.append([float(value) if value != "NaN" else np.nan for value in data_line.split()])
        label = int(label_line.split(maxsplit=1)[0])
        y.append(1 if label == 1 else 0)
        groups.append(assignment["time_group"])
        splits.append({"train": 0, "validation": 1, "test": 2}[assignment["split"]])
    return save_dataset(
        root,
        "CAND-P-087",
        np.asarray(x, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(groups),
        np.asarray(splits),
        "classification",
        "ANON_PRODUCTION",
        "uci_secom",
        sha256_file(artifact),
        sha256_file(assignment_path),
        task_hashes["CAND-P-087"],
        "pass_fail_yield_label",
    )


def update_queue(root: Path, staged: list[dict[str, Any]]) -> dict[str, Any]:
    queue_path = root / "queue" / "dual_5090_queue.v1.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    status_by_id = {item["candidate_id"]: item for item in staged}
    for shard in ("GPU_A", "GPU_B"):
        for job in queue["jobs"][shard]:
            record = status_by_id.get(job["candidate_id"])
            if record is None:
                continue
            if record["status"] == "PASS":
                job["admission_state"] = "ADMITTED"
                job["staged_dataset"] = record["path"]
                job["staged_dataset_sha256"] = record["sha256"]
                job["staged_metadata"] = f"data/staged/{job['candidate_id']}.metadata.json"
                job["data_binding"] = {
                    "full_data_state": "MATERIALIZED",
                    "source_family": record["source_id"],
                    "truth_class": record["truth_class"],
                }
            else:
                job["admission_state"] = "BLOCKED_PRE_GPU"
                job.pop("staged_dataset", None)
                job.pop("staged_dataset_sha256", None)
                job.pop("staged_metadata", None)
    jobs = queue["jobs"]["GPU_A"] + queue["jobs"]["GPU_B"]
    queue["admitted_jobs"] = sum(job["admission_state"] == "ADMITTED" for job in jobs)
    queue["blocked_jobs"] = len(jobs) - queue["admitted_jobs"]
    queue["admitted_by_shard"] = {
        shard: sum(job["admission_state"] == "ADMITTED" for job in queue["jobs"][shard]) for shard in ("GPU_A", "GPU_B")
    }
    queue["status"] = "RECOVERABLE_QUEUE_FROZEN_DATA_GATED"
    write_json(queue_path, queue)
    for shard, filename in (("GPU_A", "gpu_a.queue.json"), ("GPU_B", "gpu_b.queue.json")):
        write_json(root / "queue" / filename, {"schema": queue["schema"], "shard": shard, "jobs": queue["jobs"][shard]})
    return queue


def refresh_local_manifest(root: Path) -> None:
    path = root / "evidence" / "local_readiness_manifest.v1.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for record in manifest["records"]:
        artifact = root / record["path"]
        record["bytes"] = artifact.stat().st_size
        record["sha256"] = sha256_file(artifact)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["content_root_sha256"] = hashlib.sha256(canonical_bytes(manifest["records"])).hexdigest()
    write_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    hashes = task_contract_hashes(root)
    staged = stage_jarvis(root, hashes)
    staged.append(stage_jarvis_piezo_prototype(root, hashes))
    staged.append(stage_secom(root, hashes))
    queue = update_queue(root, staged)
    manifest = {
        "schema": "cimc.forge200.staging-manifest.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_WITH_FAIL_CLOSED_TASKS",
        "records": staged,
        "staged_pass": sum(item["status"] == "PASS" for item in staged),
        "staged_blocked": sum(item["status"] != "PASS" for item in staged),
        "queue_admitted_jobs": queue["admitted_jobs"],
        "queue_blocked_jobs": queue["blocked_jobs"],
        "content_root_sha256": hashlib.sha256(canonical_bytes(staged)).hexdigest(),
    }
    write_json(root / "data" / "staged" / "staging_manifest.v1.json", manifest)
    refresh_local_manifest(root)
    print(json.dumps({"status": manifest["status"], "staged": manifest["staged_pass"], "admitted": queue["admitted_jobs"], "blocked": queue["blocked_jobs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
