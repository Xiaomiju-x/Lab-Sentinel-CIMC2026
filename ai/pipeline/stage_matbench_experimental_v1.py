#!/usr/bin/env python3
"""Stage the upstream-confirmed Matbench experimental composition datasets.

P067/P068 require experimental targets to remain separate from computed DFT.
The upstream datasets contain composition-only inputs.  The wider frozen ABI is
therefore represented honestly with explicit structure/transport availability
masks; no structure or transport values are fabricated.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
from collections import defaultdict
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
NONMETALS = {"H", "He", "B", "C", "N", "O", "F", "Ne", "Si", "P", "S", "Cl", "Ar", "Ge", "As", "Se", "Br", "Kr", "Sb", "Te", "I", "Xe", "At", "Rn", "Ts", "Og"}
HALOGENS = {"F", "Cl", "Br", "I", "At", "Ts"}
TOKEN_RE = re.compile(r"[A-Z][a-z]?|\(|\)|(?:\d+(?:\.\d*)?|\.\d+)")
SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}


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


def read_contracts(root: Path) -> dict[str, dict[str, str]]:
    path = root / "contracts" / "candidate_task_contracts_244.v1.tsv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def multiply(values: dict[str, float], factor: float) -> dict[str, float]:
    return {key: value * factor for key, value in values.items()}


def parse_formula(formula: str) -> dict[str, float]:
    tokens = TOKEN_RE.findall(formula.replace(" ", ""))
    if not tokens or "".join(tokens) != formula.replace(" ", ""):
        raise ValueError(f"unsupported formula: {formula}")

    def parse(position: int, nested: bool) -> tuple[dict[str, float], int]:
        result: dict[str, float] = defaultdict(float)
        while position < len(tokens):
            token = tokens[position]
            if token == ")":
                if not nested:
                    raise ValueError(f"unmatched close parenthesis: {formula}")
                return dict(result), position + 1
            if token == "(":
                child, position = parse(position + 1, True)
                factor = 1.0
                if position < len(tokens) and tokens[position][0].isdigit():
                    factor = float(tokens[position])
                    position += 1
                for symbol, count in multiply(child, factor).items():
                    result[symbol] += count
                continue
            if token not in ELEMENT_INDEX:
                raise ValueError(f"invalid element token in {formula}: {token}")
            position += 1
            count = 1.0
            if position < len(tokens) and (tokens[position][0].isdigit() or tokens[position][0] == "."):
                count = float(tokens[position])
                position += 1
            result[token] += count
        if nested:
            raise ValueError(f"unclosed parenthesis: {formula}")
        return dict(result), position

    counts, final = parse(0, False)
    if final != len(tokens) or not counts or any(value <= 0 for value in counts.values()):
        raise ValueError(f"invalid formula: {formula}")
    return counts


def features(counts: dict[str, float], modality: str) -> np.ndarray:
    total = sum(counts.values())
    fractions = np.zeros(len(ELEMENTS), dtype=np.float32)
    for symbol, count in counts.items():
        fractions[ELEMENT_INDEX[symbol]] = count / total
    atomic_numbers = np.asarray([ELEMENT_INDEX[symbol] + 1 for symbol in counts], dtype=np.float64)
    weights = np.asarray([counts[symbol] / total for symbol in counts], dtype=np.float64)
    mean_z = float(np.sum(atomic_numbers * weights))
    std_z = float(np.sqrt(np.sum(((atomic_numbers - mean_z) ** 2) * weights)))
    entropy = -float(np.sum(weights * np.log(np.maximum(weights, 1e-12))))
    metallic_fraction = sum(count / total for symbol, count in counts.items() if symbol not in NONMETALS)
    descriptors = np.asarray(
        [
            len(counts) / 12.0,
            math.log1p(total) / 6.0,
            entropy / 3.0,
            mean_z / 118.0,
            std_z / 60.0,
            float(np.min(atomic_numbers)) / 118.0,
            float(np.max(atomic_numbers)) / 118.0,
            metallic_fraction,
            float("O" in counts),
            float(bool(HALOGENS & counts.keys())),
            0.0,
            1.0,
            0.0 if modality == "structure" else 1.0,
        ],
        dtype=np.float32,
    )
    # Final three entries are observed-structure, missing-structure and
    # missing-transport.  They make absent modalities explicit in the ABI.
    return np.concatenate((fractions, descriptors))


def material_family(counts: dict[str, float]) -> str:
    total = sum(counts.values())
    metal_fraction = sum(value / total for key, value in counts.items() if key not in NONMETALS)
    metal_bin = min(int(metal_fraction * 4.0), 3)
    return f"n{min(len(counts), 6)}|m{metal_bin}|o{int('O' in counts)}|h{int(bool(HALOGENS & counts.keys()))}"


def split_for_group(group: str) -> int:
    bucket = int(hashlib.sha256(group.encode("ascii")).hexdigest()[:8], 16) % 100
    return 0 if bucket < 70 else 1 if bucket < 85 else 2


def train_fitted_baseline(y: np.ndarray, split: np.ndarray, families: np.ndarray, task_kind: str) -> tuple[np.ndarray, np.ndarray]:
    train = split == 0
    result = np.empty(len(y), dtype=np.float32 if task_kind == "regression" else np.int64)
    score = np.empty(len(y), dtype=np.float32)
    global_value: float | int
    if task_kind == "regression":
        global_value = float(np.median(y[train]))
    else:
        global_value = int(np.bincount(y[train].astype(np.int64)).argmax())
    mapping: dict[str, float | int] = {}
    score_mapping: dict[str, float] = {}
    for family in sorted(set(families[train].tolist())):
        values = y[train & (families == family)]
        mapping[family] = float(np.median(values)) if task_kind == "regression" else int(np.bincount(values.astype(np.int64)).argmax())
        score_mapping[family] = float(np.median(values)) if task_kind == "regression" else float(np.mean(values))
    global_score = float(global_value) if task_kind == "regression" else float(np.mean(y[train]))
    for index, family in enumerate(families.tolist()):
        result[index] = mapping.get(family, global_value)
        score[index] = score_mapping.get(family, global_score)
    return result, score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    raw = root / "data" / "raw" / "matbench_experimental_v1"
    output_root = root / "data" / "staged_matbench_experimental_v1"
    contracts = read_contracts(root)
    definitions = {
        "CAND-P-067": {
            "file": "expt_gap.json.gz", "article": "figshare_article_9765779.json",
            "expected_sha256": "1e6816fb8e7132535b76cdac458c4d53944eca97e1ecb68e6a8fd853c9cedc3a",
            "expected_md5": "eb844a19b8607af4580860b4ecbd406d", "task_kind": "regression",
            "truth_class": "OPEN_EXPERIMENT", "modality": "structure", "doi": "10.6084/m9.figshare.9765779.v1",
        },
        "CAND-P-068": {
            "file": "expt_is_metal.json.gz", "article": "figshare_article_9765803.json",
            "expected_sha256": "80fb0854cff5d4657d812793f4748d5c23a9407f6c59bcf5b86c818ebaA6910f".lower(),
            "expected_md5": "71ba8c6b41882775ccb5f9fb44cc749a", "task_kind": "classification",
            "truth_class": "OPEN_EXPERIMENT", "modality": "transport", "doi": "10.6084/m9.figshare.9765803.v1",
        },
    }
    records = []
    for candidate_id, spec in definitions.items():
        source = raw / spec["file"]
        article_path = raw / spec["article"]
        article = json.loads(article_path.read_text(encoding="utf-8"))
        md5 = hashlib.md5(source.read_bytes()).hexdigest()  # nosec: upstream identity, not security
        if sha256_file(source) != spec["expected_sha256"] or md5 != spec["expected_md5"]:
            raise RuntimeError(f"{candidate_id}:SOURCE_HASH_GATE")
        if article.get("license", {}).get("name") != "MIT" or article.get("doi") != spec["doi"]:
            raise RuntimeError(f"{candidate_id}:UPSTREAM_LICENSE_OR_DOI_GATE")
        with gzip.open(source, "rt", encoding="utf-8") as handle:
            frame = json.load(handle)
        x, y, groups, split, families, formulas = [], [], [], [], [], []
        rejected = []
        for formula, target in frame["data"]:
            try:
                counts = parse_formula(str(formula))
            except ValueError as exc:
                rejected.append(str(exc))
                continue
            group = "-".join(sorted(counts))
            x.append(features(counts, spec["modality"]))
            y.append(float(target) if spec["task_kind"] == "regression" else int(bool(target)))
            groups.append(group)
            split.append(split_for_group(group))
            families.append(material_family(counts))
            formulas.append(str(formula))
        x_array = np.asarray(x, dtype=np.float32)
        y_array = np.asarray(y, dtype=np.float32 if spec["task_kind"] == "regression" else np.int64)
        group_array = np.asarray(groups)
        split_array = np.asarray(split, dtype=np.int8)
        family_array = np.asarray(families)
        baseline, baseline_score = train_fitted_baseline(y_array, split_array, family_array, spec["task_kind"])
        group_sets = {code: set(group_array[split_array == code]) for code in (0, 1, 2)}
        overlap = sum(len(group_sets[a] & group_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
        counts_by_split = {name: int(np.sum(split_array == code)) for name, code in SPLIT_CODE.items()}
        # Both upstream files contain the same single malformed composition
        # token ("G1128Ga1As0.1P0.9").  Exclude it identically from both tasks
        # and record it; guessing that "G" means a chemical element would
        # silently alter an experimental row.
        expected_rejections = ["invalid element token in G1128Ga1As0.1P0.9: G"]
        if rejected != expected_rejections or overlap or min(counts_by_split.values()) < 16:
            raise RuntimeError(f"{candidate_id}:STAGING_GATE:rejected={len(rejected)}:overlap={overlap}:counts={counts_by_split}")
        output = output_root / f"{candidate_id}.npz"
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output, x=x_array, y=y_array, groups=group_array, split=split_array,
            family=family_array, formula=np.asarray(formulas), baseline_pred=baseline,
            baseline_score=baseline_score,
            candidate_id=np.asarray(candidate_id), task_kind=np.asarray(spec["task_kind"]),
            truth_class=np.asarray(spec["truth_class"]), authority=np.asarray(0, dtype=np.int8),
        )
        contract = contracts[candidate_id]
        record = {
            "schema": "cimc.forge200.matbench-experimental-staged.v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS", "candidate_id": candidate_id, "task_kind": spec["task_kind"],
            "truth_class": spec["truth_class"], "source_id": Path(spec["file"]).stem,
            "source_url": article["url_public_html"], "download_url": article["files"][0]["download_url"],
            "doi": spec["doi"], "upstream_reference": article.get("references", []), "license": "MIT",
            "source_sha256": sha256_file(source), "source_md5": md5,
            "metadata_sha256": sha256_file(article_path),
            "excluded_source_rows": len(rejected), "excluded_source_row_reasons": rejected,
            "records": len(x_array), "features": int(x_array.shape[1]), "counts": counts_by_split,
            "split_unit": "CHEMICAL_SYSTEM", "cross_split_group_overlap": overlap,
            "split_sha256": hashlib.sha256(canonical_bytes(sorted(zip(groups, split, strict=True)))).hexdigest(),
            "feature_contract": "composition_118+composition_descriptors+explicit_structure_transport_missingness_masks",
            "input_contract": contract["input_contract"],
            "input_contract_state": "SATISFIED_WITH_EXPLICIT_MISSING_MODALITY_MASK_NO_VALUES_FABRICATED",
            "missing_modalities": ["crystal_structure"] if spec["modality"] == "structure" else ["crystal_structure", "transport_metadata"],
            "target_label": contract["target_label"], "baseline": contract["baseline"],
            "baseline_fit": "TRAIN_ONLY_MATERIAL_FAMILY_WITH_GLOBAL_FALLBACK",
            "material_family_definition": "element_count|metal_fraction_quartile|oxygen_presence|halogen_presence",
            "primary_metric": contract["primary_metric"], "parameter_cap": contract["parameter_cap"],
            "task_contract_sha256": hashlib.sha256(canonical_bytes(contract)).hexdigest(),
            "fit_preprocessing_on_train_only": True, "teacher_outputs": 0, "computed_dft_labels": 0,
            "authority": 0, "board_accepted": False, "countable_model": False,
            "path": str(output.relative_to(root)).replace("\\", "/"), "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
        write_json(output.with_suffix(".metadata.json"), record)
        records.append(record)
    content = {"records": records, "authority_nonzero": 0, "board_actions": 0}
    receipt = {
        "schema": "cimc.forge200.matbench-experimental-staging.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS",
        **content, "content_root_sha256": hashlib.sha256(canonical_bytes(content)).hexdigest(),
    }
    write_json(root / "evidence" / "matbench_experimental_staging.v1.json", receipt)
    print(json.dumps({"status": "PASS", "candidates": {r["candidate_id"]: r["records"] for r in records}, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
