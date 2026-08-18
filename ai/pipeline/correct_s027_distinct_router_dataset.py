#!/usr/bin/env python3
"""Replace the collided S027 dataset with its full routing contract.

The first GPU pass accidentally gave S027 the same six-domain examples and
labels as S001, producing byte-identical learned weights.  This corrective set
adds explicit cross-domain and OOD classes while preserving PMCID-family split
isolation.  OOD rows are controlled negatives and are never described as
experimental truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from build_rag_training_sets import (
    DOMAIN_IDS,
    SPLIT_CODE,
    atomic_sentence,
    canonical_bytes,
    choose_cross,
    classification_rows,
    sha256_file,
    write_json,
)


OOD_BY_SPLIT = {
    0: (
        "guitar chord progression with fingering and tempo",
        "garden irrigation calendar and soil planting depth",
    ),
    1: (
        "municipal zoning appeal and parcel boundary hearing",
        "poetry meter scansion and historical language usage",
    ),
    2: (
        "restaurant menu planning and pastry decoration",
        "bird migration observation and field identification",
    ),
}


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build_examples(rows: list[dict[str, Any]]) -> list[tuple[str, int, str, int]]:
    examples: list[tuple[str, int, str, int]] = []
    task_cues = ("retrieve evidence", "answer with citation", "classify source domain")
    for index, row in enumerate(rows):
        split = SPLIT_CODE[row["split"]]
        sentence = atomic_sentence(row["text"])
        cue = task_cues[index % len(task_cues)]
        examples.append(
            (
                f"REQUEST {cue} QUERY {row['title']} {row['section']} {sentence}",
                DOMAIN_IDS[row["domain"]],
                row["pmcid"],
                split,
            )
        )

        peer = choose_cross(rows, index, different_domain=True)
        if SPLIT_CODE[peer["split"]] != split:
            raise RuntimeError("cross-domain peer crossed split")
        examples.append(
            (
                "REQUEST compare cross-domain constraints "
                f"SOURCE_A {row['domain']} {sentence} "
                f"SOURCE_B {peer['domain']} {atomic_sentence(peer['text'])}",
                6,
                f"{row['pmcid']}+{peer['pmcid']}",
                split,
            )
        )

        template = OOD_BY_SPLIT[split][index % len(OOD_BY_SPLIT[split])]
        examples.append(
            (
                f"REQUEST route or abstain QUERY {template} fixture_{row['chunk_id']}",
                7,
                f"CONTROLLED_OOD_{split}_{index % len(OOD_BY_SPLIT[split])}",
                split,
            )
        )
    return examples


def update_queue(root: Path, metadata: dict[str, Any]) -> None:
    queue_path = root / "queue" / "dual_5090_queue.v1.json"
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    found = False
    for shard in ("GPU_A", "GPU_B"):
        for job in queue["jobs"][shard]:
            if job["candidate_id"] != "CAND-S-027":
                continue
            found = True
            job["admission_state"] = "ADMITTED"
            job["staged_dataset"] = metadata["path"]
            job["staged_dataset_sha256"] = metadata["sha256"]
            job["staged_metadata"] = "data/staged/CAND-S-027.metadata.json"
            job["data_binding"] = {
                "full_data_state": "MATERIALIZED_CORRECTIVE_DISTINCT_ROUTER",
                "source_family": metadata["source_id"],
                "truth_class": metadata["truth_class"],
                "claim_state": metadata["claim_state"],
            }
    if not found:
        raise RuntimeError("S027 missing from queue")
    write_json(queue_path, queue)
    for shard, filename in (("GPU_A", "gpu_a.queue.json"), ("GPU_B", "gpu_b.queue.json")):
        write_json(
            root / "queue" / filename,
            {"schema": queue["schema"], "shard": shard, "jobs": queue["jobs"][shard]},
        )


def update_ledgers(root: Path, metadata: dict[str, Any]) -> None:
    bindings_path = root / "data" / "ledgers" / "task_source_bindings.v1.json"
    bindings = json.loads(bindings_path.read_text(encoding="utf-8"))
    record = next(item for item in bindings["records"] if item["candidate_id"] == "CAND-S-027")
    record.update(
        {
            "full_data_state": "MATERIALIZED_CORRECTIVE_DISTINCT_ROUTER",
            "local_build_status": "PASS",
            "source_family": metadata["source_id"],
            "truth_class": metadata["truth_class"],
            "split_sha256": metadata["split_sha256"],
            "staged_dataset_sha256": metadata["sha256"],
        }
    )
    write_json(bindings_path, bindings)

    manifest_path = root / "data" / "staged" / "rag_staging_manifest.v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["records"] = [
        metadata if item["candidate_id"] == "CAND-S-027" else item
        for item in manifest["records"]
    ]
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["content_root_sha256"] = hashlib.sha256(
        canonical_bytes(manifest["records"])
    ).hexdigest()
    manifest["corrective_note"] = (
        "S027 now has domain0..5, cross-domain6, OOD7 labels and no longer "
        "duplicates S001. Controlled OOD rows are not experimental truth."
    )
    write_json(manifest_path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    corpus_path = root / "data" / "corpora" / "ccby_multidomain_v1.jsonl"
    rows = [
        json.loads(line)
        for line in corpus_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    split_map = sorted({(row["pmcid"], row["split"]) for row in rows})
    split_sha = hashlib.sha256(canonical_bytes(split_map)).hexdigest()
    contracts = read_tsv(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    contract = next(item for item in contracts if item["candidate_id"] == "CAND-S-027")
    task_hash = hashlib.sha256(canonical_bytes(contract)).hexdigest()
    previous_path = root / "data" / "staged" / "CAND-S-027.npz"
    previous_sha = sha256_file(previous_path)
    examples = build_examples(rows)
    groups_by_split = {
        split: {group for _, _, group, item_split in examples if item_split == split}
        for split in (0, 1, 2)
    }
    if any(
        groups_by_split[left] & groups_by_split[right]
        for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise RuntimeError("S027 corrective group leakage")
    metadata = classification_rows(
        root,
        "CAND-S-027",
        examples,
        {"CAND-S-027": task_hash},
        sha256_file(corpus_path),
        split_sha,
        truth_class=(
            "LICENSED_IN_DOMAIN_PLUS_STRUCTURE_DERIVED_CROSS_DOMAIN_AND_"
            "CONTROLLED_OOD_FIXTURE"
        ),
        rule=(
            "domain_namespace_0_to_5_cross_document_same_split_pair_6_"
            "split_distinct_controlled_OOD_7"
        ),
    )
    metadata.update(
        {
            "class_contract": {
                **{str(value): name for name, value in DOMAIN_IDS.items()},
                "6": "CROSS_DOMAIN",
                "7": "OOD_ABSTAIN",
            },
            "controlled_ood_is_experimental_truth": False,
            "corrects_weight_collision_with": "CAND-S-001",
        }
    )
    write_json(root / "data" / "staged" / "CAND-S-027.metadata.json", metadata)
    update_queue(root, metadata)
    update_ledgers(root, metadata)
    receipt = {
        "schema": "cimc.forge200.s027-dataset-correction.v1",
        "status": "PASS_DISTINCT_ROUTER_DATASET_GPU_RETRAIN_REQUIRED",
        "candidate_id": "CAND-S-027",
        "previous_dataset_sha256": previous_sha,
        "corrected_dataset_sha256": metadata["sha256"],
        "records": metadata["records"],
        "counts": metadata["counts"],
        "cross_split_group_overlap": 0,
        "authority": 0,
        "board_accepted": False,
        "utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(root / "evidence" / "s027_distinct_router_correction.v1.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
