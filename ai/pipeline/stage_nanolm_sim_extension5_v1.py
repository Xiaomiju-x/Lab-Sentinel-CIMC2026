#!/usr/bin/env python3
"""Stage five independent, structured-simulation NanoLM extension datasets."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from build_nanolm_contract_exact_v3 import SPLIT_CODE, encode_example
from build_teacher_distillation_cache import ContractTokenizer
from gpu_train_job import canonical_bytes, sha256_file, write_json
from nanolm_architecture import MAX_GENERATION_TOKENS, config_for_candidate


SPECS = {
    "CAND-G-002": ("FURNACE", ("stable", "heating", "cooling", "sensor-fault"), ("observe", "hold", "refuse", "escalate")),
    "CAND-G-027": ("HYP", ("phase-lag", "thermal-gradient", "dopant-loss", "sensor-bias"), ("xrd", "thermal-map", "pl-repeat", "reference-check")),
    "CAND-G-028": ("COUNTER", ("measurement-artifact", "batch-variance", "site-change", "model-mismatch"), ("repeat", "cross-sensor", "new-batch", "ablation")),
    "CAND-G-029": ("NEXT", ("xrd", "pl", "sem", "thermal-map"), ("stop-on-match", "stop-on-conflict", "stop-on-budget", "stop-on-uncertainty")),
    "CAND-G-030": ("SYNTH", ("supported", "mixed", "unresolved", "rejected"), ("low-risk", "medium-risk", "high-risk", "insufficient-evidence")),
}


def task_contracts(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["candidate_id"]: row for row in csv.DictReader(handle, delimiter="\t")}


def make_examples(candidate_id: str) -> list[dict[str, Any]]:
    field, values, actions = SPECS[candidate_id]
    examples: list[dict[str, Any]] = []
    split_families = (("train", 256), ("validation", 96), ("test", 96))
    running = 0
    for split, family_count in split_families:
        for local_index in range(family_count):
            family = f"{candidate_id}-SIM-{split.upper()}-{local_index:03d}"
            value = values[(running + local_index) % len(values)]
            action = actions[(running * 3 + local_index) % len(actions)]
            evidence_id = f"SIM-{int(candidate_id[-3:]):03d}-{running + local_index:04d}"
            claim = f"{field}:{value} ACT:{action}"
            prompt = (
                f"TASK {candidate_id}\nQUESTION Return one supported simulated decision.\n"
                f"SOURCE[1] {evidence_id}\nSOURCE_STATE STRUCTURED_SIMULATION\n"
                f"EVIDENCE_CARD[1] {field}:{value} ACTION:{action} AUTHORITY:0\nANSWER "
            )
            common = {
                "group": family,
                "split": split,
                "source_chunk_id": evidence_id,
                "source_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "claim": claim,
                "baseline_extract": "REFUSE unsupported [1].",
                "source_numbers": "0|1",
            }
            examples.append(
                {
                    **common,
                    "prompt": prompt,
                    "target": f"[1] {claim} SCOPE:SIM_ONLY",
                    "is_refusal": 0,
                    "negative_claim": "",
                }
            )
            wrong_value = values[(values.index(value) + 1) % len(values)]
            examples.append(
                {
                    **common,
                    "prompt": prompt.replace("Return one supported simulated decision.", f"Is {field}:{wrong_value} supported?"),
                    "target": "REFUSE unsupported [1].",
                    "is_refusal": 1,
                    "negative_claim": f"{field}:{wrong_value}",
                }
            )
        running += family_count
    return examples


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "contracts" / "nanolm_sim_extension5.v1.json"
    extension = json.loads(contract_path.read_text(encoding="utf-8"))
    if extension["status"] != "PRETRAIN_FROZEN" or extension["authority"] != 0:
        raise RuntimeError("NANOLM_SIM5_CONTRACT_GATE")
    tokenizer_path = root / extension["tokenizer"]
    tokenizer = ContractTokenizer(tokenizer_path)
    contracts = task_contracts(root / "contracts" / "candidate_task_contracts_244.v1.tsv")
    stage_root = root / "data" / "staged_nanolm_sim_extension5_v1"
    stage_root.mkdir(parents=True, exist_ok=True)
    receipt_records = []
    for candidate_id in extension["candidate_ids"]:
        examples = make_examples(candidate_id)
        encoded = [encode_example(tokenizer, item) for item in examples]
        if max(item["target_length"] for item in encoded) > MAX_GENERATION_TOKENS:
            raise RuntimeError(f"{candidate_id}:TARGET_LENGTH_GATE")
        arrays: dict[str, Any] = {
            key: np.asarray([item[key] for item in encoded])
            for key in ("x", "y", "loss_mask", "prompt_tokens", "prompt_length", "target_tokens", "target_length")
        }
        arrays.update(
            {
                "groups": np.asarray([item["group"] for item in examples]),
                "split": np.asarray([SPLIT_CODE[item["split"]] for item in examples], dtype=np.int8),
                "is_refusal": np.asarray([item["is_refusal"] for item in examples], dtype=np.uint8),
                "source_chunk_id": np.asarray([item["source_chunk_id"] for item in examples]),
                "claim_text": np.asarray([item["claim"] for item in examples]),
                "baseline_extract_text": np.asarray([item["baseline_extract"] for item in examples]),
                "source_numbers": np.asarray([item["source_numbers"] for item in examples]),
                "negative_claim_text": np.asarray([item["negative_claim"] for item in examples]),
                "candidate_id": np.asarray(candidate_id),
                "task_kind": np.asarray("nano_transformer_lm"),
                "truth_class": np.asarray("STRUCTURED_SIMULATION"),
                "authority": np.asarray(0, dtype=np.int8),
            }
        )
        counts = {name: int(np.sum(arrays["split"] == code)) for name, code in SPLIT_CODE.items()}
        group_sets = {code: set(arrays["groups"][arrays["split"] == code].tolist()) for code in SPLIT_CODE.values()}
        overlap = sum(len(group_sets[left] & group_sets[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
        if counts != extension["split_counts"] or overlap:
            raise RuntimeError(f"{candidate_id}:SPLIT_GATE:{counts}:{overlap}")
        data_path = stage_root / f"{candidate_id}.npz"
        np.savez_compressed(data_path, **arrays)
        original = contracts[candidate_id]
        metadata = {
            "schema": "cimc.forge200.staged-nanolm-sim-extension.v1",
            "status": "PASS_CONTRACT_SHAPED_SOURCE_SUPERVISED",
            "candidate_id": candidate_id,
            "task_kind": "nano_transformer_lm",
            "truth_class": "STRUCTURED_SIMULATION",
            "claim_state": "SIMULATED_EVIDENCE_CARD_NOT_EXPERT_REVIEW_OR_EXPERIMENTAL_GROUND_TRUTH",
            "public_claim_scope": "SIM_ONLY",
            "original_task_contract_status": extension["original_task_contract_status"],
            "path": str(data_path.relative_to(root)).replace("\\", "/"),
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "records": len(examples),
            "split_counts": counts,
            "split_unit": extension["split_unit"],
            "cross_split_group_overlap": overlap,
            "tokenizer_path": extension["tokenizer"],
            "tokenizer_sha256": sha256_file(tokenizer_path),
            "source_path": "contracts/nanolm_sim_extension5.v1.json",
            "source_sha256": sha256_file(contract_path),
            "task_contract_sha256": hashlib.sha256(canonical_bytes(original)).hexdigest(),
            "input_contract_state": "SIM_EXTENSION_SHAPED_ORIGINAL_EXPERT_CONTRACT_UNCHANGED",
            "exact_metric_fields": ["sim_task_field", "citation", "scope", "refusal"],
            "architecture": config_for_candidate(candidate_id).to_dict(),
            "teacher_outputs": 0,
            "teacher_promoted_to_ground_truth": False,
            "authority": 0,
            "board_accepted": False,
            "countable_model": False,
        }
        write_json(data_path.with_suffix(".metadata.json"), metadata)
        receipt_records.append(metadata)
    receipt = {
        "schema": "cimc.forge200.nanollm-sim-extension5-staging.v1",
        "status": "PASS_5_SIM_ONLY_NANOLM_DATASETS_FROZEN",
        "candidate_count": 5,
        "records": receipt_records,
        "original_exact_contract_promotions": 0,
        "authority_nonzero": 0,
        "board_actions": 0,
        "content_root_sha256": hashlib.sha256(canonical_bytes(receipt_records)).hexdigest(),
    }
    write_json(root / "evidence" / "nanolm_sim_extension5_staging.v1.json", receipt)
    print(json.dumps({"status": receipt["status"], "candidates": extension["candidate_ids"], "records_each": 896, "content_root_sha256": receipt["content_root_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
