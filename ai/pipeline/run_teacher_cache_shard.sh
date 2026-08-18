#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?candidate work root required}"
MODEL_PATH="${2:?teacher model path required}"
OUTPUT_ROOT="${3:?output root required}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-4B}"
MODEL_REVISION="${MODEL_REVISION:-master}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"
MANIFEST_CACHE="${OUTPUT_ROOT}/teacher_model_files.v1.json"

for raw_number in $(seq 1 26); do
  printf -v number "%03d" "${raw_number}"
  candidate="CAND-G-${number}"
  receipt="${OUTPUT_ROOT}/${candidate}/receipt.json"
  if [[ -f "${receipt}" ]] && grep -q 'PASS_TRAIN_ONLY_TEACHER_CACHE' "${receipt}"; then
    continue
  fi
  "${PYTHON_BIN}" "${ROOT}/pipeline/build_teacher_distillation_cache.py" \
    --root "${ROOT}" \
    --model-path "${MODEL_PATH}" \
    --model-id "${MODEL_ID}" \
    --model-revision "${MODEL_REVISION}" \
    --candidate-id "${candidate}" \
    --output-root "${OUTPUT_ROOT}" \
    --device cuda:0 \
    --batch-size 4 \
    --model-manifest-cache "${MANIFEST_CACHE}" \
    > "${OUTPUT_ROOT}/logs/${candidate}.log" 2>&1
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
records = []
for number in range(1, 27):
    candidate = f"CAND-G-{number:03d}"
    receipt_path = root / candidate / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "PASS_TRAIN_ONLY_TEACHER_CACHE":
        raise SystemExit(f"{candidate}: {receipt['status']}")
    records.append(
        {
            "candidate_id": candidate,
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "cache_sha256": receipt["cache_sha256"],
            "train_records_processed": receipt["train_records_processed"],
            "soft_positions": receipt["soft_positions"],
            "validation_records_seen": receipt["validation_records_seen"],
            "test_records_seen": receipt["test_records_seen"],
            "authority": receipt["authority"],
        }
    )
canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
manifest = {
    "schema": "cimc.forge200.teacher-cache-shard.v1",
    "status": "PASS",
    "candidate_count": len(records),
    "train_records_processed": sum(item["train_records_processed"] for item in records),
    "soft_positions": sum(item["soft_positions"] for item in records),
    "validation_records_seen": sum(item["validation_records_seen"] for item in records),
    "test_records_seen": sum(item["test_records_seen"] for item in records),
    "authority_nonzero": sum(item["authority"] != 0 for item in records),
    "teacher_promoted_to_ground_truth": 0,
    "records": records,
    "content_root_sha256": hashlib.sha256(canonical).hexdigest(),
}
(root / "manifest.v1.json").write_text(
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, sort_keys=True))
PY
