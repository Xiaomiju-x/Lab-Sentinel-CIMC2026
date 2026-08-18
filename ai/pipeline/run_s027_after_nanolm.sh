#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?candidate work root required}"
WAIT_MANIFEST="${2:?completed NanoLM shard manifest required}"
OUTPUT_ROOT="${3:?corrective output root required}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"

mkdir -p "${OUTPUT_ROOT}/logs"
while [[ ! -f "${WAIT_MANIFEST}" ]]; do
  sleep 5
done

"${PYTHON_BIN}" "${ROOT}/pipeline/gpu_train_job.py" \
  --candidate-id CAND-S-027 \
  --root "${ROOT}" \
  --artifact-root "${OUTPUT_ROOT}" \
  --device cuda:0 \
  --batch-size 1024 \
  --max-epochs 100 \
  --checkpoint-epochs 5 \
  --early-stop-patience 12 \
  --resume \
  > "${OUTPUT_ROOT}/logs/CAND-S-027.log" 2>&1

test -f "${OUTPUT_ROOT}/CAND-S-027/promotion_receipt.json"
