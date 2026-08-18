#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?candidate work root required}"
OUTPUT_ROOT="${2:?artifact output root required}"
TEACHER_CACHE="${3:?primary teacher cache root required}"
BRIDGE_CACHE="${4:?bridge cache root required}"
CANDIDATE_CSV="${5:?comma-separated candidate ids required}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda3/bin/python}"
MAX_EPOCHS="${MAX_EPOCHS:-60}"
MIN_EPOCHS="${MIN_EPOCHS:-40}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-12}"
QAT_EPOCHS="${QAT_EPOCHS:-6}"

mkdir -p "${OUTPUT_ROOT}" "${OUTPUT_ROOT}/logs"
IFS=',' read -r -a candidates <<< "${CANDIDATE_CSV}"

for candidate in "${candidates[@]}"; do
  receipt="${OUTPUT_ROOT}/${candidate}/promotion_receipt.json"
  if [[ -f "${receipt}" ]] && grep -q 'HOST_GPU_TRAINED_CORRECTIVE_EXACT_BASELINE_AND_BOARD_PENDING' "${receipt}"; then
    continue
  fi
  "${PYTHON_BIN}" "${ROOT}/pipeline/gpu_train_nanolm_v2_job.py" \
    --candidate-id "${candidate}" \
    --root "${ROOT}" \
    --artifact-root "${OUTPUT_ROOT}" \
    --teacher-cache-root "${TEACHER_CACHE}" \
    --bridge-cache-root "${BRIDGE_CACHE}" \
    --device cuda:0 \
    --batch-size 32 \
    --max-epochs "${MAX_EPOCHS}" \
    --min-epochs "${MIN_EPOCHS}" \
    --early-stop-patience "${EARLY_STOP_PATIENCE}" \
    --checkpoint-epochs 2 \
    --qat-epochs "${QAT_EPOCHS}" \
    --resume \
    > "${OUTPUT_ROOT}/logs/${candidate}.log" 2>&1
done

"${PYTHON_BIN}" - "${OUTPUT_ROOT}" "${CANDIDATE_CSV}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
candidates = sys.argv[2].split(",")
records = []
for candidate in candidates:
    receipt_path = root / candidate / "promotion_receipt.json"
    eval_path = root / candidate / "eval_grouped.json"
    quant_path = root / candidate / "quantization_parity.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    quant = json.loads(quant_path.read_text(encoding="utf-8"))
    if receipt["status"] != "HOST_GPU_TRAINED_CORRECTIVE_EXACT_BASELINE_AND_BOARD_PENDING":
        raise SystemExit(f"{candidate}: {receipt['status']}")
    records.append(
        {
            "candidate_id": candidate,
            "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "package_sha256": receipt["package"]["sha256"],
            "package_bytes": receipt["package"]["bytes"],
            "parameters": receipt["parameter_count"],
            "three_seed_count": receipt["three_seed_count"],
            "baseline_proxy_pass": evaluation["baseline_proxy_pass"],
            "mean_primary_composite": evaluation["mean_primary_composite"],
            "token_parity": quant["token_parity"],
            "sequence_parity": quant["sequence_parity"],
            "runtime_seconds": receipt["runtime_seconds"],
            "exact_contract_baseline_pending": receipt["exact_contract_baseline_pending"],
            "board_accepted": receipt["board_accepted"],
            "countable_model": receipt["countable_model"],
            "authority": receipt["authority"],
        }
    )
canonical = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
manifest = {
    "schema": "cimc.forge200.nanollm-training-shard.v2",
    "status": "PASS_TRAINED_CORRECTIVE_EXACT_BASELINE_AND_BOARD_PENDING",
    "candidate_count": len(records),
    "proxy_baseline_pass": sum(item["baseline_proxy_pass"] for item in records),
    "w8_token_parity_pass": sum(item["token_parity"] >= 0.95 for item in records),
    "w8_sequence_parity_exact": sum(item["sequence_parity"] == 1.0 for item in records),
    "package_bytes": sum(item["package_bytes"] for item in records),
    "runtime_seconds": sum(item["runtime_seconds"] for item in records),
    "exact_contract_baseline_pending": len(records),
    "board_accepted": 0,
    "countable_models": 0,
    "authority_nonzero": sum(item["authority"] != 0 for item in records),
    "records": records,
    "content_root_sha256": hashlib.sha256(canonical).hexdigest(),
}
(root / "manifest.v2.json").write_text(
    json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(manifest, sort_keys=True))
PY
