# Forge200 dual RTX5090 runbook

Status: `LOCAL_PREPARATION / NO_REMOTE_CONNECTION_PERFORMED`.

## Safety and identity gates

- Use only newly supplied SSH endpoints. The scripts do not contain or open SSH connections.
- Each instance must report project `CIMC`, one NVIDIA RTX5090 with at least 28 GiB visible VRAM, and enough local disk for its shard.
- GPU-A and GPU-B are independent workers. Do not use cross-public-network DDP.
- Every new model has `authority=0`; no artifact is board accepted in this phase.

## Environment

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --requirement pipeline/requirements-gpu-cu128.lock.txt
python pipeline/gpu_queue_worker.py --shard GPU_A --mode audit
python pipeline/gpu_queue_worker.py --shard GPU_B --mode audit
```

## Pilot and resume

Run the pilot for 30–60 minutes only after both shard audits pass:

```bash
python pipeline/gpu_queue_worker.py --shard GPU_A --mode pilot --artifact-root artifacts/cloud5090 --pilot-jobs 12 --pilot-epochs 40 --max-minutes 60 --resume
python pipeline/gpu_queue_worker.py --shard GPU_B --mode pilot --artifact-root artifacts/cloud5090 --pilot-jobs 8 --pilot-epochs 40 --max-minutes 60 --resume
```

At two elapsed hours, recompute ETA from completed work. Pause before the full queue if projected dual-card wall time exceeds 10 hours. Both shards contain admitted source-bound tasks; pre-GPU rejected jobs remain excluded and fixture results never count as trained models.

```bash
python pipeline/reestimate_gpu_eta.py --artifact-root artifacts/cloud5090 --elapsed-hours 2
```

Full workers, after pilot approval:

```bash
python pipeline/gpu_queue_worker.py --shard GPU_A --mode full --artifact-root artifacts/cloud5090 --resume
python pipeline/gpu_queue_worker.py --shard GPU_B --mode full --artifact-root artifacts/cloud5090 --resume
```

Continuously copy back `worker_*.state.json`, per-candidate directories, and `transfer_*.json`. Verify all returned hashes before releasing an instance.

If an hourly instance must be cloned or replaced, follow `pipeline/GPU_INSTANCE_MIGRATION.md`. Never reuse an old endpoint without newly supplied SSH details.
