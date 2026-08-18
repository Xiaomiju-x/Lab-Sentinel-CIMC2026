"""
AI-3 TinyTransformer — Synthetic Temperature Time-Series Data Generator

Generates labeled sliding-window sequences from simulated sintering runs for
5-class temperature-curve classification:
  Class 0: normal       正常烧结曲线
  Class 1: fast_ramp    升温过快 (ramp rate × 4)
  Class 2: undertemp    烧结不达温 (sinter temp × 0.92)
  Class 3: temp_drift   温漂 (+60°C offset after sinter ramp)
  Class 4: slow_ramp    升温过慢 (ramp rate × 0.25)

Output files (in same directory):
  X_seq.npy          (N_train, SEQ_LEN=64, N_FEAT=8)  float32, z-scored
  y_seq.npy          (N_train,)                         int32 labels 0-4
  X_seq_test.npy     (N_test,  SEQ_LEN,    N_FEAT)      same normalisation
  y_seq_test.npy     (N_test,)
  ai3_data_stats.json  mu/std per feature + class map (for MCU inference)

Run: cd CIMC/model/ai3_sintering_transformer && python synth_data_ai3.py
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np

# ── import simulate_host from ai2 dir ─────────────────────────────────────────
_AI2_DIR = Path(__file__).parents[1] / "ai2_env_ae"
sys.path.insert(0, str(_AI2_DIR))
from synth_data import simulate_host, PROFILES_PATH  # noqa: E402

SEED = 42

SEQ_LEN  = 64   # time steps per window (64 consecutive simulated minutes)
STRIDE   = 16   # sliding-window stride

# Group-A temperature features (indices 0-7 of the 32-D feature vector)
FEAT_IDX = list(range(8))
#   [0] temp_current    [1] temp_target     [2] temp_deviation  [3] temp_gradient
#   [4] temp_rms_60s    [5] temp_std_60s    [6] hold_min_cum    [7] stage_id
N_FEAT = len(FEAT_IDX)  # 8

CLASS_NAMES = ["normal", "fast_ramp", "undertemp", "temp_drift", "slow_ramp"]
N_CLASSES   = len(CLASS_NAMES)


# ── anomaly generators ────────────────────────────────────────────────────────

def _make_fast_ramp(profile: dict, rng: np.random.Generator) -> np.ndarray:
    p = copy.deepcopy(profile)
    p["calcine"]["ramp_C_per_min"] = max(p["calcine"]["ramp_C_per_min"] * 4, 8)
    p["sinter"]["ramp_C_per_min"]  = max(p["sinter"]["ramp_C_per_min"]  * 4, 8)
    return simulate_host(p, rng)


def _make_undertemp(profile: dict, rng: np.random.Generator) -> np.ndarray:
    """
    Inject a large negative temperature deviation during sinter hold (stage_id=4).
    Using profile-target reduction is undetectable (furnace follows wrong target);
    injecting a visible -100 °C deviation during sinter hold is detectable in any
    window that overlaps the sinter phase.
    """
    arr = simulate_host(copy.deepcopy(profile), rng)
    # stage_id is feat[7], stored as stage/5 — sinter hold is stage 4 → value 0.8
    sinter_mask = np.abs(arr[:, 7] - 0.8) < 0.05   # stage_id ≈ 0.8 ± 0.05
    if sinter_mask.any():
        arr[sinter_mask, 2] -= 1.00    # temp_deviation /100 → −100 °C below target
        arr[sinter_mask, 0] -= 0.0625  # temp_current  /1600 → −100 °C absolute
    return arr


def _make_temp_drift(profile: dict, rng: np.random.Generator) -> np.ndarray:
    arr = simulate_host(copy.deepcopy(profile), rng)
    # inject +60 °C offset: inflate temp_deviation (feature [2]) and temp_current (feature [0])
    arr[:, 2] += 0.60   # temp_deviation / 100 → +60 °C
    arr[:, 0] += 0.04   # temp_current  / 1600 → +64 °C
    return arr


def _make_slow_ramp(profile: dict, rng: np.random.Generator) -> np.ndarray:
    p = copy.deepcopy(profile)
    p["calcine"]["ramp_C_per_min"] = max(p["calcine"]["ramp_C_per_min"] * 0.25, 0.5)
    p["sinter"]["ramp_C_per_min"]  = max(p["sinter"]["ramp_C_per_min"]  * 0.25, 0.5)
    return simulate_host(p, rng)


_ANOMALY_FN = {
    "fast_ramp":  _make_fast_ramp,
    "undertemp":  _make_undertemp,
    "temp_drift": _make_temp_drift,
    "slow_ramp":  _make_slow_ramp,
}


# ── window extraction ─────────────────────────────────────────────────────────

def _extract_windows(arr: np.ndarray, label: int):
    """Extract (SEQ_LEN, N_FEAT) windows from run array (T, 32). Returns (X, y)."""
    T    = arr.shape[0]
    feats = arr[:, FEAT_IDX]          # (T, N_FEAT)
    xs, ys = [], []
    for start in range(0, T - SEQ_LEN + 1, STRIDE):
        xs.append(feats[start : start + SEQ_LEN])
        ys.append(label)
    if not xs:
        return (np.zeros((0, SEQ_LEN, N_FEAT), dtype=np.float32),
                np.zeros(0, dtype=np.int32))
    return (np.array(xs, dtype=np.float32),
            np.full(len(xs), label, dtype=np.int32))


# ── main generator ────────────────────────────────────────────────────────────

def generate(seed: int = SEED, out_dir: Path = None):
    if out_dir is None:
        out_dir = Path(__file__).parent

    rng = np.random.default_rng(seed)
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    host_keys = [k for k in profiles if not k.startswith("_")]
    print(f"Generating AI-3 data for {len(host_keys)} host profiles ...")

    # Normal: 7 train noise levels + 1 test noise level per host
    NORMAL_TRAIN_NOISE = [0.5, 0.8, 1.0, 1.0, 1.0, 1.0, 1.2]
    NORMAL_TEST_NOISE  = [1.5]

    # Anomaly: 3 train noise levels + 1 test noise level per type per host
    ANOMALY_TYPES       = ["fast_ramp", "undertemp", "temp_drift", "slow_ramp"]
    ANOMALY_TRAIN_NOISE = [0.8, 1.0, 1.2]
    ANOMALY_TEST_NOISE  = [1.0]

    tr_X, tr_y = [], []
    te_X, te_y = [], []
    counts_tr = [0] * N_CLASSES
    counts_te = [0] * N_CLASSES

    for host in host_keys:
        p = profiles[host]

        # ── normal runs ───────────────────────────────────────────
        for noise in NORMAL_TRAIN_NOISE:
            try:
                arr = simulate_host(p, rng, noise_scale=noise)
                X, y = _extract_windows(arr, 0)
                tr_X.append(X); tr_y.append(y); counts_tr[0] += len(X)
            except Exception as exc:
                print(f"  [warn] {host} normal noise={noise}: {exc}")

        for noise in NORMAL_TEST_NOISE:
            try:
                arr = simulate_host(p, rng, noise_scale=noise)
                X, y = _extract_windows(arr, 0)
                te_X.append(X); te_y.append(y); counts_te[0] += len(X)
            except Exception as exc:
                print(f"  [warn] {host} normal test noise={noise}: {exc}")

        # ── anomaly runs ──────────────────────────────────────────
        for cls_idx, atype in enumerate(ANOMALY_TYPES, start=1):
            fn = _ANOMALY_FN[atype]

            for noise in ANOMALY_TRAIN_NOISE:
                try:
                    arr = fn(p, rng)
                    # mild noise perturbation on temperature features
                    arr[:, :8] += rng.normal(0, 0.005 * noise,
                                             (arr.shape[0], 8)).astype(np.float32)
                    X, y = _extract_windows(arr, cls_idx)
                    tr_X.append(X); tr_y.append(y); counts_tr[cls_idx] += len(X)
                except Exception as exc:
                    print(f"  [warn] {host} {atype} train noise={noise}: {exc}")

            for noise in ANOMALY_TEST_NOISE:
                try:
                    arr = fn(p, rng)
                    arr[:, :8] += rng.normal(0, 0.005 * noise,
                                             (arr.shape[0], 8)).astype(np.float32)
                    X, y = _extract_windows(arr, cls_idx)
                    te_X.append(X); te_y.append(y); counts_te[cls_idx] += len(X)
                except Exception as exc:
                    print(f"  [warn] {host} {atype} test noise={noise}: {exc}")

    X_tr = np.concatenate(tr_X, axis=0)
    y_tr = np.concatenate(tr_y, axis=0)
    X_te = np.concatenate(te_X, axis=0)
    y_te = np.concatenate(te_y, axis=0)

    # ── z-score normalise using training statistics ───────────────
    X_flat = X_tr.reshape(-1, N_FEAT)
    mu  = X_flat.mean(axis=0).astype(np.float32)
    std = (X_flat.std(axis=0) + 1e-8).astype(np.float32)

    X_tr = ((X_tr - mu) / std).astype(np.float32)
    X_te = ((X_te - mu) / std).astype(np.float32)

    # ── save ──────────────────────────────────────────────────────
    np.save(out_dir / "X_seq.npy",      X_tr)
    np.save(out_dir / "y_seq.npy",      y_tr)
    np.save(out_dir / "X_seq_test.npy", X_te)
    np.save(out_dir / "y_seq_test.npy", y_te)

    stats = {
        "mu":          mu.tolist(),
        "std":         std.tolist(),
        "feat_idx":    FEAT_IDX,
        "feat_names": ["temp_current", "temp_target", "temp_deviation", "temp_gradient",
                       "temp_rms_60s", "temp_std_60s", "hold_min_cum", "stage_id"],
        "class_names": CLASS_NAMES,
        "seq_len":     SEQ_LEN,
        "n_feat":      N_FEAT,
    }
    (out_dir / "ai3_data_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8")

    print(f"\nTrain  shape={X_tr.shape}  class counts: {counts_tr}")
    print(f"Test   shape={X_te.shape}  class counts: {counts_te}")
    print(f"\nSaved to {out_dir}:")
    print("  X_seq.npy / y_seq.npy         (train, z-scored)")
    print("  X_seq_test.npy / y_seq_test.npy  (held-out test)")
    print("  ai3_data_stats.json              mu/std + class map")
    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    generate()
