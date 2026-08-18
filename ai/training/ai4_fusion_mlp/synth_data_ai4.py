"""
AI-4 Fusion MLP — Synthetic Multi-Modal Feature Generator

Simulates the 16-D fusion input that the MCU assembles at runtime from the
outputs of AI-1, AI-2, AI-3, and the DOA sub-system, then labels each sample
with one of 4 risk levels.

16-D feature layout
--------------------
[0:4]   AI-1 softmax probs:  [p_empty, p_loaded, p_sintering, p_done]
[4]     AI-2 anomaly ratio:  score / q_hat  (1.0 = at threshold)
[5]     AI-2 temp feat residual   (normalised 0-1)
[6]     AI-2 vib  feat residual   (normalised 0-1)
[7]     AI-2 gas  feat residual   (normalised 0-1)
[8:13]  AI-3 softmax probs:  [p_normal, p_fast_ramp, p_undertemp, p_temp_drift, p_slow_ramp]
[13]    DOA anomaly flag      (0 = quiet, 1 = anomalous sound detected)
[14]    DOA intensity         (normalised 0-1)
[15]    sintering progress    (0-1)

Risk levels (output classes)
-----------------------------
  0  良品     (good)       — no anomaly, sintering proceeding normally
  1  疑似废品  (suspected)  — one mild anomaly signal; continue but alert
  2  废品     (bad)        — clear anomaly; halt and log failure
  3  严重异常  (critical)   — multiple or severe anomalies; trigger robot + alarm

Output:
  X_fusion.npy          (N_train, 16)  float32
  y_fusion.npy          (N_train,)     int32
  X_fusion_test.npy     (N_test,  16)
  y_fusion_test.npy     (N_test,)
  ai4_data_info.json    feature layout + class map

Run: cd CIMC/model/ai4_fusion_mlp && python synth_data_ai4.py
"""

import json
import numpy as np
from pathlib import Path

SEED       = 42
N_FEAT     = 16
N_CLASSES  = 4
CLASS_NAMES = ["good", "suspected", "bad", "critical"]

OUT_DIR = Path(__file__).parent


# ── helpers ───────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def _noisy_softmax(peak_class: int, n_classes: int, rng, concentration: float = 4.0) -> np.ndarray:
    """Dirichlet-like softmax centred on peak_class."""
    alpha = np.ones(n_classes) * 0.5
    alpha[peak_class] = concentration
    return rng.dirichlet(alpha).astype(np.float32)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# ── per-class sample generator ────────────────────────────────────────────────

def _sample_good(rng, n: int) -> np.ndarray:
    """Normal sintering — all signals clean."""
    rows = []
    for _ in range(n):
        stage = rng.choice([2, 3])           # sintering or done
        ai1   = _noisy_softmax(stage, 4, rng, concentration=6.0)
        ai2_ratio = rng.uniform(0.0, 0.6)    # well below q_hat
        ai2_resid = _clamp01(rng.normal(0.1, 0.05, 3).astype(np.float32))
        ai3   = _noisy_softmax(0, 5, rng, concentration=6.0)  # normal
        doa_flag = 0.0
        doa_int  = rng.uniform(0.0, 0.15)
        progress = rng.uniform(0.0, 1.0)
        rows.append([*ai1, ai2_ratio, *ai2_resid, *ai3, doa_flag, doa_int, progress])
    return np.array(rows, dtype=np.float32)


def _sample_suspected(rng, n: int) -> np.ndarray:
    """One mild anomaly signal present."""
    rows = []
    anomaly_types = ["ai2_mild", "ai3_minor", "doa_weak", "ai1_wrong_stage"]
    for _ in range(n):
        atype = rng.choice(anomaly_types)

        ai1 = _noisy_softmax(rng.choice([1, 2, 3]), 4, rng, concentration=3.0)
        ai2_ratio = rng.uniform(0.3, 0.8)
        ai2_resid = _clamp01(rng.normal(0.15, 0.08, 3).astype(np.float32))
        ai3 = _noisy_softmax(0, 5, rng, concentration=3.0)
        doa_flag = 0.0
        doa_int  = rng.uniform(0.0, 0.2)
        progress = rng.uniform(0.0, 1.0)

        if atype == "ai2_mild":
            ai2_ratio = rng.uniform(0.9, 1.3)   # slightly above threshold
            ai2_resid[rng.integers(3)] = rng.uniform(0.4, 0.7)
        elif atype == "ai3_minor":
            ai3 = _noisy_softmax(rng.choice([1, 2, 4]), 5, rng, concentration=2.5)
        elif atype == "doa_weak":
            doa_flag = 1.0
            doa_int  = rng.uniform(0.2, 0.5)
        elif atype == "ai1_wrong_stage":
            stage = rng.choice([0, 1])           # empty/loaded during sintering
            ai1 = _noisy_softmax(stage, 4, rng, concentration=3.0)

        rows.append([*ai1, ai2_ratio, *ai2_resid, *ai3, doa_flag, doa_int, progress])
    return np.array(rows, dtype=np.float32)


def _sample_bad(rng, n: int) -> np.ndarray:
    """Clear single anomaly — batch quality compromised."""
    rows = []
    anomaly_types = ["ai2_high", "ai3_confirmed", "doa_strong", "ai2_ai3_both"]
    for _ in range(n):
        atype = rng.choice(anomaly_types)

        ai1 = _noisy_softmax(rng.choice([1, 2, 3]), 4, rng, concentration=3.0)
        ai2_ratio = rng.uniform(0.5, 1.2)
        ai2_resid = _clamp01(rng.normal(0.2, 0.1, 3).astype(np.float32))
        ai3 = _noisy_softmax(0, 5, rng, concentration=3.0)
        doa_flag = 0.0
        doa_int  = rng.uniform(0.0, 0.3)
        progress = rng.uniform(0.0, 1.0)

        if atype == "ai2_high":
            ai2_ratio = rng.uniform(1.5, 3.0)
            ai2_resid = _clamp01(rng.normal(0.6, 0.15, 3).astype(np.float32))
        elif atype == "ai3_confirmed":
            cls = rng.choice([1, 2, 3, 4])      # fast_ramp / undertemp / temp_drift / slow_ramp
            ai3 = _noisy_softmax(cls, 5, rng, concentration=5.0)
        elif atype == "doa_strong":
            doa_flag = 1.0
            doa_int  = rng.uniform(0.6, 1.0)
        elif atype == "ai2_ai3_both":
            ai2_ratio = rng.uniform(1.2, 2.0)
            cls = rng.choice([1, 2, 4])
            ai3 = _noisy_softmax(cls, 5, rng, concentration=4.0)

        rows.append([*ai1, ai2_ratio, *ai2_resid, *ai3, doa_flag, doa_int, progress])
    return np.array(rows, dtype=np.float32)


def _sample_critical(rng, n: int) -> np.ndarray:
    """Multiple or severe anomalies — immediate intervention required."""
    rows = []
    for _ in range(n):
        # Multiple simultaneous anomalies
        ai1 = _noisy_softmax(rng.choice([0, 1]), 4, rng, concentration=3.0)  # wrong state
        ai2_ratio = rng.uniform(2.0, 6.0)       # far above threshold
        ai2_resid = _clamp01(rng.normal(0.8, 0.15, 3).astype(np.float32))
        # AI-3 severe: temp_drift (cls3) most critical; also fast_ramp (cls1)
        cls = rng.choice([1, 3], p=[0.4, 0.6])
        ai3 = _noisy_softmax(cls, 5, rng, concentration=5.0)
        doa_flag = float(rng.random() > 0.3)     # 70% also have sound anomaly
        doa_int  = rng.uniform(0.5, 1.0) if doa_flag else rng.uniform(0.0, 0.3)
        progress = rng.uniform(0.1, 0.9)         # critical mid-run

        rows.append([*ai1, ai2_ratio, *ai2_resid, *ai3, doa_flag, doa_int, progress])
    return np.array(rows, dtype=np.float32)


_SAMPLERS = [_sample_good, _sample_suspected, _sample_bad, _sample_critical]


# ── main generator ────────────────────────────────────────────────────────────

def generate(seed: int = SEED, n_per_class_train: int = 2000,
             n_per_class_test: int = 500, out_dir: Path = None):
    if out_dir is None:
        out_dir = OUT_DIR

    rng = np.random.default_rng(seed)

    X_tr, y_tr, X_te, y_te = [], [], [], []

    print(f"Generating AI-4 fusion data  ({n_per_class_train} train + {n_per_class_test} test per class) ...")
    for cls_idx, (cls_name, sampler) in enumerate(zip(CLASS_NAMES, _SAMPLERS)):
        n_total = n_per_class_train + n_per_class_test
        X = sampler(rng, n_total)
        y = np.full(n_total, cls_idx, dtype=np.int32)
        X_tr.append(X[:n_per_class_train]);  y_tr.append(y[:n_per_class_train])
        X_te.append(X[n_per_class_train:]);  y_te.append(y[n_per_class_train:])
        print(f"  class {cls_idx} {cls_name:12s}: {n_per_class_train} train + {n_per_class_test} test")

    X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
    X_te = np.concatenate(X_te); y_te = np.concatenate(y_te)

    # Shuffle train
    idx = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[idx], y_tr[idx]

    # Clip ai2_ratio at 6 to avoid extreme outliers
    X_tr[:, 4] = np.clip(X_tr[:, 4], 0, 6)
    X_te[:, 4] = np.clip(X_te[:, 4], 0, 6)

    np.save(out_dir / "X_fusion.npy",      X_tr)
    np.save(out_dir / "y_fusion.npy",      y_tr)
    np.save(out_dir / "X_fusion_test.npy", X_te)
    np.save(out_dir / "y_fusion_test.npy", y_te)

    info = {
        "n_feat":      N_FEAT,
        "n_classes":   N_CLASSES,
        "class_names": CLASS_NAMES,
        "feat_layout": {
            "0:4":  "AI-1 softmax probs [empty, loaded, sintering, done]",
            "4":    "AI-2 anomaly_ratio = score / q_hat",
            "5":    "AI-2 temp feature residual (0-1)",
            "6":    "AI-2 vib  feature residual (0-1)",
            "7":    "AI-2 gas  feature residual (0-1)",
            "8:13": "AI-3 softmax probs [normal, fast_ramp, undertemp, temp_drift, slow_ramp]",
            "13":   "DOA anomaly flag (0/1)",
            "14":   "DOA intensity (0-1)",
            "15":   "sintering progress (0-1)",
        },
    }
    (out_dir / "ai4_data_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8")

    print(f"\nSaved to {out_dir}:")
    print(f"  X_fusion.npy / y_fusion.npy                 {X_tr.shape}")
    print(f"  X_fusion_test.npy / y_fusion_test.npy       {X_te.shape}")
    print(f"  ai4_data_info.json")
    return X_tr, y_tr, X_te, y_te


if __name__ == "__main__":
    generate()
