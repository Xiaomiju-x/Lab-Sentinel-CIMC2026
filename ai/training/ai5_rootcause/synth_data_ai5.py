"""
AI-5 Root-Cause Diagnoser — Synthetic Diagnostic-Feature Generator

AI-1/2/3/4 detect THAT a sintering run is anomalous. AI-5 maps the upstream-AI
*signature* (+ raw furnace/gas/humidity context) to a NAMED process root cause,
so the operator gets WHY + WHAT-TO-DO, not just a risk level.

This simulates the 27-D diagnostic vector the MCU assembles at runtime from the
live snapshot (lab_ai_snapshot_t + env_result + gas_safety_eval + furnace_sim),
then labels each sample with one of 9 root-cause classes. Per-class feature
ranges are GROUNDED, not invented:
  - AI-1/2/3 channels reuse the AI-4 fusion layout (synth_data_ai4.py).
  - gas species + onset temperature come from gas_safety_table.h (distilled from
    raw_materials.json): Cr2O3->Cr6+ @>900C oxidizing, carbonate/nitrate/ammonium
    ->CO2/NOx/NH3/HF at their onsets, MoO3->vapor sublimation @>700C.
  - furnace stage / atmosphere / progress mirror furnace_sim.h (FURN_* + atm_code).
  - the 9 classes + their trigger signatures are the measure-first-gated taxonomy
    in taxonomy.json (build_taxonomy.py).

27-D feature layout  (all 0-1 except idx 4, which is 0-6 like AI-4)
-------------------------------------------------------------------
[0:4]   AI-1 softmax probs        [empty, loaded, sintering, done]
[4]     AI-2 anomaly ratio        score / q_hat (0-6, clipped)
[5]     AI-2 temp feat residual   (0-1)
[6]     AI-2 vib  feat residual   (0-1)
[7]     AI-2 gas  feat residual   (0-1)
[8:13]  AI-3 softmax probs        [normal, fast_ramp, undertemp, temp_drift, slow_ramp]
[13]    temp_norm                 furnace T / 1600 C
[14]    ramp_norm                 |dT/dt| / 20 C/min   (profile baseline 5 -> 0.25)
[15]    gas_sev_norm              gas_safety max_sev / 3
[16:23] gas species one-hot       [none, NH3, HF, NOx, CO2, vapor, Cr6+]  (GAS_* enum)
[23]    humidity_norm             SHT30 RH / 100
[24]    mq135_rise                MQ-135 elevation over baseline (0-1, soft)
[25]    atmosphere_oxidizing      1 = air/oxidizing, 0 = reducing/inert
[26]    progress                  0-1 across the run

9 root-cause classes (output)
-----------------------------
  0 NORMAL              7 GRIND_INHOMOGENEITY
  1 RAMP_TOO_FAST       8 SOAK_TEMP_DRIFT
  2 UNDER_TEMPERATURE
  3 OXIDIZE_CR6
  4 DECOMP_GAS_SURGE
  5 VOLATILIZATION_LOSS
  6 MOISTURE_HYDROXYL

Output: X_ai5.npy / y_ai5.npy / X_ai5_test.npy / y_ai5_test.npy / ai5_data_info.json
Run: cd CIMC/model/ai5_rootcause && python synth_data_ai5.py
"""
import json
import numpy as np
from pathlib import Path

SEED = 42
N_FEAT = 27
CLASS_NAMES = ["NORMAL", "RAMP_TOO_FAST", "UNDER_TEMPERATURE", "OXIDIZE_CR6",
               "DECOMP_GAS_SURGE", "VOLATILIZATION_LOSS", "MOISTURE_HYDROXYL",
               "GRIND_INHOMOGENEITY", "SOAK_TEMP_DRIFT"]
N_CLASSES = len(CLASS_NAMES)

# gas species enum -- must match gas_safety_table.h GAS_*
GAS_NONE, GAS_NH3, GAS_HF, GAS_NOX, GAS_CO2, GAS_VAPOR, GAS_CR6 = range(7)
N_GAS = 7

OUT_DIR = Path(__file__).parent


# ── helpers (shared style with synth_data_ai4.py) ───────────────────────────
def _noisy_softmax(peak, n, rng, conc=4.0):
    a = np.ones(n) * 0.5
    a[peak] = conc
    return rng.dirichlet(a).astype(np.float32)


def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _gas_onehot(species, rng, sharp=0.9):
    """soft one-hot: dominant species `sharp`, rest share the remainder."""
    v = rng.uniform(0.0, 0.04, N_GAS).astype(np.float32)
    v[species] += sharp
    v = v / (v.sum() + 1e-9)
    return v


def _row(ai1, ai2_ratio, ai2_resid, ai3, temp_norm, ramp_norm, gas_sev_norm,
         gas_oh, humidity, mq_rise, atm_ox, progress):
    return [*ai1, float(ai2_ratio), *ai2_resid, *ai3, float(temp_norm),
            float(ramp_norm), float(gas_sev_norm), *gas_oh, float(humidity),
            float(mq_rise), float(atm_ox), float(progress)]


# ── per-class samplers (grounded; see module docstring) ─────────────────────
def s_normal(rng, n):
    out = []
    for _ in range(n):
        stage = rng.choice([2, 3])                       # sintering / done
        ai1 = _noisy_softmax(stage, 4, rng, 6.0)
        ratio = rng.uniform(0.0, 0.7)
        resid = _clamp01(rng.normal(0.10, 0.05, 3))
        ai3 = _noisy_softmax(0, 5, rng, 6.0)             # normal curve
        temp = rng.uniform(0.05, 0.95)
        ramp = rng.uniform(0.18, 0.32)                   # ~5 C/min
        # benign CO2 during calcine occasionally, else none
        if rng.random() < 0.25:
            gas, sev = GAS_CO2, 1
        else:
            gas, sev = GAS_NONE, 0
        gas_oh = _gas_onehot(gas, rng)
        humid = rng.uniform(0.20, 0.50)
        mq = rng.uniform(0.0, 0.20)
        atm = 1.0 if rng.random() < 0.85 else 0.0
        prog = rng.uniform(0.0, 1.0)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, sev / 3.0, gas_oh,
                        humid, mq, atm, prog))
    return np.array(out, dtype=np.float32)


def s_ramp_fast(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(rng.choice([1, 2]), 4, rng, 3.0)
        ratio = rng.uniform(1.0, 2.5)
        resid = _clamp01(rng.normal(0.12, 0.06, 3))
        resid[0] = rng.uniform(0.40, 0.80)               # temp residual high
        ai3 = _noisy_softmax(1, 5, rng, 5.0)             # fast_ramp
        temp = rng.uniform(0.20, 0.70)                   # mid-ramp
        ramp = rng.uniform(0.50, 1.00)                   # 10-20 C/min
        gas, sev = (GAS_CO2, 1) if rng.random() < 0.2 else (GAS_NONE, 0)
        gas_oh = _gas_onehot(gas, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, sev / 3.0, gas_oh,
                        rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.25), 1.0,
                        rng.uniform(0.05, 0.50)))
    return np.array(out, dtype=np.float32)


def s_under_temp(rng, n):
    out = []
    for _ in range(n):
        # reaction incomplete -> stuck loaded/sintering, never 'done'
        ai1 = _noisy_softmax(rng.choice([1, 2]), 4, rng, 3.5)
        ratio = rng.uniform(1.0, 2.5)
        resid = _clamp01(rng.normal(0.12, 0.06, 3))
        resid[0] = rng.uniform(0.40, 0.70)
        ai3 = _noisy_softmax(2, 5, rng, 5.0)             # undertemp
        temp = rng.uniform(0.40, 0.78)                   # peaked low (<1250C)
        ramp = rng.uniform(0.12, 0.35)
        gas_oh = _gas_onehot(GAS_NONE, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 0.0, gas_oh,
                        rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.2), 1.0,
                        rng.uniform(0.60, 1.0)))           # late but cold
    return np.array(out, dtype=np.float32)


def s_oxidize_cr6(rng, n):
    out = []
    for _ in range(n):
        # AI-3 curve may look normal -- chemistry fault, not a thermal-curve fault
        ai1 = _noisy_softmax(rng.choice([2, 3]), 4, rng, 3.0)
        ratio = rng.uniform(1.2, 3.0)
        resid = _clamp01(rng.normal(0.15, 0.07, 3))
        resid[2] = rng.uniform(0.50, 0.90)               # gas residual high
        ai3 = _noisy_softmax(rng.choice([0, 4]), 5, rng, 2.5)
        temp = rng.uniform(0.60, 0.95)                   # >900C (norm>0.56)
        ramp = rng.uniform(0.10, 0.35)
        gas_oh = _gas_onehot(GAS_CR6, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 2.0 / 3.0, gas_oh,
                        rng.uniform(0.1, 0.35), rng.uniform(0.50, 1.0),
                        1.0,                               # oxidizing is the trigger
                        rng.uniform(0.40, 0.95)))
    return np.array(out, dtype=np.float32)


def s_decomp_gas(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(rng.choice([1, 2]), 4, rng, 3.0)
        ratio = rng.uniform(1.2, 3.0)
        resid = _clamp01(rng.normal(0.15, 0.07, 3))
        resid[2] = rng.uniform(0.50, 0.90)               # gas residual high
        ai3 = _noisy_softmax(rng.choice([0, 1]), 5, rng, 2.5)  # may co-occur w/ fast ramp
        temp = rng.uniform(0.10, 0.56)                   # onset 170-900C
        ramp = rng.uniform(0.20, 0.70)
        species = rng.choice([GAS_NH3, GAS_HF, GAS_NOX, GAS_CO2])
        sev = {GAS_NH3: 2, GAS_HF: 3, GAS_NOX: 3, GAS_CO2: 1}[species]
        gas_oh = _gas_onehot(species, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, sev / 3.0, gas_oh,
                        rng.uniform(0.2, 0.5), rng.uniform(0.50, 1.0), 1.0,
                        rng.uniform(0.05, 0.45)))          # early ramp / calcine
    return np.array(out, dtype=np.float32)


def s_volatil(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(rng.choice([2, 3]), 4, rng, 3.0)
        ratio = rng.uniform(1.0, 2.0)
        resid = _clamp01(rng.normal(0.15, 0.07, 3))
        resid[2] = rng.uniform(0.40, 0.70)
        ai3 = _noisy_softmax(0, 5, rng, 2.5)
        temp = rng.uniform(0.60, 0.95)                   # high-T sublimation
        ramp = rng.uniform(0.10, 0.35)
        gas_oh = _gas_onehot(GAS_VAPOR, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 2.0 / 3.0, gas_oh,
                        rng.uniform(0.1, 0.35), rng.uniform(0.30, 0.70), 1.0,
                        rng.uniform(0.50, 0.95)))
    return np.array(out, dtype=np.float32)


def s_moisture(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(rng.choice([0, 1]), 4, rng, 3.5)   # empty/loaded
        ratio = rng.uniform(0.6, 1.3)                    # mild
        resid = _clamp01(rng.normal(0.20, 0.08, 3))
        resid[0] = rng.uniform(0.20, 0.45)
        ai3 = _noisy_softmax(0, 5, rng, 4.0)
        temp = rng.uniform(0.02, 0.25)                   # room / early, cold
        ramp = rng.uniform(0.0, 0.30)
        gas_oh = _gas_onehot(GAS_NONE, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 0.0, gas_oh,
                        rng.uniform(0.60, 0.95),           # humidity HIGH
                        rng.uniform(0.0, 0.30),
                        1.0 if rng.random() < 0.6 else 0.0,
                        rng.uniform(0.0, 0.20)))
    return np.array(out, dtype=np.float32)


def s_grind(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(1, 4, rng, 4.0)             # loaded (powder)
        ratio = rng.uniform(1.0, 2.5)
        resid = _clamp01(rng.normal(0.12, 0.06, 3))
        resid[1] = rng.uniform(0.50, 0.90)               # VIB residual high
        ai3 = _noisy_softmax(0, 5, rng, 2.5)             # thermal curve normal
        temp = rng.uniform(0.05, 0.30)                   # near-room (grind stage)
        ramp = rng.uniform(0.0, 0.20)
        gas_oh = _gas_onehot(GAS_NONE, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 0.0, gas_oh,
                        rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.25), 1.0,
                        rng.uniform(0.30, 0.60)))          # grind between calcine & ramp2
    return np.array(out, dtype=np.float32)


def s_soak_drift(rng, n):
    out = []
    for _ in range(n):
        ai1 = _noisy_softmax(2, 4, rng, 3.5)             # sintering
        ratio = rng.uniform(1.0, 2.5)
        resid = _clamp01(rng.normal(0.12, 0.06, 3))
        resid[0] = rng.uniform(0.40, 0.70)               # temp residual elevated
        ai3 = _noisy_softmax(3, 5, rng, 5.0)             # temp_drift
        temp = rng.uniform(0.55, 0.95)                   # at a hold setpoint
        ramp = rng.uniform(0.0, 0.15)                    # holding -> low ramp
        gas_oh = _gas_onehot(GAS_NONE, rng)
        out.append(_row(ai1, ratio, resid, ai3, temp, ramp, 0.0, gas_oh,
                        rng.uniform(0.2, 0.5), rng.uniform(0.0, 0.2), 1.0,
                        rng.uniform(0.40, 0.90)))          # during a soak
    return np.array(out, dtype=np.float32)


SAMPLERS = [s_normal, s_ramp_fast, s_under_temp, s_oxidize_cr6, s_decomp_gas,
            s_volatil, s_moisture, s_grind, s_soak_drift]


def generate(seed=SEED, n_train=2200, n_test=600, out_dir=None):
    out_dir = out_dir or OUT_DIR
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = [], [], [], []
    print(f"Generating AI-5 root-cause data ({n_train} train + {n_test} test per class)...")
    for ci, (name, samp) in enumerate(zip(CLASS_NAMES, SAMPLERS)):
        X = samp(rng, n_train + n_test)
        y = np.full(n_train + n_test, ci, dtype=np.int32)
        Xtr.append(X[:n_train]); ytr.append(y[:n_train])
        Xte.append(X[n_train:]); yte.append(y[n_train:])
        print(f"  {ci} {name:20s}: {n_train}+{n_test}")
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    Xte = np.concatenate(Xte); yte = np.concatenate(yte)
    idx = rng.permutation(len(Xtr)); Xtr, ytr = Xtr[idx], ytr[idx]
    # clip ratio col 4 to 0-6 like AI-4
    Xtr[:, 4] = np.clip(Xtr[:, 4], 0, 6); Xte[:, 4] = np.clip(Xte[:, 4], 0, 6)

    np.save(out_dir / "X_ai5.npy", Xtr); np.save(out_dir / "y_ai5.npy", ytr)
    np.save(out_dir / "X_ai5_test.npy", Xte); np.save(out_dir / "y_ai5_test.npy", yte)
    info = {
        "n_feat": N_FEAT, "n_classes": N_CLASSES, "class_names": CLASS_NAMES,
        "gas_enum": ["none", "NH3", "HF", "NOx", "CO2", "vapor", "Cr6+"],
        "feat_layout": {
            "0:4": "AI-1 softmax [empty,loaded,sintering,done]",
            "4": "AI-2 anomaly ratio (0-6)",
            "5": "AI-2 temp residual", "6": "AI-2 vib residual", "7": "AI-2 gas residual",
            "8:13": "AI-3 softmax [normal,fast_ramp,undertemp,temp_drift,slow_ramp]",
            "13": "temp_norm T/1600C", "14": "ramp_norm |dT/dt|/20", "15": "gas_sev/3",
            "16:23": "gas species one-hot [none,NH3,HF,NOx,CO2,vapor,Cr6+]",
            "23": "humidity RH/100", "24": "mq135 rise (0-1)",
            "25": "atmosphere oxidizing (0/1)", "26": "progress (0-1)",
        },
        "grounding": "gas species + onset from gas_safety_table.h; stages/atmosphere "
                     "from furnace_sim.h; AI-3 anomaly classes from synth_data_ai3.py; "
                     "9-class taxonomy measure-first-gated in taxonomy.json",
    }
    (out_dir / "ai5_data_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"\nSaved: X_ai5 {Xtr.shape}, test {Xte.shape}, ai5_data_info.json")
    return Xtr, ytr, Xte, yte


if __name__ == "__main__":
    generate()
