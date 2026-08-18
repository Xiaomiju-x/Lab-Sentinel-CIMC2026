"""
AI-2 Sintering AE — Synthetic Data Generator
Reads sintering_profiles.json (13 host families), simulates full sintering
time series (calcine → grind → sinter → cool) and emits a 32-D feature
vector every minute.

32-D layout
-----------
Group A – Temperature (8)
  [0]  temp_current     current furnace temp (°C, normalised /1600)
  [1]  temp_target      profile target at this minute (/1600)
  [2]  temp_deviation   (current - target) / 100
  [3]  temp_gradient    dT/dt last minute (°C/min, /20)
  [4]  temp_rms_60s     RMS of last 60 samples (/1600)
  [5]  temp_std_60s     std of last 60 samples (/50)
  [6]  hold_min_cum     cumulative minutes within ±10 °C of hold target (/600)
  [7]  stage_id         0=ramp1 1=calcine_hold 2=grind 3=ramp2 4=sinter_hold 5=cool (/5)

Group B – Gas / atmosphere (6)
  [8]  mq135_raw        simulated MQ-135 ADC (0-1 normalised)
  [9]  mq135_avg5       5-min moving average
  [10] atmosphere_code  0=air 1=N2 2=H2_N2 3=Ar (/3)
  [11] mq135_std        std of last 5 min (/0.05)
  [12] mq135_trend      linear slope of last 5 min (/0.01)
  [13] atm_match        1=atmosphere matches profile expectation, 0=anomaly

Group C – Room environment (4)
  [14] room_temp        (°C - 25) / 5
  [15] room_humidity    (% - 50) / 10
  [16] room_temp_grad   °C/min / 0.2
  [17] room_hum_grad    %/min / 0.5

Group D – Vibration ADXL345 (8)
  [18] vib_x_rms        /0.1g
  [19] vib_y_rms        /0.1g
  [20] vib_z_rms        /0.1g
  [21] vib_x_cen        spectral centroid x /500 Hz
  [22] vib_y_cen        spectral centroid y /500 Hz
  [23] vib_z_cen        spectral centroid z /500 Hz
  [24] vib_energy       total energy /0.3
  [25] grinding_flag    1 during grinding step, else 0

Group E – Vision AI-1 proxy logits (4)
  [26] vis_cls0_empty
  [27] vis_cls1_loaded
  [28] vis_cls2_sintering
  [29] vis_cls3_done
  (stage-conditioned probabilities with noise, stand-in until AI-1 is trained)

Group F – Progress (2)
  [30] progress         0→1 over total process duration
  [31] cum_dev          cumulative |temp_deviation| / 1000
"""

import json
import math
import random
import numpy as np
from pathlib import Path

PROFILES_PATH = Path(__file__).parents[3] / "predict_engine" / "sintering_profiles.json"
SEED = 42

ATM_CODE = {"air": 0, "N2": 1, "H2-N2": 2, "Ar": 3,
            "95%N2+5%H2": 2, "H2/Ar": 2, "1%H2/Ar": 2,
            "O2": 0, "N2/H2": 2}


def _atm_code(atm_str: str) -> int:
    for k, v in ATM_CODE.items():
        if k.lower() in atm_str.lower():
            return v
    return 0


def _mq135_baseline(atm_code: int) -> float:
    """Baseline MQ-135 ADC for each atmosphere (0-1)."""
    return {0: 0.55, 1: 0.25, 2: 0.15, 3: 0.20}.get(atm_code, 0.55)


def simulate_host(profile: dict, rng: np.random.Generator,
                  noise_scale: float = 1.0) -> np.ndarray:
    """
    Simulate one full sintering run for a host profile.
    Returns array of shape (T, 32) where T = total minutes.
    """
    calc = profile["calcine"]
    sint = profile["sinter"]

    t_room = 25.0
    ramp1_min = int((calc["temp_C"] - t_room) / max(calc["ramp_C_per_min"], 1))
    hold1_min = int(calc["hours"] * 60)
    grind_min = 30
    ramp2_min = int((sint["temp_C"] - t_room) / max(sint["ramp_C_per_min"], 1))
    hold2_min = int(sint["hours"] * 60)
    cool_min = int((sint["temp_C"] - t_room) / 3)   # 3 °C/min cooling

    stage_lens = [ramp1_min, hold1_min, grind_min, ramp2_min, hold2_min, cool_min]
    total_min = sum(stage_lens)

    atm_calc = _atm_code(calc["atmosphere"])
    atm_sint = _atm_code(sint["atmosphere"])

    rows = []
    temp_hist = [t_room] * 60
    mq_hist = [_mq135_baseline(atm_calc)] * 5
    hold_cum = 0.0
    cum_dev = 0.0
    prev_room_temp = t_room + rng.normal(0, 0.5)
    prev_room_hum = 50.0 + rng.normal(0, 3)

    t_abs = 0
    for stage_id, stage_len in enumerate(stage_lens):
        atm_code = atm_calc if stage_id < 3 else atm_sint
        hold_target = calc["temp_C"] if stage_id == 1 else sint["temp_C"] if stage_id == 4 else 0.0

        for step in range(stage_len):
            # --- target profile temperature ---
            if stage_id == 0:   # ramp1
                t_target = t_room + (calc["temp_C"] - t_room) * (step / max(stage_len - 1, 1))
            elif stage_id == 1: # calcine hold
                t_target = float(calc["temp_C"])
            elif stage_id == 2: # grind (room temp)
                t_target = t_room
            elif stage_id == 3: # ramp2
                t_target = t_room + (sint["temp_C"] - t_room) * (step / max(stage_len - 1, 1))
            elif stage_id == 4: # sinter hold
                t_target = float(sint["temp_C"])
            else:               # cooling
                t_target = sint["temp_C"] - 3.0 * step
                t_target = max(t_target, t_room)

            # current temp with noise
            t_current = t_target + rng.normal(0, 3.0 * noise_scale)
            temp_hist = temp_hist[1:] + [t_current]
            temp_dev = t_current - t_target

            # hold counter
            if stage_id in (1, 4) and abs(temp_dev) <= 10:
                hold_cum += 1
            cum_dev += abs(temp_dev) / 100.0

            # gradient (last minute)
            temp_grad = (temp_hist[-1] - temp_hist[-2]) if len(temp_hist) >= 2 else 0.0

            # Group A
            A = [
                t_current / 1600.0,
                t_target / 1600.0,
                temp_dev / 100.0,
                temp_grad / 20.0,
                float(np.sqrt(np.mean(np.array(temp_hist[-60:]) ** 2))) / 1600.0,
                float(np.std(temp_hist[-60:])) / 50.0,
                hold_cum / 600.0,
                stage_id / 5.0,
            ]

            # Group B – MQ-135
            mq_base = _mq135_baseline(atm_code)
            mq_raw = mq_base + rng.normal(0, 0.02 * noise_scale)
            mq_raw = float(np.clip(mq_raw, 0, 1))
            mq_hist = mq_hist[1:] + [mq_raw]
            mq_avg5 = float(np.mean(mq_hist))
            mq_std = float(np.std(mq_hist)) / 0.05
            mq_slope = float(np.polyfit(range(len(mq_hist)), mq_hist, 1)[0]) / 0.01
            B = [mq_raw, mq_avg5, atm_code / 3.0, mq_std, mq_slope, 1.0]

            # Group C – room environment
            room_temp = prev_room_temp + rng.normal(0, 0.05 * noise_scale)
            room_hum = prev_room_hum + rng.normal(0, 0.2 * noise_scale)
            C = [
                (room_temp - 25.0) / 5.0,
                (room_hum - 50.0) / 10.0,
                (room_temp - prev_room_temp) / 0.2,
                (room_hum - prev_room_hum) / 0.5,
            ]
            prev_room_temp = room_temp
            prev_room_hum = room_hum

            # Group D – vibration
            is_grinding = float(stage_id == 2)
            vib_base = 0.05 + 0.3 * is_grinding
            vib_xyz = rng.normal(vib_base, 0.01 * noise_scale, 3)
            cen_base = 50.0 + 150.0 * is_grinding
            cen_xyz = rng.normal(cen_base, 5.0 * noise_scale, 3)
            energy = float(np.sum(vib_xyz ** 2)) / 0.3
            D = list(vib_xyz / 0.1) + list(cen_xyz / 500.0) + [energy, is_grinding]

            # Group E – vision proxy logits (softmax-like)
            stage_to_logit = {
                0: [0.1, 0.7, 0.1, 0.1],  # ramp1: crucible loaded
                1: [0.1, 0.6, 0.2, 0.1],  # calcine hold: loaded
                2: [0.6, 0.3, 0.05, 0.05], # grind: mostly empty view
                3: [0.1, 0.2, 0.6, 0.1],  # ramp2: sintering
                4: [0.05, 0.1, 0.8, 0.05], # sinter hold: sintering
                5: [0.1, 0.1, 0.1, 0.7],  # cooling: done
            }
            vis_raw = np.array(stage_to_logit[stage_id]) + rng.normal(0, 0.05 * noise_scale, 4)
            vis_raw = np.clip(vis_raw, 0, 1)
            E = list(vis_raw / vis_raw.sum())

            # Group F – progress
            F = [t_abs / max(total_min - 1, 1), min(cum_dev / 1000.0, 1.0)]

            row = A + B + C + D + E + F
            assert len(row) == 32, f"Feature dim mismatch: {len(row)}"
            rows.append(row)
            t_abs += 1

    return np.array(rows, dtype=np.float32)


def make_anomaly(profile: dict, rng: np.random.Generator, anomaly_type: str) -> np.ndarray:
    """
    Generate anomalous run for evaluation (not used in AE training).
    anomaly_type: 'fast_ramp' | 'wrong_atm' | 'temp_drift' | 'undertemp' | 'vib_spike'
    """
    import copy
    p = copy.deepcopy(profile)

    if anomaly_type == "fast_ramp":
        p["calcine"]["ramp_C_per_min"] *= 4
        p["sinter"]["ramp_C_per_min"] *= 4
    elif anomaly_type == "wrong_atm":
        p["sinter"]["atmosphere"] = "air"  # force air for a reducing host
    elif anomaly_type == "temp_drift":
        # inject +60°C drift after sinter ramp
        arr = simulate_host(p, rng, noise_scale=1.0)
        arr[:, 2] += 0.6   # inflate temp_deviation feature
        return arr
    elif anomaly_type == "undertemp":
        p["sinter"]["temp_C"] = int(p["sinter"]["temp_C"] * 0.92)
    elif anomaly_type == "vib_spike":
        arr = simulate_host(p, rng, noise_scale=1.0)
        n = len(arr)
        spike_start = rng.integers(n // 4, 3 * n // 4)
        arr[spike_start:spike_start + 10, 18:26] *= 8  # vib group spike
        return arr

    return simulate_host(p, rng, noise_scale=1.0)


def generate(seed: int = SEED):
    rng = np.random.default_rng(seed)
    profiles = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))

    normal_runs = []
    anomaly_runs = []

    host_keys = [k for k in profiles if not k.startswith("_")]
    print(f"Generating normal runs for {len(host_keys)} hosts ...")

    for host in host_keys:
        p = profiles[host]
        # 5 normal runs per host with slight noise variation
        for noise in [0.8, 1.0, 1.0, 1.0, 1.2]:
            arr = simulate_host(p, rng, noise_scale=noise)
            normal_runs.append(arr)
            print(f"  {host:20s}  noise={noise:.1f}  T={len(arr):4d} min")

        # 1 anomaly run per type per host (eval only)
        for atype in ["fast_ramp", "wrong_atm", "undertemp", "vib_spike"]:
            try:
                arr = make_anomaly(p, rng, atype)
                anomaly_runs.append((host, atype, arr))
            except Exception as e:
                print(f"  [warn] {host} {atype}: {e}")

    X_normal = np.concatenate(normal_runs, axis=0)
    print(f"\nNormal dataset: {X_normal.shape}  (samples x 32)")
    print(f"Anomaly runs  : {len(anomaly_runs)}")

    out_dir = Path(__file__).parent
    np.save(out_dir / "X_normal.npy", X_normal)

    # save anomaly as dict of arrays for eval
    anomaly_dict = {f"{h}_{t}": a for h, t, a in anomaly_runs}
    np.savez(out_dir / "X_anomaly.npz", **anomaly_dict)

    print(f"Saved → {out_dir}/X_normal.npy  (shape {X_normal.shape})")
    print(f"Saved → {out_dir}/X_anomaly.npz  ({len(anomaly_dict)} runs)")
    return X_normal


if __name__ == "__main__":
    generate()
