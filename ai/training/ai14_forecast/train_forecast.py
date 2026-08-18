"""Train and export the AI-14 process-minute furnace forecaster.

R2.1 protocol changes:
* complete furnace runs, not overlapping windows, are the split unit;
* train/dev/calibration/locked_test run IDs are disjoint and deterministic;
* the shipping gate is evaluated on locked runs with paired run-level metrics;
* exported artifacts bind the split protocol, metrics, weights, and golden vectors.

The model remains the existing 24->32->32->12 MLP. Inputs are 24 consecutive
process-minute temperatures and outputs are the next 12 process minutes. This
script mirrors ``furnace_sim.c`` and does not claim physical-furnace accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# Required by torch deterministic CUDA GEMM on CUDA >= 10.2. It must be set
# before torch creates the first CuBLAS handle.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
CIMC = ROOT / "CIMC"
OUT = CIMC / "firmware" / "ai_models_c"
HOST = CIMC / "model" / "host_test"
ARTIFACTS = CIMC / "model" / "ai14_forecast" / "artifacts"

PROTOCOL_VERSION = "AI14_R2.1_PROCESS_MINUTE_V1"
PROTOCOL_HEX = 0x00020100
RNG_SEED = 0xA14C0DE

# These constants must match furnace_sim.c.
T_ROOM, CALCINE_C, SINTER_C = 25.0, 900.0, 1500.0
BASE_RAMP, COOL_RATE = 5.0, 3.0
CALCINE_MIN, SINTER_MIN, GRIND_MIN = 240, 360, 30
TNORM = 1600.0
WIN = 24
HOR = 12
HIDDEN = 32

ANOM_NONE, ANOM_FAST, ANOM_SLOW, ANOM_DRIFT, ANOM_UNDER = range(5)
ANOMALY_NAMES = {
    ANOM_NONE: "normal",
    ANOM_FAST: "fast_ramp",
    ANOM_SLOW: "slow_ramp",
    ANOM_DRIFT: "sensor_drift",
    ANOM_UNDER: "undertemp",
}

# Each split contains every scenario family. A trajectory lineage appears in one
# split only; windows inherit their parent run's split.
SPLIT_COUNTS = {
    "train": 20,
    "dev": 5,
    "calibration": 5,
    "locked_test": 10,
}


class _LCG:
    """Bit-exact integer implementation of furnace_sim.c's LCG."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF

    def _u(self) -> float:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return ((self.s >> 8) & 0xFFFFFF) / float(0x1000000) - 0.5

    def noise3(self) -> float:
        return (self._u() + self._u() + self._u() + self._u()) * 3.0


def ramp_rate(anomaly: int) -> float:
    if anomaly == ANOM_FAST:
        return BASE_RAMP * 4.0
    if anomaly == ANOM_SLOW:
        return max(BASE_RAMP * 0.25, 0.5)
    return BASE_RAMP


def stage_bounds(anomaly: int) -> list[int]:
    rate = ramp_rate(anomaly)
    ramp1 = int((CALCINE_C - T_ROOM) / max(rate, 1.0))
    ramp2 = int((SINTER_C - T_ROOM) / max(rate, 1.0))
    cool = int((SINTER_C - T_ROOM) / COOL_RATE)
    bounds = [0] * 6
    bounds[0] = ramp1
    bounds[1] = bounds[0] + CALCINE_MIN
    bounds[2] = bounds[1] + GRIND_MIN
    bounds[3] = bounds[2] + ramp2
    bounds[4] = bounds[3] + SINTER_MIN
    bounds[5] = bounds[4] + cool
    return bounds


def target_temp(stage: int, step: int, stage_len: int) -> float:
    if stage == 0:
        frac = step / (stage_len - 1) if stage_len > 1 else 1.0
        return T_ROOM + (CALCINE_C - T_ROOM) * frac
    if stage == 1:
        return CALCINE_C
    if stage == 2:
        return T_ROOM
    if stage == 3:
        frac = step / (stage_len - 1) if stage_len > 1 else 1.0
        return T_ROOM + (SINTER_C - T_ROOM) * frac
    if stage == 4:
        return SINTER_C
    return max(SINTER_C - COOL_RATE * step, T_ROOM)


def generate_trajectory(anomaly: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    bounds = stage_bounds(anomaly)
    lcg = _LCG(seed)
    temps: list[float] = []
    knees: list[int] = []
    for minute in range(bounds[5]):
        previous = 0
        stage, step, stage_len = 5, 0, 1
        for candidate in range(6):
            if minute < bounds[candidate]:
                stage = candidate
                step = minute - previous
                stage_len = bounds[candidate] - previous
                break
            previous = bounds[candidate]
        value = target_temp(stage, step, stage_len) + lcg.noise3()
        if anomaly == ANOM_DRIFT:
            value += 64.0
        elif anomaly == ANOM_UNDER and stage == 4:
            value -= 100.0
        temps.append(value)
        knees.append(int(any(abs(minute - boundary) <= 6 for boundary in bounds[:5])))
    return np.asarray(temps, np.float32), np.asarray(knees, np.uint8)


@dataclass(frozen=True)
class Run:
    run_id: str
    split: str
    anomaly: int
    seed: int
    temps: np.ndarray
    knees: np.ndarray


def build_runs() -> list[Run]:
    runs: list[Run] = []
    for anomaly in sorted(ANOMALY_NAMES):
        ordinal = 0
        for split, count in SPLIT_COUNTS.items():
            for _ in range(count):
                ordinal += 1
                seed = (0xA1400000 + anomaly * 0x10000 + ordinal) & 0xFFFFFFFF
                run_id = f"a{anomaly}_{ANOMALY_NAMES[anomaly]}_{ordinal:03d}_{seed:08x}"
                temps, knees = generate_trajectory(anomaly, seed)
                runs.append(Run(run_id, split, anomaly, seed, temps, knees))
    return runs


def windows_for(runs: Iterable[Run]) -> dict[str, np.ndarray]:
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    knee_rows: list[int] = []
    run_rows: list[str] = []
    minute_rows: list[int] = []
    anomaly_rows: list[int] = []
    for run in runs:
        normalised = run.temps / TNORM
        for start in range(len(normalised) - WIN - HOR + 1):
            x_rows.append(normalised[start : start + WIN])
            y_rows.append(normalised[start + WIN : start + WIN + HOR])
            knee_rows.append(int(run.knees[start + WIN : start + WIN + HOR].any()))
            run_rows.append(run.run_id)
            minute_rows.append(start + WIN)
            anomaly_rows.append(run.anomaly)
    return {
        "x": np.asarray(x_rows, np.float32),
        "y": np.asarray(y_rows, np.float32),
        "knee": np.asarray(knee_rows, np.uint8),
        "run_id": np.asarray(run_rows),
        "minute": np.asarray(minute_rows, np.int32),
        "anomaly": np.asarray(anomaly_rows, np.uint8),
    }


class ForecastMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WIN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HOR),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def baselines(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    last = x[:, -1:]
    slope = x[:, -1:] - x[:, -2:-1]
    steps = np.arange(1, HOR + 1, dtype=np.float32)[None, :]
    return np.repeat(last, HOR, axis=1), last + slope * steps


def mae_c(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction - truth)) * TNORM)


def predict(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            batch = torch.from_numpy(x[start : start + 8192]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def run_metrics(data: dict[str, np.ndarray], prediction: np.ndarray) -> dict:
    persistence, linear = baselines(data["x"])
    knee = data["knee"].astype(bool)
    rows = []
    improvements = []
    for run_id in sorted(set(data["run_id"].tolist())):
        mask = data["run_id"] == run_id
        knee_mask = mask & knee
        anomaly = int(data["anomaly"][np.flatnonzero(mask)[0]])
        row = {
            "run_id": run_id,
            "anomaly": ANOMALY_NAMES[anomaly],
            "windows": int(mask.sum()),
            "knee_windows": int(knee_mask.sum()),
            "model_mae_c": mae_c(prediction[mask], data["y"][mask]),
            "persistence_mae_c": mae_c(persistence[mask], data["y"][mask]),
            "linear_mae_c": mae_c(linear[mask], data["y"][mask]),
            "model_knee_mae_c": mae_c(prediction[knee_mask], data["y"][knee_mask]),
            "linear_knee_mae_c": mae_c(linear[knee_mask], data["y"][knee_mask]),
            "run_max_abs_error_c": float(
                np.max(np.abs(prediction[mask] - data["y"][mask])) * TNORM
            ),
        }
        row["knee_improvement_c"] = row["linear_knee_mae_c"] - row["model_knee_mae_c"]
        improvements.append(row["knee_improvement_c"])
        rows.append(row)
    return {
        "windows": int(len(data["x"])),
        "knee_windows": int(knee.sum()),
        "overall_mae_c": {
            "model": mae_c(prediction, data["y"]),
            "persistence": mae_c(persistence, data["y"]),
            "linear": mae_c(linear, data["y"]),
        },
        "knee_mae_c": {
            "model": mae_c(prediction[knee], data["y"][knee]),
            "persistence": mae_c(persistence[knee], data["y"][knee]),
            "linear": mae_c(linear[knee], data["y"][knee]),
        },
        "run_rows": rows,
        "run_knee_improvements_c": improvements,
    }


def bootstrap_lower(values: np.ndarray, seed: int, samples: int = 20000) -> float:
    rng = np.random.default_rng(seed)
    means = np.empty(samples, np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = rng.integers(0, len(values), size=(count, len(values)))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.05, method="lower"))


def exact_sign_p(successes: int, n: int) -> float:
    return float(sum(math.comb(n, k) for k in range(successes, n + 1)) / (2**n))


def finite_sample_quantile(values: np.ndarray, coverage: float) -> float:
    rank = min(len(values), math.ceil((len(values) + 1) * coverage))
    return float(np.sort(values)[rank - 1])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def c_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, np.float32).reshape(-1)
    body = ", ".join(f"{value:.8e}f" for value in flat)
    return f"static const float {name}[{flat.size}] = {{ {body} }};\n"


def split_manifest(runs: list[Run]) -> dict:
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "split_unit": "complete_furnace_run",
        "window_minutes": WIN,
        "horizon_minutes": HOR,
        "normalisation_c": TNORM,
        "runs": [
            {
                "run_id": run.run_id,
                "split": run.split,
                "anomaly": ANOMALY_NAMES[run.anomaly],
                "seed": run.seed,
                "minutes": int(len(run.temps)),
            }
            for run in runs
        ],
    }
    manifest["protocol_root"] = canonical_sha256(manifest)
    return manifest


def export_headers(
    model: ForecastMLP,
    locked: dict[str, np.ndarray],
    locked_prediction: np.ndarray,
    protocol_root: str,
) -> tuple[Path, Path]:
    layers = model.net
    weights = [
        layers[0].weight.detach().cpu().numpy(),
        layers[0].bias.detach().cpu().numpy(),
        layers[2].weight.detach().cpu().numpy(),
        layers[2].bias.detach().cpu().numpy(),
        layers[4].weight.detach().cpu().numpy(),
        layers[4].bias.detach().cpu().numpy(),
    ]
    names = ["ai14_w0", "ai14_b0", "ai14_w1", "ai14_b1", "ai14_w2", "ai14_b2"]
    weight_header = OUT / "ai14_forecast_weights.h"
    with weight_header.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("/* AUTO-GENERATED AI-14 R2.1 process-minute forecaster.\n")
        handle.write(" * Simulation-trained decision support; board validation pending.\n")
        handle.write(f" * Split protocol root: {protocol_root}\n */\n")
        handle.write("#ifndef AI14_FORECAST_WEIGHTS_H\n#define AI14_FORECAST_WEIGHTS_H\n\n")
        handle.write(f"#define AI14_PROTOCOL_VERSION 0x{PROTOCOL_HEX:08X}UL\n")
        handle.write("#define AI14_CADENCE_PROCESS_MINUTES 1U\n")
        handle.write(f"#define AI14_WIN {WIN}\n#define AI14_HID {HIDDEN}\n#define AI14_HOR {HOR}\n")
        handle.write(f"#define AI14_TNORM {TNORM:.1f}f\n\n")
        for name, value in zip(names, weights):
            handle.write(c_array(name, value))
        handle.write("\n#endif\n")

    # One knee and one non-knee vector for each scenario family.
    picks: list[int] = []
    for anomaly in sorted(ANOMALY_NAMES):
        mask = locked["anomaly"] == anomaly
        for want_knee in (0, 1):
            candidates = np.flatnonzero(mask & (locked["knee"] == want_knee))
            picks.append(int(candidates[len(candidates) // 2]))
    golden_header = OUT / "ai14_forecast_golden.h"
    with golden_header.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("/* AUTO-GENERATED AI-14 R2.1 host golden vectors. */\n")
        handle.write("#ifndef AI14_FORECAST_GOLDEN_H\n#define AI14_FORECAST_GOLDEN_H\n\n")
        handle.write(f"#define AI14_NG {len(picks)}\n\n")
        handle.write(c_array("ai14_g_in", locked["x"][picks]))
        handle.write(c_array("ai14_g_out", locked_prediction[picks]))
        handle.write("\n#endif\n")
    shutil.copy2(golden_header, HOST / golden_header.name)
    return weight_header, golden_header


def main() -> None:
    torch.manual_seed(RNG_SEED)
    np.random.seed(RNG_SEED & 0xFFFFFFFF)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RNG_SEED)
    torch.use_deterministic_algorithms(True)

    OUT.mkdir(parents=True, exist_ok=True)
    HOST.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    runs = build_runs()
    manifest = split_manifest(runs)
    split_path = ARTIFACTS / "ai14_r21_split_manifest.json"
    split_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    split_ids = {
        split: {run.run_id for run in runs if run.split == split} for split in SPLIT_COUNTS
    }
    overlap = {
        f"{left}__{right}": sorted(split_ids[left] & split_ids[right])
        for i, left in enumerate(SPLIT_COUNTS)
        for right in list(SPLIT_COUNTS)[i + 1 :]
    }
    if any(overlap.values()):
        raise RuntimeError(f"run leakage detected: {overlap}")

    datasets = {
        split: windows_for(run for run in runs if run.split == split) for split in SPLIT_COUNTS
    }
    for split, data in datasets.items():
        print(
            f"[data] {split:11s}: {len(split_ids[split]):3d} runs, "
            f"{len(data['x']):7d} windows, knees={int(data['knee'].sum()):5d}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={device} protocol={PROTOCOL_VERSION}")
    model = ForecastMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    objective = nn.SmoothL1Loss()
    train_tensor = TensorDataset(
        torch.from_numpy(datasets["train"]["x"]),
        torch.from_numpy(datasets["train"]["y"]),
    )
    loader_generator = torch.Generator().manual_seed(RNG_SEED)
    loader = DataLoader(
        train_tensor,
        batch_size=4096,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
        drop_last=False,
    )

    best_state = None
    best_dev = float("inf")
    stale = 0
    max_epochs = 160
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x_batch, y_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x_batch.to(device)), y_batch.to(device))
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            dev_prediction = predict(model, datasets["dev"]["x"], device)
            dev_mae = mae_c(dev_prediction, datasets["dev"]["y"])
            print(f"[train] epoch={epoch:03d} dev_mae_c={dev_mae:.4f}")
            if dev_mae < best_dev - 1e-4:
                best_dev = dev_mae
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 10:
                print(f"[train] early stop at epoch={epoch}")
                break
    if best_state is None:
        raise RuntimeError("training did not produce a dev checkpoint")
    model.load_state_dict(best_state)
    model.to(device)

    predictions = {split: predict(model, data["x"], device) for split, data in datasets.items()}
    metrics = {
        split: run_metrics(datasets[split], predictions[split])
        for split in ("dev", "calibration", "locked_test")
    }

    locked_improvement = np.asarray(
        metrics["locked_test"]["run_knee_improvements_c"], np.float64
    )
    improved = int((locked_improvement > 0.0).sum())
    lower95 = bootstrap_lower(locked_improvement, RNG_SEED ^ 0xB0057)
    sign_p = exact_sign_p(improved, len(locked_improvement))

    calibration_run_max = np.asarray(
        [row["run_max_abs_error_c"] for row in metrics["calibration"]["run_rows"]],
        np.float64,
    )
    locked_run_max = np.asarray(
        [row["run_max_abs_error_c"] for row in metrics["locked_test"]["run_rows"]],
        np.float64,
    )
    q90_run_max = finite_sample_quantile(calibration_run_max, 0.90)
    diagnostic_coverage = float(np.mean(locked_run_max <= q90_run_max))

    gate = {
        "run_split_disjoint": not any(overlap.values()),
        "locked_run_count": len(locked_improvement),
        "locked_mean_knee_improvement_c": float(locked_improvement.mean()),
        "locked_bootstrap_one_sided_95_lower_improvement_c": lower95,
        "locked_runs_improved": improved,
        "locked_exact_sign_test_one_sided_p": sign_p,
        "model_beats_linear_at_knees": bool(lower95 > 0.0 and sign_p < 0.05),
        "model_beats_persistence_overall": bool(
            metrics["locked_test"]["overall_mae_c"]["model"]
            < metrics["locked_test"]["overall_mae_c"]["persistence"]
        ),
    }
    gate["all_pass"] = all(
        gate[key]
        for key in (
            "run_split_disjoint",
            "model_beats_linear_at_knees",
            "model_beats_persistence_overall",
        )
    )

    print("[locked] overall MAE C:", metrics["locked_test"]["overall_mae_c"])
    print("[locked] knee MAE C:", metrics["locked_test"]["knee_mae_c"])
    print(
        "[locked] paired knee improvement "
        f"mean={locked_improvement.mean():.3f}C lower95={lower95:.3f}C "
        f"improved={improved}/{len(locked_improvement)} p={sign_p:.6g}"
    )
    if not gate["all_pass"]:
        print(f"[gate] FAILED: {gate}")
        sys.exit(1)

    weight_header, golden_header = export_headers(
        model,
        datasets["locked_test"],
        predictions["locked_test"],
        manifest["protocol_root"],
    )
    report = {
        "status": "HOST_TRAINING_VERIFIED_BOARD_FLASH_PENDING",
        "protocol": PROTOCOL_VERSION,
        "protocol_root": manifest["protocol_root"],
        "split_manifest_sha256": sha256_file(split_path),
        "split_counts_per_anomaly": SPLIT_COUNTS,
        "overlap_audit": overlap,
        "best_dev_mae_c": best_dev,
        "metrics": metrics,
        "paired_locked_gate": gate,
        "run_level_diagnostic_conformal": {
            "claim": "diagnostic_only_not_r3_formula_coverage",
            "score": "per_run_max_absolute_forecast_error_c",
            "target": 0.90,
            "n_calibration_runs": int(len(calibration_run_max)),
            "n_locked_runs": int(len(locked_run_max)),
            "q90_c": q90_run_max,
            "locked_coverage": diagnostic_coverage,
        },
        "artifacts": {
            "weights": str(weight_header.relative_to(CIMC)).replace("\\", "/"),
            "weights_sha256": sha256_file(weight_header),
            "golden": str(golden_header.relative_to(CIMC)).replace("\\", "/"),
            "golden_sha256": sha256_file(golden_header),
        },
        "boundaries": {
            "source": "furnace_sim.c_matched_synthetic_trajectories",
            "physical_furnace_accuracy_claimed": False,
            "board_validation_complete": False,
            "action_authority": 0,
        },
    }
    report_path = ARTIFACTS / "ai14_r21_evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[gate] PASS protocol_root={manifest['protocol_root']}")
    print(f"[export] {weight_header}")
    print(f"[export] {golden_header}")
    print(f"[evidence] {report_path}")


if __name__ == "__main__":
    main()
