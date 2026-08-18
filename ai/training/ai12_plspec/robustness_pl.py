"""
robustness_pl.py — AI-12 PL dopant-classifier robustness sweep (CIMC Lab-Sentinel)
==================================================================================
Population robustness of the on-chip AI-12 PL dopant classifier on the FULL 281
REAL Fluoromax emission spectra (not a demo subset), under the same three spectral
perturbations the firmware injects live on the Robust HMI page:
  - noise     : additive Gaussian, sigma in units of the [0,1]-normalised spectrum
  - occlusion : zero a central band of `w` bins (a dead/clipped detector region)
  - baseline  : additive linear tilt (drifting baseline / lamp ageing)

Metric = top-1 dopant accuracy (Cr / Ni / Cr+Ni) retained vs the clean spectrum.
Deterministic RNG -> reproducible numbers for docs/robustness_matrix.md and the HMI
reference panel. Reuses the exact train_plspec.py loader + model (no re-train).

Run:  cd CIMC/model/ai12_plspec && python robustness_pl.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train_plspec import load_dataset, SpecMLP, GRID_N   # noqa: E402

HERE = Path(__file__).parent


def load_model():
    ck = torch.load(HERE / "ai12_plspec.pt", map_location="cpu", weights_only=True)
    m = SpecMLP()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m


def predict(m, X):
    with torch.no_grad():
        return m(torch.tensor(X.astype(np.float32))).argmax(1).numpy()


def perturb(X, mode, amt, rng):
    Y = X.copy()
    n = X.shape[1]
    if mode == "noise":
        Y = np.clip(Y + amt * rng.standard_normal(Y.shape).astype(np.float32), 0.0, 1.0)
    elif mode == "occ":
        s = max(0, n // 2 - int(amt) // 2)
        Y[:, s:s + int(amt)] = 0.0
    elif mode == "base":
        tilt = (amt * (np.linspace(0.0, 1.0, n).astype(np.float32) - 0.5))
        Y = np.clip(Y + tilt[None, :], 0.0, 1.0)
    return Y


def main():
    X, y, _ = load_dataset()
    m = load_model()
    clean = predict(m, X)
    base_acc = float((clean == y).mean())
    print(f"[AI-12] clean top-1 acc = {base_acc * 100:.1f}%  (n={len(X)}, GRID_N={GRID_N})\n")

    rng = np.random.default_rng(1234)
    print("noise (Gaussian sigma):")
    for amt in [0.0, 0.03, 0.06, 0.10, 0.15]:
        p = predict(m, perturb(X, "noise", amt, rng))
        print(f"  sigma {amt:.2f} -> {(p == y).mean() * 100:5.1f}%")
    print("occlusion (central band, bins of 64):")
    for amt in [0, 5, 10, 15, 20]:
        p = predict(m, perturb(X, "occ", amt, rng))
        print(f"  {amt:2d} bins -> {(p == y).mean() * 100:5.1f}%")
    print("baseline drift (linear tilt amplitude):")
    for amt in [0.0, 0.10, 0.20, 0.30]:
        p = predict(m, perturb(X, "base", amt, rng))
        print(f"  {amt:.2f}    -> {(p == y).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
