"""
probe_pl_models.py — measure-first probes for two CANDIDATE PL models (CIMC)
===========================================================================
Before writing any C engine, prove (or disprove) that each candidate is learnable
AND non-redundant. This is the same discipline that killed the AI-13 Dq/B regressor
(it was algebra from AI-6) — see pattern_measure_first_model_rejection.

Reuses AI-12's exact 64-pt spectrum pipeline (spectrum_numerical/, labels.csv).

  Probe A — PL host-ID (2-class): only 2 hosts exist in the lab set
            (Y3ZnGa3GeO12 x177, NaY2Ga2InGe2O12 x104). Is the emission shape
            host-separable beyond the 63% majority baseline? 5-fold CV.

  Probe B — PL lambda_em regressor: labels.csv already has lambda_max. The trivial
            baseline is argmax(spectrum)->grid wavelength. A learned regressor is only
            worth shipping if it BEATS argmax — e.g. sub-bin interpolation, or
            robustness when the spectrum is noised. Otherwise it's redundant -> REJECT.

Run:  cd CIMC/model && python probe_pl_models.py
"""
import sys
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "ai12_plspec"))
from train_plspec import load_dataset, GRID, GRID_LO, GRID_HI, GRID_N  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SN = ROOT / "spectrum_numerical"


def load_aux():
    """Return host_id (0/1), lambda_max aligned to load_dataset()'s kept order."""
    X, y, paths = load_dataset()
    rows = {r["path"].replace("\\", "/"): r
            for r in csv.DictReader(open(SN / "data" / "labels.csv", encoding="utf-8"))}
    # normalise label keys to posix and index by basename-path suffix
    rows = {k.replace("\\", "/"): v for k, v in rows.items()}
    hosts, lam = [], []
    host_names = {}
    for p in paths:
        pp = p.replace("\\", "/")
        r = next((v for k, v in rows.items() if pp.endswith(k)), None)
        if r is None:
            raise RuntimeError(f"no label row for {p}")
        h = r["host"]
        host_names.setdefault(h, len(host_names))
        hosts.append(host_names[h])
        lam.append(float(r["lambda_max"]))
    return (np.asarray(X, np.float32), np.asarray(hosts, np.int64),
            np.asarray(lam, np.float32), host_names)


class TinyMLP(nn.Module):
    def __init__(self, out, hid=24):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(GRID_N, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, out))

    def forward(self, x):
        return self.net(x)


def kfold(n, k=5, seed=0):
    idx = np.random.RandomState(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def probe_host(X, h):
    print("\n=== Probe A: PL host-ID (2-class) ===")
    n = len(X)
    maj = max(np.bincount(h)) / n
    folds = kfold(n)
    accs = []
    for f in range(5):
        te = folds[f]; tr = np.setdiff1d(np.arange(n), te)
        m = TinyMLP(2)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-3)
        xt, yt = torch.tensor(X[tr]), torch.tensor(h[tr])
        for _ in range(300):
            opt.zero_grad(); loss = nn.functional.cross_entropy(m(xt), yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            pred = m(torch.tensor(X[te])).argmax(1).numpy()
        accs.append((pred == h[te]).mean())
    acc = float(np.mean(accs))
    print(f"  majority baseline {maj*100:.1f}%   5-fold CV {acc*100:.1f}% (+/-{np.std(accs)*100:.1f})")
    edge = acc - maj
    verdict = "SHIP" if edge > 0.10 else ("WEAK-PRIOR" if edge > 0.03 else "REJECT")
    print(f"  edge over baseline = {edge*100:+.1f} pts  -> {verdict}")
    return verdict, acc, maj


def probe_lambda(X, lam):
    print("\n=== Probe B: PL lambda_em regressor vs argmax ===")
    grid = GRID.astype(np.float32)
    # trivial baseline: argmax bin -> wavelength
    argmax_lam = grid[np.argmax(X, axis=1)]
    mae_argmax = float(np.mean(np.abs(argmax_lam - lam)))
    bin_w = (GRID_HI - GRID_LO) / (GRID_N - 1)
    print(f"  grid bin width {bin_w:.1f} nm (argmax quantisation ~+/-{bin_w/2:.1f} nm)")
    print(f"  argmax baseline MAE = {mae_argmax:.2f} nm")

    # learned regressor (clean)
    n = len(X); folds = kfold(n)
    lam_n = (lam - 600.0) / 1050.0
    preds = np.zeros(n, np.float32)
    for f in range(5):
        te = folds[f]; tr = np.setdiff1d(np.arange(n), te)
        m = TinyMLP(1)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-4)
        xt, yt = torch.tensor(X[tr]), torch.tensor(lam_n[tr, None])
        for _ in range(400):
            opt.zero_grad(); loss = nn.functional.smooth_l1_loss(m(xt), yt)
            loss.backward(); opt.step()
        with torch.no_grad():
            preds[te] = (m(torch.tensor(X[te])).numpy().ravel() * 1050.0 + 600.0)
    mae_reg = float(np.mean(np.abs(preds - lam)))
    print(f"  learned regressor MAE (clean, 5-fold) = {mae_reg:.2f} nm")

    # robustness: add Gaussian noise to spectra, re-evaluate both
    rng = np.random.RandomState(1)
    Xn = np.clip(X + rng.normal(0, 0.05, X.shape).astype(np.float32), 0, None)
    Xn = Xn / (Xn.max(1, keepdims=True) + 1e-6)
    argmax_lam_n = grid[np.argmax(Xn, axis=1)]
    mae_argmax_noisy = float(np.mean(np.abs(argmax_lam_n - lam)))
    print(f"  noisy (sigma=0.05): argmax MAE = {mae_argmax_noisy:.2f} nm")

    # verdict: regressor must beat argmax clearly to be non-redundant
    edge = mae_argmax - mae_reg
    verdict = "SHIP" if edge > 5.0 else "REJECT (redundant with argmax)"
    print(f"  regressor edge over argmax (clean) = {edge:+.2f} nm  -> {verdict}")
    return verdict, mae_argmax, mae_reg


def main():
    torch.manual_seed(0); np.random.seed(0)
    X, h, lam, hn = load_aux()
    print(f"[data] {len(X)} spectra; hosts {hn}; lambda range {lam.min():.0f}-{lam.max():.0f} nm")
    va, *_ = probe_host(X, h)
    vb, *_ = probe_lambda(X, lam)
    print("\n=== measure-first summary ===")
    print(f"  PL host-ID     : {va}")
    print(f"  PL lambda regr : {vb}")


if __name__ == "__main__":
    main()
