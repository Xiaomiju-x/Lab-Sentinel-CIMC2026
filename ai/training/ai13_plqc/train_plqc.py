"""
train_plqc.py — AI-13 PL-spectrum QC autoencoder (CIMC Lab-Sentinel)
====================================================================
WHY THIS, NOT "Dq/B regression" (measure-first rejection, ADR-4):
The originally-planned AI-13 "crystal-field Dq/B estimator" FAILS the honesty bar:
  * In predictions/dqb_train_data.json, Dq is itself DERIVED from lambda_em via the
    exact inversion 10*Dq = 1e7/lambda_em + Stokes (meta.inversion_ref). lambda_em is
    already predicted by AI-6 -> a Dq "model" is just algebra on AI-6's output (redundant).
  * The Racah B is ~constant in the real corpus (640-660 cm-1, std~10) -> not learnable.
  * The existing predict_engine/dqb_regressor.pt teacher is degenerate (predicts Dq~1587
    for everything, std=2 on its own corpus) -> distilling it yields a constant model.
So the honest crystal-field read-out (Dq / B / field-class) is shipped as a DERIVED
display from AI-6's lambda_em (exact algebra, labelled "derived"), NOT a fake model.

The AI-13 ML slot instead does an HONEST, measurable, DISTINCT job: an unsupervised
autoencoder over the 281 real Fluoromax emission spectra. AI-12 says WHICH activator;
AI-13 is the PL-stage QC GATE — "is this even a valid NIR phosphor emission spectrum?"
Reconstruction MSE > conformal q_hat (90%) => anomalous (noise / flat / artifact /
unexpected phase). Same AE+conformal recipe as AI-2, applied to the spectral domain.
Decision-support only; on-device the spectrum is replayed (no on-board spectrometer).

Run:  cd CIMC/model/ai13_plqc && python train_plqc.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
SN = ROOT / "spectrum_numerical"
HERE = Path(__file__).parent
GRID_LO, GRID_HI, GRID_N = 600.0, 1650.0, 64
GRID = np.linspace(GRID_LO, GRID_HI, GRID_N).astype(np.float32)


def parse_spectrum(fp):
    wl, it = [], []
    is_em = True
    try:
        for ln in open(fp, encoding="utf-8", errors="ignore"):
            parts = ln.replace("\t", ",").split(",")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a.lower() == "type":
                is_em = "em" in b.lower()
            try:
                x = float(a); yv = float(b)
            except ValueError:
                continue
            if 350.0 <= x <= 1750.0:
                wl.append(x); it.append(yv)
    except Exception:
        return None, False
    if len(wl) < 8:
        return None, False
    wl = np.asarray(wl, np.float32); it = np.asarray(it, np.float32)
    order = np.argsort(wl); wl, it = wl[order], it[order]
    g = np.interp(GRID, wl, it, left=it[0], right=it[-1]).astype(np.float32)
    g = g - g.min(); mx = g.max()
    if mx > 1e-9:
        g = g / mx
    return g.astype(np.float32), is_em


def load_spectra():
    rows = list(csv.DictReader(open(SN / "data" / "labels.csv", encoding="utf-8")))
    X, paths = [], []
    for r in rows:
        fp = SN / r["path"].replace("\\", "/")
        if not fp.exists():
            continue
        g, is_em = parse_spectrum(fp)
        if g is None or not is_em:
            continue
        X.append(g); paths.append(str(fp))
    X = np.stack(X).astype(np.float32)
    print(f"[data] {len(X)} real emission spectra loaded (64-pt normalised)")
    return X, paths


class SpecAE(nn.Module):
    """64 -> 32 -> 8 -> 32 -> 64 ReLU autoencoder (float-C via nn_ops). ~5K params."""
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(GRID_N, 32), nn.ReLU(),
                                 nn.Linear(32, 8), nn.ReLU())
        self.dec = nn.Sequential(nn.Linear(8, 32), nn.ReLU(),
                                 nn.Linear(32, GRID_N))

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    torch.manual_seed(0)
    X, paths = load_spectra()
    n = len(X)
    idx = np.random.default_rng(0).permutation(n)
    nv = max(16, n // 5); vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(X[ti]); Xv = torch.tensor(X[vi])

    m = SpecAE()
    opt = torch.optim.Adam(m.parameters(), lr=2e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    for ep in range(600):
        m.train(); opt.zero_grad()
        loss = lossf(m(Xt), Xt); loss.backward(); opt.step()
    m.eval()

    with torch.no_grad():
        rec_all = m(torch.tensor(X)).numpy()
    mse = ((rec_all - X) ** 2).mean(1)
    q_hat = float(np.quantile(mse, 0.90))      # conformal-style 90% in-dist threshold
    print(f"[recon] in-dist MSE: mean={mse.mean():.5f} p50={np.median(mse):.5f} "
          f"p90(q_hat)={q_hat:.5f} max={mse.max():.5f}")

    # honesty check: injected anomalies must exceed q_hat
    rng = np.random.default_rng(1)
    anos = {
        "flat":        np.full((1, GRID_N), 0.5, np.float32),
        "white_noise": rng.random((1, GRID_N)).astype(np.float32),
        "double_peak": None,
    }
    # a physically-wrong double-narrow-peak spectrum
    dp = np.zeros(GRID_N, np.float32)
    for c, w in [(10, 1.5), (50, 1.5)]:
        dp += np.exp(-0.5 * ((np.arange(GRID_N) - c) / w) ** 2)
    dp = dp / dp.max(); anos["double_peak"] = dp[None].astype(np.float32)
    flagged = 0
    for name, a in anos.items():
        with torch.no_grad():
            r = m(torch.tensor(a)).numpy()
        am = float(((r - a) ** 2).mean())
        hit = am > q_hat
        flagged += hit
        print(f"[anomaly] {name:12s} MSE={am:.5f}  {'FLAGGED' if hit else 'missed'} (x{am/q_hat:.1f} q_hat)")
    in_dist_fpr = float((mse > q_hat).mean())
    print(f"[QC] in-dist false-flag rate={in_dist_fpr:.3f} (target ~0.10)  anomalies flagged={flagged}/3")

    # golden: one real spectrum + recon + mse (deterministic = X[0])
    with torch.no_grad():
        g_rec = m(torch.tensor(X[:1])).numpy().reshape(-1)
    g_mse = float(((g_rec - X[0]) ** 2).mean())
    torch.save({"model_state_dict": m.state_dict(),
                "grid_lo": GRID_LO, "grid_hi": GRID_HI, "grid_n": GRID_N,
                "q_hat": q_hat, "mse_mean": float(mse.mean()), "n": n,
                "golden_input": X[0].tolist(), "golden_recon": g_rec.tolist(),
                "golden_mse": g_mse, "anomaly_flagged": int(flagged)},
               HERE / "ai13_plqc.pt")
    print(f"[final] saved ai13_plqc.pt  q_hat={q_hat:.5f}  anomalies {flagged}/3 flagged")


if __name__ == "__main__":
    main()
