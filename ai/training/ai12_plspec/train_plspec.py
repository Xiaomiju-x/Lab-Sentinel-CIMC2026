"""
train_plspec.py — AI-12 PL-emission dopant classifier (CIMC Lab-Sentinel)
=========================================================================
Opens the XRD project's UNTAPPED 281 real Fluoromax PL emission spectra
(spectrum_numerical/, labelled in spectrum_numerical/data/labels.csv by activator:
cr=162 / ni=54 / cr_ni=65) and trains an on-chip classifier that reads an emission
spectrum SHAPE and names the activator (Cr3+ / Ni2+ / Cr3+&Ni2+).

This is genuine ML (the broadband ~800nm Cr3+ d-d band vs the ~1300nm Ni2+ band vs
a mixed profile are physically different shapes) on REAL measured data, NOT an
analytic peak-pick. Edge role: a PL-stage QC gate — "the measured emission matches
the loaded recipe's dopant" / "unexpected Ni signature -> cross-contamination".

On the MCU the PL spectrum comes from the lab Fluoromax spectrometer (the sentinel
has no on-board spectrometer), so the demo REPLAYS stored real spectra — the same honest
approach as furnace_sim replaying real sintering curves. Decision-support only.

Pipeline: raw (wavelength,intensity) -> resample to fixed 64-pt grid over
[GRID_LO,GRID_HI] nm -> baseline-subtract + peak-normalise to [0,1] -> MLP 64->32->16->3.

Run:  cd CIMC/model/ai12_plspec && python train_plspec.py
"""
import csv
import os
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
CLASSES = ["cr", "ni", "cr_ni"]                 # dopant_id 0/1/2


def parse_spectrum(fp):
    """Return (wl[], inten[]) from a Fluoromax CSV (header lines then numeric rows)."""
    wl, it = [], []
    is_emission = True
    try:
        for ln in open(fp, encoding="utf-8", errors="ignore"):
            parts = ln.replace("\t", ",").split(",")
            if len(parts) < 2:
                continue
            a, b = parts[0].strip(), parts[1].strip()
            if a.lower() == "type":
                is_emission = "emission" in b.lower() or "em" in b.lower()
            try:
                x = float(a); yv = float(b)
            except ValueError:
                continue
            if 350.0 <= x <= 1750.0:
                wl.append(x); it.append(yv)
    except Exception:
        return None, None, False
    if len(wl) < 8:
        return None, None, False
    return np.asarray(wl, np.float32), np.asarray(it, np.float32), is_emission


def resample_norm(wl, it):
    order = np.argsort(wl)
    wl, it = wl[order], it[order]
    g = np.interp(GRID, wl, it, left=it[0], right=it[-1]).astype(np.float32)
    g = g - g.min()
    mx = g.max()
    if mx > 1e-9:
        g = g / mx
    return g.astype(np.float32)


def load_dataset():
    rows = list(csv.DictReader(open(SN / "data" / "labels.csv", encoding="utf-8")))
    X, y, kept_paths = [], [], []
    skipped = 0
    for r in rows:
        fp = SN / r["path"].replace("\\", "/")
        did = r.get("dopant_id")
        if did is None or did == "":
            continue
        cls = int(did)
        if cls not in (0, 1, 2) or not fp.exists():
            skipped += 1
            continue
        wl, it, is_em = parse_spectrum(fp)
        if wl is None or not is_em:
            skipped += 1
            continue
        X.append(resample_norm(wl, it)); y.append(cls); kept_paths.append(str(fp))
    X = np.stack(X).astype(np.float32); y = np.asarray(y, np.int64)
    print(f"[data] kept {len(X)} emission spectra (skipped {skipped})  "
          f"cr={int((y==0).sum())} ni={int((y==1).sum())} cr_ni={int((y==2).sum())}")
    return X, y, kept_paths


class SpecMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GRID_N, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
            nn.Linear(16, 3),
        )

    def forward(self, x): return self.net(x)


def train_one(Xtr, ytr, epochs=400, lr=3e-3, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    m = SpecMLP()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xt), yt); loss.backward(); opt.step()
    m.eval()
    return m


def stratified_kfold(X, y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for c in (0, 1, 2):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[j % k].append(ii)
    accs = []
    cm = np.zeros((3, 3), int)
    for f in range(k):
        te = np.array(folds[f]); tr = np.array([i for i in range(len(X)) if i not in set(te.tolist())])
        m = train_one(X[tr], y[tr], seed=seed + f)
        with torch.no_grad():
            pred = m(torch.tensor(X[te])).argmax(1).numpy()
        accs.append(float((pred == y[te]).mean()))
        for t, p in zip(y[te], pred):
            cm[t, p] += 1
    return float(np.mean(accs)), float(np.std(accs)), cm


def main():
    X, y, paths = load_dataset()
    n = len(X)
    base = max((y == c).sum() for c in (0, 1, 2)) / n
    acc, sd, cm = stratified_kfold(X, y, k=5)
    print(f"[5-fold CV] acc={acc:.3f} +/- {sd:.3f}  (majority baseline {base:.3f})")
    print(f"[confusion] rows=true cols=pred  cr/ni/cr_ni\n{cm}")

    model = train_one(X, y, seed=0)                 # final on all data
    # representative spectrum per class for golden/demo replay (first of each class)
    reps = {}
    for c in (0, 1, 2):
        i = int(np.where(y == c)[0][0]); reps[c] = (X[i], paths[i])
    torch.save({"model_state_dict": model.state_dict(),
                "grid_lo": GRID_LO, "grid_hi": GRID_HI, "grid_n": GRID_N,
                "cv_acc": acc, "cv_sd": sd, "baseline": base, "n": n,
                "confusion": cm.tolist(),
                "reps": {int(c): reps[c][0].tolist() for c in reps},
                "rep_paths": {int(c): reps[c][1] for c in reps}},
               HERE / "ai12_plspec.pt")
    print(f"[final] saved ai12_plspec.pt  CV={acc:.3f} vs baseline {base:.3f}")


if __name__ == "__main__":
    main()
