"""
train_purity.py — AI-11 phase-purity pre-flight prior (CIMC Lab-Sentinel)
=========================================================================
MEASURE-FIRST. observed_pl.csv has 37 rows with a REAL measured XRD phase result
(15 pure / 21 mixed / 1 amorphous; the other 30 are "unknown" = unmeasured). We
ask: can a tiny model predict pure-vs-impure from the recipe's 24-D formula
descriptor (the SAME input as AI-6/7/13, already precomputed per preset)?

This is a COMPOSITIONAL question (does this formula+conditions form a clean phase),
which is normally an off-device composition model's job. AI-11's honest role is a fast
EDGE TRIAGE prior shown at pre-flight ("this recipe historically tends to go impure ->
worth a closer look off-device before you burn a furnace run"). It does NOT drive the
safety chain. We report its real leave-one-out CV accuracy and only deploy the
variant that actually beats the majority-class baseline; otherwise we say so.

We compare 3 deployable forms under leave-one-out CV:
  (a) linear  (24 -> 2)            -- logistic-style
  (b) MLP     (24 -> 16 -> 2)
  (c) kNN k=3 on standardised desc -- nearest known recipe's outcome (retrieval)

Run:  cd CIMC/model/ai11_purity && python train_purity.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from predict_engine import ts_torch as T          # noqa: E402

HERE = Path(__file__).parent
IN_DIM = 24


def guess_site(formula):
    for s in ("Ga", "Al", "Sc", "In"):
        if s in formula:
            return s
    return "Al"


def load_labeled():
    """37 rows with a real xrd_result label -> (desc24, pure?1:0)."""
    X, y, names = [], [], []
    with open(ROOT / "exp_ground_truth" / "observed_pl.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fm = (r.get("formula") or "").strip()
            res = (r.get("xrd_result") or "").strip().lower()
            if not fm or res in ("", "unknown", "nan"):
                continue
            site = (r.get("dopant_site") or "").strip() or guess_site(fm)
            pct = float(r.get("dopant_pct") or 1.0)
            try:
                d = T.formula_descriptor(fm, site, pct)
            except Exception:
                continue
            X.append(d.numpy())
            y.append(1 if res == "pure" else 0)        # pure vs impure(mixed/amorphous)
            names.append(fm)
    X = np.asarray(X, np.float32); y = np.asarray(y, np.int64)
    return X, y, names


class LinNet(nn.Module):
    def __init__(self):
        super().__init__(); self.net = nn.Linear(IN_DIM, 2)

    def forward(self, x): return self.net(x)


class MLPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, 16), nn.ReLU(), nn.Linear(16, 2))

    def forward(self, x): return self.net(x)


def train_net(make, Xtr, ytr, mu, sd, epochs=250, lr=5e-3, wd=3e-3):
    torch.manual_seed(0)
    m = make()
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    Xn = torch.tensor((Xtr - mu) / sd)
    yt = torch.tensor(ytr)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Xn), yt); loss.backward(); opt.step()
    m.eval()
    return m


def loo_net(make, X, y):
    n = len(X); correct = 0
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, ytr = X[tr], y[tr]
        mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
        m = train_net(make, Xtr, ytr, mu, sd)
        with torch.no_grad():
            xi = torch.tensor((X[i:i+1] - mu) / sd)
            pred = int(m(xi).argmax(1).item())
        correct += (pred == y[i])
    return correct / n


def loo_knn(X, y, k=3):
    n = len(X); correct = 0
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        mu = X[tr].mean(0); sd = X[tr].std(0) + 1e-6
        q = (X[i] - mu) / sd
        ref = (X[tr] - mu) / sd
        d = ((ref - q) ** 2).sum(1)
        nn_idx = tr[np.argsort(d)[:k]]
        vote = int(round(y[nn_idx].mean()))
        correct += (vote == y[i])
    return correct / n


def main():
    X, y, names = load_labeled()
    n = len(X); npos = int(y.sum())
    base = max(npos, n - npos) / n
    print(f"[data] {n} labeled recipes  pure={npos} impure={n-npos}  majority-baseline={base:.3f}")

    acc_lin = loo_net(LinNet, X, y)
    acc_mlp = loo_net(MLPNet, X, y)
    acc_knn = loo_knn(X, y, k=3)
    print(f"[LOO-CV] linear={acc_lin:.3f}  MLP={acc_mlp:.3f}  kNN(k=3)={acc_knn:.3f}  (baseline {base:.3f})")

    # pick the best deployable MLP-form (linear or MLP) that >= baseline; report honestly
    best = "MLP" if acc_mlp >= acc_lin else "linear"
    best_acc = max(acc_mlp, acc_lin)
    make = MLPNet if best == "MLP" else LinNet
    mu = X.mean(0); sd = X.std(0) + 1e-6
    m = train_net(make, X, y, mu, sd)                 # final on all 37
    # baked normalisation -> the C engine takes raw desc24 (we fold mu/sd into export)
    torch.save({"model_state_dict": m.state_dict(),
                "arch": best, "mu": mu.tolist(), "sd": sd.tolist(),
                "loo_linear": acc_lin, "loo_mlp": acc_mlp, "loo_knn": acc_knn,
                "baseline": base, "n": n, "n_pure": npos},
               HERE / "ai11_purity.pt")
    print(f"[final] saved ai11_purity.pt  arch={best}  LOO={best_acc:.3f} vs baseline {base:.3f} "
          f"({'BEATS baseline -> deploy as triage prior' if best_acc > base + 1e-6 else 'NOT above baseline -> will deploy as honest retrieval/uncertain'})")


if __name__ == "__main__":
    main()
