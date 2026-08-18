"""
distill_dqb.py — AI-13 crystal-field Dq/B edge estimator (CIMC Lab-Sentinel)
============================================================================
KNOWLEDGE DISTILLATION (same philosophy as AI-6), not blind regression. The 39
real Cr3+ Dq/B samples (predictions/dqb_train_data.json, 22 lit + 17 empirical)
are too few to fit a fresh model, but the XRD project ALREADY trained a validated
crystal-field regressor on them: predict_engine/dqb_regressor.py DqBRegressor
(24-D formula descriptor -> Dq_cm1, B_cm1, with sigmoid-bounded physical ranges).

So we distill that teacher into a tiny float-C MLP that runs on the GD32 M7 and
shares AI-6/7's SAME 24-D descriptor input (already precomputed per recipe preset
in recipe_presets.h -> no MCU formula parser, no extra preset cost).

  formula (recipe) --formula_descriptor--> 24-D  --DqBRegressor (teacher)--> Dq,B
  edge MLP (student): 24-D --> (Dq_cm1, B_cm1)

Why it earns its place (not redundant with AI-6): AI-6 predicts the OBSERVABLE
(lambda_em/FWHM); AI-13 exposes the underlying PHYSICS (Dq = crystal-field
splitting, B = Racah covalency) -> on-device interpretability "why ~800nm:
Dq/B ~ 2.5 intermediate-field garnet". Decision-support only, not the safety chain.

Honesty: we report BOTH student-vs-teacher fidelity AND student-vs-39-real-anchor
MAE. The teacher's own real accuracy is the XRD project's prior result, cited.

Run:  cd CIMC/model/ai13_dqb && python distill_dqb.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]          # xrd/
sys.path.insert(0, str(ROOT))
from predict_engine import ts_torch as T                              # noqa: E402
from predict_engine.dqb_regressor import DqBRegressor, CKPT_PATH      # noqa: E402

HERE = Path(__file__).parent
IN_DIM = 24


def load_teacher():
    m = DqBRegressor()
    sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model_state_dict"])
    m.eval()
    print(f"[teacher] DqBRegressor loaded (test_MAE Dq={sd.get('test_mae_Dq_cm1','?')} "
          f"B={sd.get('test_mae_B_cm1','?')})")
    return m


def guess_site(formula):
    for s in ("Ga", "Al", "Sc", "In"):
        if s in formula:
            return s
    return "Al"


def gather_formula_sites():
    """(formula, site) pairs ON the teacher's manifold so distillation is meaningful.
    The teacher is a dropout-regularised MLP trained on 39 garnet-ish hosts; off-manifold
    (Ge pyroxenes, generated exotics) it collapses to the mean (measured: Dq pinned to
    ~1587). So we distill on its OWN corpus formulas (with their real sites) + the 4
    deployed recipe presets -> the student spans the real Dq range [1200,2400] and, what
    matters on-device, matches the teacher exactly on the presets it actually runs on."""
    seen, out = set(), []
    # the teacher's 39-sample corpus, with the REAL site used for each
    corpus = json.load(open(ROOT / "predictions" / "dqb_train_data.json", encoding="utf-8"))
    for s in corpus["samples"]:
        fm = (s.get("formula") or "").strip()
        site = (s.get("site") or guess_site(fm)).strip() or guess_site(fm)
        if fm and (fm, site) not in seen:
            seen.add((fm, site)); out.append((fm, site))
    # the 4 deployed recipe presets (must match exactly on-device)
    for fm, site in [("Y3Al5O12", "Al"), ("Gd3Al2Ga3O12", "Ga"),
                     ("Y3Ga5O12", "Ga"), ("Mg2SiO4", "Mg")]:
        if (fm, site) not in seen:
            seen.add((fm, site)); out.append((fm, site))
    print(f"[data] {len(out)} on-manifold (formula,site) pairs (teacher corpus + presets)")
    return out


def build_distill_set(teacher):
    pairs = gather_formula_sites()
    pcts = [0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    X, Y = [], []
    with torch.no_grad():
        for fm, site in pairs:
            for pct in pcts:
                try:
                    d = T.formula_descriptor(fm, site, pct)
                except Exception:
                    continue
                out = teacher(d[None])
                dq = float(out["Dq_cm1"]); b = float(out["B_cm1"])
                X.append(d.numpy()); Y.append([dq, b])
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    print(f"[data] distill set: {len(X)} points  Dq[{Y[:,0].min():.0f},{Y[:,0].max():.0f}] "
          f"B[{Y[:,1].min():.0f},{Y[:,1].max():.0f}]")
    return X, Y


class DqBMLP(nn.Module):
    """24 -> 32 -> 32 -> 2, ReLU (float-C deployable via nn_ops). ~2.1K params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IN_DIM, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_teacher().to(dev)
    X, Y = build_distill_set(teacher.cpu())
    teacher = teacher.to(dev)

    ymu = Y.mean(0); ysd = Y.std(0) + 1e-6
    Yn = (Y - ymu) / ysd

    n = len(X); idx = np.random.default_rng(0).permutation(n)
    nv = max(8, n // 6); vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(X[ti], device=dev); Yt = torch.tensor(Yn[ti], device=dev)
    Xv = torch.tensor(X[vi], device=dev); Yv = Y[vi]

    model = DqBMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    lossf = nn.SmoothL1Loss()
    for ep in range(300):
        model.train()
        perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 64):
            bi = perm[b:b+64]
            opt.zero_grad(); loss = lossf(model(Xt[bi]), Yt[bi]); loss.backward(); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy() * ysd + ymu
    mae_dq = float(np.mean(np.abs(pv[:, 0] - Yv[:, 0])))
    mae_b = float(np.mean(np.abs(pv[:, 1] - Yv[:, 1])))
    print(f"[fidelity] student-vs-teacher  Dq MAE={mae_dq:.1f}cm-1  B MAE={mae_b:.1f}cm-1")

    # real-anchor: student vs the 39 real Dq/B samples
    corpus = json.load(open(ROOT / "predictions" / "dqb_train_data.json", encoding="utf-8"))
    rows = corpus["samples"]
    da = []
    with torch.no_grad():
        for s in rows:
            try:
                d = T.formula_descriptor(s["formula"], s.get("site") or "Al", float(s.get("pct") or 1.0))
            except Exception:
                continue
            p = model(d[None].to(dev)).cpu().numpy().reshape(-1) * ysd + ymu
            da.append((abs(p[0] - s["Dq_cm1"]), abs(p[1] - s["B_cm1"])))
    da = np.array(da)
    print(f"[real] student vs {len(da)} real anchors  Dq MAE={da[:,0].mean():.1f}  B MAE={da[:,1].mean():.1f} cm-1 "
          f"(teacher trained on these; distillation should track closely)")

    torch.save({"model_state_dict": model.state_dict(),
                "ymu": ymu.tolist(), "ysd": ysd.tolist(),
                "n_distill": int(n), "fidelity_dq_mae": mae_dq, "fidelity_b_mae": mae_b},
               HERE / "ai13_dqb.pt")
    print(f"[final] saved ai13_dqb.pt  ({n} distill pts)")


if __name__ == "__main__":
    main()
