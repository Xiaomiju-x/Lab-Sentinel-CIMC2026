"""
distill_thermal.py — AI-7 thermal-quenching edge surrogate (CIMC Lab-Sentinel)
==============================================================================
Same distillation idea as AI-6, but for the OTHER big NIR-phosphor failure mode:
thermal quenching. The teacher is the XRD project's validated Struck-Fonger model
(predict_engine/ts_torch.py thermal_quenching_struck_fonger): given the crystal-
field activation energy (Ea = 0.12*E_4T2, E_4T2 = 10*Dq) + Huang-Rhys (S, hbar_omega),
it returns I(T)/I(T_ref) — the fraction of emission surviving at temperature T.

We distill formula -> thermal_stability_pct@150C (= 100 * I(423K)/I(300K)) into a
tiny float-C MLP. Teacher params come from the SAME validated TSPredictor used by
AI-6, so the chain is: formula --descriptor--> TSPredictor --(Dq,S,hbar_omega)-->
Struck-Fonger --> stability%.

Blind regression on the 14 real thermal_stability rows is hopeless (same as AI-6's
17-row lambda finding), so we distill the validated physics and report student-vs-
teacher fidelity + student-vs-14-real anchors.

Output is decision-support ("this composition keeps ~XX% emission at 150C"); it does
NOT drive the safety chain. ~1.3K params.

Run:  cd CIMC/model/ai7_thermal && python distill_thermal.py
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
sys.path.insert(0, str(ROOT / "CIMC" / "model" / "ai6_ts_optical"))
from distill_ts_optical import load_teacher, gather_formulas, guess_site  # reuse  # noqa: E402

HERE = Path(__file__).parent
IN_DIM = 24
T_HOT = 423.15      # 150 C
T_REF = 300.0       # room


def teacher_thermal(teacher, desc):
    """formula descriptor -> thermal_stability_pct@150C via TSPredictor + Struck-Fonger."""
    with torch.no_grad():
        o = teacher(desc[None])
        E_4T2 = o["E_4T2_cm1"]; S = o["S"]; hw = o["hbar_omega_cm1"]
        q = T.thermal_quenching_struck_fonger(E_4T2, S, hw,
                                              torch.tensor([T_HOT]), torch.tensor([T_REF]))
        return float(q["I_ratio"]) * 100.0


class ThermalMLP(nn.Module):
    """24 -> 16 -> 16 -> 1 (thermal_stability_pct). ~1.3K params."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(IN_DIM, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = load_teacher()
    formulas = gather_formulas()
    pcts = [0.5, 0.75, 1.0, 1.5, 2.0]
    X, Y = [], []
    for fm in formulas:
        site = guess_site(fm)
        for pct in pcts:
            try:
                d = T.formula_descriptor(fm, site, pct)
            except Exception:
                continue
            ts = teacher_thermal(teacher, d)
            if 0.0 < ts <= 100.5:
                X.append(d.numpy()); Y.append(ts)
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    print(f"[data] {len(X)} distill pts  thermal_stab%[{Y.min():.1f},{Y.max():.1f}] mean {Y.mean():.1f}")

    ymu, ysd = Y.mean(), Y.std() + 1e-6
    Yn = (Y - ymu) / ysd
    n = len(X); idx = np.random.default_rng(0).permutation(n); nv = max(8, n // 6)
    vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(X[ti], device=dev); Yt = torch.tensor(Yn[ti], device=dev)
    Xv = torch.tensor(X[vi], device=dev); Yv = Y[vi]

    model = ThermalMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    lossf = nn.SmoothL1Loss()
    for ep in range(300):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 64):
            bi = perm[b:b+64]
            opt.zero_grad(); loss = lossf(model(Xt[bi]), Yt[bi]); loss.backward(); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy() * ysd + ymu
    print(f"[fidelity] student-vs-teacher thermal_stab MAE={np.mean(np.abs(pv-Yv)):.2f}%")

    # real anchors: 14 rows with thermal_stability_pct_at_150C
    anchors = []
    with open(ROOT / "exp_ground_truth" / "observed_pl.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("thermal_stability_pct_at_150C") or "").strip()
            if v not in ("", "nan") and (r.get("formula") or "").strip():
                try:
                    site = (r.get("dopant_site") or guess_site(r["formula"])).strip() or guess_site(r["formula"])
                    d = T.formula_descriptor(r["formula"].strip(), site, float(r.get("dopant_pct") or 1.0))
                    anchors.append((d, float(v)))
                except Exception:
                    pass
    if anchors:
        with torch.no_grad():
            Xa = torch.stack([a[0] for a in anchors]).to(dev)
            pa = model(Xa).cpu().numpy() * ysd + ymu
        ya = np.array([a[1] for a in anchors])
        print(f"[real] student vs {len(anchors)} real thermal_stab anchors  MAE={np.mean(np.abs(pa-ya)):.1f}%")

    torch.save({"model_state_dict": model.state_dict(), "ymu": float(ymu), "ysd": float(ysd)},
               HERE / "ai7_thermal.pt")
    print(f"[final] saved ai7_thermal.pt ({len(X)} pts, {len(ti)} train)")


if __name__ == "__main__":
    main()
