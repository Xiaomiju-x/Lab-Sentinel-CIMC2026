"""
distill_ts_optical.py — AI-6 optical edge surrogate (CIMC Lab-Sentinel)
=======================================================================
KNOWLEDGE DISTILLATION, not blind regression. The 17 real lambda_em rows are far
too few to fit a model (measured: blind LOO MAE 42nm > 25nm predict-mean baseline).
So instead we distill the XRD project's ALREADY-VALIDATED differentiable
Tanabe-Sugano predictor (predict_engine/ts_torch.py TSPredictor, lambda_em MAE
6.2nm on 17 lit samples) into a tiny float-C MLP that runs on the GD32 M7.

Pipeline:
  formula (recipe) --formula_descriptor--> 24-D feature
                   --TSPredictor (teacher)--> lambda_em, FWHM   (the labels)
  edge MLP (student): 24-D --> (lambda_em_nm, FWHM_nm)

Training formulas come from REAL XRD assets (observed_pl.csv + candidate_pool.json
+ generated nir_v2), each x several dopant_pct values -> a few hundred CONSISTENT
teacher-labelled points (smooth function -> distills cleanly, unlike 17 noisy rows).

Honesty: the student reproduces the TEACHER (validated 6.2nm); we report BOTH the
student-vs-teacher fidelity AND the student-vs-17-real-anchors MAE. The teacher's
own real-world accuracy (6.2nm) is the XRD project's prior result, cited not re-won.

The 24-D descriptor is recomputable on the MCU (element fractions + a small Shannon
radius table — both tiny). Output is decision-support ("this batch should emit
~810nm"), shown on the HMI; it does NOT drive the safety chain.

Run:  cd CIMC/model/ai6_ts_optical && python distill_ts_optical.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]      # xrd/
sys.path.insert(0, str(ROOT))
from predict_engine import ts_torch as T          # noqa: E402

HERE = Path(__file__).parent
IN_DIM = 24
OUT = ["lambda_em_nm", "fwhm_nm"]


# ----------------------------------------------------------------------------- teacher
def load_teacher():
    m = T.TSPredictor()
    sd = torch.load(ROOT / "predict_engine" / "ts_torch.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(sd["model_state_dict"])
    m.eval()
    print(f"[teacher] TSPredictor loaded (train mae={sd.get('mae','?')})")
    return m


def guess_site(formula):
    """Cr3+ sits octahedral: prefer Ga/Al/Sc present in the host."""
    for s in ("Ga", "Al", "Sc", "In"):
        if s in formula:
            return s
    return "Al"


def gather_formulas():
    """Real XRD formulas: observed_pl + candidate_pool + generated nir_v2."""
    fs = set()
    # observed_pl.csv
    import csv
    with open(ROOT / "exp_ground_truth" / "observed_pl.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("formula") or "").strip():
                fs.add(r["formula"].strip())
    # candidate_pool.json (dict of lists)
    try:
        cp = json.load(open(ROOT / "crystal_data_shared" / "candidate_pool.json", encoding="utf-8"))
        for v in cp.values():
            for c in (v if isinstance(v, list) else [v]):
                if isinstance(c, dict) and c.get("formula"):
                    fs.add(c["formula"].strip())
    except Exception as e:
        print("[data] candidate_pool skipped:", e)
    # generated nir_v2
    gen = ROOT / "crystal_data_shared" / "generated" / "nir_v2"
    if gen.is_dir():
        for p in gen.glob("*.cif"):
            fs.add(p.stem.split("_")[0])
    fs = sorted(f for f in fs if any(c.isupper() for c in f))
    print(f"[data] {len(fs)} unique real formulas")
    return fs


def build_distill_set(teacher):
    """Each formula x several dopant_pct -> (24-D desc, lambda_em, FWHM) via teacher."""
    formulas = gather_formulas()
    pcts = [0.5, 0.75, 1.0, 1.5, 2.0]
    X, Y = [], []
    with torch.no_grad():
        for fm in formulas:
            site = guess_site(fm)
            for pct in pcts:
                try:
                    d = T.formula_descriptor(fm, site, pct)
                except Exception:
                    continue
                out = teacher(d[None])
                le = float(out["lambda_em_nm"]); fw = float(out["fwhm_nm"])
                if not (300.0 < le < 1600.0 and 10.0 < fw < 400.0):
                    continue                      # drop non-physical teacher outputs
                X.append(d.numpy()); Y.append([le, fw])
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    print(f"[data] distill set: {len(X)} points  lambda_em[{Y[:,0].min():.0f},{Y[:,0].max():.0f}] "
          f"FWHM[{Y[:,1].min():.0f},{Y[:,1].max():.0f}]")
    return X, Y


# ----------------------------------------------------------------------------- student
class OpticalMLP(nn.Module):
    """24 -> 32 -> 32 -> 2, ReLU. Tiny float-C deployable (~2.1K params)."""
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
    teacher = load_teacher()
    X, Y = build_distill_set(teacher)

    # output normalisation (baked into export later): standardise per target
    ymu = Y.mean(0); ysd = Y.std(0) + 1e-6
    Yn = (Y - ymu) / ysd

    n = len(X); idx = np.random.default_rng(0).permutation(n)
    nv = max(8, n // 6); vi, ti = idx[:nv], idx[nv:]
    Xt = torch.tensor(X[ti], device=dev); Yt = torch.tensor(Yn[ti], device=dev)
    Xv = torch.tensor(X[vi], device=dev); Yv = Y[vi]

    model = OpticalMLP().to(dev)
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
    # validation: student vs teacher (fidelity)
    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy() * ysd + ymu
    mae_le = float(np.mean(np.abs(pv[:, 0] - Yv[:, 0])))
    mae_fw = float(np.mean(np.abs(pv[:, 1] - Yv[:, 1])))
    print(f"[fidelity] student-vs-teacher  lambda_em MAE={mae_le:.2f}nm  FWHM MAE={mae_fw:.2f}nm")

    # real-anchor: student vs the 17 real observed lambda_em
    import csv
    anchors = []
    with open(ROOT / "exp_ground_truth" / "observed_pl.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            v = (r.get("lambda_em_nm") or "").strip()
            if v not in ("", "nan") and (r.get("formula") or "").strip():
                try:
                    site = (r.get("dopant_site") or guess_site(r["formula"])).strip() or guess_site(r["formula"])
                    pct = float(r.get("dopant_pct") or 1.0)
                    d = T.formula_descriptor(r["formula"].strip(), site, pct)
                    anchors.append((d, float(v)))
                except Exception:
                    pass
    if anchors:
        with torch.no_grad():
            Xa = torch.stack([a[0] for a in anchors]).to(dev)
            pa = (model(Xa).cpu().numpy() * ysd + ymu)[:, 0]
        ya = np.array([a[1] for a in anchors])
        print(f"[real] student vs {len(anchors)} real lambda_em anchors  MAE={np.mean(np.abs(pa-ya)):.1f}nm "
              f"(teacher's own real MAE ~6.2nm is the XRD project's prior validation)")

    torch.save({"model_state_dict": model.state_dict(),
                "ymu": ymu.tolist(), "ysd": ysd.tolist(),
                "n_train": int(len(ti)), "fidelity_lambda_mae": mae_le,
                "fidelity_fwhm_mae": mae_fw},
               HERE / "ai6_optical.pt")
    print(f"[final] saved ai6_optical.pt  ({len(X)} distill pts, {len(ti)} train)")


if __name__ == "__main__":
    main()
