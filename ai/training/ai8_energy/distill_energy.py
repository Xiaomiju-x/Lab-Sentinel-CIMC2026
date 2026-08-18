"""
distill_energy.py — AI-8 energy & carbon edge estimator (CIMC Lab-Sentinel)
===========================================================================
Fills the Siemens-flavoured "energy efficiency / carbon" gap (flagged 2026-05-31
audit) with a tiny float-C MLP that predicts a sintering batch's electricity (kWh)
and CO2 (kg) from the recipe, BEFORE it runs.

Honesty / sources:
  • Labels come from a PHYSICS energy model of a NOMINAL benchtop muffle furnace
    (rated P_max = 5 kW, reaches 1600 C). NOT a fabricated citation — it is an
    explicitly-nominal lab-furnace spec; the per-batch number is a planning
    estimate, deliberately labelled "nominal".
  • The on-device LIVE energy is the controller's ACTUAL duty cycle u integrated
    over time (E = Σ u·P_max·dt) — that part is MEASURED, not modelled. AI-8 is
    the pre-run estimate; the HMI shows estimate vs live.
  • Grid carbon factor: 0.5703 kgCO2/kWh — China national grid average emission
    factor (生态环境部 / MEE 2022 全国电网平均). Cited, not invented.
  • Recipe schedules come from the 13 REAL host profiles in
    predict_engine/sintering_profiles.json (4 with real DOIs).

Physics model (lumped, documented):
  hold power at setpoint T:  P_hold(T) = P_MAX * ((T-T_amb)/(T_max-T_amb))**1.5
                             (radiation-weighted steady loss; fraction of rated)
  hold energy   = P_hold(T) * hold_hours
  ramp energy   = C_TH * dT  +  0.5 * P_hold(T) * ramp_hours   (heat mass + losses)
  total kWh = Σ_segments(calcine, sinter)   (cooling = natural, ~0)

Run:  cd CIMC/model/ai8_energy && python distill_energy.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent

# ---- nominal benchtop muffle-furnace power model (documented constants) -------
P_MAX = 5.0          # kW rated
T_AMB = 25.0         # C
T_MAX = 1600.0       # C
C_TH = 0.05          # kWh per degC heated (furnace body + load thermal mass, nominal)
RAMP_DEFAULT = 5.0   # C/min (sintering_profiles default ramp)
CO2_PER_KWH = 0.5703  # kg CO2 / kWh — China national grid avg (MEE 2022)


def p_hold(T):
    return P_MAX * (max(T - T_AMB, 0.0) / (T_MAX - T_AMB)) ** 1.5


def segment_energy(T_from, T_to, hold_h, ramp_C_per_min=RAMP_DEFAULT):
    dT = max(T_to - T_from, 0.0)
    ramp_h = (dT / ramp_C_per_min) / 60.0 if ramp_C_per_min > 0 else 0.0
    ramp_E = C_TH * dT + 0.5 * p_hold(T_to) * ramp_h
    hold_E = p_hold(T_to) * hold_h
    return ramp_E + hold_E


def recipe_energy_kwh(calc_T, calc_h, sint_T, sint_h, ramp=RAMP_DEFAULT):
    e = segment_energy(T_AMB, calc_T, calc_h, ramp)        # ambient -> calcine
    e += segment_energy(calc_T, sint_T, sint_h, ramp)      # calcine -> sinter
    return e


def load_profiles():
    d = json.load(open(ROOT / "predict_engine" / "sintering_profiles.json", encoding="utf-8"))
    out = []
    for k, v in d.items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        c, s = v.get("calcine", {}), v.get("sinter", {})
        try:
            out.append((k, float(c["temp_C"]), float(c["hours"]),
                        float(s["temp_C"]), float(s["hours"]),
                        float(s.get("ramp_C_per_min", RAMP_DEFAULT))))
        except Exception:
            pass
    return out


def feats(calc_T, calc_h, sint_T, sint_h, ramp):
    return np.array([calc_T / 1600.0, calc_h / 12.0, sint_T / 1600.0,
                     sint_h / 12.0, ramp / 10.0], np.float32)


class EnergyMLP(nn.Module):
    """5 -> 16 -> 16 -> 1 (kWh). ~0.4K params. CO2 = kWh * factor (in C, no model)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    torch.manual_seed(0)
    profs = load_profiles()
    print(f"[data] {len(profs)} real host profiles")
    rng = np.random.default_rng(0)
    X, Y = [], []
    for (_, cT, cH, sT, sH, rp) in profs:
        for _ in range(60):                      # sample physical variations
            dcT = cT + rng.uniform(-80, 80); dcH = max(0.5, cH + rng.uniform(-1, 1))
            dsT = sT + rng.uniform(-80, 80); dsH = max(1.0, sH + rng.uniform(-2, 2))
            drp = float(np.clip(rp + rng.uniform(-2, 2), 1.0, 12.0))
            X.append(feats(dcT, dcH, dsT, dsH, drp))
            Y.append(recipe_energy_kwh(dcT, dcH, dsT, dsH, drp))
    X = np.asarray(X, np.float32); Y = np.asarray(Y, np.float32)
    print(f"[data] {len(X)} pts  kWh[{Y.min():.1f},{Y.max():.1f}] mean {Y.mean():.1f}  "
          f"-> CO2[{Y.min()*CO2_PER_KWH:.1f},{Y.max()*CO2_PER_KWH:.1f}]kg")

    ymu, ysd = Y.mean(), Y.std() + 1e-6
    Yn = (Y - ymu) / ysd
    n = len(X); idx = rng.permutation(n); nv = n // 6
    vi, ti = idx[:nv], idx[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(X[ti], device=dev); Yt = torch.tensor(Yn[ti], device=dev)
    Xv = torch.tensor(X[vi], device=dev); Yv = Y[vi]
    model = EnergyMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-6)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 250)
    lossf = nn.SmoothL1Loss()
    for ep in range(250):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 64):
            bi = perm[b:b+64]
            opt.zero_grad(); loss = lossf(model(Xt[bi]), Yt[bi]); loss.backward(); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        pv = model(Xv).cpu().numpy() * ysd + ymu
    mae = float(np.mean(np.abs(pv - Yv)))
    print(f"[fidelity] energy MLP vs physics  MAE={mae:.3f} kWh ({mae/Y.mean()*100:.1f}% of mean)")
    # show the 13 nominal profiles' predicted energy/carbon
    print("[profiles] nominal energy/carbon:")
    with torch.no_grad():
        for (name, cT, cH, sT, sH, rp) in profs:
            phys = recipe_energy_kwh(cT, cH, sT, sH, rp)
            pred = float(model(torch.tensor(feats(cT, cH, sT, sH, rp), device=dev)[None]).cpu()) * ysd + ymu
            print(f"   {name:16s} phys={phys:5.1f}kWh pred={pred:5.1f}kWh  CO2={phys*CO2_PER_KWH:5.1f}kg")
    torch.save({"model_state_dict": model.state_dict(), "ymu": float(ymu), "ysd": float(ysd),
                "co2_per_kwh": CO2_PER_KWH, "p_max_kw": P_MAX, "mae_kwh": mae},
               HERE / "ai8_energy.pt")
    print(f"[final] saved ai8_energy.pt")


if __name__ == "__main__":
    main()
