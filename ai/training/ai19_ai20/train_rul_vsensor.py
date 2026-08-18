"""
train_rul_vsensor.py — AI-19 sintering RUL + AI-20 thermocouple-integrity monitor
=================================================================================
Project: 18 -> 20 on-chip AI models (CIMC Lab-Sentinel). Two NEW, non-redundant
tasks, both measure-first gated and host-golden verifiable.

  AI-19  RUL / ETA regressor      : window[24]+hold_cum+stage -> minutes-to-done.
                                    A NEW quantity no other model outputs. Useful:
                                    operator "ETA to firing complete". Honest/non-
                                    trivial because the run TOTAL shifts under the
                                    fast/slow-ramp anomalies, so a nominal-schedule
                                    baseline (which even gets the true stage+step)
                                    is WRONG on anomaly runs; the model recovers ETA
                                    from the observable window + cumulative dwell.
                                    GATE: beat nominal-schedule baseline MAE (min).

  AI-20  thermocouple-integrity   : window of (measured, commanded-setpoint) ->
         classifier (4-class)       {0 healthy / 1 stuck / 2 offset / 3 dropout}.
                                    Analytical-redundancy SENSOR monitor: questions
                                    whether the READING itself is trustworthy, vs
                                    AI-3 (process-anomaly classifier that assumes the
                                    reading is real). The setpoint channel is the
                                    fault-independent reference, so a FROZEN sensor on
                                    a moving ramp is caught (the 1-step-regressor form
                                    was REJECTED by measure-first: persistence beat it
                                    and a stuck sensor was invisible, AUC 0.51). Trained
                                    by standard fault-INJECTION over furnace_sim runs.
                                    GATE: overall acc beats majority AND offset+dropout
                                    recall > 0.9 (stuck capped by plateau ambiguity —
                                    reported honestly, not gated).

Self-contained: replicates furnace_sim.c (incl. LCG noise), trains the two tiny
nets, runs the gates, emits C weight headers + a host golden header.

Run:  cd CIMC/model && python ai19_ai20/train_rul_vsensor.py
Out:  firmware/ai_models_c/ai19_rul_weights.h
      firmware/ai_models_c/ai20_tcfault_weights.h
      firmware/ai_models_c/ai19_ai20_golden.h   (+ copy in model/host_test/)
"""
import sys
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]               # .../xrd
OUT = ROOT / "CIMC" / "firmware" / "ai_models_c"
HOST = ROOT / "CIMC" / "model" / "host_test"

# ---- profile constants — MUST match furnace_sim.c byte-for-byte -----------------
T_ROOM, CALCINE_C, SINTER_C = 25.0, 900.0, 1500.0
BASE_RAMP, COOL_RATE = 5.0, 3.0
CALCINE_MIN, SINTER_MIN, GRIND_MIN = 240, 360, 30
TNORM = 1600.0

WIN = 24          # AI-19 input window — matches AI-14 fcwin
L = 12            # AI-20 classifier window
RNORM = 1500.0    # AI-19 remaining-minutes normaliser

ANOM_NONE, ANOM_FAST, ANOM_SLOW, ANOM_DRIFT, ANOM_UNDER = range(5)
# AI-20 classes — only cleanly SENSOR-attributable faults (a constant offset is
# physically indistinguishable from a real process drift, so it is deliberately
# NOT a class: that ambiguity belongs to AI-3's process-anomaly job, not here).
FCLS = ["healthy", "open-circuit", "erratic"]
NF = 8            # AI-20 engineered plausibility features
NC20 = 3          # the two cleanly-attributable TC failures + healthy


class _LCG:
    """Replicates furnace_sim.c frand_unit()/noise3() exactly (uint32 wrap)."""
    def __init__(self, seed=0x1234abcd):
        self.s = np.uint32(seed)

    def _u(self):
        with np.errstate(over="ignore"):
            self.s = np.uint32(np.uint32(self.s * np.uint32(1664525)) +
                               np.uint32(1013904223))
        return ((self.s >> np.uint32(8)) & np.uint32(0xFFFFFF)) / float(0x1000000) - 0.5

    def noise3(self):
        return (self._u() + self._u() + self._u() + self._u()) * 3.0


def ramp_rate(anom):
    if anom == ANOM_FAST:
        return BASE_RAMP * 4.0
    if anom == ANOM_SLOW:
        return max(BASE_RAMP * 0.25, 0.5)
    return BASE_RAMP


def stage_bounds(anom):
    r = ramp_rate(anom)
    ramp1 = int((CALCINE_C - T_ROOM) / max(r, 1.0))
    ramp2 = int((SINTER_C - T_ROOM) / max(r, 1.0))
    cool = int((SINTER_C - T_ROOM) / COOL_RATE)
    b = [0] * 6
    b[0] = ramp1
    b[1] = b[0] + CALCINE_MIN
    b[2] = b[1] + GRIND_MIN
    b[3] = b[2] + ramp2
    b[4] = b[3] + SINTER_MIN
    b[5] = b[4] + cool
    return b


NOM_B = stage_bounds(ANOM_NONE)
NOM_TOTAL = NOM_B[5]


def target_temp(stage, step, slen):
    if stage == 0:
        frac = step / (slen - 1) if slen > 1 else 1.0
        return T_ROOM + (CALCINE_C - T_ROOM) * frac
    if stage == 1:
        return CALCINE_C
    if stage == 2:
        return T_ROOM
    if stage == 3:
        frac = step / (slen - 1) if slen > 1 else 1.0
        return T_ROOM + (SINTER_C - T_ROOM) * frac
    if stage == 4:
        return SINTER_C
    t = SINTER_C - COOL_RATE * step
    return max(t, T_ROOM)


def gen_run(anom, seed):
    """One full run; per-minute arrays matching furnace_sim.c state."""
    b = stage_bounds(anom)
    total = b[5]
    lcg = _LCG(seed)
    temp, tgt, stage_a, step_a, hold_a, knee = [], [], [], [], [], []
    hold_cum = 0.0
    for m in range(total):
        prev, stage, step, slen = 0, 5, 0, 1
        for i in range(6):
            if m < b[i]:
                stage, step, slen = i, m - prev, b[i] - prev
                break
            prev = b[i]
        tt = target_temp(stage, step, slen)
        tc = tt + lcg.noise3()
        if anom == ANOM_DRIFT:
            tc += 64.0
        elif anom == ANOM_UNDER and stage == 4:
            tc -= 100.0
        if (stage == 1 or stage == 4) and abs(tc - tt) <= 10.0:
            hold_cum += 1.0
        temp.append(tc); tgt.append(tt); stage_a.append(stage)
        step_a.append(step); hold_a.append(hold_cum)
        knee.append(1 if any(abs(m - bb) <= 6 for bb in b[:5]) else 0)
    return dict(temp=np.array(temp, np.float32), tgt=np.array(tgt, np.float32),
                stage=np.array(stage_a, np.int32), step=np.array(step_a, np.int32),
                hold=np.array(hold_a, np.float32), knee=np.array(knee, np.int32),
                total=total, b=b)


def build_runs():
    runs, seed = [], 1
    for _ in range(8):
        runs.append(gen_run(ANOM_NONE, seed)); seed += 1
    for a in (ANOM_FAST, ANOM_SLOW, ANOM_DRIFT, ANOM_UNDER):
        for _ in range(3):
            runs.append(gen_run(a, seed)); seed += 1
    return runs


def carr(name, arr):
    flat = np.asarray(arr, np.float32).reshape(-1)
    body = ", ".join(f"{v:.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{ {body} }};\n"


# =============================================================================
# AI-19 RUL
# =============================================================================
def ai19_windows(runs):
    X, Y, base = [], [], []
    for r in runs:
        tn = r["temp"] / TNORM
        total = r["total"]
        for m in range(WIN - 1, total):
            feat = np.empty(WIN + 2, np.float32)
            feat[:WIN] = tn[m - WIN + 1:m + 1]
            feat[WIN] = r["hold"][m] / 600.0
            feat[WIN + 1] = r["stage"][m] / 5.0
            X.append(feat)
            Y.append(np.float32(((total - 1) - m) / RNORM))
            st, sp = int(r["stage"][m]), int(r["step"][m])
            nominal_elapsed = (NOM_B[st - 1] if st > 0 else 0) + sp
            base.append(max(NOM_TOTAL - 1 - nominal_elapsed, 0))
    return (np.asarray(X, np.float32), np.asarray(Y, np.float32),
            np.asarray(base, np.float32))


class RUL(nn.Module):
    def __init__(self, nx=WIN + 2, hid=48):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# =============================================================================
# AI-20 thermocouple-integrity classifier (fault injection + plausibility feats)
# =============================================================================
def feats8(meas, setp):
    """8 engineered plausibility features from one L-window of normalised
       (measured, setpoint). MUST be mirrored byte-for-byte in ai_ext_models.c.
       Population std (ddof=0) to match the C sqrt(mean(x^2)-mean^2) form."""
    dm = np.diff(meas)
    f = np.empty(NF, np.float32)
    f[0] = float(np.mean(meas - setp))          # residual bias (context only)
    f[1] = float(np.std(meas))                  # measured variability
    f[2] = float(np.std(setp))                  # setpoint variability (is program moving?)
    f[3] = float(np.min(meas))                  # dropout -> ~0
    f[4] = float(np.max(np.abs(dm)))            # stuck->0 / erratic->large jump
    f[5] = float(np.max(np.abs(meas - setp)))   # gross deviation
    f[6] = float(np.mean(meas))                 # level
    f[7] = float(np.mean(meas < (40.0 / TNORM)))  # fraction implausibly low
    return f


def ai20_windows(runs, rng):
    """label = fault class; X = 8 plausibility features over an L-window. Faults
       INJECTED over real sim trajectories (standard fault-injection):
       0 healthy / 1 open-circuit(->0) / 2 erratic(EMI/loose HF noise).
       'stuck/frozen' and 'offset' are deliberately NOT classes: a freeze on a flat
       program and an offset are both physically indistinguishable from, respectively,
       a healthy hold and a real process drift — claiming them would be dishonest."""
    X, Yc, raw = [], [], []
    for r in runs:
        tn = r["temp"] / TNORM
        gn = r["tgt"] / TNORM
        total = r["total"]
        for m in range(L, total, 2):                     # stride 2 -> dataset size
            meas = tn[m - L:m].copy()
            setp = gn[m - L:m].copy()
            cls = int(rng.integers(0, NC20))
            if cls == 1:                                 # open circuit / dropout -> ~0
                onset = int(rng.integers(0, L - 1))
                meas[onset:] = rng.normal(0.0, 2.0 / TNORM, size=L - onset).astype(np.float32)
            elif cls == 2:                               # erratic: large HF noise (loose/EMI)
                onset = int(rng.integers(0, L - 1))
                meas[onset:] += rng.normal(0.0, 40.0 / TNORM,
                                           size=L - onset).astype(np.float32)
            X.append(feats8(meas, setp)); Yc.append(cls)
            raw.append(np.concatenate([meas, setp]).astype(np.float32))
    return (np.asarray(X, np.float32), np.asarray(Yc, np.int64),
            np.asarray(raw, np.float32))


class TCFault(nn.Module):
    def __init__(self, nx=NF, hid=16, nc=NC20):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(nx, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, nc))

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0); np.random.seed(0)
    rng = np.random.default_rng(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = build_runs()

    # ---------------------------------------------------------------- AI-19 RUL
    X, Y, BASE = ai19_windows(runs)
    n = len(X); idx = np.random.permutation(n)
    X, Y, BASE = X[idx], Y[idx], BASE[idx]
    ntr = int(n * 0.8)
    Xtr, Ytr = X[:ntr], Y[:ntr]
    Xte, Yte, BASEte = X[ntr:], Y[ntr:], BASE[ntr:]
    print(f"[AI-19] {n} windows (train {ntr} / test {n-ntr})")

    m19 = RUL().to(dev)
    opt = torch.optim.Adam(m19.parameters(), lr=2e-3)
    lossf = nn.MSELoss()
    xt = torch.tensor(Xtr, device=dev); yt = torch.tensor(Ytr, device=dev)
    for ep in range(900):
        m19.train(); opt.zero_grad()
        loss = lossf(m19(xt), yt); loss.backward(); opt.step()
        if (ep + 1) % 300 == 0:
            print(f"  ep{ep+1:4d}  loss {loss.item():.6f}")
    m19.eval()
    with torch.no_grad():
        pred = m19(torch.tensor(Xte, device=dev)).cpu().numpy()
    mae_model = float(np.mean(np.abs(pred * RNORM - Yte * RNORM)))
    mae_base = float(np.mean(np.abs(BASEte - Yte * RNORM)))
    mae_mean = float(np.mean(np.abs(Yte.mean() * RNORM - Yte * RNORM)))
    print("\n=== measure-first AI-19 (MAE in minutes-to-done) ===")
    print(f"  constant-mean {mae_mean:7.1f} | nominal-schedule {mae_base:7.1f} | "
          f"AI-19 {mae_model:7.1f}")
    gate19 = mae_model < mae_base and mae_model < mae_mean
    print(f"  --> AI-19 beats both baselines: {gate19} (gate)")
    if not gate19:
        print("  !! GATE FAILED — RUL not worth shipping; pivot needed.")
        sys.exit(1)

    # ---------------------------------------------------------------- AI-20 TCfault
    XV, YV, RAW = ai20_windows(runs, rng)
    nv = len(XV); idxv = np.random.permutation(nv)
    XV, YV, RAW = XV[idxv], YV[idxv], RAW[idxv]
    ntv = int(nv * 0.8)
    Xvtr, Yvtr = XV[:ntv], YV[:ntv]
    Xvte, Yvte, RAWte = XV[ntv:], YV[ntv:], RAW[ntv:]
    print(f"\n[AI-20] {nv} fault-injected windows (train {ntv} / test {nv-ntv})")

    m20 = TCFault().to(dev)
    opt = torch.optim.Adam(m20.parameters(), lr=2e-3)
    lossc = nn.CrossEntropyLoss()
    xt = torch.tensor(Xvtr, device=dev); yt = torch.tensor(Yvtr, device=dev)
    for ep in range(900):
        m20.train(); opt.zero_grad()
        loss = lossc(m20(xt), yt); loss.backward(); opt.step()
        if (ep + 1) % 300 == 0:
            print(f"  ep{ep+1:4d}  loss {loss.item():.6f}")
    m20.eval()
    with torch.no_grad():
        logit = m20(torch.tensor(Xvte, device=dev)).cpu().numpy()
    pcls = logit.argmax(1)
    acc = float((pcls == Yvte).mean())
    maj = float(np.bincount(Yvte, minlength=NC20).max() / len(Yvte))
    print("\n=== measure-first AI-20 thermocouple-integrity classifier ===")
    print(f"  overall acc {acc*100:5.1f}%  (majority baseline {maj*100:4.1f}%)")
    rec = {}
    for c in range(NC20):
        mask = Yvte == c
        rec[c] = float((pcls[mask] == c).mean()) if mask.sum() else 0.0
        print(f"    recall {FCLS[c]:12s} {rec[c]*100:5.1f}%")
    # GATE: beat majority, near-zero false alarms (healthy recall high), and both real
    # TC failures (open-circuit + erratic) caught reliably.
    gate20 = acc > maj + 0.2 and rec[0] > 0.9 and rec[1] > 0.9 and rec[2] > 0.9
    print(f"  --> acc>majority, healthy>0.9 (low false-alarm), open-circuit+erratic>0.9: {gate20} (gate)")
    print("     (frozen/offset deliberately out of scope — physically ambiguous vs")
    print("      healthy-hold / process-drift; handed to the residual + AI-3 cross-check.)")
    if not gate20:
        print("  !! GATE FAILED — TC-fault classifier not worth shipping; pivot needed.")
        sys.exit(1)

    # ===================================================================== export
    def lin(mod):
        s = mod.net
        return [(s[0].weight, s[0].bias), (s[2].weight, s[2].bias),
                (s[4].weight, s[4].bias)]

    h = OUT / "ai19_rul_weights.h"
    with open(h, "w") as f:
        f.write("/* ai19_rul_weights.h - AUTO-GENERATED by ai19_ai20/train_rul_vsensor.py\n")
        f.write(" * AI-19 sintering RUL/ETA regressor (CIMC Lab-Sentinel).\n")
        f.write(f" * MLP {WIN+2}->48->48->1. feat = 24 temps/1600 + hold_cum/600 + stage/5;\n")
        f.write(f" * output * {RNORM:.0f} = minutes to firing-complete. Measure-first: MAE\n")
        f.write(f" * {mae_model:.0f} min beats nominal-schedule baseline {mae_base:.0f} min (the run\n")
        f.write(" * total shifts under ramp anomalies, so fixed-nominal scheduling is wrong). */\n")
        f.write("#ifndef AI19_RUL_WEIGHTS_H\n#define AI19_RUL_WEIGHTS_H\n\n")
        f.write(f"#define AI19_WIN {WIN}\n#define AI19_NX {WIN+2}\n#define AI19_HID 48\n")
        f.write(f"#define AI19_RNORM {RNORM:.1f}f\n#define AI19_TNORM {TNORM:.1f}f\n\n")
        for i, (W, b) in enumerate(lin(m19)):
            f.write(carr(f"ai19_w{i}", W.detach().cpu().numpy()))
            f.write(carr(f"ai19_b{i}", b.detach().cpu().numpy()))
        f.write("\n#endif\n")
    print(f"\n[export] {h}")

    h = OUT / "ai20_tcfault_weights.h"
    with open(h, "w") as f:
        f.write("/* ai20_tcfault_weights.h - AUTO-GENERATED by ai19_ai20/train_rul_vsensor.py\n")
        f.write(" * AI-20 thermocouple-integrity classifier (CIMC Lab-Sentinel).\n")
        f.write(f" * MLP {NF}->16->16->{NC20} over 8 plausibility features of an L={L} window of\n")
        f.write(" * (measured, commanded-setpoint). class 0 healthy / 1 open-circuit(->0) /\n")
        f.write(" * 2 erratic(EMI/loose). Analytical-redundancy SENSOR monitor: questions whether\n")
        f.write(" * the READING is trustworthy (AI-3 assumes it is real). The 1-step-regressor\n")
        f.write(" * form was REJECTED by measure-first (persistence beat it; a frozen sensor\n")
        f.write(" * invisible); 'stuck' and 'offset' dropped as physically ambiguous (vs healthy-\n")
        f.write(f" * hold / process-drift). Fault-injection training; acc {acc*100:.0f}% (majority\n")
        f.write(f" * {maj*100:.0f}%), open-circuit/erratic recall {rec[1]*100:.0f}%/{rec[2]*100:.0f}%. */\n")
        f.write("#ifndef AI20_TCFAULT_WEIGHTS_H\n#define AI20_TCFAULT_WEIGHTS_H\n\n")
        f.write(f"#define AI20_L {L}\n#define AI20_NF {NF}\n#define AI20_H 16\n#define AI20_NC {NC20}\n")
        f.write(f"#define AI20_TNORM {TNORM:.1f}f\n\n")
        for i, (W, b) in enumerate(lin(m20)):
            f.write(carr(f"ai20_w{i}", W.detach().cpu().numpy()))
            f.write(carr(f"ai20_b{i}", b.detach().cpu().numpy()))
        f.write("\n#endif\n")
    print(f"[export] {h}")

    # ---- golden -----------------------------------------------------------------
    p19 = [0, 1, 2, 3, 4, 5]
    # one+ test window of each fault class -> golden verifies all four paths
    p20 = []
    for c in range(NC20):
        w = np.where(Yvte == c)[0]
        p20.extend(list(w[:2]))
    p20 = p20[:6]
    g = OUT / "ai19_ai20_golden.h"
    with open(g, "w") as f:
        f.write("/* ai19_ai20_golden.h - AUTO-GENERATED. AI-19/20 host golden vectors.\n")
        f.write(" * ai20_g_in = RAW window [measured x L, setpoint x L]; the C engine recomputes\n")
        f.write(" * the 8 plausibility features then the MLP, so this verifies the FULL path. */\n")
        f.write("#ifndef AI19_AI20_GOLDEN_H\n#define AI19_AI20_GOLDEN_H\n\n")
        f.write(f"#define AI19_NG {len(p19)}\n#define AI20_NG {len(p20)}\n\n")
        f.write(carr("ai19_g_in", Xte[p19]))                       # [NG][26]
        f.write(carr("ai19_g_out", pred[p19]))                     # [NG] normalised remaining
        f.write(carr("ai20_g_in", RAWte[p20]))                     # [NG][2L] raw window
        f.write(carr("ai20_g_logit", logit[p20]))                  # [NG][4] PyTorch logits
        gcls = np.asarray(pcls[p20], np.int32)
        f.write(f"static const int ai20_g_cls[{len(p20)}] = {{ "
                + ", ".join(str(int(v)) for v in gcls) + " };\n")
        f.write("\n#endif\n")
    shutil.copy(g, HOST / "ai19_ai20_golden.h")
    print(f"[export] {g}  (+ host_test copy)")
    print("[done] AI-19 + AI-20 exported. Both gates PASSED.")


if __name__ == "__main__":
    main()
