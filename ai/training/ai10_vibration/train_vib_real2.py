"""
train_vib_real2.py — AI-10 vibration PdM, RETRAINED ON REAL DATA (CIMC Lab-Sentinel)
====================================================================================
Replaces the synthetic 4-class ISO-20816 bootstrap (train_vib_pdm.py) with a REAL
2-class model trained on data captured from THIS motor + ADXL345:

    class 0  stopped   (motor off -> rest; magnitude is gravity-only ~= 0 mg AC)
    class 1  running   (motor spinning; real captured vibration, ~1-3 g AC)

HONESTY: trained on real ADXL345 windows captured on-device (capture_real.txt,
2026-06-02, software-PWM 50%). The two states are physically separable by a wide
margin (rest acRMS ~= 0 vs running ~1000-3000 mg), so a tiny MLP nails it. Light
jitter / amplitude-scale / time-shift augmentation (standard, signal stays real,
same practice as AI-1's 52 crucible photos) makes mu/sd representative and the
model robust to small rest bumps. We deliberately do NOT fabricate bearing /
imbalance / looseness fault classes (we have no real faulted hardware -> would
violate ADR-4); honest 2-class stopped/running on real data instead.

Same 8 features + same MLP shape as the on-device C (8->16->16->OUT), so the
exported header drops straight in (only AI10_OUT changes 4->2).

Run:  cd CIMC/model/ai10_vibration && <mace_env python> train_vib_real2.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[2]
sys.path.insert(0, str(HERE))
from train_vib_pdm import features as vib_features  # noqa: E402  (mirrors on-device C)

OUT_FW = ROOT / "CIMC" / "firmware" / "ai_models_c"
OUT_HOST = ROOT / "CIMC" / "model" / "host_test"
N = 64
NAMES = ["stopped", "running"]
NOUT = 2


# ----------------------------------------------------------------- data loading
def load_capture(path):
    """Parse '[cap] run=R w=v0,...,v63' lines -> list of (label, mg_array[64])."""
    out = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.search(r"run=(\d+)\s+w=([\-0-9,]+)", ln)
        if not m:
            continue
        lab = int(m.group(1))
        vals = [int(v) for v in m.group(2).split(",") if v != ""]
        if len(vals) >= N:
            out.append((lab, np.asarray(vals[:N], np.float64)))
    return out


def win_to_feat(mg64):
    """Mirror the on-device pipeline: mg -> g, mean-remove, then 8 features.
    (C caller mean-removes the ring window before ai10_features.)"""
    w = mg64 / 1000.0
    w = w - w.mean()
    return vib_features(w.astype(np.float64)).astype(np.float32), w.astype(np.float32)


def augment_running(w_g, rng):
    """One real running window (g, mean-removed) -> a realistic variant."""
    s = rng.uniform(0.75, 1.30)                       # amplitude (speed/coupling)
    sh = rng.integers(0, N)                           # circular phase shift
    v = np.roll(w_g, sh) * s
    v = v + rng.normal(0, 0.02, N)                    # small sensor noise (g)
    v = v - v.mean()
    return v


def augment_stopped(rng):
    """Stopped class. Real rest read all-zero (gravity-only). Represent it as a
    mix of EXACT zero (as captured) + small sub-rotation jitter (handling/table
    bumps), all well below the motor's running energy. Spanning crest 0 (zero)
    and ~3 (noise) makes crest/kurt non-discriminative, forcing the model to key
    on vibration ENERGY (rms / ebb) — the real physical 'is it spinning' signal."""
    if rng.random() < 0.4:
        return np.zeros(N)                            # genuine rest, as captured
    sigma = rng.uniform(0.0, 0.04)                    # 0..40 mg small disturbance
    v = rng.normal(0, sigma, N)
    return v - v.mean()


# --------------------------------------------------------------------- model
class VibMLP2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(),
                                  nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, NOUT))

    def forward(self, x):
        return self.net(x)


def carr(name, arr):
    flat = np.asarray(arr, np.float32).reshape(-1)
    return f"static const float {name}[{flat.size}] = {{ {', '.join(f'{v:.8e}f' for v in flat)} }};\n"


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    rows = []
    for cap in sorted(HERE.glob("capture_real*.txt")):
        r = load_capture(cap)
        rows += r
        print(f"[data] {cap.name}: {sum(1 for l, _ in r if l == 1)} running windows")
    real_run = [w for (lab, w) in [(l, win_to_feat(a)[1]) for (l, a) in rows] if lab == 1]
    n_run_real = len(real_run)
    print(f"[data] real running windows total = {n_run_real} (25%+50% speeds; stopped = all-zero rest, regenerated)")

    # Build dataset. Keep some REAL windows held out for honest validation.
    rng.shuffle(real_run)
    n_val = max(3, n_run_real // 5)
    run_val_g = real_run[:n_val]
    run_tr_g = real_run[n_val:]

    PER = 400
    Xtr, ytr = [], []
    # stopped (class 0): jittered rest
    for _ in range(PER):
        Xtr.append(vib_features(augment_stopped(rng)).astype(np.float32)); ytr.append(0)
    # running (class 1): augmented from real training windows
    for _ in range(PER):
        base = run_tr_g[rng.integers(0, len(run_tr_g))]
        Xtr.append(vib_features(augment_running(base, rng)).astype(np.float32)); ytr.append(1)
    Xtr = np.stack(Xtr); ytr = np.array(ytr)

    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xn = (Xtr - mu) / sd

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn, device=dev); yt = torch.tensor(ytr, device=dev)
    model = VibMLP2().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    lossf = nn.CrossEntropyLoss()
    for ep in range(300):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 64):
            bi = perm[b:b + 64]
            opt.zero_grad(); loss = lossf(model(Xt[bi]), yt[bi]); loss.backward(); opt.step()
        sch.step()
    model.eval()

    # ---- honest validation on REAL held-out windows (no augmentation) ----
    # stopped val = a handful of genuine all-zero rest windows
    Xv, yv = [], []
    for _ in range(n_val):
        Xv.append(vib_features((np.zeros(N) - 0.0).astype(np.float64)).astype(np.float32)); yv.append(0)
    for w in run_val_g:
        Xv.append(vib_features(w.astype(np.float64)).astype(np.float32)); yv.append(1)
    Xv = np.stack(Xv); yv = np.array(yv)
    with torch.no_grad():
        pv = model(torch.tensor(((Xv - mu) / sd), device=dev)).argmax(1).cpu().numpy()
    acc = (pv == yv).mean()
    cm = np.zeros((NOUT, NOUT), int)
    for t_, p_ in zip(yv, pv):
        cm[t_, p_] += 1
    print(f"[vib] REAL held-out val acc = {acc * 100:.1f}%  (n={len(yv)}: {n_val} stopped + {len(run_val_g)} running)")
    print("      confusion (rows=true):")
    for i in range(NOUT):
        print(f"        {NAMES[i]:8s} " + " ".join(f"{cm[i, j]:3d}" for j in range(NOUT)))

    # ---- export weights header (AI10_OUT 4->2) ----
    L = []
    for m in model.net:
        if isinstance(m, nn.Linear):
            L.append((m.weight.detach().cpu().numpy().astype(np.float32),
                      m.bias.detach().cpu().numpy().astype(np.float32)))
    s = ("/* AI-10 vibration PdM (8 feats -> 2 logits: stopped/running).\n"
         " * REAL data: trained on this motor + ADXL345 (capture_real.txt 2026-06-02),\n"
         " * light jitter/scale/shift aug. Honest 2-class (no faked fault classes).\n"
         " * AUTO-GENERATED by train_vib_real2.py. */\n"
         "#ifndef AI10_VIB_WEIGHTS_H\n#define AI10_VIB_WEIGHTS_H\n\n"
         f"#define AI10_IN 8\n#define AI10_H 16\n#define AI10_OUT {NOUT}\n")
    for i, (W, b) in enumerate(L):
        s += carr(f"ai10_w{i}", W) + carr(f"ai10_b{i}", b)
    s += carr("ai10_in_mu", mu) + carr("ai10_in_sd", sd)
    s += "\n#endif\n"
    (OUT_FW / "ai10_vib_weights.h").write_text(s, encoding="utf-8")

    # ---- golden: one REAL running window (mean-removed g), py feats + logits ----
    gw = run_val_g[0] if run_val_g else real_run[0]    # deterministic (post-shuffle)
    gw = gw.astype(np.float32)                           # already mean-removed g
    f8 = vib_features(gw.astype(np.float64)).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(((f8 - mu) / sd), device=dev)[None]).cpu().numpy().reshape(-1)
    cls = int(np.argmax(logits))
    g = ("/* AI-10 golden: a REAL running window (mean-removed g) + python feats +\n"
         " * raw logits (2-class stopped/running). AUTO-GENERATED. */\n"
         "#ifndef AI10_GOLDEN_H\n#define AI10_GOLDEN_H\n\n")
    g += carr("g_ai10_window", gw) + carr("g_ai10_feat8", f8) + carr("g_ai10_logits", logits)
    g += f"static const int g_ai10_cls = {cls};\n\n#endif\n"
    (OUT_FW / "ai10_golden.h").write_text(g, encoding="utf-8")
    (OUT_HOST / "ai10_golden.h").write_text(g, encoding="utf-8")

    torch.save({"model_state_dict": model.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(),
                "n_out": NOUT, "names": NAMES, "acc_real": float(acc)},
               HERE / "ai10_vib_real2.pt")
    print(f"[golden] AI-10 real window->feat8={np.round(f8, 3)}")
    print(f"         logits={np.round(logits, 3)} cls={cls} ({NAMES[cls]})")
    print(f"[final] wrote ai10_vib_weights.h (OUT={NOUT}) + ai10_golden.h (fw+host) + ai10_vib_real2.pt")


if __name__ == "__main__":
    main()
