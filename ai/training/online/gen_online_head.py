"""
gen_online_head.py — generate the on-device ONLINE-LEARNING risk head
(weights init + golden update sequence) for the GD32 firmware.

What this is (TinyML continual learning, honestly scoped):
  A linear softmax classifier  risk = softmax(W·f + b),  f in R^16, 4 classes
  {good,warn,bad,crit}. It is SEEDED on the PC (a quick fit on synthetic
  furnace features) and then ADAPTED ON-CHIP by one SGD step per operator
  correction (forward + backward + weight update all on the M7). "越用越准":
  the head learns this furnace/operator's calls without any cloud retraining.

  We do NOT claim to train the whole nano-LM on-chip (infeasible). We train a
  real (tiny) last layer with real gradients on-device — the achievable, honest
  TinyML on-device-learning result for a Cortex-M7.

Golden contract: the firmware copies the const init W/b into RAM at boot, then
applies the SAME golden (feature,label) update stream; the resulting RAM weights
must match `online_golden.h` to ~1e-5, and predictions must match. Proves the C
forward+backward+SGD reproduces numpy.

Outputs (-> ../../firmware/ai_models_c/):
  online_head_weights.h   init W[4][16], b[4], lr, dims
  online_golden.h         update stream + expected post-update W/b + predict checks

Run:  python gen_online_head.py        (numpy only, RTX/torch not needed)
"""
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / "firmware" / "ai_models_c"
OUT.mkdir(parents=True, exist_ok=True)

F, K, LR = 16, 4, 0.10
rng = np.random.default_rng(20260603)


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _sample_one():
    """One realistic 16-D feature vector in the SAME space the firmware's
    online_build_feat() emits (normalized [0,1] flags + temp/dev/etc.), with a
    physics-based severity -> {good,warn,bad,crit} label. Feature layout (must
    match lab_sentinel.c online_build_feat exactly):
      f0 risk/3  f1 duty  f2 dev/200  f3 tc_fault  f4 tc_OC  f5 tc_SC  f6 smoke
      f7 temp/1600  f8 elem  f9 state/3  f10 OOC(run only)  f11 motor  f12 cpk
      f13 seg  f14 overshoot  f15 bias=1 """
    f = np.zeros(F, np.float32)
    f[15] = 1.0
    if rng.random() < 0.33:
        # IDLE / healthy standby -> good. f10 is gated to 0 at idle (firmware too).
        f[7] = rng.uniform(20.0, 70.0) / 1600.0       # near room temp
        f[8] = rng.uniform(0.85, 1.0)                 # element health high
        f[9] = 0.0                                    # idle
        f[12] = 0.0
        label = 0
        f[0] = 0.0
        return f, label
    # ---- running scenario ----
    f[9] = 1.0 / 3.0                                  # state = run
    f[1] = rng.uniform(0.1, 1.0)                      # heater duty
    f[7] = rng.uniform(0.18, 0.95)                    # temp normalized
    f[8] = rng.uniform(0.45, 1.0)                     # element health
    f[11] = 1.0 if rng.random() < 0.5 else 0.0        # grinder motor
    f[13] = rng.uniform(0.0, 1.0)                     # recipe segment
    f[12] = rng.uniform(0.0, 1.2)                     # cpk
    dev = rng.normal(0.0, 22.0)                       # tracking deviation (deg)
    f[2] = dev / 200.0
    f[14] = 1.0 if dev > 5.0 else 0.0                 # overshoot
    if rng.random() < 0.06: f[3] = 1.0; f[4] = 1.0    # TC open-circuit
    if rng.random() < 0.05: f[3] = 1.0; f[5] = 1.0    # TC short
    if rng.random() < 0.05: f[6] = 1.0                # smoke / off-gas
    f[10] = 1.0 if rng.random() < 0.30 else 0.0       # out-of-control (run only)
    # physics severity from the feature vector
    sev = (3.0 * max(f[3], f[4], f[5])                # TC fault -> crit
           + 2.2 * f[6]                               # smoke
           + 2.0 * abs(f[2])                          # |deviation|
           + 1.2 * (f[14] * f[7])                     # overshoot at high temp
           + 1.0 * f[10]                              # OOC during run
           + 0.9 * (1.0 - f[8])                       # element degradation
           + rng.normal(0.0, 0.22))
    label = 0 if sev < 0.6 else 1 if sev < 1.5 else 2 if sev < 2.6 else 3
    # controller risk f0 is a real upstream input: correlated with the truth but
    # noisy/sometimes wrong (that disagreement is exactly what the operator TEACHes).
    f0 = label / 3.0 + rng.normal(0.0, 0.18)
    if rng.random() < 0.12:                           # controller disagrees
        f0 = rng.uniform(0.0, 1.0)
    f[0] = float(np.clip(f0, 0.0, 1.0))
    return f, int(label)


def synth(n):
    """n realistic (feature, label) pairs (see _sample_one)."""
    X = np.zeros((n, F), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        X[i], y[i] = _sample_one()
    return X, y


def fit_init(X, y, epochs=120, lr=0.2):
    W = np.zeros((K, F), np.float32); b = np.zeros(K, np.float32)
    for _ in range(epochs):
        for i in range(len(X)):
            p = softmax(W @ X[i] + b)
            g = p.copy(); g[y[i]] -= 1.0
            W -= lr * np.outer(g, X[i]); b -= lr * g
    return W, b


def carr_f(name, a, shape2=None):
    a = np.asarray(a, np.float32)
    if shape2:
        rows = ",".join("{" + ",".join(f"{float(v):.7e}f" for v in r) + "}" for r in a)
        return f"static const float {name}[{a.shape[0]}][{a.shape[1]}] = {{{rows}}};\n"
    return f"static const float {name}[{a.size}] = {{{','.join(f'{float(v):.7e}f' for v in a.reshape(-1))}}};\n"


def main():
    Xtr, ytr = synth(600)
    W0, b0 = fit_init(Xtr, ytr)
    acc0 = (np.array([softmax(W0 @ x + b0).argmax() for x in Xtr]) == ytr).mean()
    hist = np.bincount(ytr, minlength=K)
    print(f"init head fit acc {acc0:.3f}  class hist good/warn/bad/crit = {hist.tolist()}")

    # sanity: the seed MUST predict good(0) on the exact firmware idle vector
    # (matches online_build_feat at idle: temp~25/1600, elem=1, f10 gated to 0, bias=1)
    idle = np.zeros(F, np.float32); idle[7] = 25.0 / 1600.0; idle[8] = 1.0; idle[15] = 1.0
    idle_p = softmax(W0 @ idle + b0)
    print(f"idle-vector pred = {int(idle_p.argmax())} (want 0=good)  probs={np.round(idle_p,3).tolist()}")
    assert int(idle_p.argmax()) == 0, "seed mispredicts idle furnace -> retune severity"

    # golden online stream: M operator corrections (feature, true label)
    Xup, yup = synth(24)
    W, b = W0.copy().astype(np.float64), b0.copy().astype(np.float64)  # f64 ref accum
    for i in range(len(Xup)):
        p = softmax(W @ Xup[i] + b)
        g = p.copy(); g[yup[i]] -= 1.0
        W -= LR * np.outer(g, Xup[i]); b -= LR * g
    Wf, bf = W.astype(np.float32), b.astype(np.float32)

    # predict checks on 4 probe features (after adaptation)
    Xpb, ypb = synth(4)
    preds = [int(softmax(Wf @ x + bf).argmax()) for x in Xpb]

    h = ("/* online_head_weights.h - AUTO-GENERATED by gen_online_head.py. Do not edit.\n"
         " * On-device online-learning risk head (linear softmax, 16->4). Init seeded on\n"
         " * PC; copied to RAM at boot and adapted by 1 SGD step per operator correction. */\n"
         "#ifndef ONLINE_HEAD_WEIGHTS_H\n#define ONLINE_HEAD_WEIGHTS_H\n\n")
    h += f"#define OL_F {F}\n#define OL_K {K}\n#define OL_LR {LR:.6f}f\n\n"
    h += carr_f("ol_W0", W0, shape2=True)
    h += carr_f("ol_b0", b0)
    h += "\n#endif\n"
    (OUT / "online_head_weights.h").write_text(h, encoding="utf-8")

    g = ("/* online_golden.h - AUTO-GENERATED. Update stream + expected post-update W/b. */\n"
         "#ifndef ONLINE_GOLDEN_H\n#define ONLINE_GOLDEN_H\n\n")
    g += f"#define OL_NUP {len(Xup)}\n"
    g += carr_f("ol_up_x", Xup, shape2=True)
    g += f"static const int ol_up_y[OL_NUP] = {{{','.join(str(int(v)) for v in yup)}}};\n\n"
    g += carr_f("ol_W_exp", Wf, shape2=True)
    g += carr_f("ol_b_exp", bf)
    g += f"\n#define OL_NPROBE {len(Xpb)}\n"
    g += carr_f("ol_probe_x", Xpb, shape2=True)
    g += f"static const int ol_probe_pred[OL_NPROBE] = {{{','.join(str(v) for v in preds)}}};\n"
    g += "\n#endif\n"
    (OUT / "online_golden.h").write_text(g, encoding="utf-8")
    print("wrote online_head_weights.h / online_golden.h ->", OUT)
    print("golden probe preds:", preds)


if __name__ == "__main__":
    main()
