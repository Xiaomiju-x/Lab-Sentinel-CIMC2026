"""
train_vib_pdm.py — AI-10 vibration PdM classifier (CIMC Lab-Sentinel)
=====================================================================
The one NEW model on a REAL sensor modality (ADXL345, already on the board at
200 Hz) rather than XRD data: rotating-equipment predictive maintenance for the
grinding motor / furnace fan / pump. 4 classes:
  0 normal      low broadband + tiny 1x (rotation) line
  1 imbalance   strong 1x line
  2 bearing     high-frequency band energy (BPFO/BPFI) + sidebands
  3 looseness   1x + strong harmonics (2x, 3x, 4x)

HONESTY: trained on SYNTHETIC vibration whose fault signatures are TEXTBOOK ISO
20816 / rotating-machinery diagnostics (imbalance=1x, looseness=harmonics,
bearing=high-freq) — NOT real ADXL345 data yet. The board has the ADXL345 + the
eccentric vibration motor (procurement P0 #10) to collect REAL data for these 4
classes; this model is the synthetic-bootstrap (exactly like AI-1 was before the
real crucible photos) and will be refined on real data once hardware is wired.
Clearly labelled as such on the HMI; decision-support, not safety-chain.

Features (8, computed on a 64-sample ADXL345 window on-device, cheap):
  rms, crest_factor, kurtosis, E_1x, E_2x, E_3x, E_highband, E_broadband
(all normalised). Tiny MLP 8->16->16->4 (~0.5K params), float-C.

Run:  cd CIMC/model/ai10_vibration && python train_vib_pdm.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
FS = 200.0          # ADXL345 sample rate (Hz)
N = 64              # window
F_ROT = 25.0        # nominal rotation freq (Hz), 1500 rpm


def synth_window(cls, rng):
    """Generate one 64-sample vibration window for class cls (textbook signatures)."""
    t = np.arange(N) / FS
    x = rng.normal(0, 0.05, N)                       # broadband floor
    w1 = 2 * np.pi * F_ROT
    if cls == 0:        # normal
        x += 0.10 * np.sin(w1 * t + rng.uniform(0, 6.28))
    elif cls == 1:      # imbalance — dominant 1x
        x += rng.uniform(0.6, 1.0) * np.sin(w1 * t + rng.uniform(0, 6.28))
    elif cls == 2:      # bearing — high-freq band (BPFO ~ 3.5x..) + noise bursts
        fb = rng.uniform(70, 95)                      # high-freq defect band (Hz)
        x += rng.uniform(0.3, 0.6) * np.sin(2 * np.pi * fb * t + rng.uniform(0, 6.28))
        x += rng.normal(0, 0.15, N) * (rng.random(N) > 0.6)   # impulsive bursts
        x += 0.10 * np.sin(w1 * t)
    else:               # looseness — 1x + strong harmonics
        x += 0.4 * np.sin(w1 * t)
        for h in (2, 3, 4):
            x += rng.uniform(0.2, 0.45) * np.sin(h * w1 * t + rng.uniform(0, 6.28))
    return x.astype(np.float64)


def features(x):
    """8 cheap features (mirror the on-device C)."""
    rms = np.sqrt(np.mean(x * x)) + 1e-9
    peak = np.max(np.abs(x))
    crest = peak / rms
    kurt = np.mean(((x - x.mean()) / (x.std() + 1e-9)) ** 4)
    sp = np.abs(np.fft.rfft(x * np.hanning(N)))        # magnitude spectrum
    freqs = np.fft.rfftfreq(N, 1 / FS)

    def band(f0, f1):
        m = (freqs >= f0) & (freqs < f1)
        return float(np.sum(sp[m] ** 2))
    e1 = band(F_ROT - 6, F_ROT + 6)
    e2 = band(2 * F_ROT - 6, 2 * F_ROT + 6)
    e3 = band(3 * F_ROT - 6, 3 * F_ROT + 6)
    ehi = band(65, 100)
    ebb = band(0, 100) + 1e-9
    return np.array([rms, crest, kurt, e1 / ebb, e2 / ebb, e3 / ebb, ehi / ebb, ebb], np.float32)


class VibMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(8, 16), nn.ReLU(),
                                 nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    NPER = 600
    X, y = [], []
    for cls in range(4):
        for _ in range(NPER):
            X.append(features(synth_window(cls, rng))); y.append(cls)
    X = np.stack(X); y = np.array(y)
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xn = (X - mu) / sd
    idx = rng.permutation(len(y)); nv = len(y) // 5
    vi, ti = idx[:nv], idx[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xt = torch.tensor(Xn[ti], device=dev); yt = torch.tensor(y[ti], device=dev)
    Xv = torch.tensor(Xn[vi], device=dev); yv = y[vi]
    model = VibMLP().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 200)
    lossf = nn.CrossEntropyLoss()
    for ep in range(200):
        model.train(); perm = torch.randperm(len(Xt), device=dev)
        for b in range(0, len(Xt), 64):
            bi = perm[b:b+64]
            opt.zero_grad(); loss = lossf(model(Xt[bi]), yt[bi]); loss.backward(); opt.step()
        sch.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).cpu().numpy()
    acc = (pred == yv).mean()
    names = ["normal", "imbalance", "bearing", "looseness"]
    cm = np.zeros((4, 4), int)
    for t_, p_ in zip(yv, pred):
        cm[t_, p_] += 1
    print(f"[vib] synthetic val acc = {acc*100:.1f}%  (textbook ISO20816 signatures)")
    print("      confusion (rows=true):")
    for i in range(4):
        print(f"        {names[i]:10s} " + " ".join(f"{cm[i,j]:3d}" for j in range(4)))
    torch.save({"model_state_dict": model.state_dict(), "mu": mu.tolist(), "sd": sd.tolist(),
                "acc_synth": float(acc)}, HERE / "ai10_vib.pt")
    print(f"[final] saved ai10_vib.pt  (synthetic; real ADXL345 refine pending hardware)")


if __name__ == "__main__":
    main()
