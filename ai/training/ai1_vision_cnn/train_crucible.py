"""
train_crucible.py — train AI-1 crucible 4-class CNN (CIMC Lab-Sentinel)
=======================================================================
Runs on the local RTX 4060 laptop (CIMC models are tiny — no 5090 needed).

Data source (auto):
  • If  data/crucible/{empty,loaded,sintering,done}/*.png|jpg  exists  -> REAL data
    (torchvision ImageFolder, resized to 64x64, with augmentation).
  • Else -> synthetic placeholder (synth_crucible.py) to validate the pipeline.

Pixels are fed as float [0,1] CHW — NO mean/std normalisation, so the firmware
just does (RGB565 -> /255) with nothing else (matches export golden).

Outputs:
  crucible_cnn.pt        state_dict (consumed by export_weights_to_c.py)
  crucible_config.json   arch dims + class names + data source + val accuracy

Run:  cd CIMC/model/ai1_vision_cnn && python train_crucible.py [--epochs 40]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from crucible_cnn import CrucibleCNN, CLASS_NAMES, IN_HW, N_CLS

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "crucible"


def load_real():
    """Return (X,y) float32 CHW[0,1] / int64 from data/crucible/<class>/, or None."""
    have = DATA_DIR.is_dir() and any(
        (DATA_DIR / c).is_dir() and any((DATA_DIR / c).glob("*"))
        for c in CLASS_NAMES
    )
    if not have:
        return None
    from torchvision import datasets, transforms
    tf = transforms.Compose([
        transforms.Resize((IN_HW, IN_HW)),
        transforms.ToTensor(),                      # -> [0,1] CHW
    ])
    # enforce our fixed class order via a class_to_idx remap
    ds = datasets.ImageFolder(str(DATA_DIR), transform=tf)
    remap = {ds.class_to_idx[c]: i for i, c in enumerate(CLASS_NAMES) if c in ds.class_to_idx}
    X = torch.stack([ds[i][0] for i in range(len(ds))]).numpy().astype(np.float32)
    y = np.array([remap[ds[i][1]] for i in range(len(ds))], dtype=np.int64)
    print(f"[data] REAL dataset: {len(y)} imgs, per-class {np.bincount(y, minlength=N_CLS)}")
    return X, y


def load_data():
    real = load_real()
    if real is not None:
        return real, "real"
    from synth_crucible import make_synth
    X, y = make_synth(n_per_class=500, seed=1234)
    print(f"[data] SYNTHETIC placeholder: {len(y)} imgs (collect real data into "
          f"{DATA_DIR}/<class>/ to override)")
    return (X, y), "synthetic"


def augment(xb, rng):
    """Light on-the-fly augmentation: hflip + brightness + gaussian noise."""
    if rng.random() < 0.5:
        xb = torch.flip(xb, dims=[3])
    xb = xb * (0.85 + 0.30 * rng.random()) + (rng.random() - 0.5) * 0.05
    xb = xb + torch.randn_like(xb) * 0.02
    return xb.clamp_(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    (X, y), src = load_data()

    g = torch.Generator().manual_seed(0)
    n = len(y)
    idx = torch.randperm(n, generator=g)
    n_val = max(1, int(0.15 * n))
    vi, ti = idx[:n_val], idx[n_val:]
    Xt = torch.from_numpy(X)[ti]; yt = torch.from_numpy(y)[ti]
    Xv = torch.from_numpy(X)[vi].to(dev); yv = torch.from_numpy(y)[vi].to(dev)

    model = CrucibleCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(0)

    best = 0.0
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(yt))
        tot = 0.0
        for b in range(0, len(yt), args.bs):
            bi = perm[b:b + args.bs]
            xb = augment(Xt[bi].clone(), rng).to(dev)
            yb = yt[bi].to(dev)
            opt.zero_grad()
            out = model(xb)
            loss = lossf(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(bi)
        sched.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        best = max(best, acc)
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} loss={tot/len(yt):.4f} val_acc={acc*100:.1f}%")

    # final confusion matrix on val
    model.eval()
    with torch.no_grad():
        pred = model(Xv).argmax(1).cpu().numpy()
    gtv = yv.cpu().numpy()
    cm = np.zeros((N_CLS, N_CLS), dtype=int)
    for t, p in zip(gtv, pred):
        cm[t, p] += 1
    print("confusion (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:5]:>6}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:>9} " + " ".join(f"{cm[i, j]:6d}" for j in range(N_CLS)))

    torch.save(model.state_dict(), HERE / "crucible_cnn.pt")
    cfg = {
        "arch": "CrucibleCNN", "in_ch": 3, "in_hw": IN_HW, "n_cls": N_CLS,
        "ch": [8, 16, 24, 32], "class_names": CLASS_NAMES,
        "input_norm": "none ([0,1] from RGB565/255)", "data_source": src,
        "val_acc": round(best, 4), "n_images": int(n),
    }
    (HERE / "crucible_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"\nsaved crucible_cnn.pt + crucible_config.json (src={src}, best_val={best*100:.1f}%)")


if __name__ == "__main__":
    main()
