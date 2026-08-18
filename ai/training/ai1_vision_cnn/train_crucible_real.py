"""
train_crucible_real.py — AI-1 crucible 3-class CNN on REAL phone photos (CIMC)
==============================================================================
Trains the deployed CrucibleCNN (3-class: empty / loaded / done) on the real
crucible photos the PI lab phone-shot into CIMC/手机拍摄数据/ (copied to
data/crucible/{empty,loaded,done}/ by the prep step).

Why this is its own script (not train_crucible.py): only ~52 real images, so a
single random 15% val split is too noisy to trust. This does:
  1. a colour-separability report (loaded=white vs done=yellow is the hard pair —
     they only differ in R/G-vs-B balance, so we measure it BEFORE training);
  2. stratified 5-fold cross-validation for an HONEST accuracy estimate
     (mean ± std + an aggregated out-of-fold confusion matrix);
  3. a FINAL model trained on ALL images for deployment (export_crucible_to_c.py).

Augmentation is HUE-LOCKED (hue=0): brightness/contrast/mild-saturation + crop/
rotate/flip/blur/noise expand the tiny set and bridge the phone↔OV5640 gap, but
never rotate the colour wheel — colour IS the loaded/done discriminator.

Pixels are float [0,1] CHW, NO mean/std normalisation (firmware does RGB565/255
and nothing else — matches the export golden).

Run:  cd CIMC/model/ai1_vision_cnn && python train_crucible_real.py [--epochs 90]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from crucible_cnn import CrucibleCNN, CLASS_NAMES, IN_HW, N_CLS

HERE = Path(__file__).parent
DATA_DIR = HERE / "data" / "crucible"

# ---- training augmentation: hue LOCKED to 0 (preserve white/yellow) -----------
TRAIN_TF = transforms.Compose([
    transforms.Resize((IN_HW, IN_HW)),
    transforms.RandomResizedCrop(IN_HW, scale=(0.75, 1.0), ratio=(0.85, 1.18)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.30, contrast=0.25, saturation=0.15, hue=0.0),
    transforms.GaussianBlur(3, sigma=(0.1, 1.2)),   # OV5640 softer optics
    transforms.ToTensor(),                          # -> [0,1] CHW
    transforms.Lambda(lambda t: (t + torch.randn_like(t) * 0.02).clamp_(0, 1)),
])
EVAL_TF = transforms.Compose([
    transforms.Resize((IN_HW, IN_HW)),
    transforms.ToTensor(),
])


def list_dataset():
    """Return [(path, label)], asserting all 3 class dirs exist with images."""
    items = []
    for lbl, cname in enumerate(CLASS_NAMES):
        d = DATA_DIR / cname
        files = sorted([p for p in d.glob("*")
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png")]) if d.is_dir() else []
        if not files:
            raise SystemExit(f"[data] no images in {d} — run the copy step first")
        items += [(p, lbl) for p in files]
    counts = np.bincount([l for _, l in items], minlength=N_CLS)
    print(f"[data] REAL crucible photos: {len(items)} imgs  per-class "
          + "  ".join(f"{c}={n}" for c, n in zip(CLASS_NAMES, counts)))
    return items, counts


class CrucibleDS(torch.utils.data.Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


def colour_report(items):
    """Mean RGB of the central 40% crop per class — quantifies loaded/done split."""
    print("\n[colour] central-crop mean RGB per class (loaded white vs done yellow):")
    by = {c: [] for c in range(N_CLS)}
    for p, y in items:
        a = np.asarray(Image.open(p).convert("RGB").resize((64, 64)), dtype=np.float32) / 255.0
        c = a[19:45, 19:45]                       # central ~40%
        by[y].append(c.reshape(-1, 3).mean(0))
    stats = {}
    for y in range(N_CLS):
        m = np.mean(by[y], 0)
        stats[y] = m
        rg_minus_b = (m[0] + m[1]) / 2 - m[2]
        print(f"  {CLASS_NAMES[y]:8s} R={m[0]:.3f} G={m[1]:.3f} B={m[2]:.3f}  "
              f"brightness={m.mean():.3f}  (R+G)/2-B={rg_minus_b:+.3f}  (yellowness)")
    # the discriminating axis for loaded vs done is yellowness ((R+G)/2 - B)
    if N_CLS >= 3:
        yl = (stats[1][0] + stats[1][1]) / 2 - stats[1][2]
        yd = (stats[2][0] + stats[2][1]) / 2 - stats[2][2]
        print(f"  -> loaded yellowness {yl:+.3f} vs done yellowness {yd:+.3f}  "
              f"(gap {yd - yl:+.3f}; done should be MORE yellow)")


def stratified_folds(items, k, seed=0):
    rng = np.random.default_rng(seed)
    by = {y: [i for i, (_, l) in enumerate(items) if l == y] for y in range(N_CLS)}
    for v in by.values():
        rng.shuffle(v)
    folds = [[] for _ in range(k)]
    for y, idxs in by.items():
        for j, i in enumerate(idxs):
            folds[j % k].append(i)
    return folds


def train_one(items, tr_idx, va_idx, args, dev, cls_w):
    tr = torch.utils.data.DataLoader(
        CrucibleDS([items[i] for i in tr_idx], TRAIN_TF),
        batch_size=args.bs, shuffle=True, num_workers=0)
    model = CrucibleCNN().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    lossf = nn.CrossEntropyLoss(weight=cls_w.to(dev), label_smoothing=0.05)
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
        sched.step()
    # eval (deterministic transform)
    model.eval()
    preds = []
    if va_idx is not None:
        va = torch.utils.data.DataLoader(CrucibleDS([items[i] for i in va_idx], EVAL_TF),
                                         batch_size=64)
        with torch.no_grad():
            for xb, yb in va:
                preds.append(model(xb.to(dev)).argmax(1).cpu().numpy())
        preds = np.concatenate(preds) if preds else np.array([])
    return model, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    items, counts = list_dataset()
    colour_report(items)
    cls_w = torch.tensor((counts.sum() / (N_CLS * np.maximum(counts, 1))).astype(np.float32))

    # ---- stratified k-fold CV (honest accuracy on so few images) ----
    print(f"\n[cv] stratified {args.folds}-fold (dev={dev}) ...")
    folds = stratified_folds(items, args.folds)
    cm = np.zeros((N_CLS, N_CLS), dtype=int)
    accs = []
    for f in range(args.folds):
        va_idx = folds[f]
        tr_idx = [i for g in range(args.folds) if g != f for i in folds[g]]
        _, preds = train_one(items, tr_idx, va_idx, args, dev, cls_w)
        gt = np.array([items[i][1] for i in va_idx])
        for t, p in zip(gt, preds):
            cm[t, p] += 1
        a = float((preds == gt).mean())
        accs.append(a)
        print(f"  fold{f}: val={len(va_idx)} acc={a*100:.1f}%")
    print(f"[cv] mean acc = {np.mean(accs)*100:.1f}% +/- {np.std(accs)*100:.1f}%")
    print("[cv] aggregated out-of-fold confusion (rows=true, cols=pred):")
    print("        " + " ".join(f"{c[:6]:>7}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c:>8} " + " ".join(f"{cm[i, j]:7d}" for j in range(N_CLS)))

    # ---- FINAL model on ALL data (for deployment) ----
    print("\n[final] training on ALL images for deployment ...")
    model, _ = train_one(items, list(range(len(items))), None, args, dev, cls_w)
    torch.save(model.state_dict(), HERE / "crucible_cnn.pt")
    cfg = {
        "arch": "CrucibleCNN", "in_ch": 3, "in_hw": IN_HW, "n_cls": N_CLS,
        "ch": [8, 16, 24, 32], "class_names": CLASS_NAMES,
        "input_norm": "none ([0,1] from RGB565/255)", "data_source": "real_phone",
        "cv_acc_mean": round(float(np.mean(accs)), 4),
        "cv_acc_std": round(float(np.std(accs)), 4),
        "n_images": int(len(items)),
        "per_class": {c: int(n) for c, n in zip(CLASS_NAMES, counts)},
    }
    (HERE / "crucible_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"[final] saved crucible_cnn.pt + crucible_config.json  "
          f"(CV {np.mean(accs)*100:.1f}%, classes={CLASS_NAMES})")


if __name__ == "__main__":
    main()
