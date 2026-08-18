"""
AI-5 Root-Cause Diagnoser — Train + Export

Architecture (N_FEAT=27 -> N_CLASSES=9):
  Linear(27 -> 32) -> BatchNorm -> ReLU -> Dropout(0.2)
  Linear(32 -> 16) -> BatchNorm -> ReLU -> Dropout(0.2)
  Linear(16 -> 9)
  ~1.6 K params -> ~6 KB float32 in Flash

AI-5 maps the upstream-AI signature (AI-1/2/3/4 outputs + raw furnace/gas/humidity
context, 27-D) to one of 9 NAMED process root causes (taxonomy.json). It is the
"why + what-to-do" layer on top of the "what's-wrong" detectors.

Run: cd CIMC/model/ai5_rootcause && python train_ai5.py
Outputs:
  ai5_diagnose.pt    PyTorch checkpoint
  ai5_diagnose.onnx  ONNX opset 11
  ai5_config.json    class names + feature layout + accuracy for the MCU
  eval_report.txt    per-class precision/recall/F1 + confusion matrix
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent
N_FEAT, N_CLASSES = 27, 9


class DiagMLP(nn.Module):
    def __init__(self, n_feat=N_FEAT, n_classes=N_CLASSES, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),     nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_or_generate():
    if not (OUT_DIR / "X_ai5.npy").exists():
        print("X_ai5.npy not found - running synth_data_ai5.py ...")
        from synth_data_ai5 import generate
        generate(out_dir=OUT_DIR)
    return (np.load(OUT_DIR / "X_ai5.npy"), np.load(OUT_DIR / "y_ai5.npy"),
            np.load(OUT_DIR / "X_ai5_test.npy"), np.load(OUT_DIR / "y_ai5_test.npy"))


def train_model(X, y, epochs=250, batch=128, lr=5e-3):
    n = len(X); n_val = max(int(n * 0.15), 64)
    idx = np.random.permutation(n)
    Xtr = torch.tensor(X[idx[n_val:]]).to(DEVICE)
    ytr = torch.tensor(y[idx[n_val:]], dtype=torch.long).to(DEVICE)
    Xv = torch.tensor(X[idx[:n_val]]).to(DEVICE)
    yv = torch.tensor(y[idx[:n_val]], dtype=torch.long).to(DEVICE)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    model = DiagMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best, best_state, patience, no_imp = 0.0, None, 30, 0
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            loss = crit(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            va = (model(Xv).argmax(1) == yv).float().mean().item()
        sched.step()
        if ep % 50 == 0 or ep == 1:
            print(f"  epoch {ep:4d}/{epochs}  val_acc={va:.4f}")
        if va > best + 1e-5:
            best, best_state, no_imp = va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  early stop @ {ep}  best_val_acc={best:.4f}"); break
    print(f"trained in {time.time()-t0:.1f}s  best_val_acc={best:.4f}")
    model.load_state_dict(best_state)
    return model, best


def evaluate(model, X, y, names, report, label=""):
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X).to(DEVICE)).argmax(1).cpu().numpy()
    acc = float((preds == y).mean())
    report.append(f"\n[{label}] overall accuracy: {acc*100:.2f}%  (n={len(y)})")
    report.append(f"\n{'Class':20s} {'Prec':>7s} {'Rec':>7s} {'F1':>7s} {'N':>6s}")
    report.append("-" * 52)
    for i, nm in enumerate(names):
        m = (y == i); pm = (preds == i)
        tp = (m & pm).sum(); fp = (~m & pm).sum(); fn = (m & ~pm).sum()
        prec = tp / (tp + fp + 1e-9); rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        report.append(f"  {nm:18s} {prec:7.3f} {rec:7.3f} {f1:7.3f} {int(m.sum()):6d}")
    report.append(f"\n[{label}] confusion (rows=true, cols=pred):")
    report.append(" " * 20 + "".join(f"{i:>5d}" for i in range(len(names))))
    for i, nm in enumerate(names):
        row = f"  {i} {nm[:15]:15s}"
        for j in range(len(names)):
            row += f"{int(((y==i)&(preds==j)).sum()):5d}"
        report.append(row)
    return acc


def export_onnx(model, path):
    """Optional (for the GD32-AI-Tool/TFLite claim); the firmware C export reads
    the .pt directly, so a missing `onnx` package is non-fatal."""
    try:
        model.eval().cpu()
        torch.onnx.export(model, torch.zeros(1, N_FEAT), str(path),
                          input_names=["diag_features"], output_names=["rootcause_logits"],
                          opset_version=11,
                          dynamic_axes={"diag_features": {0: "batch"}, "rootcause_logits": {0: "batch"}})
        print(f"ONNX exported: {path}")
    except Exception as e:
        print(f"ONNX export skipped ({type(e).__name__}: {e}); regenerate in mace_env if needed.")


def main():
    print(f"Device: {DEVICE}")
    Xtr, ytr, Xte, yte = load_or_generate()
    info = json.loads((OUT_DIR / "ai5_data_info.json").read_text(encoding="utf-8"))
    names = info["class_names"]
    print(f"train {Xtr.shape} counts {np.bincount(ytr).tolist()}")
    n_params = sum(p.numel() for p in DiagMLP().parameters())
    print(f"params: {n_params:,d} (~{max(1, n_params*4//1024)} KB float32)")

    report = ["AI-5 Root-Cause Diagnoser - Evaluation Report", "=" * 56,
              f"params={n_params}  arch=27-32-16-9 (BN folded at export)"]
    model, best_val = train_model(Xtr, ytr)
    torch.save({"model": model.state_dict(), "class_names": names}, OUT_DIR / "ai5_diagnose.pt")
    evaluate(model, Xtr, ytr, names, report, "train")
    te_acc = evaluate(model, Xte, yte, names, report, "test")
    (OUT_DIR / "eval_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("\n".join(report))

    cfg = {"n_feat": N_FEAT, "n_classes": N_CLASSES, "class_names": names,
           "feat_layout": info.get("feat_layout", {}), "gas_enum": info.get("gas_enum"),
           "best_val_acc": float(best_val), "test_acc": float(te_acc),
           "accuracy_benchmark": "synthetic process-root-cause test set (9 grounded classes)",
           "role": "diagnoses WHY a run is anomalous (named root cause + action), on top "
                   "of AI-1/2/3/4 which detect THAT it is anomalous",
           "taxonomy": "CIMC/model/ai5_rootcause/taxonomy.json (measure-first gated)"}
    (OUT_DIR / "ai5_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"config -> {OUT_DIR}/ai5_config.json  test_acc={te_acc*100:.2f}%")
    export_onnx(model, OUT_DIR / "ai5_diagnose.onnx")


if __name__ == "__main__":
    main()
