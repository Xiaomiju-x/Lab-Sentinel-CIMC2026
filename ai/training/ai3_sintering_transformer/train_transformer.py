"""
AI-3 Sintering Curve TinyTransformer — Train + Evaluate + ONNX + TFLite Export

Architecture  (SEQ_LEN=64, N_FEAT=8, D_MODEL=32):
  Linear(8 → 32) + sinusoidal PE
  2 × [pre-norm TinySelfAttention(heads=2) + FFN(32→64→32, ReLU) + LayerNorm]
  Global average pool  →  Linear(32 → 5)
  ~17 K parameters  →  ~17 KB INT8

Run: cd CIMC/model/ai3_sintering_transformer && python train_transformer.py
Outputs:
  ai3_transformer.pt               PyTorch checkpoint + normalisation stats
  ai3_transformer.onnx             ONNX opset 11
  ai3_transformer_float32.tflite   TFLite float32 (if tinynn installed)
  ai3_config.json                  class names + norm stats for MCU
  eval_report.txt                  per-class accuracy + confusion matrix
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent

SEQ_LEN   = 64
N_FEAT    = 8
D_MODEL   = 64     # 32→64 for more capacity
N_HEAD    = 4      # 2→4 heads
N_LAYERS  = 2
DIM_FF    = 128    # 64→128 feedforward
N_CLASSES = 5
DROPOUT   = 0.1


# ── model ─────────────────────────────────────────────────────────────────────

class TinySelfAttention(nn.Module):
    """Manual multi-head attention — clean ONNX opset-11 export."""

    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head  = n_head
        self.d_head  = d_model // n_head
        self.scale   = self.d_head ** -0.5
        self.q_proj  = nn.Linear(d_model, d_model, bias=False)
        self.k_proj  = nn.Linear(d_model, d_model, bias=False)
        self.v_proj  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TinyTransformerBlock(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, d_model: int, n_head: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = TinySelfAttention(d_model, n_head)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TinyTransformer(nn.Module):
    """
    Sintering curve 5-class classifier.
    Input:  (batch, SEQ_LEN=64, N_FEAT=8)   z-scored temperature features
    Output: (batch, N_CLASSES=5)             raw logits
    ~17 K params
    """

    def __init__(self,
                 n_feat   = N_FEAT,
                 d_model  = D_MODEL,
                 n_head   = N_HEAD,
                 n_layers = N_LAYERS,
                 dim_ff   = DIM_FF,
                 n_classes= N_CLASSES,
                 seq_len  = SEQ_LEN,
                 dropout  = DROPOUT):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)

        # Fixed sinusoidal PE baked as a buffer — seq_len is constant for MCU
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, d_model)

        self.blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, n_head, dim_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) + self.pe          # (B, T, D)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)        # global average pool → (B, D)
        return self.head(x)                 # (B, n_classes)


# ── data helpers ──────────────────────────────────────────────────────────────

def load_or_generate():
    X_path = OUT_DIR / "X_seq.npy"
    y_path = OUT_DIR / "y_seq.npy"
    if not X_path.exists() or not y_path.exists():
        print("X_seq.npy not found — running synth_data_ai3.py ...")
        from synth_data_ai3 import generate
        generate(out_dir=OUT_DIR)
    X_tr = np.load(X_path)
    y_tr = np.load(y_path)
    X_te = np.load(OUT_DIR / "X_seq_test.npy") if (OUT_DIR / "X_seq_test.npy").exists() else None
    y_te = np.load(OUT_DIR / "y_seq_test.npy") if (OUT_DIR / "y_seq_test.npy").exists() else None
    return X_tr, y_tr, X_te, y_te


# ── training ──────────────────────────────────────────────────────────────────

def train_model(X: np.ndarray, y: np.ndarray,
                epochs: int = 150, batch: int = 64, lr: float = 1e-3):
    n     = len(X)
    n_val = max(int(n * 0.15), 64)
    idx   = np.random.permutation(n)

    X_tr  = torch.tensor(X[idx[n_val:]]).to(DEVICE)
    y_tr  = torch.tensor(y[idx[n_val:]], dtype=torch.long).to(DEVICE)
    X_val = torch.tensor(X[idx[:n_val]]).to(DEVICE)
    y_val = torch.tensor(y[idx[:n_val]], dtype=torch.long).to(DEVICE)

    # class weights to handle slight imbalance
    counts  = np.bincount(y[idx[n_val:]], minlength=N_CLASSES).astype(np.float32)
    weights = torch.tensor(counts.sum() / (N_CLASSES * counts + 1e-8)).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch, shuffle=True)
    model  = TinyTransformer().to(DEVICE)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.CrossEntropyLoss(weight=weights)

    best_val_acc, best_state, patience, no_imp = 0.0, None, 30, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss, tr_ok = 0.0, 0
        for xb, yb in loader:
            logits = model(xb)
            loss   = crit(logits, yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * len(xb)
            tr_ok   += (logits.argmax(1) == yb).sum().item()
        tr_loss /= len(X_tr)
        tr_acc   = tr_ok / len(X_tr)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_acc    = (val_logits.argmax(1) == y_val).float().mean().item()

        sched.step()

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{epochs}  train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}"
                  f"  tr_loss={tr_loss:.4f}")

        if val_acc > best_val_acc + 1e-5:
            best_val_acc = val_acc
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp       = 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop @ epoch {ep}  best_val_acc={best_val_acc:.4f}")
                break

    print(f"Training done in {time.time()-t0:.1f}s  best_val_acc={best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return model, best_val_acc


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             class_names: list, report: list, label: str = ""):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X).to(DEVICE)).cpu().numpy()
    preds = logits.argmax(axis=1)
    acc   = float((preds == y).mean())

    tag = f"[{label}] " if label else ""
    report.append(f"\n{tag}Overall accuracy: {acc*100:.2f}%  (n={len(y)})")
    report.append(f"\n{'Class':15s}  {'Prec':>7s}  {'Rec':>7s}  {'F1':>7s}  {'N':>6s}")
    report.append("-" * 50)

    for i, name in enumerate(class_names):
        mask  = (y == i)
        pred_mask = (preds == i)
        n_true = mask.sum()
        tp = (mask & pred_mask).sum()
        fp = (~mask & pred_mask).sum()
        fn = (mask & ~pred_mask).sum()
        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)
        report.append(f"  {name:13s}  {prec:7.3f}  {rec:7.3f}  {f1:7.3f}  {n_true:6d}")

    # confusion matrix (manual, no sklearn needed)
    report.append(f"\n{tag}Confusion matrix (rows=true, cols=pred):")
    header = " " * 15 + "".join(f"{c[:9]:>10s}" for c in class_names)
    report.append(header)
    for i, name in enumerate(class_names):
        row_str = f"  {name[:13]:13s}"
        for j in range(len(class_names)):
            cnt = int(((y == i) & (preds == j)).sum())
            row_str += f"{cnt:10d}"
        report.append(row_str)

    return acc


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: nn.Module, path: Path):
    model.eval().cpu()
    dummy = torch.zeros(1, SEQ_LEN, N_FEAT)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["temperature_sequence"],
        output_names=["class_logits"],
        opset_version=11,
        dynamic_axes={
            "temperature_sequence": {0: "batch"},
            "class_logits":         {0: "batch"},
        },
    )
    print(f"ONNX exported: {path}")


# ── TFLite export ─────────────────────────────────────────────────────────────

def export_tflite(model: nn.Module, path: Path):
    try:
        from tinynn.converter import TFLiteConverter
    except ImportError:
        print("tinynn not found — skip TFLite export (pip install tinynn)")
        return
    model.eval().cpu()
    dummy = torch.zeros(1, SEQ_LEN, N_FEAT)
    converter = TFLiteConverter(model=model, dummy_input=dummy, tflite_path=str(path))
    converter.convert()
    size_kb = path.stat().st_size / 1024
    print(f"TFLite exported: {path}  ({size_kb:.1f} KB)")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Device: {DEVICE}")

    # 1. data
    print("\n[1] Loading / generating data ...")
    X_tr, y_tr, X_te, y_te = load_or_generate()
    print(f"    Train: {X_tr.shape}  class counts: {np.bincount(y_tr).tolist()}")
    if X_te is not None:
        print(f"    Test:  {X_te.shape}  class counts: {np.bincount(y_te).tolist()}")

    stats_path = OUT_DIR / "ai3_data_stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}
    class_names = stats.get("class_names", ["normal","fast_ramp","undertemp","temp_drift","slow_ramp"])
    mu  = np.array(stats["mu"],  dtype=np.float32) if "mu"  in stats else None
    std = np.array(stats["std"], dtype=np.float32) if "std" in stats else None

    # 2. train
    print("\n[2] Training TinyTransformer ...")
    n_params = sum(p.numel() for p in TinyTransformer().parameters())
    print(f"    Parameters: {n_params:,d}  (~{n_params * 1 // 1024} KB INT8)")
    model, best_val_acc = train_model(X_tr, y_tr)

    torch.save({
        "model":       model.state_dict(),
        "mu":          mu,
        "std":         std,
        "class_names": class_names,
        "seq_len":     SEQ_LEN,
        "n_feat":      N_FEAT,
    }, OUT_DIR / "ai3_transformer.pt")
    print(f"    Saved → {OUT_DIR}/ai3_transformer.pt")

    # 3. evaluate
    print("\n[3] Evaluation ...")
    report = ["AI-3 Sintering Curve TinyTransformer — Evaluation Report", "=" * 62]
    tr_acc = evaluate(model, X_tr, y_tr, class_names, report, label="train")
    if X_te is not None:
        te_acc = evaluate(model, X_te, y_te, class_names, report, label="test")
    else:
        te_acc = tr_acc

    report_txt = "\n".join(report) + "\n"
    (OUT_DIR / "eval_report.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)

    # 4. config for MCU
    config = {
        "seq_len":     SEQ_LEN,
        "n_feat":      N_FEAT,
        "d_model":     D_MODEL,
        "n_head":      N_HEAD,
        "n_layers":    N_LAYERS,
        "dim_ff":      DIM_FF,
        "n_classes":   N_CLASSES,
        "class_names": class_names,
        "feat_names":  stats.get("feat_names", [f"feat_{i}" for i in range(N_FEAT)]),
        "feat_idx":    stats.get("feat_idx",   list(range(N_FEAT))),
        "mu":          (mu.tolist()  if mu  is not None else None),
        "std":         (std.tolist() if std is not None else None),
        "best_val_acc": float(best_val_acc),
        "test_acc":     float(te_acc),
    }
    (OUT_DIR / "ai3_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    print(f"Config saved → {OUT_DIR}/ai3_config.json")

    # 5. ONNX
    print("\n[4] Exporting ONNX ...")
    export_onnx(model, OUT_DIR / "ai3_transformer.onnx")

    # 6. TFLite
    print("\n[5] Exporting TFLite ...")
    export_tflite(model, OUT_DIR / "ai3_transformer_float32.tflite")

    print("\n=== All done ===")
    print(f"  ai3_transformer.pt              PyTorch checkpoint + norm stats")
    print(f"  ai3_transformer.onnx            ONNX opset 11")
    print(f"  ai3_transformer_float32.tflite  TFLite float32 (if tinynn installed)")
    print(f"  ai3_config.json                 class names + norm stats for MCU")
    print(f"  eval_report.txt                 per-class accuracy + confusion matrix")


if __name__ == "__main__":
    main()
