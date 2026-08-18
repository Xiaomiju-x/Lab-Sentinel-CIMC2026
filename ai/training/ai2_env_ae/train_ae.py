"""
AI-2 Sintering AE — Train + Conformal Calibration + ONNX + TFLite INT8 export

Run: cd CIMC/model/ai2_env_ae && python train_ae.py
Outputs:
  ai2_ae.pt              PyTorch checkpoint
  ai2_ae.onnx            ONNX opset 11 (for GD32 AI Tool / TinyNeuralNetwork)
  ai2_ae_float32.tflite  TFLite float32 (via TinyNeuralNetwork)
  ai2_ae_q_hat.json      Conformal q_hat (90%) + feature normalisation stats
  eval_report.txt        Anomaly detection metrics on X_anomaly.npz
"""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = Path(__file__).parent


# ── model ─────────────────────────────────────────────────────────────────────
class SinterAE(nn.Module):
    """32-16-8-16-32 symmetric AE. ~5 KB INT8."""

    def __init__(self, in_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 16), nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, in_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# ── data helpers ──────────────────────────────────────────────────────────────
def load_or_generate():
    npy = OUT_DIR / "X_normal.npy"
    if not npy.exists():
        print("X_normal.npy not found — running synth_data.py ...")
        from synth_data import generate
        generate()
    return np.load(npy)


def normalise(X: np.ndarray):
    """Z-score normalisation per feature. Returns (X_norm, mean, std)."""
    mu = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    return ((X - mu) / std).astype(np.float32), mu.astype(np.float32), std.astype(np.float32)


# ── training ──────────────────────────────────────────────────────────────────
def train(X_norm: np.ndarray, epochs: int = 150, batch: int = 64, lr: float = 1e-3):
    n = len(X_norm)
    n_val = max(int(n * 0.15), 64)
    idx = np.random.permutation(n)
    X_tr = torch.tensor(X_norm[idx[n_val:]])
    X_val = torch.tensor(X_norm[idx[:n_val]])

    loader = DataLoader(TensorDataset(X_tr), batch_size=batch, shuffle=True)
    model = SinterAE().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    best_val, best_state, patience, no_imp = math.inf, None, 20, 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            loss = criterion(model(xb), xb)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(X_tr)

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val.to(DEVICE)), X_val.to(DEVICE)).item()

        sched.step()

        if ep % 20 == 0 or ep == 1:
            print(f"  Epoch {ep:4d}/{epochs}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val - 1e-7:
            best_val, best_state, no_imp = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_imp += 1
            if no_imp >= patience:
                print(f"  Early stop at epoch {ep} (best val={best_val:.6f})")
                break

    print(f"Training done in {time.time()-t0:.1f}s  best_val_mse={best_val:.6f}")
    model.load_state_dict(best_state)
    return model


# ── conformal calibration ─────────────────────────────────────────────────────
def conformal_calibrate(model: nn.Module, X_norm: np.ndarray, alpha: float = 0.10):
    """
    Split Conformal CI (1-alpha = 90% coverage).
    Returns q_hat: scalar threshold; errors > q_hat → anomaly.
    Also returns per-feature mean absolute error for attribution.
    """
    model.eval()
    X_t = torch.tensor(X_norm).to(DEVICE)
    with torch.no_grad():
        X_hat = model(X_t).cpu().numpy()
    errors = np.mean((X_norm - X_hat) ** 2, axis=1)   # MSE per sample
    n = len(errors)
    q_level = math.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(errors, q_level))

    per_feat_mae = float(np.mean(np.abs(X_norm - X_hat)))
    feat_mae = np.mean(np.abs(X_norm - X_hat), axis=0).tolist()

    print(f"Conformal q_hat (90% CI): {q_hat:.6f}  (n={n}, q_level={q_level:.4f})")
    return q_hat, feat_mae, per_feat_mae


# ── anomaly evaluation ────────────────────────────────────────────────────────
def eval_anomaly(model: nn.Module, X_norm_train: np.ndarray, mu, std,
                 q_hat: float, report_lines: list):
    npz_path = OUT_DIR / "X_anomaly.npz"
    if not npz_path.exists():
        report_lines.append("X_anomaly.npz not found — skip anomaly eval")
        return

    data = np.load(npz_path)
    model.eval()

    report_lines.append(f"\n{'Run':35s}  {'mean_score':>12s}  {'detected%':>10s}")
    report_lines.append("-" * 62)

    # normal baseline score
    X_t = torch.tensor(X_norm_train).to(DEVICE)
    with torch.no_grad():
        X_hat = model(X_t).cpu().numpy()
    normal_scores = np.mean((X_norm_train - X_hat) ** 2, axis=1)
    normal_det = float((normal_scores > q_hat).mean())
    report_lines.append(f"  {'[normal_train]':33s}  {normal_scores.mean():12.6f}  {normal_det*100:9.1f}%  (FPR)")

    for key in sorted(data.files):
        X_anom_raw = data[key].astype(np.float32)
        X_anom_norm = ((X_anom_raw - mu) / std).astype(np.float32)
        X_t = torch.tensor(X_anom_norm).to(DEVICE)
        with torch.no_grad():
            X_hat = model(X_t).cpu().numpy()
        scores = np.mean((X_anom_norm - X_hat) ** 2, axis=1)
        detected = float((scores > q_hat).mean())
        report_lines.append(f"  {key:33s}  {scores.mean():12.6f}  {detected*100:9.1f}%")


# ── export ONNX ───────────────────────────────────────────────────────────────
def export_onnx(model: nn.Module, path: Path):
    model.eval().cpu()
    dummy = torch.zeros(1, 32)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=["features"], output_names=["reconstruction"],
        opset_version=11,
        dynamic_axes={"features": {0: "batch"}, "reconstruction": {0: "batch"}},
    )
    print(f"ONNX exported: {path}")


# ── export TFLite via TinyNeuralNetwork ───────────────────────────────────────
def export_tflite(model: nn.Module, path: Path):
    try:
        from tinynn.converter import TFLiteConverter
    except ImportError:
        print("tinynn not found — skip TFLite export (run: pip install tinynn or use mace_env)")
        return

    model.eval().cpu()
    dummy = torch.zeros(1, 32)
    converter = TFLiteConverter(model=model, dummy_input=dummy,
                                tflite_path=str(path))
    converter.convert()
    size_kb = path.stat().st_size / 1024
    print(f"TFLite exported: {path}  ({size_kb:.1f} KB)")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # 1. data
    print("\n[1] Loading / generating synthetic data ...")
    X_raw = load_or_generate()
    X_norm, mu, std = normalise(X_raw)
    print(f"    shape={X_norm.shape}  mean={X_norm.mean():.4f}  std={X_norm.std():.4f}")

    # 2. train
    print("\n[2] Training SinterAE ...")
    model = train(X_norm)
    torch.save({"model": model.state_dict(), "mu": mu, "std": std},
               OUT_DIR / "ai2_ae.pt")
    print(f"    Saved → {OUT_DIR}/ai2_ae.pt")

    # 3. conformal
    print("\n[3] Conformal calibration (90% CI) ...")
    q_hat, feat_mae, global_mae = conformal_calibrate(model, X_norm)
    q_hat_data = {
        "q_hat_90": q_hat,
        "global_mae": global_mae,
        "feat_mae": feat_mae,
        "mu": mu.tolist(),
        "std": std.tolist(),
        "feat_names": [
            "temp_current","temp_target","temp_deviation","temp_gradient",
            "temp_rms_60s","temp_std_60s","hold_min_cum","stage_id",
            "mq135_raw","mq135_avg5","atmosphere_code","mq135_std",
            "mq135_trend","atm_match",
            "room_temp","room_humidity","room_temp_grad","room_hum_grad",
            "vib_x_rms","vib_y_rms","vib_z_rms",
            "vib_x_cen","vib_y_cen","vib_z_cen","vib_energy","grinding_flag",
            "vis_cls0","vis_cls1","vis_cls2","vis_cls3",
            "progress","cum_dev",
        ],
    }
    (OUT_DIR / "ai2_ae_q_hat.json").write_text(
        json.dumps(q_hat_data, indent=2), encoding="utf-8")
    print(f"    Saved → {OUT_DIR}/ai2_ae_q_hat.json")

    # 4. anomaly eval
    print("\n[4] Anomaly evaluation ...")
    report = ["AI-2 Sintering AE — Anomaly Evaluation Report", "=" * 62]
    eval_anomaly(model, X_norm, mu, std, q_hat, report)
    report_txt = "\n".join(report) + "\n"
    (OUT_DIR / "eval_report.txt").write_text(report_txt, encoding="utf-8")
    print(report_txt)

    # 5. ONNX
    print("\n[5] Exporting ONNX ...")
    export_onnx(model, OUT_DIR / "ai2_ae.onnx")

    # 6. TFLite
    print("\n[6] Exporting TFLite ...")
    export_tflite(model, OUT_DIR / "ai2_ae_float32.tflite")

    print("\n=== All done ===")
    print(f"  ai2_ae.pt              PyTorch checkpoint + normalisation stats")
    print(f"  ai2_ae.onnx            ONNX opset 11")
    print(f"  ai2_ae_float32.tflite  TFLite float32 (if tinynn installed)")
    print(f"  ai2_ae_q_hat.json      Conformal q_hat + feat names for MCU")
    print(f"  eval_report.txt        Anomaly detection report")


if __name__ == "__main__":
    main()
