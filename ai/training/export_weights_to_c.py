"""
export_weights_to_c.py  —  CIMC Lab-Sentinel
================================================================
Extract trained weights of AI-1/2/3/4 into hand-written-C inference
headers for the GD32H759 firmware (firmware/ai_models_c/).

Why hand-written float32 C (not GD32 AI Tool):
  GD32 Embedded AI Tool GUI produced 0-byte .ai files (Phase 0 note).
  ADR-1 sanctioned fallback = hand-written CMSIS-NN / float kernels.
  These models are KB-scale; float32 on the M7 FPU @600MHz runs each
  in <5 ms and reproduces PyTorch byte-for-byte.

Outputs (into firmware/ai_models_c/):
  ai1_cnn_weights.h          AI-1 TinyCNN (MNIST weights; crucible head pending data)
  ai2_ae_weights.h           AI-2 Sintering AE + mu/std + conformal q_hat + feat_mae
  ai3_transformer_weights.h  AI-3 TinyTransformer + mu/std
  ai4_fusion_weights.h       AI-4 Fusion MLP (BatchNorm folded into Linear)
  ai_golden.h                Deterministic input + expected output for on-chip self-test

Run:  cd CIMC/model && python export_weights_to_c.py
      (needs torch; use mace_env)

No fabricated constants: every number below is a trained-weight value or a
documented architecture dim from the matching train_*.py.
"""

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).parent
OUT  = HERE.parent / "firmware" / "ai_models_c"
OUT.mkdir(parents=True, exist_ok=True)

GOLDEN_SEED = 1234


# ───────────────────────── model definitions (must match train_*.py) ─────────

class TinyCNN(nn.Module):            # AI-1  (train_mnist_hello.py)
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(4, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(8 * 7 * 7, 32), nn.ReLU(), nn.Linear(32, num_classes),
        )

    def forward(self, x):
        f = self.features(x)
        emb = self.classifier[1](self.classifier[0](f))   # Linear(392->32)
        emb = self.classifier[2](emb)                      # ReLU  -> 32D embedding
        out = self.classifier[3](emb)                      # Linear(32->10)
        return out, emb


class SinterAE(nn.Module):           # AI-2  (train_ae.py)
    def __init__(self, in_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(in_dim, 16), nn.ReLU(), nn.Linear(16, 8))
        self.decoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, in_dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TinySelfAttention(nn.Module):  # AI-3  (train_transformer.py)
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.scale  = self.d_head ** -0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out  = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class TinyTransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = TinySelfAttention(d_model, n_head)
        self.ff    = nn.Sequential(nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Linear(dim_ff, d_model))
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, n_feat=8, d_model=64, n_head=4, n_layers=2, dim_ff=128,
                 n_classes=5, seq_len=64, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(n_feat, d_model)
        pe  = torch.zeros(seq_len, d_model)
        pos = torch.arange(seq_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.blocks = nn.ModuleList([TinyTransformerBlock(d_model, n_head, dim_ff, dropout)
                                     for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.proj(x) + self.pe
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        return self.head(x)


class FusionMLP(nn.Module):          # AI-4  (train_fusion_mlp.py)
    def __init__(self, n_feat=16, n_classes=4, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 16),     nn.BatchNorm1d(16), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ───────────────────────── C emit helpers ─────────────────────────

def carr(name, arr):
    """Format a numpy array as a flat C float array initialiser string."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    vals = ", ".join(f"{float(v):.8e}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{ {vals} }};\n"


def header_top(macro, title):
    return (f"/* {title}\n"
            " * AUTO-GENERATED by CIMC/model/export_weights_to_c.py — do not edit by hand.\n"
            " * Weights are trained float32 values; stored in Flash (.rodata).\n"
            " */\n"
            f"#ifndef {macro}\n#define {macro}\n\n")


def header_bot(macro):
    return f"\n#endif /* {macro} */\n"


def fold_bn_into_linear(lin_w, lin_b, bn_w, bn_b, bn_mean, bn_var, eps=1e-5):
    """Fold BatchNorm1d (applied AFTER a Linear) back into that Linear.
       y = bn(Wx+b) = gamma*((Wx+b)-mean)/sqrt(var+eps) + beta
       -> W' = (gamma/sqrt(var+eps))[:,None] * W ;  b' = (b-mean)*s + beta
    """
    s = bn_w / np.sqrt(bn_var + eps)
    Wf = lin_w * s[:, None]
    bf = (lin_b - bn_mean) * s + bn_b
    return Wf.astype(np.float32), bf.astype(np.float32)


def sd_np(sd, key):
    return sd[key].detach().cpu().numpy().astype(np.float32)


# ───────────────────────── per-model export ─────────────────────────

def export_ai1(golden):
    sd = torch.load(HERE / "ai1_vision_cnn" / "tiny_cnn_mnist.pt", map_location="cpu", weights_only=True)
    m = TinyCNN(); m.load_state_dict(sd); m.eval()

    rng = np.random.default_rng(GOLDEN_SEED)
    x = rng.uniform(0.0, 1.0, size=(1, 1, 28, 28)).astype(np.float32)
    with torch.no_grad():
        logits, emb = m(torch.from_numpy(x))
    golden["ai1_input"]  = x.reshape(-1)
    golden["ai1_logits"] = logits.numpy().reshape(-1)
    golden["ai1_emb"]    = emb.numpy().reshape(-1)

    s = header_top("AI1_CNN_WEIGHTS_H", "AI-1 TinyCNN weights (Conv4-MP-Conv8-MP-FC32-FC10, MNIST 97.3%)")
    s += "/* Architecture: 1x28x28 ->[conv3x3 pad1]4x28x28 ->relu ->mp2 4x14x14\n"
    s += " *               ->[conv3x3 pad1]8x14x14 ->relu ->mp2 8x7x7 ->flatten 392\n"
    s += " *               ->fc 32 ->relu(=embedding) ->fc 10 (logits)\n"
    s += " * NOTE: weights are MNIST digits; crucible 4-class head awaits camera+dataset.\n"
    s += " *       The on-chip CNN engine itself is validated by ai_golden.h. */\n\n"
    s += "#define AI1_IN_CH 1\n#define AI1_IN_H 28\n#define AI1_IN_W 28\n"
    s += "#define AI1_C0 4\n#define AI1_C1 8\n#define AI1_FLAT 392\n#define AI1_EMB 32\n#define AI1_NCLS 10\n\n"
    s += carr("ai1_conv0_w", sd_np(sd, "features.0.weight"))   # [4,1,3,3]
    s += carr("ai1_conv0_b", sd_np(sd, "features.0.bias"))
    s += carr("ai1_conv1_w", sd_np(sd, "features.3.weight"))   # [8,4,3,3]
    s += carr("ai1_conv1_b", sd_np(sd, "features.3.bias"))
    s += carr("ai1_fc0_w",   sd_np(sd, "classifier.1.weight")) # [32,392]
    s += carr("ai1_fc0_b",   sd_np(sd, "classifier.1.bias"))
    s += carr("ai1_fc1_w",   sd_np(sd, "classifier.3.weight")) # [10,32]
    s += carr("ai1_fc1_b",   sd_np(sd, "classifier.3.bias"))
    s += header_bot("AI1_CNN_WEIGHTS_H")
    (OUT / "ai1_cnn_weights.h").write_text(s, encoding="utf-8")
    print("  ai1_cnn_weights.h written")


def export_ai2(golden):
    ck = torch.load(HERE / "ai2_env_ae" / "ai2_ae.pt", map_location="cpu", weights_only=True)
    sd = ck["model"]
    m = SinterAE(); m.load_state_dict(sd); m.eval()
    qh = json.loads((HERE / "ai2_env_ae" / "ai2_ae_q_hat.json").read_text(encoding="utf-8"))
    mu  = np.array(qh["mu"],  dtype=np.float32)
    std = np.array(qh["std"], dtype=np.float32)
    feat_mae = np.array(qh["feat_mae"], dtype=np.float32)
    q_hat = float(qh["q_hat_90"])

    rng = np.random.default_rng(GOLDEN_SEED + 2)
    # golden input is in NORMALISED space (what the AE actually sees)
    xn = rng.standard_normal((1, 32)).astype(np.float32)
    with torch.no_grad():
        rec = m(torch.from_numpy(xn)).numpy().reshape(-1)
    mse = float(np.mean((xn.reshape(-1) - rec) ** 2))
    golden["ai2_input_norm"] = xn.reshape(-1)
    golden["ai2_recon"]      = rec
    golden["ai2_mse"]        = np.array([mse], dtype=np.float32)

    s = header_top("AI2_AE_WEIGHTS_H", "AI-2 Sintering AE 32-16-8-16-32 + conformal q_hat")
    s += "#define AI2_DIM 32\n#define AI2_H1 16\n#define AI2_H2 8\n\n"
    s += f"#define AI2_QHAT_90 {q_hat:.8e}f   /* MSE>q_hat => anomaly (90%% CI) */\n\n"
    s += carr("ai2_enc0_w", sd_np(sd, "encoder.0.weight"))  # [16,32]
    s += carr("ai2_enc0_b", sd_np(sd, "encoder.0.bias"))
    s += carr("ai2_enc1_w", sd_np(sd, "encoder.2.weight"))  # [8,16]
    s += carr("ai2_enc1_b", sd_np(sd, "encoder.2.bias"))
    s += carr("ai2_dec0_w", sd_np(sd, "decoder.0.weight"))  # [16,8]
    s += carr("ai2_dec0_b", sd_np(sd, "decoder.0.bias"))
    s += carr("ai2_dec1_w", sd_np(sd, "decoder.2.weight"))  # [32,16]
    s += carr("ai2_dec1_b", sd_np(sd, "decoder.2.bias"))
    s += carr("ai2_mu",  mu)
    s += carr("ai2_std", std)
    s += carr("ai2_feat_mae", feat_mae)   # baseline per-feature MAE for attribution normalisation
    s += header_bot("AI2_AE_WEIGHTS_H")
    (OUT / "ai2_ae_weights.h").write_text(s, encoding="utf-8")
    print("  ai2_ae_weights.h written")


def export_ai3(golden):
    ck = torch.load(HERE / "ai3_sintering_transformer" / "ai3_transformer.pt", map_location="cpu", weights_only=True)
    sd = ck["model"]
    m = TinyTransformer(); m.load_state_dict(sd); m.eval()
    cfg = json.loads((HERE / "ai3_sintering_transformer" / "ai3_config.json").read_text(encoding="utf-8"))
    mu  = np.array(cfg["mu"],  dtype=np.float32)
    std = np.array(cfg["std"], dtype=np.float32)

    rng = np.random.default_rng(GOLDEN_SEED + 3)
    # golden input in NORMALISED space (64 x 8)
    xn = rng.standard_normal((1, 64, 8)).astype(np.float32)
    with torch.no_grad():
        logits = m(torch.from_numpy(xn)).numpy().reshape(-1)
    golden["ai3_input_norm"] = xn.reshape(-1)
    golden["ai3_logits"]     = logits

    D, H, FF = 64, 4, 128
    s = header_top("AI3_TRANSFORMER_WEIGHTS_H", "AI-3 Sintering-curve TinyTransformer (seq64 x feat8, 2 blocks, 4 heads)")
    s += f"#define AI3_SEQ 64\n#define AI3_FEAT 8\n#define AI3_DMODEL {D}\n#define AI3_NHEAD {H}\n"
    s += f"#define AI3_DHEAD {D // H}\n#define AI3_FF {FF}\n#define AI3_NLAYER 2\n#define AI3_NCLS 5\n\n"
    s += carr("ai3_proj_w", sd_np(sd, "proj.weight"))   # [64,8]
    s += carr("ai3_proj_b", sd_np(sd, "proj.bias"))
    s += carr("ai3_pe",     sd_np(sd, "pe").reshape(64, 64))   # positional encoding [64,64]
    for b in range(2):
        p = f"blocks.{b}."
        s += carr(f"ai3_b{b}_n1_w", sd_np(sd, p + "norm1.weight"))
        s += carr(f"ai3_b{b}_n1_b", sd_np(sd, p + "norm1.bias"))
        s += carr(f"ai3_b{b}_q_w",  sd_np(sd, p + "attn.q_proj.weight"))   # [64,64] no bias
        s += carr(f"ai3_b{b}_k_w",  sd_np(sd, p + "attn.k_proj.weight"))
        s += carr(f"ai3_b{b}_v_w",  sd_np(sd, p + "attn.v_proj.weight"))
        s += carr(f"ai3_b{b}_o_w",  sd_np(sd, p + "attn.out_proj.weight"))
        s += carr(f"ai3_b{b}_o_b",  sd_np(sd, p + "attn.out_proj.bias"))
        s += carr(f"ai3_b{b}_n2_w", sd_np(sd, p + "norm2.weight"))
        s += carr(f"ai3_b{b}_n2_b", sd_np(sd, p + "norm2.bias"))
        s += carr(f"ai3_b{b}_ff0_w", sd_np(sd, p + "ff.0.weight"))         # [128,64]
        s += carr(f"ai3_b{b}_ff0_b", sd_np(sd, p + "ff.0.bias"))
        s += carr(f"ai3_b{b}_ff1_w", sd_np(sd, p + "ff.2.weight"))         # [64,128]
        s += carr(f"ai3_b{b}_ff1_b", sd_np(sd, p + "ff.2.bias"))
    s += carr("ai3_normf_w", sd_np(sd, "norm.weight"))
    s += carr("ai3_normf_b", sd_np(sd, "norm.bias"))
    s += carr("ai3_head_w",  sd_np(sd, "head.weight"))    # [5,64]
    s += carr("ai3_head_b",  sd_np(sd, "head.bias"))
    s += carr("ai3_mu",  mu)
    s += carr("ai3_std", std)
    s += header_bot("AI3_TRANSFORMER_WEIGHTS_H")
    (OUT / "ai3_transformer_weights.h").write_text(s, encoding="utf-8")
    print("  ai3_transformer_weights.h written")


def export_ai4(golden):
    ck = torch.load(HERE / "ai4_fusion_mlp" / "ai4_fusion.pt", map_location="cpu", weights_only=True)
    sd = ck["model"]
    m = FusionMLP(); m.load_state_dict(sd); m.eval()

    # locate indices: Linear at net.0/net.4/net.8, BN at net.1/net.5
    fc0_w, fc0_b = sd_np(sd, "net.0.weight"), sd_np(sd, "net.0.bias")
    bn0 = (sd_np(sd, "net.1.weight"), sd_np(sd, "net.1.bias"),
           sd_np(sd, "net.1.running_mean"), sd_np(sd, "net.1.running_var"))
    fc1_w, fc1_b = sd_np(sd, "net.4.weight"), sd_np(sd, "net.4.bias")
    bn1 = (sd_np(sd, "net.5.weight"), sd_np(sd, "net.5.bias"),
           sd_np(sd, "net.5.running_mean"), sd_np(sd, "net.5.running_var"))
    fc2_w, fc2_b = sd_np(sd, "net.8.weight"), sd_np(sd, "net.8.bias")

    fc0_w, fc0_b = fold_bn_into_linear(fc0_w, fc0_b, *bn0)
    fc1_w, fc1_b = fold_bn_into_linear(fc1_w, fc1_b, *bn1)

    rng = np.random.default_rng(GOLDEN_SEED + 4)
    x = rng.uniform(0.0, 1.0, size=(1, 16)).astype(np.float32)
    with torch.no_grad():
        ref = m(torch.from_numpy(x)).numpy().reshape(-1)
    # verify the folded weights reproduce the original eval-mode forward
    h = np.maximum(fc0_w @ x.reshape(-1) + fc0_b, 0.0)
    h = np.maximum(fc1_w @ h + fc1_b, 0.0)
    man = fc2_w @ h + fc2_b
    err = float(np.max(np.abs(man - ref)))
    assert err < 1e-4, f"AI-4 BN-fold mismatch: {err}"
    print(f"  AI-4 BN-fold verified  max|folded-orig|={err:.2e}")
    golden["ai4_input"]  = x.reshape(-1)
    golden["ai4_logits"] = ref

    s = header_top("AI4_FUSION_WEIGHTS_H", "AI-4 Fusion MLP 16-32-16-4 (BatchNorm folded into Linear)")
    s += "#define AI4_IN 16\n#define AI4_H0 32\n#define AI4_H1 16\n#define AI4_NCLS 4\n\n"
    s += carr("ai4_fc0_w", fc0_w)   # [32,16]
    s += carr("ai4_fc0_b", fc0_b)
    s += carr("ai4_fc1_w", fc1_w)   # [16,32]
    s += carr("ai4_fc1_b", fc1_b)
    s += carr("ai4_fc2_w", fc2_w)   # [4,16]
    s += carr("ai4_fc2_b", fc2_b)
    s += header_bot("AI4_FUSION_WEIGHTS_H")
    (OUT / "ai4_fusion_weights.h").write_text(s, encoding="utf-8")
    print("  ai4_fusion_weights.h written")


def export_golden(golden):
    s = header_top("AI_GOLDEN_H", "Golden test vectors for on-chip inference self-test")
    s += "/* Each pair: a fixed input + the PyTorch (eval-mode) expected output.\n"
    s += " * The firmware runs its C engine on the input and checks max|err| < tol.\n"
    s += " * AI-2/AI-3 inputs are already in NORMALISED space (post mu/std). */\n\n"
    for k, v in golden.items():
        s += carr("g_" + k, v)
    s += header_bot("AI_GOLDEN_H")
    (OUT / "ai_golden.h").write_text(s, encoding="utf-8")
    print("  ai_golden.h written")


def main():
    print(f"Output dir: {OUT}")
    golden = {}
    export_ai1(golden)
    export_ai2(golden)
    export_ai3(golden)
    export_ai4(golden)
    export_golden(golden)
    print("\nGolden vector summary:")
    for k, v in golden.items():
        a = np.asarray(v)
        print(f"  {k:18s} shape={a.shape}  range=[{a.min():.4f},{a.max():.4f}]")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
