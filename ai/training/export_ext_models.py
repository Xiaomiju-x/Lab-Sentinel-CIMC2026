"""
export_ext_models.py — emit float-C headers for AI-11/12/13 (CIMC Lab-Sentinel)
===============================================================================
Project: 11 -> 14 on-chip AI models. Three NEW models grounded in REAL XRD data:

  AI-11 phase-purity prior   24-D formula descriptor -> pure / impure  (+P(pure))
                             (37 real observed_pl phase labels; LOO 70% vs 60% base)
  AI-12 PL dopant classifier 64-pt emission spectrum  -> Cr / Ni / Cr+Ni
                             (281 real Fluoromax spectra; 5-fold CV 98.2%)
  AI-13 PL-QC autoencoder    64-pt emission spectrum  -> recon + MSE; anomaly if >q_hat
                             (281 real spectra, AE+conformal; 3/3 injected anomalies)

Emits weight headers into firmware/ai_models_c/ + a host golden header into BOTH
firmware/ai_models_c/ and model/host_test/. Mirrors export_new_models.py style.

AI-11's per-feature standardisation (mu/sd) is FOLDED into layer-0 weights (same as
BN folding) so the C engine runs straight on the raw 24-D descriptor (no mu/sd array).

Run:  cd CIMC/model && python export_ext_models.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "CIMC" / "model" / "ai11_purity"))
sys.path.insert(0, str(ROOT / "CIMC" / "model" / "ai12_plspec"))
sys.path.insert(0, str(ROOT / "CIMC" / "model" / "ai13_plqc"))

from predict_engine import ts_torch as T            # noqa: E402
from train_purity import MLPNet, LinNet             # noqa: E402
from train_plspec import SpecMLP                     # noqa: E402
from train_plqc import SpecAE                         # noqa: E402

OUT = ROOT / "CIMC" / "firmware" / "ai_models_c"
HOST = ROOT / "CIMC" / "model" / "host_test"
M = ROOT / "CIMC" / "model"


def carr(name, arr):
    flat = np.asarray(arr, np.float32).reshape(-1)
    return f"static const float {name}[{flat.size}] = {{ {', '.join(f'{v:.8e}f' for v in flat)} }};\n"


def lin(model_seq):
    """list of (W[out,in], b[out]) for nn.Linear layers in a Sequential."""
    L = []
    for m in model_seq:
        if isinstance(m, torch.nn.Linear):
            L.append((m.weight.detach().numpy().astype(np.float64),
                      m.bias.detach().numpy().astype(np.float64)))
    return L


# --------------------------------------------------------------------- AI-11 purity
def export_purity():
    d = torch.load(M / "ai11_purity" / "ai11_purity.pt", map_location="cpu", weights_only=True)
    arch = d.get("arch", "MLP")
    net = (MLPNet() if arch == "MLP" else LinNet())
    net.load_state_dict(d["model_state_dict"]); net.eval()
    L = lin(net.net)
    mu = np.array(d["mu"], np.float64); sd = np.array(d["sd"], np.float64)
    # fold standardisation (x-mu)/sd into layer 0:  W0' = W0/sd ; b0' = b0 - W0.(mu/sd)
    W0, b0 = L[0]
    W0f = W0 / sd[None, :]
    b0f = b0 - (W0 * (mu / sd)[None, :]).sum(1)
    s = ("/* AI-11 phase-purity pre-flight prior (24-D formula descriptor -> 2 logits:\n"
         " * 0=impure 1=pure). 37 REAL observed_pl phase labels (15 pure/22 impure).\n"
         " * Honest: leave-one-out CV %.1f%% vs %.1f%% majority baseline -> a weak-but-real\n"
         " * EDGE TRIAGE prior; deep compositional call stays off-device. Std folded\n"
         " * into layer 0 (runs on raw descriptor). AUTO-GENERATED. */\n"
         "#ifndef AI11_PURITY_WEIGHTS_H\n#define AI11_PURITY_WEIGHTS_H\n\n"
         "#define AI11_IN 24\n#define AI11_H %d\n#define AI11_OUT 2\n"
         % (d["loo_mlp"] * 100, d["baseline"] * 100, W0.shape[0]))
    s += carr("ai11_w0", W0f) + carr("ai11_b0", b0f)
    s += carr("ai11_w1", L[1][0]) + carr("ai11_b1", L[1][1])
    s += "\n#endif\n"
    (OUT / "ai11_purity_weights.h").write_text(s, encoding="utf-8")

    # golden: YAG:Cr preset descriptor -> logits/probs (post-fold engine must match torch)
    dvec = T.formula_descriptor("Y3Al5O12", "Al", 1.0).numpy().astype(np.float32)
    with torch.no_grad():
        logit = net(torch.tensor((dvec - mu) / sd, dtype=torch.float32)[None]).numpy().reshape(-1)
    p = np.exp(logit - logit.max()); p = p / p.sum()
    return dvec, logit, float(p[1]), int(np.argmax(logit)), d


# --------------------------------------------------------------------- AI-12 PL class
def export_plclass():
    d = torch.load(M / "ai12_plspec" / "ai12_plspec.pt", map_location="cpu", weights_only=True)
    net = SpecMLP(); net.load_state_dict(d["model_state_dict"]); net.eval()
    L = lin(net.net)
    gn = int(d["grid_n"])
    s = ("/* AI-12 PL-emission dopant classifier (%d-pt normalised emission spectrum ->\n"
         " * 3 logits: 0=Cr 1=Ni 2=Cr+Ni). 281 REAL Fluoromax spectra, 5-fold CV %.1f%%.\n"
         " * Input = baseline-subtracted + peak-normalised resample over [%g,%g] nm.\n"
         " * AUTO-GENERATED. */\n#ifndef AI12_PLSPEC_WEIGHTS_H\n#define AI12_PLSPEC_WEIGHTS_H\n\n"
         "#define AI12_IN %d\n#define AI12_H1 32\n#define AI12_H2 16\n#define AI12_OUT 3\n"
         "#define AI12_GRID_LO %gf\n#define AI12_GRID_HI %gf\n"
         % (gn, d["cv_acc"] * 100, d["grid_lo"], d["grid_hi"], gn, d["grid_lo"], d["grid_hi"]))
    for i, (W, b) in enumerate(L):
        s += carr(f"ai12_w{i}", W) + carr(f"ai12_b{i}", b)
    s += "\n#endif\n"
    (OUT / "ai12_plspec_weights.h").write_text(s, encoding="utf-8")

    # demo + golden spectra: one real spectrum per class (cr/ni/cr_ni)
    reps = d["reps"]  # {0:[...],1:[...],2:[...]}
    names = ["Cr", "Ni", "Cr+Ni"]
    dm = ("/* AI-12/13 demo spectra: one REAL emission spectrum per dopant class,\n"
          " * %d-pt normalised (replayed on the PL screen — the sentinel has no spectrometer,\n"
          " * same honest replay as furnace_sim). AUTO-GENERATED. */\n"
          "#ifndef AI12_DEMO_SPECTRA_H\n#define AI12_DEMO_SPECTRA_H\n\n#define DEMO_SPEC_N %d\n\n" % (gn, gn))
    arr = np.array([reps[str(c)] if str(c) in reps else reps[c] for c in (0, 1, 2)], np.float32)
    dm += carr("demo_spec", arr)
    dm += 'static const char *const demo_spec_name[3] = { "Cr", "Ni", "Cr+Ni" };\n'
    dm += "\n#endif\n"
    (OUT / "ai12_demo_spectra.h").write_text(dm, encoding="utf-8")
    (HOST / "ai12_demo_spectra.h").write_text(dm, encoding="utf-8")

    # golden = class-0 (Cr) rep spectrum -> logits/argmax
    g_in = arr[0]
    with torch.no_grad():
        g_logit = net(torch.tensor(g_in)[None]).numpy().reshape(-1)
    return g_in, g_logit, int(np.argmax(g_logit)), names, d


# --------------------------------------------------------------------- AI-13 PL-QC AE
def export_plqc():
    d = torch.load(M / "ai13_plqc" / "ai13_plqc.pt", map_location="cpu", weights_only=True)
    net = SpecAE(); net.load_state_dict(d["model_state_dict"]); net.eval()
    enc = lin(net.enc); dec = lin(net.dec)
    gn = int(d["grid_n"])
    s = ("/* AI-13 PL-QC autoencoder (%d-pt emission spectrum -> recon; anomaly if MSE>q_hat).\n"
         " * 281 REAL spectra, AE+conformal (same recipe as AI-2). in-dist q_hat(90%%)=%.6f,\n"
         " * 3/3 injected anomalies flagged at 10%% in-dist false-flag. Decision-support QC gate.\n"
         " * AUTO-GENERATED. */\n#ifndef AI13_PLQC_WEIGHTS_H\n#define AI13_PLQC_WEIGHTS_H\n\n"
         "#define AI13_IN %d\n#define AI13_H 32\n#define AI13_Z 8\n"
         % (gn, d["q_hat"], gn))
    s += carr("ai13_e0w", enc[0][0]) + carr("ai13_e0b", enc[0][1])
    s += carr("ai13_e1w", enc[1][0]) + carr("ai13_e1b", enc[1][1])
    s += carr("ai13_d0w", dec[0][0]) + carr("ai13_d0b", dec[0][1])
    s += carr("ai13_d1w", dec[1][0]) + carr("ai13_d1b", dec[1][1])
    s += f"static const float ai13_q_hat = {d['q_hat']:.8e}f;\n"
    s += "\n#endif\n"
    (OUT / "ai13_plqc_weights.h").write_text(s, encoding="utf-8")
    return np.array(d["golden_input"], np.float32), np.array(d["golden_recon"], np.float32), \
        float(d["golden_mse"]), d


def main():
    a11_desc, a11_logit, a11_ppure, a11_cls, d11 = export_purity()
    a12_in, a12_logit, a12_cls, names, d12 = export_plclass()
    a13_in, a13_rec, a13_mse, d13 = export_plqc()

    g = ("/* ext_models_golden.h — golden I/O for AI-11/12/13 engines (PyTorch eval).\n"
         " * AUTO-GENERATED by export_ext_models.py. */\n"
         "#ifndef EXT_MODELS_GOLDEN_H\n#define EXT_MODELS_GOLDEN_H\n\n")
    g += carr("g_ai11_desc", a11_desc)
    g += carr("g_ai11_logits", a11_logit)
    g += f"static const float g_ai11_ppure = {a11_ppure:.8e}f;\nstatic const int g_ai11_cls = {a11_cls};\n\n"
    g += carr("g_ai12_spec", a12_in)
    g += carr("g_ai12_logits", a12_logit)
    g += f"static const int g_ai12_cls = {a12_cls};\n\n"
    g += carr("g_ai13_spec", a13_in)
    g += carr("g_ai13_recon", a13_rec)
    g += f"static const float g_ai13_mse = {a13_mse:.8e}f;\n\n"
    g += "#endif\n"
    (OUT / "ext_models_golden.h").write_text(g, encoding="utf-8")
    (HOST / "ext_models_golden.h").write_text(g, encoding="utf-8")

    print("[ext] AI-11 purity  LOO=%.3f base=%.3f  P(pure|YAG)=%.3f cls=%d"
          % (d11["loo_mlp"], d11["baseline"], a11_ppure, a11_cls))
    print("[ext] AI-12 plclass CV=%.3f  golden cls=%d (%s)" % (d12["cv_acc"], a12_cls, names[a12_cls]))
    print("[ext] AI-13 plqc    q_hat=%.6f  golden MSE=%.6f (%s)"
          % (d13["q_hat"], a13_mse, "in-dist OK" if a13_mse < d13["q_hat"] else "above q_hat"))
    print("[done] wrote ai11/ai12/ai13 weights + demo spectra + ext_models_golden ->", OUT)


if __name__ == "__main__":
    main()
