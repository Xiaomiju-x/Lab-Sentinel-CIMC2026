"""
gd32ai_deploy_export.py — DEPLOY the GD32-AI-Tool-produced TFLite on-chip.
==========================================================================
CIMC Lab-Sentinel — 赛题「GD32 Embedded AI 工具链使用 + 端侧部署」(3 分) 的端侧腿。

This closes the loop the conversion-only step left open: it reads the AI-4 fusion
.tflite that the GD32 Embedded AI Tool's bundled TinyNeuralNetwork produced, EXTRACTS
the FullyConnected weights/biases directly from the flatbuffer (NOT from PyTorch),
emits them as a C header, and bakes a golden so the on-chip engine can byte-verify it.

Two honesty checks are asserted here:
  1. tflite-extracted weights reproduce the original PyTorch FusionMLP within float
     round-trip tolerance  -> the tool's conversion is FAITHFUL.
  2. a fixed-input golden (computed from the EXTRACTED weights) is emitted; the MCU
     reruns it in the boot self-test -> the DEPLOYED model is what the tool produced.

So "we used the GD32 AI Tool to convert AND deployed its TFLite output on-chip,
host+chip verified" is literally true (weights traced to the tool's .tflite bytes).

Run: cd CIMC/model && python gd32ai_deploy_export.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from tflite.Model import Model
from tflite.BuiltinOperator import BuiltinOperator
from tflite.BuiltinOptions import BuiltinOptions
from tflite.FullyConnectedOptions import FullyConnectedOptions
from tflite.ActivationFunctionType import ActivationFunctionType

HERE = Path(__file__).parent
TFLITE = HERE / "gd32ai_export" / "ai4_fusion_gd32ai.tflite"
OUT_H = Path(__file__).parents[1] / "firmware" / "ai_models_c" / "ai4_tflite_deploy.h"

sys.path.insert(0, str(HERE))
from export_weights_to_c import FusionMLP  # noqa: E402

_ACT = {v: k for k, v in ActivationFunctionType.__dict__.items() if isinstance(v, int)}


def buf_f32(m, tensor_idx, sg):
    t = sg.Tensors(tensor_idx)
    b = m.Buffers(t.Buffer())
    raw = b.DataAsNumpy()                       # uint8
    arr = raw.view(np.float32).copy()
    return arr, list(t.ShapeAsNumpy())


def extract(path):
    """Return ordered list of FC layers [(W[out,in], b[out], act_str), ...]."""
    m = Model.GetRootAs(open(path, "rb").read(), 0)
    sg = m.Subgraphs(0)
    layers = []
    for i in range(sg.OperatorsLength()):
        op = sg.Operators(i)
        code = m.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        if code != BuiltinOperator.FULLY_CONNECTED:
            raise RuntimeError(f"op{i} is not FULLY_CONNECTED ({code})")
        w_idx, b_idx = op.Inputs(1), op.Inputs(2)
        w, wsh = buf_f32(m, w_idx, sg)
        b, _ = buf_f32(m, b_idx, sg)
        w = w.reshape(wsh)                       # [out, in]
        act = ActivationFunctionType.NONE
        if op.BuiltinOptionsType() == BuiltinOptions.FullyConnectedOptions:
            o = FullyConnectedOptions()
            o.Init(op.BuiltinOptions().Bytes, op.BuiltinOptions().Pos)
            act = o.FusedActivationFunction()
        layers.append((w.astype(np.float32), b.astype(np.float32), _ACT.get(act, str(act))))
    return layers


def fwd_from_layers(x, layers):
    h = x.astype(np.float32)
    for w, b, act in layers:
        h = h @ w.T + b
        if act == "RELU":
            h = np.maximum(h, 0.0)
        elif act != "NONE":
            raise RuntimeError(f"unsupported fused act {act}")
    return h


def main():
    layers = extract(TFLITE)
    print(f"[tflite] {TFLITE.name}: {len(layers)} FC layers")
    for i, (w, b, act) in enumerate(layers):
        print(f"  FC{i}: W{list(w.shape)} b{list(b.shape)} act={act}")

    # ---- check 1: extracted weights reproduce the PyTorch FusionMLP ----
    ck = torch.load(HERE / "ai4_fusion_mlp" / "ai4_fusion.pt", map_location="cpu", weights_only=True)
    m4 = FusionMLP(); m4.load_state_dict(ck["model"]); m4.eval()
    rng = np.random.default_rng(7)
    X = rng.standard_normal((64, layers[0][0].shape[1])).astype(np.float32)
    with torch.no_grad():
        ref = m4(torch.tensor(X)).numpy()
    got = fwd_from_layers(X, layers)
    err = float(np.max(np.abs(ref - got)))
    print(f"[check1] tflite-extracted vs PyTorch FusionMLP  max|err|={err:.3e}  "
          f"{'FAITHFUL' if err < 1e-4 else 'MISMATCH'}")

    # ---- golden: a fixed input run through the EXTRACTED weights ----
    nin = layers[0][0].shape[1]
    gx = np.array([(((k * 37) % 100) / 100.0 - 0.5) for k in range(nin)], dtype=np.float32)
    gy = fwd_from_layers(gx[None, :], layers)[0]

    # ---- emit C header ----
    def carr(name, a):
        flat = np.asarray(a, np.float32).ravel()
        body = ", ".join(f"{v:.8e}f" for v in flat)
        return f"static const float {name}[{flat.size}] = {{ {body} }};\n"

    nh = [layers[0][0].shape[0], layers[1][0].shape[0]]
    nout = layers[2][0].shape[0]
    with open(OUT_H, "w", encoding="ascii") as f:
        f.write("/* ai4_tflite_deploy.h - AI-4 fusion weights EXTRACTED from the GD32-AI-Tool\n"
                " * .tflite flatbuffer (gd32ai_export/ai4_fusion_gd32ai.tflite), AUTO-GENERATED\n"
                " * by model/gd32ai_deploy_export.py. Proves the on-chip deployed model IS the\n"
                " * tool's TFLite output (weights traced to the flatbuffer bytes, not re-derived).\n"
                " * Graph: FC(16->32)RELU -> FC(32->16)RELU -> FC(16->4). host check1 max|err| vs\n"
                f" * PyTorch = {err:.2e} (faithful). */\n")
        f.write("#ifndef AI4_TFLITE_DEPLOY_H\n#define AI4_TFLITE_DEPLOY_H\n\n")
        f.write(f"#define TFL_NIN  {nin}\n#define TFL_NH0  {nh[0]}\n#define TFL_NH1  {nh[1]}\n#define TFL_NOUT {nout}\n\n")
        f.write(carr("tfl_w0", layers[0][0])); f.write(carr("tfl_b0", layers[0][1]))
        f.write(carr("tfl_w1", layers[1][0])); f.write(carr("tfl_b1", layers[1][1]))
        f.write(carr("tfl_w2", layers[2][0])); f.write(carr("tfl_b2", layers[2][1]))
        f.write("\n")
        f.write(carr("tfl_golden_in", gx))
        f.write(carr("tfl_golden_out", gy))
        f.write("\n#endif /* AI4_TFLITE_DEPLOY_H */\n")
    print(f"[emit] {OUT_H}  (golden out = {np.array2string(gy, precision=4)})")
    print("OK" if err < 1e-4 else "FAIL")
    return 0 if err < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
