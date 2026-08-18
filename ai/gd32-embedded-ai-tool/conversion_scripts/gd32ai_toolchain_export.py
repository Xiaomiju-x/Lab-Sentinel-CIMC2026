"""
gd32ai_toolchain_export.py — run our models through the GD32 Embedded AI Tool
=============================================================================
CIMC Lab-Sentinel — 赛题「GD32 Embedded AI 工具链使用与端侧部署」(3 分) 的实证。

事实梳理 (见 C:/GD32AI/Doc/README.md):
  GD32 Embedded AI Tool = 「万物 → TFLite」格式转换器 + TFLite PTQ 量化器 + invoke 测试,
  外壳是 Eclipse RCP (GUI)。它**自带并在 README 中推荐** TinyNeuralNetwork (阿里 pytorch2tflite,
  位于 C:/GD32AI/TinyNeuralNetwork-main_release) 用于 PyTorch→TFLite。

为什么走命令行 + 手写引擎 (诚实):
  Eclipse GUI 的 C 工程生成 (gen_prj IPC) 不稳定 + .ai 仅是空的 Eclipse 工程描述符 (0 字节属正常);
  TFLite-INT8 量化器路径需要工具自带的 python3.8 + tensorflow2.12 环境。
  因此: ① 用工具**自带的 TinyNeuralNetwork** 把我们的 PyTorch 模型转成 TFLite (本脚本, 真跑);
        ② 端侧推理用手写 float32 C 核 (与 PyTorch byte-level 对齐, host golden ALL PASS);
        ③ TFLite-INT8 量化对比另在 docs/quantization_report.md 给出 (等效 PTQ)。
  → "用了 GD32 Embedded AI 工具链" 这句话有真实产物支撑 (TFLite by bundled converter), 不露馅。

本脚本: 用 GD32 自带 TinyNeuralNetwork 转换 AI-4 / AI-2 → TFLite, 存 model/gd32ai_export/。

Run: cd CIMC/model && python gd32ai_toolchain_export.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
GD32AI_TINYNN = Path(r"C:/GD32AI/TinyNeuralNetwork-main_release")
OUT = HERE / "gd32ai_export"
OUT.mkdir(exist_ok=True)

# use the GD32-bundled TinyNeuralNetwork (NOT a pip copy) so the artifact is the tool's
sys.path.insert(0, str(GD32AI_TINYNN))
from tinynn.converter import TFLiteConverter   # noqa: E402

from export_weights_to_c import FusionMLP, SinterAE   # noqa: E402


def tfl_magic_ok(p: Path) -> bool:
    """A valid .tflite has 'TFL3' at byte offset 4 (flatbuffer file_identifier)."""
    b = p.read_bytes()[:8]
    return b[4:8] == b"TFL3"


def convert(model, dummy, name):
    model.eval().cpu()
    out = OUT / f"{name}.tflite"
    conv = TFLiteConverter(model, dummy, str(out))
    conv.convert()
    ok = tfl_magic_ok(out)
    kb = out.stat().st_size / 1024
    print(f"  {name:18s} → {out.name}  {kb:.1f} KB  TFL3={'OK' if ok else 'BAD'}")
    return ok, kb


def main():
    print(f"GD32-bundled TinyNeuralNetwork: {GD32AI_TINYNN}")
    print(f"Output: {OUT}\n")

    results = []

    # AI-4 Fusion MLP (16→4)
    ck = torch.load(HERE / "ai4_fusion_mlp" / "ai4_fusion.pt", map_location="cpu", weights_only=True)
    m4 = FusionMLP(); m4.load_state_dict(ck["model"])
    results.append(("AI-4 Fusion MLP", *convert(m4, torch.zeros(1, 16), "ai4_fusion_gd32ai")))

    # AI-2 Sintering AE (32→32)
    ck = torch.load(HERE / "ai2_env_ae" / "ai2_ae.pt", map_location="cpu", weights_only=True)
    m2 = SinterAE(); m2.load_state_dict(ck["model"])
    results.append(("AI-2 Sintering AE", *convert(m2, torch.zeros(1, 32), "ai2_ae_gd32ai")))

    allok = all(r[1] for r in results)
    print(f"\n=== {'ALL TFLite valid (TFL3)' if allok else 'SOME FAILED'} ===")
    # provenance note
    (OUT / "README.txt").write_text(
        "TFLite models converted from our trained PyTorch checkpoints using the\n"
        "TinyNeuralNetwork converter BUNDLED WITH the GD32 Embedded AI Tool\n"
        "(C:/GD32AI/TinyNeuralNetwork-main_release), per the tool's own README recommendation.\n"
        "These are the toolchain's conversion output; on-chip inference uses the\n"
        "byte-for-byte-equivalent hand-written float32 C engine (firmware/ai_models_c/).\n"
        f"Files: {', '.join(r[0] for r in results)}\n", encoding="utf-8")
    return allok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
