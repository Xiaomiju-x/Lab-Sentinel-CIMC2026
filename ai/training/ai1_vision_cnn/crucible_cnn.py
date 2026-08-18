"""
crucible_cnn.py — AI-1 crucible-state classifier (CIMC Lab-Sentinel)
====================================================================
This is the REAL task model: a 3-class crucible state from a 64x64 RGB crop of
the OV5640 frame, trained on real phone-shot crucible photos (CIMC/手机拍摄数据).

Classes (fixed order — must match firmware + dataset folders):
  0 empty    空坩埚 / 坩埚外侧      (dark interior visible, white wall on dark table)
  1 loaded   装研磨好的粉末(偏白)   (whitish/grey ground precursor, pre-sinter)
  2 done     装烧制好的粉末(偏黄)   (yellowish sintered product)

Why only 3 (was 4): the "sintering" (炉内红热) state is INSIDE the 1500 C furnace
and cannot be photographed from outside, so it is not a visually-classifiable
class. The firmware keeps a separate 4-stage *furnace-stage proxy* (empty→loading
→firing→done) for the sim-batch HMI animation + AI-4/AI-5 — that is a process-
stage signal, NOT this visual CNN. loaded(white) vs done(yellow) separate on the
R/G-vs-B colour balance, so colour MUST be preserved (no hue jitter in aug).

Architecture (kept tiny + float-C-deployable on GD32H759 M7):
  in 3x64x64
  -> [conv3x3 s2 p1] 8 x32x32  +BN +ReLU
  -> [conv3x3 s2 p1] 16x16x16  +BN +ReLU
  -> [conv3x3 s2 p1] 24x 8x 8  +BN +ReLU
  -> [conv3x3 s2 p1] 32x 4x 4  +BN +ReLU
  -> GAP -> 32  -> FC 4 (logits)

Why stride-2 convs (not conv+maxpool): halves spatial size IN the conv, so the
largest activation is 32x32x8 (32 KB) — no 64x64x8 (131 KB) transient. Keeps the
whole forward inside ~80 KB SRAM scratch on-chip.

Why color (RGB not grayscale): the red/orange GLOW is the single strongest cue for
"sintering", and powder-white vs dark-empty separate on brightness — both live in
the channels. Grayscale would throw the glow away.

BatchNorm is folded into the (bias-less) conv at export time, exactly like AI-4.

~12 K params (~48 KB Flash float32), ~3.9 M MACs (~26 ms on M7 with weight-stationary
batched kernels, see pattern_mcu_gemm_cache_thrash). Runs fine in the 5 fps vision_task.
"""

import torch
import torch.nn as nn

IN_CH   = 3
IN_HW   = 64
N_CLS   = 3
CH      = [8, 16, 24, 32]                     # conv output channels per stage
CLASS_NAMES = ["empty", "loaded", "done"]
CLASS_ZH    = ["空坩埚", "装料(偏白)", "烧成(偏黄)"]


def _block(i, o):
    """3x3 stride-2 conv (no bias) + BN + ReLU. BN folds into the conv at export."""
    return nn.Sequential(
        nn.Conv2d(i, o, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )


class CrucibleCNN(nn.Module):
    def __init__(self, in_ch=IN_CH, n_cls=N_CLS, ch=CH):
        super().__init__()
        self.features = nn.Sequential(
            _block(in_ch, ch[0]),     # 64 -> 32
            _block(ch[0], ch[1]),     # 32 -> 16
            _block(ch[1], ch[2]),     # 16 -> 8
            _block(ch[2], ch[3]),     #  8 -> 4
        )
        self.head = nn.Linear(ch[3], n_cls)

    def forward(self, x):
        f = self.features(x)          # [B, 32, 4, 4]
        g = f.mean(dim=(2, 3))        # global average pool -> [B, 32]
        return self.head(g)           # logits [B, 4]
