"""
cam_crucible.py — Class Activation Map for the AI-1 crucible CNN (CIMC)
======================================================================
卖点 7「可解释性」. The crucible net ends in GAP -> FC, so CAM needs NO gradients:
the heatmap for class c is just the FC-row-weighted sum of the last conv feature
maps (32 x 4x4). That is exactly what the on-chip engine can compute from s_bufA
(post-conv3) + cru_fc_w[c] in ~512 MACs — so this same CAM can live on the GD32
LVGL touchscreen (tap an alarm -> see where the CNN looked).

This PC script validates the math and renders demo overlays for the report.

Run:  cd CIMC/model/ai1_vision_cnn && python cam_crucible.py
Out:  ../../docs/figures/crucible_cam.png
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crucible_cnn import CrucibleCNN, CLASS_NAMES
from synth_crucible import _render

HERE = Path(__file__).parent


def cam_for(model, x):
    """Return (pred, cam64) — cam64 is a [64,64] heatmap in [0,1] for the pred class.
    Mirrors the on-chip path: feature maps (32x4x4) weighted by FC row of pred class."""
    with torch.no_grad():
        f = model.features(x)            # [1,32,4,4]
        g = f.mean(dim=(2, 3))           # GAP
        logits = model.head(g)           # [1,4]
        pred = int(logits.argmax(1))
        w = model.head.weight[pred]      # [32]
        cam = (w[None, :, None, None] * f).sum(1)[0]   # [4,4]
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-6)
        cam64 = F.interpolate(cam[None, None], size=(64, 64),
                              mode="bilinear", align_corners=False)[0, 0]
    return pred, cam64.numpy()


def main():
    sd = torch.load(HERE / "crucible_cnn.pt", map_location="cpu", weights_only=True)
    m = CrucibleCNN(); m.load_state_dict(sd); m.eval()

    rng = np.random.default_rng(7)
    fig, axes = plt.subplots(4, 2, figsize=(5, 10))
    for cls in range(4):
        im = _render(cls, rng)                          # HWC [0,1]
        x = torch.from_numpy(np.transpose(im, (2, 0, 1)))[None].float()
        pred, cam = cam_for(m, x)
        axes[cls, 0].imshow(im); axes[cls, 0].axis("off")
        axes[cls, 0].set_title(f"in: {CLASS_NAMES[cls]}", fontsize=9)
        axes[cls, 1].imshow(im); axes[cls, 1].imshow(cam, cmap="jet", alpha=0.5)
        axes[cls, 1].axis("off")
        ok = "OK" if pred == cls else "MISS"
        axes[cls, 1].set_title(f"CAM -> pred {CLASS_NAMES[pred]} [{ok}]", fontsize=9)
    fig.suptitle("AI-1 crucible CAM (GAP+FC, no-grad) — synthetic", fontsize=11)
    fig.tight_layout()
    out = HERE.parent.parent / "docs" / "figures" / "crucible_cam.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
