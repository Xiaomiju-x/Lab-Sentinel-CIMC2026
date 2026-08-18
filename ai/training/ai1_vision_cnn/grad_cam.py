"""
grad_cam.py — Grad-CAM explainability for AI-1 crucible CNN (CIMC Lab-Sentinel)
===============================================================================
赛题 (7) 模型可解释性 (附加 5 分): 可视化 AI-1 在判定坩埚状态时"看"的区域。

Grad-CAM (Selvaraju 2017): for target class c, weight each feature-map channel k of
a chosen conv layer by the gradient of the class score w.r.t. that map (GAP'd), then
ReLU the weighted sum -> a coarse spatial heatmap of class-discriminative regions.

We hook the 3rd conv stage (24 ch x 8x8) for an 8x8 heatmap (the final stage is only
4x4). Heatmaps are upsampled to 64x64 and overlaid on the input crucible image.

Honesty: the AI-1 crucible head is trained on PROCEDURAL-SYNTHETIC crucible images
(synth_crucible.py) until the real OV5640 dataset is collected — Grad-CAM here proves
the EXPLAINABILITY mechanism + that the CNN keys on the physically-correct cues
(empty=dark cavity, loaded=powder fill, sintering=hot glow core, done=dark block).
Once real images swap in, the exact same script produces the field heatmaps shown on
the HMI when an operator taps an alarm.

Output: ../../docs/figures/gradcam_crucible.png   (4-panel: input | heatmap overlay)

Run: cd CIMC/model/ai1_vision_cnn && python grad_cam.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from crucible_cnn import CrucibleCNN, CLASS_NAMES
from synth_crucible import make_synth

HERE = Path(__file__).parent
OUT  = HERE / ".." / ".." / "docs" / "figures" / "gradcam_crucible.png"
TARGET_STAGE = 2          # features[2] -> [B,24,8,8]


class GradCAM:
    def __init__(self, model, target_module):
        self.model = model.eval()
        self.acts = None
        self.grads = None
        target_module.register_forward_hook(self._fwd)
        target_module.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, o):
        self.acts = o.detach()

    def _bwd(self, _m, _gi, go):
        self.grads = go[0].detach()

    def __call__(self, x, cls):
        logits = self.model(x)                       # [1,4]
        self.model.zero_grad()
        logits[0, cls].backward()
        # alpha_k = GAP of gradients ; cam = ReLU(sum_k alpha_k * A_k)
        alpha = self.grads.mean(dim=(2, 3), keepdim=True)        # [1,C,1,1]
        cam = F.relu((alpha * self.acts).sum(dim=1, keepdim=True))  # [1,1,8,8]
        cam = F.interpolate(cam, size=(64, 64), mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        return cam, int(logits.argmax(1).item())


def main():
    pt = HERE / "crucible_cnn.pt"
    if not pt.exists():
        print("crucible_cnn.pt not found — run train_crucible.py first.")
        return
    model = CrucibleCNN()
    model.load_state_dict(torch.load(pt, map_location="cpu", weights_only=True))
    cam = GradCAM(model, model.features[TARGET_STAGE])

    # one representative synthetic image per class (fixed seed → reproducible figure)
    X, y = make_synth(n_per_class=1, seed=7)
    order = [int(np.where(y == c)[0][0]) for c in range(4)]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for col, c in enumerate(range(4)):
        idx = order[col]
        x = torch.from_numpy(X[idx:idx + 1])         # [1,3,64,64]
        heat, pred = cam(x.clone().requires_grad_(True), c)
        img = np.transpose(X[idx], (1, 2, 0))         # CHW->HWC

        axes[0, col].imshow(np.clip(img, 0, 1))
        axes[0, col].set_title(f"input: {CLASS_NAMES[c]}", fontsize=11)
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(img, 0, 1))
        axes[1, col].imshow(heat, cmap="jet", alpha=0.5)
        ok = "OK" if pred == c else f"pred={CLASS_NAMES[pred]}"
        axes[1, col].set_title(f"Grad-CAM ({ok})", fontsize=11)
        axes[1, col].axis("off")

    fig.suptitle("AI-1 Crucible CNN — Grad-CAM class-discriminative regions "
                 "(stage-3, 8x8 -> 64x64)", fontsize=13)
    fig.text(0.5, 0.01,
             "synthetic crucible images (synth_crucible.py); same script produces field "
             "heatmaps on real OV5640 frames for the HMI alarm-inspect view",
             ha="center", fontsize=8, style="italic")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")

    # also report energy-in-disc (sanity: CAM should concentrate on the crucible disc)
    yy, xx = np.mgrid[0:64, 0:64]
    disc = np.sqrt((yy - 31.5) ** 2 + (xx - 31.5) ** 2) <= 0.42 * 64
    for col, c in enumerate(range(4)):
        idx = order[col]
        x = torch.from_numpy(X[idx:idx + 1]).clone().requires_grad_(True)
        heat, _ = cam(x, c)
        frac = heat[disc].sum() / (heat.sum() + 1e-8)
        print(f"  {CLASS_NAMES[c]:10s}: {frac*100:.0f}% of Grad-CAM energy on crucible disc")


if __name__ == "__main__":
    main()
