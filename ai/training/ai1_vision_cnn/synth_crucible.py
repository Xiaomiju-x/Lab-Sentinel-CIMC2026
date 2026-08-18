"""
synth_crucible.py — synthetic 4-class crucible images (CIMC Lab-Sentinel)
=========================================================================
Procedurally renders crucible-state images so the train -> export -> C-engine ->
golden pipeline can be validated BEFORE the real OV5640 dataset is collected. The
4 classes are deliberately separable by the same cues the real states differ on
(brightness / fill texture / glow color / block darkness), so a model that learns
these is a reasonable placeholder until real data swaps in.

NOT a substitute for real data — purely to prove the machinery end-to-end and give
a >MNIST placeholder. Once data/crucible/<class>/*.png exists, train_crucible.py
uses that instead automatically.

Each image: 64x64x3 float in [0,1], a centered crucible disc on a dim bench.
"""

import numpy as np

IN_HW = 64


def _disc_mask(h=IN_HW, w=IN_HW, r_frac=0.42):
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return rr <= (r_frac * min(h, w))


_DISC = _disc_mask()


def _render(cls, rng):
    """Render one 64x64x3 [0,1] image for class cls with random variation."""
    img = np.zeros((IN_HW, IN_HW, 3), dtype=np.float32)
    # dim metal bench background with slight warm tint + lighting jitter
    bg = rng.uniform(0.08, 0.22)
    img[:] = np.array([bg, bg, bg * 1.05], dtype=np.float32)

    inside = _DISC
    if cls == 0:        # empty: dark interior (shadowed cavity)
        v = rng.uniform(0.02, 0.10)
        img[inside] = [v, v, v * 1.1]
    elif cls == 1:      # loaded: white/grey powder, granular texture
        base = rng.uniform(0.55, 0.85)
        tex = rng.normal(0.0, 0.07, size=(IN_HW, IN_HW))
        for c in range(3):
            ch = np.clip(base + tex, 0, 1)
            img[:, :, c][inside] = ch[inside]
    elif cls == 2:      # sintering: red/orange glow, radial hot core
        yy, xx = np.mgrid[0:IN_HW, 0:IN_HW]
        cy, cx = (IN_HW - 1) / 2.0, (IN_HW - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (0.42 * IN_HW)
        glow = np.clip(1.0 - rr, 0, 1) ** rng.uniform(1.2, 2.2)
        r = np.clip(0.55 + 0.45 * glow, 0, 1)
        g = np.clip(0.20 + 0.55 * glow, 0, 1)
        b = np.clip(0.02 + 0.12 * glow, 0, 1)
        img[:, :, 0][inside] = r[inside]
        img[:, :, 1][inside] = g[inside]
        img[:, :, 2][inside] = b[inside]
    else:               # done: darkened sintered block, low-sat grey-brown
        base = rng.uniform(0.18, 0.38)
        tex = rng.normal(0.0, 0.05, size=(IN_HW, IN_HW))
        img[:, :, 0][inside] = np.clip(base + 0.05 + tex, 0, 1)[inside]
        img[:, :, 1][inside] = np.clip(base + tex, 0, 1)[inside]
        img[:, :, 2][inside] = np.clip(base - 0.04 + tex, 0, 1)[inside]

    # global sensor-ish noise + brightness jitter (domain robustness)
    img = img * rng.uniform(0.85, 1.15) + rng.normal(0.0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def make_synth(n_per_class=400, seed=0):
    """Return (X[N,3,64,64] float32, y[N] int64)."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls in range(4):
        for _ in range(n_per_class):
            im = _render(cls, rng)                 # HWC
            X.append(np.transpose(im, (2, 0, 1)))  # -> CHW
            y.append(cls)
    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int64)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


if __name__ == "__main__":
    X, y = make_synth(8)
    print("synth shape", X.shape, "labels", np.bincount(y))
    print("range", X.min(), X.max())
