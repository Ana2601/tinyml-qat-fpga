"""Synthetic 8x8 "tiny image" classifier dataset.

Offline, deterministic, and fast to train on M1 - a stand-in for an MLPerf
Tiny-style classification task (e.g., gesture / key-phrase / tiny image).

Eight classes combine a bright-quadrant location with an intensity level,
so the model must read both geometry and brightness:

  class = quadrant (0..3) * 2 + (bright if 1.0 else 0.55)

Each image gets a 1-pixel boundary jitter and gaussian noise so the classes
overlap - hard enough that aggressive (2-bit) quantization measurably hurts.

Images are float32 in [0, 1] (single channel), 1024 train / 256 test.
"""
import os
import random

import numpy as np
import torch

H = W = 8
N_CLASSES = 8
N_TRAIN = 1024
N_TEST = 256
NOISE = 0.18

DATA_CACHE = os.path.join(os.path.dirname(__file__), "data")


def _img(cls: int, rng: random.Random) -> np.ndarray:
    img = np.zeros((H, W), dtype=np.float64)
    quad = cls // 2
    level = 1.0 if cls % 2 else 0.55
    jit = rng.randint(-1, 1)                       # boundary jitter
    ys = (slice(0, 4 + jit), slice(3 + jit, 4 + jit + jit), slice(4 + jit, H),
          slice(max(3 + jit, 0), H))
    xs = (slice(0, 4 + jit), slice(3 + jit, 4 + jit + jit), slice(4 + jit, W),
          slice(max(3 + jit, 0), W))
    img[ys[quad], xs[quad]] = level
    img += np.random.default_rng(int(rng.random() * 2**32)).normal(
        0.0, NOISE, img.shape)
    return np.clip(img, 0.0, 1.0)


def make_dataset() -> tuple[torch.Tensor, torch.Tensor,
                            torch.Tensor, torch.Tensor]:
    cache = os.path.join(DATA_CACHE, "tiny_class.pt")
    if os.path.exists(cache):
        d = torch.load(cache)
        return d["x_tr"], d["y_tr"], d["x_te"], d["y_te"]

    rng = random.Random(0)
    xs_tr, ys_tr = [], []
    for i in range(N_TRAIN):
        cls = rng.randrange(N_CLASSES)
        xs_tr.append(_img(cls, rng))
        ys_tr.append(cls)
    xs_te, ys_te = [], []
    for i in range(N_TEST):
        cls = rng.randrange(N_CLASSES)
        xs_te.append(_img(cls, rng))
        ys_te.append(cls)

    x_tr = torch.tensor(np.stack(xs_tr)[:, None].astype(np.float32))
    x_te = torch.tensor(np.stack(xs_te)[:, None].astype(np.float32))
    y_tr = torch.tensor(ys_tr, dtype=torch.long)
    y_te = torch.tensor(ys_te, dtype=torch.long)

    os.makedirs(DATA_CACHE, exist_ok=True)
    torch.save({"x_tr": x_tr, "y_tr": y_tr, "x_te": x_te, "y_te": y_te}, cache)
    return x_tr, y_tr, x_te, y_te