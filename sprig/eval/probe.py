"""3-conv-block CNN probe: shape (8-way) + color (8-way) heads.

Used for the compositional-holdout score: trained on a probe dataset that
DOES include the held-out combos (leakage-safe by directory separation),
then applied to model generations.

Labeled image dir formats accepted by `train` (checked in this order):
1. memmap pair: `images.u8` [N,64,64,3] + `labels.npy` int64 [N,2]
   (shape_idx, color_idx) in the canonical prompts.SHAPES / prompts.COLORS order;
2. flat PNG files named `<shape>_<color>_<anything>.png`.
"""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sprig.eval.prompts import COLORS, SHAPES


class ProbeCNN(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        w = width
        self.blocks = nn.Sequential(
            nn.Conv2d(3, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.MaxPool2d(2),                                                     # 32
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(),
            nn.MaxPool2d(2),                                                     # 16
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.Conv2d(4 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.MaxPool2d(2),                                                     # 8
        )
        self.shape_head = nn.Linear(4 * w, len(SHAPES))
        self.color_head = nn.Linear(4 * w, len(COLORS))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.blocks(x)
        h = h.mean(dim=(2, 3))
        return self.shape_head(h), self.color_head(h)


def _images_to_tensor(images: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    """u8 [N,64,64,3] (numpy or torch) -> float [N,3,64,64] in [-1,1]."""
    if isinstance(images, np.ndarray):
        images = torch.from_numpy(np.array(images, dtype=images.dtype))
    x = images.to(torch.float32) / 127.5 - 1.0
    return x.permute(0, 3, 1, 2).contiguous()


def _load_probe_dir(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    img_path = os.path.join(data_dir, "images.u8")
    lab_path = os.path.join(data_dir, "labels.npy")
    if os.path.exists(img_path) and os.path.exists(lab_path):
        labels = np.load(lab_path).astype(np.int64)
        n = labels.shape[0]
        images = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(n, 64, 64, 3))
        return np.asarray(images), labels
    files = sorted(glob.glob(os.path.join(data_dir, "*.png")))
    if not files:
        raise FileNotFoundError(
            "probe dir {} has neither images.u8+labels.npy nor *.png".format(data_dir)
        )
    from PIL import Image

    imgs: List[np.ndarray] = []
    labs: List[Tuple[int, int]] = []
    for f in files:
        parts = os.path.basename(f).split("_")
        shape_name, color_name = parts[0], parts[1]
        labs.append((SHAPES.index(shape_name), COLORS.index(color_name)))
        imgs.append(np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8))
    return np.stack(imgs), np.asarray(labs, dtype=np.int64)


def train(
    data_dir: str,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "cpu",
    ckpt_path: Optional[str] = None,
    seed: int = 0,
) -> str:
    """Train the probe on a labeled image dir; returns the checkpoint path."""
    images, labels = _load_probe_dir(data_dir)
    torch.manual_seed(seed)
    model = ProbeCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = images.shape[0]
    gen = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        order = gen.permutation(n)
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            x = _images_to_tensor(images[idx]).to(device)
            ys = torch.from_numpy(labels[idx, 0]).to(device)
            yc = torch.from_numpy(labels[idx, 1]).to(device)
            logit_s, logit_c = model(x)
            loss = F.cross_entropy(logit_s, ys) + F.cross_entropy(logit_c, yc)
            opt.zero_grad()
            loss.backward()
            opt.step()
    if ckpt_path is None:
        ckpt_path = os.path.join(data_dir, "probe.pt")
    torch.save(
        {"state_dict": model.state_dict(), "shapes": SHAPES, "colors": COLORS}, ckpt_path
    )
    return ckpt_path


def load_probe(ckpt_path: str, device: str = "cpu") -> ProbeCNN:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = ProbeCNN().to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def score_generations(
    images: Union[np.ndarray, torch.Tensor],
    expected_shape: str,
    expected_color: str,
    ckpt_path: Union[str, ProbeCNN],
    device: str = "cpu",
    batch_size: int = 64,
) -> float:
    """Fraction of images classified as (expected_shape AND expected_color)."""
    model = ckpt_path if isinstance(ckpt_path, ProbeCNN) else load_probe(ckpt_path, device)
    model.eval()
    target_s = SHAPES.index(expected_shape)
    target_c = COLORS.index(expected_color)
    x_all = _images_to_tensor(images)
    hits = 0
    for i in range(0, x_all.shape[0], batch_size):
        x = x_all[i:i + batch_size].to(device)
        logit_s, logit_c = model(x)
        pred_s = logit_s.argmax(dim=1)
        pred_c = logit_c.argmax(dim=1)
        hits += int(((pred_s == target_s) & (pred_c == target_c)).sum().item())
    return hits / float(x_all.shape[0])
