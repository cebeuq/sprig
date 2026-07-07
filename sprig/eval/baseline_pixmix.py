"""B0 trivial baseline: position-dependent 4-component DL mixture per pixel.

One parameter grid [64, 64, 40] (no neural net, no caption): per pixel, the
same 40-channel discretized-logistic mixture parameterization as the texel
atlas (sprig/model/dl.py layout: 4 components x contiguous blocks of
[weight, 3 means, 3 log-scales, 3 coupling coeffs]). B0's bpd measures
exactly what the grammar buys over a caption-free per-pixel model.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from sprig.model.dl import LOGSCALE_IDX, N_CH, N_COMP, dl_logprob, u8_to_unit

LOG2 = math.log(2.0)
N_PARAMS = N_CH  # 40


def _to_u8_tensor(images_u8: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if isinstance(images_u8, np.ndarray):
        images_u8 = torch.from_numpy(np.array(images_u8, dtype=images_u8.dtype))
    return images_u8


class PixMixBaseline(nn.Module):
    """params: [H, W, 40] trainable grid; no other parameters."""

    def __init__(self, height: int = 64, width: int = 64):
        super().__init__()
        p = torch.zeros(height, width, N_PARAMS)
        # spread component means so the mixture starts multi-modal
        comp_means = torch.linspace(-0.6, 0.6, N_COMP)
        for j in range(N_COMP):
            p[..., 10 * j + 1: 10 * j + 4] = comp_means[j]
        p[..., LOGSCALE_IDX] = -1.0
        self.params = nn.Parameter(p)

    def logprob_per_image(self, images_u8: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """[B] total log-likelihood (nats) per image."""
        img = _to_u8_tensor(images_u8).to(self.params.device)
        x = u8_to_unit(img).permute(0, 3, 1, 2)               # [B,3,H,W]
        # expand (view) params to the batch lead dim: dl_logprob broadcasts
        # x against params but not vice versa
        p = self.params.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        lp = dl_logprob(p, x)                                 # [B,H,W] fp32
        return lp.sum(dim=(1, 2))

    @torch.no_grad()
    def bpd(self, images_u8: Union[np.ndarray, torch.Tensor], batch_size: int = 256) -> float:
        """Mean bits/dim over images (dims = 3*H*W)."""
        n = int(images_u8.shape[0])
        h, w = self.params.shape[0], self.params.shape[1]
        total = 0.0
        for i in range(0, n, batch_size):
            total += float(self.logprob_per_image(images_u8[i:i + batch_size]).sum())
        return -total / (n * 3.0 * h * w * LOG2)


def fit(
    images_u8: Union[np.ndarray, torch.Tensor],
    steps: int = 2000,
    lr: float = 5e-2,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = 0,
) -> PixMixBaseline:
    """Fit B0 by Adam on random minibatches of training images; returns the model."""
    torch.manual_seed(seed)
    model = PixMixBaseline().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = images_u8.shape[0]
    gen = np.random.default_rng(seed)
    for _ in range(steps):
        idx = gen.integers(0, n, size=min(batch_size, n))
        if isinstance(images_u8, np.ndarray):
            batch = images_u8[idx]
        else:
            batch = images_u8[torch.from_numpy(idx)]
        nll = -model.logprob_per_image(batch).mean() / (3.0 * 64 * 64)
        opt.zero_grad()
        nll.backward()
        opt.step()
    return model
