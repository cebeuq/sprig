from __future__ import annotations

import math

import numpy as np
import torch

from sprig.eval import baseline_pixmix as b0
from sprig.model.dl import LOGSCALE_IDX, MEAN_IDX, dl_logprob


def test_dl_logprob_normalizes():
    """With 4 identical components and zero coupling, the per-channel
    distribution over the 256 bins must sum to 1."""
    params = torch.zeros(b0.N_PARAMS, 1, 1)
    params[MEAN_IDX] = 0.1   # same mean for every component/channel
    params[LOGSCALE_IDX] = -2.0
    vals = torch.arange(256, dtype=torch.float32) / 127.5 - 1.0
    x = vals[:, None, None, None].repeat(1, 3, 1, 1)  # [256,3,1,1], channels equal
    lp = dl_logprob(params.unsqueeze(0).expand(256, -1, -1, -1), x)  # [256,1,1]
    # identical channels + identical comps => lp = 3 * per-channel logprob
    total = torch.exp(lp / 3.0).sum()
    assert abs(float(total) - 1.0) < 1e-3, float(total)


def test_dl_logprob_prefers_matching_pixels():
    params = torch.zeros(b0.N_PARAMS, 1, 1)
    params[MEAN_IDX] = 0.5  # means at +0.5
    params[LOGSCALE_IDX] = -3.0
    near = dl_logprob(params.unsqueeze(0), torch.full((1, 3, 1, 1), 0.5))
    far = dl_logprob(params.unsqueeze(0), torch.full((1, 3, 1, 1), -0.9))
    assert float(near.sum()) > float(far.sum())


def _toy_images(n=64, seed=0):
    """Position-dependent but image-independent data: left half dark,
    right half bright, small noise — exactly what B0 can model."""
    rng = np.random.default_rng(seed)
    imgs = np.zeros((n, 64, 64, 3), dtype=np.int16)
    imgs[:, :, :32, :] = 40
    imgs[:, :, 32:, :] = 210
    imgs = imgs + rng.integers(-6, 7, size=imgs.shape)
    return np.clip(imgs, 0, 255).astype(np.uint8)


def test_fit_improves_bpd_and_shapes():
    imgs = _toy_images()
    fresh = b0.PixMixBaseline()
    bpd_before = fresh.bpd(imgs[:16])
    model = b0.fit(imgs, steps=150, lr=0.1, batch_size=32, seed=0)
    assert tuple(model.params.shape) == (64, 64, 40)
    bpd_after = model.bpd(imgs[:16])
    assert math.isfinite(bpd_after)
    assert bpd_after < bpd_before - 1.0, (bpd_before, bpd_after)
    # near-deterministic data should compress well below 8 bits
    assert bpd_after < 5.0


def test_bpd_accepts_torch_and_numpy():
    imgs = _toy_images(n=8)
    model = b0.PixMixBaseline()
    a = model.bpd(imgs)
    b = model.bpd(torch.from_numpy(imgs))
    assert abs(a - b) < 1e-6
