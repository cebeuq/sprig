from __future__ import annotations

import torch

from sprig.model import dl


def test_bin_probs_sum_to_one_per_channel():
    torch.manual_seed(0)
    B, H, W = 2, 5, 3
    params = torch.randn(B, dl.N_CH, H, W) * 2.0
    x_obs = dl.u8_to_unit(torch.randint(0, 256, (B, 3, H, W), dtype=torch.uint8))
    _lw, means, log_scales, coeffs = dl.unpack_dl_params(params)
    bins = dl.u8_to_unit(torch.arange(256, dtype=torch.uint8))
    for c in range(3):
        x = x_obs.unsqueeze(0).repeat(256, 1, 1, 1, 1)  # [256,B,3,H,W]
        x[:, :, c] = bins.view(256, 1, 1, 1)
        ch = dl.dl_channel_logprobs(
            means.unsqueeze(0), log_scales.unsqueeze(0), coeffs.unsqueeze(0), x
        )  # [256,B,4,3,H,W]
        probs = ch.exp()[:, :, :, c].sum(0)  # [B,4,H,W]
        assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_bin_probs_sum_extreme_scales():
    torch.manual_seed(1)
    params = torch.randn(1, dl.N_CH, 4, 4)
    for j in range(dl.N_COMP):
        params[:, 10 * j + 1 : 10 * j + 4] = torch.rand(1, 3, 4, 4) * 2 - 1
        params[:, 10 * j + 4 : 10 * j + 7] = -9.0  # clamps to -7 (sharpest)
    _lw, means, log_scales, coeffs = dl.unpack_dl_params(params)
    assert float(log_scales.min()) == -7.0
    bins = dl.u8_to_unit(torch.arange(256, dtype=torch.uint8))
    x = torch.zeros(256, 1, 3, 4, 4)
    x[:, :, 0] = bins.view(256, 1, 1, 1)
    ch = dl.dl_channel_logprobs(
        means.unsqueeze(0), log_scales.unsqueeze(0), coeffs.unsqueeze(0), x
    )
    probs = ch.exp()[:, :, :, 0].sum(0)
    assert torch.allclose(probs, torch.ones_like(probs), atol=1e-4)


def test_logprob_shape_and_finite():
    torch.manual_seed(2)
    params = torch.randn(3, 2, dl.N_CH, 8, 8)
    x = dl.u8_to_unit(torch.randint(0, 256, (3, 2, 3, 8, 8), dtype=torch.uint8))
    lp = dl.dl_logprob(params, x)
    assert lp.shape == (3, 2, 8, 8)
    assert lp.dtype == torch.float32
    assert torch.isfinite(lp).all()


def test_mean_pixels_known_values():
    params = torch.zeros(1, dl.N_CH, 2, 2)
    params[:, 0] = 5.0                      # component 0 dominates
    params[:, 1] = 0.3                      # mu_R
    params[:, 2] = -0.2                     # mu_G
    params[:, 3] = 0.5                      # mu_B
    pix = dl.dl_mean_pixels(params)
    assert pix.shape == (1, 3, 2, 2)
    assert torch.allclose(pix[:, 0], torch.full((1, 2, 2), 0.3))
    assert torch.allclose(pix[:, 1], torch.full((1, 2, 2), -0.2))
    assert torch.allclose(pix[:, 2], torch.full((1, 2, 2), 0.5))
    # Coupling: large raw alpha -> tanh ~ 1 -> G = mu_G + R.
    params[:, 7] = 20.0
    pix = dl.dl_mean_pixels(params)
    expected_g = -0.2 + float(torch.tanh(torch.tensor(20.0))) * 0.3
    assert torch.allclose(pix[:, 1], torch.full((1, 2, 2), expected_g), atol=1e-5)


def test_u8_roundtrip():
    v = torch.arange(256, dtype=torch.uint8).view(1, 16, 16)
    assert torch.equal(dl.unit_to_u8(dl.u8_to_unit(v)), v)


def test_logprob_grad_flows():
    torch.manual_seed(3)
    params = torch.randn(2, dl.N_CH, 4, 4, requires_grad=True)
    x = dl.u8_to_unit(torch.randint(0, 256, (2, 3, 4, 4), dtype=torch.uint8))
    lp = dl.dl_logprob(params, x).sum()
    lp.backward()
    assert params.grad is not None
    assert torch.isfinite(params.grad).all()
