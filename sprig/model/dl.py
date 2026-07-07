"""Discretized-logistic mixture emissions (PixelCNN++ style) with RGB coupling.

Parameter layout (the "40 channels", DESIGN.md section 4): 4 mixture components,
each a contiguous block of 10 channels, so channel index = 10*j + t for
component j in 0..3 and t in:

    t = 0      mixture-weight logit
    t = 1..3   raw means for (R, G, B)                 (image scaled to [-1, 1])
    t = 4..6   log-scales for (R, G, B)                (clamped to [-7, 2])
    t = 7..9   channel-coupling coeffs (alpha: G|R, beta: B|R, gamma: B|G),
               squashed by tanh inside this module.

Coupled means (per component): m_R = mu_R;  m_G = mu_G + alpha * x_R;
m_B = mu_B + beta * x_R + gamma * x_G, where x_R/x_G are OBSERVED values in
[-1, 1] (teacher-forced coupling, exactly as in PixelCNN++).

All math is done in fp32 regardless of input dtype.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

N_COMP = 4
CH_PER_COMP = 10
N_CH = N_COMP * CH_PER_COMP  # 40

WEIGHT_IDX = [10 * j for j in range(N_COMP)]
MEAN_IDX = [10 * j + 1 + c for j in range(N_COMP) for c in range(3)]
LOGSCALE_IDX = [10 * j + 4 + c for j in range(N_COMP) for c in range(3)]
COEFF_IDX = [10 * j + 7 + c for j in range(N_COMP) for c in range(3)]

LOGSCALE_MIN = -7.0
LOGSCALE_MAX = 2.0
_HALF_BIN = 1.0 / 255.0  # half bin width in [-1, 1] scale (256 levels)


def unpack_dl_params(
    params: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split raw params [..., 40, H, W] into mixture pieces.

    Returns (logit_w [..., 4, H, W], means [..., 4, 3, H, W],
    log_scales [..., 4, 3, H, W] clamped, coeffs [..., 4, 3, H, W] tanh-ed).
    """
    if params.shape[-3] != N_CH:
        raise ValueError("expected %d param channels, got %d" % (N_CH, params.shape[-3]))
    p = params.float()
    lead = p.shape[:-3]
    hw = p.shape[-2:]
    p = p.reshape(lead + (N_COMP, CH_PER_COMP) + hw)
    logit_w = p[..., 0, :, :]
    means = p[..., 1:4, :, :]
    log_scales = p[..., 4:7, :, :].clamp(LOGSCALE_MIN, LOGSCALE_MAX)
    coeffs = torch.tanh(p[..., 7:10, :, :])
    return logit_w, means, log_scales, coeffs


def couple_means(
    means: torch.Tensor, coeffs: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Coupled per-component means given observed pixels.

    means/coeffs: [..., 4, 3, H, W]; x: broadcastable to [..., 3, H, W],
    values in [-1, 1]. Returns coupled means [..., 4, 3, H, W].
    """
    xb = x.float().unsqueeze(-4)  # [..., 1, 3, H, W]
    x_r = xb[..., 0, :, :]
    x_g = xb[..., 1, :, :]
    m_r = means[..., 0, :, :]
    m_g = means[..., 1, :, :] + coeffs[..., 0, :, :] * x_r
    m_b = means[..., 2, :, :] + coeffs[..., 1, :, :] * x_r + coeffs[..., 2, :, :] * x_g
    m_r, m_g, m_b = torch.broadcast_tensors(m_r, m_g, m_b)
    return torch.stack([m_r, m_g, m_b], dim=-3)


def dl_channel_logprobs(
    means: torch.Tensor,
    log_scales: torch.Tensor,
    coeffs: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """Per-component, per-channel discretized-logistic log P(bin of x).

    means/log_scales/coeffs: [..., 4, 3, H, W] (log_scales already clamped,
    coeffs already tanh-ed — i.e. outputs of unpack_dl_params). x broadcastable
    to [..., 3, H, W] in [-1, 1] on the 256-level grid. Returns [..., 4, 3, H, W].
    Edge bins (0 and 255) integrate the full tails, so per-channel bin
    probabilities sum to 1.
    """
    xb = x.float().unsqueeze(-4)
    m = couple_means(means, coeffs, x)
    centered = xb - m
    inv_std = torch.exp(-log_scales)
    plus_in = inv_std * (centered + _HALF_BIN)
    min_in = inv_std * (centered - _HALF_BIN)

    log_cdf_plus = plus_in - F.softplus(plus_in)          # log sigmoid(plus_in)
    log_one_minus_cdf_min = -F.softplus(min_in)           # log(1 - sigmoid(min_in))
    cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(min_in)
    mid_in = inv_std * centered
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    log_prob_mid = torch.where(
        cdf_delta > 1e-5,
        torch.log(cdf_delta.clamp(min=1e-12)),
        log_pdf_mid - math.log(127.5),
    )
    logp = torch.where(
        xb < -0.999,
        log_cdf_plus,
        torch.where(xb > 0.999, log_one_minus_cdf_min, log_prob_mid),
    )
    return logp


def dl_logprob(params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Full mixture log-prob per pixel.

    params [..., 40, H, W] raw; x broadcastable to [..., 3, H, W] in [-1, 1].
    Returns fp32 [..., H, W]: log p(x_pixel) = logsumexp_j (log w_j +
    sum_channels log P_j(channel)).
    """
    logit_w, means, log_scales, coeffs = unpack_dl_params(params)
    ch = dl_channel_logprobs(means, log_scales, coeffs, x)      # [..., 4, 3, H, W]
    per_comp = ch.sum(dim=-3)                                    # [..., 4, H, W]
    log_w = F.log_softmax(logit_w, dim=-3)
    return torch.logsumexp(log_w + per_comp, dim=-3)


def dl_mean_pixels(params: torch.Tensor) -> torch.Tensor:
    """Deterministic 'DL-mean' rendering: per pixel pick the argmax-weight
    component, then roll out the coupled means sequentially
    (R = mu_R, G = mu_G + a R, B = mu_B + b R + c G), clamped to [-1, 1].

    params [..., 40, H, W] -> pixels [..., 3, H, W] fp32 in [-1, 1].
    """
    logit_w, means, log_scales, coeffs = unpack_dl_params(params)
    idx = logit_w.argmax(dim=-3)                                 # [..., H, W]
    gather_idx = idx.unsqueeze(-3).unsqueeze(-4).expand(
        means.shape[:-4] + (1, 3) + means.shape[-2:]
    )
    m = means.gather(-4, gather_idx).squeeze(-4)                 # [..., 3, H, W]
    cf = coeffs.gather(-4, gather_idx).squeeze(-4)
    r = m[..., 0, :, :].clamp(-1.0, 1.0)
    g = (m[..., 1, :, :] + cf[..., 0, :, :] * r).clamp(-1.0, 1.0)
    b = (m[..., 2, :, :] + cf[..., 1, :, :] * r + cf[..., 2, :, :] * g).clamp(-1.0, 1.0)
    return torch.stack([r, g, b], dim=-3)


def u8_to_unit(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 [0,255] -> fp32 [-1, 1] on the 256-level grid."""
    return x_u8.float() / 127.5 - 1.0


def unit_to_u8(x: torch.Tensor) -> torch.Tensor:
    """fp32 [-1, 1] -> uint8, rounding to the nearest of the 256 levels."""
    return ((x.clamp(-1.0, 1.0) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
