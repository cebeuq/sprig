"""Texel atlas renderer and leaf emission scoring (DESIGN.md section 4, atlas.py).

The renderer amortizes the terminal decoder into a canonical per-texel atlas of
discretized-logistic parameters: atlas [B, T_v, 40, 16, 16] (channel layout in
sprig/model/dl.py). A trainable per-texel additive bias grid [T_v, 40, 16, 16]
is added to the renderer output — the resurrection-writable parameterization
(F3.3 / M1.2): dead texels are revived by directly overwriting their bias rows.

FiLM mapping (illumination field Phi [B, 8, 16, 16] over the canvas), sampled
bilinearly at each leaf's canvas-center position -> phi [..., 8]:

    scale_c = 1 + phi[c]      for c in {0: R, 1: G, 2: B}
    shift_c =     phi[3 + c]  for c in {0: R, 1: G, 2: B}
    phi[6:8] reserved (unused in v0.1)

Each DL *mean* channel group (R/G/B means, identically across the 4 mixture
components) is modulated as mean' = mean * scale_c + shift_c. Log-scales,
weights and coupling coefficients are not modulated.
"""
from __future__ import annotations

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprig.model import dl
from sprig.model.gmt import CrossAttnBlock, caption_padding_mask

ATLAS_RES = 16
ATLAS_WIDTH = 256


_LOG_HALF_RANGE = math.log(127.5)


def _ell_chunk_math(
    pooled_g: torch.Tensor,
    crops_c: torch.Tensor,
    w_c: torch.Tensor,
    sc_c: torch.Tensor,
    sh_c: torch.Tensor,
) -> torch.Tensor:
    """One region-chunk of emission scores -> [B, ng, T_v] fp32.

    Runs under gradient checkpointing: the per-pixel intermediates are
    recomputed in backward instead of stored (storing them across all chunks
    is >90GB at batch 256). Exactly the dl.dl_logprob math with FiLM'd means
    (fp32 throughout), but unrolled over the 4 mixture components x 3 RGB
    channels so every intermediate is a [B,ng,T_v,h,w] slab instead of the
    [B,ng,T_v,4,3,h,w] block — torch.compile then fuses the whole chunk
    without materializing multi-GiB buffers (the tensorized form made
    inductor allocate the full block, ~18 GiB at B=128 leaf_chunk=49).
    """
    # NOTE: a bf16 variant (inputs cast down, fp32 pixel-sum) was tried and
    # REVERTED: eager bf16 shifts ell by tens of nats (bf16 ulp at |ell|~1e3
    # is ~8) and fails the 0.1% loss parity gate by ~40x.
    p = pooled_g.float()                                     # [B,T_v,40,h,w]
    x = crops_c.float()                                      # [B,ng,3,h,w]
    sc = sc_c.float()                                        # [B,ng,3]
    sh = sh_c.float()
    # Mixture-weight log-softmax over the 4 weight channels (pooled-sized).
    wl = [p[:, :, 10 * j] for j in range(dl.N_COMP)]         # [B,T_v,h,w] each
    wm = torch.logaddexp(torch.logaddexp(wl[0], wl[1]),
                         torch.logaddexp(wl[2], wl[3]))      # logsumexp_j
    x_ch = [x[:, :, c].unsqueeze(2) for c in range(3)]       # [B,ng,1,h,w]

    mix: Optional[torch.Tensor] = None
    for j in range(dl.N_COMP):
        acc: Optional[torch.Tensor] = None                   # sum_c log P_j(c)
        for c in range(3):
            mu = p[:, :, 10 * j + 1 + c].unsqueeze(1)        # [B,1,T_v,h,w]
            ls = p[:, :, 10 * j + 4 + c].clamp(
                dl.LOGSCALE_MIN, dl.LOGSCALE_MAX).unsqueeze(1)
            m = mu * sc[:, :, c, None, None, None] + sh[:, :, c, None, None, None]
            if c == 1:
                al = torch.tanh(p[:, :, 10 * j + 7]).unsqueeze(1)
                m = m + al * x_ch[0]
            elif c == 2:
                be = torch.tanh(p[:, :, 10 * j + 8]).unsqueeze(1)
                ga = torch.tanh(p[:, :, 10 * j + 9]).unsqueeze(1)
                m = m + be * x_ch[0] + ga * x_ch[1]
            xc = x_ch[c]
            centered = xc - m                                # [B,ng,T_v,h,w]
            inv_std = torch.exp(-ls)
            plus_in = inv_std * (centered + dl._HALF_BIN)
            min_in = inv_std * (centered - dl._HALF_BIN)
            log_cdf_plus = plus_in - F.softplus(plus_in)
            log_one_minus_cdf_min = -F.softplus(min_in)
            cdf_delta = torch.sigmoid(plus_in) - torch.sigmoid(min_in)
            mid_in = inv_std * centered
            log_pdf_mid = mid_in - ls - 2.0 * F.softplus(mid_in)
            log_prob_mid = torch.where(
                cdf_delta > 1e-5,
                torch.log(cdf_delta.clamp(min=1e-12)),
                log_pdf_mid - _LOG_HALF_RANGE,
            )
            lp = torch.where(
                xc < -0.999,
                log_cdf_plus,
                torch.where(xc > 0.999, log_one_minus_cdf_min, log_prob_mid),
            )
            acc = lp if acc is None else acc + lp
        pc = acc + (wl[j] - wm).unsqueeze(1)                 # log w_j + log P_j
        mix = pc if mix is None else torch.logaddexp(mix, pc)
    # Per-pixel importance weights (object-pixel up-weighting; all-ones in the
    # unweighted/eval path — multiplying by exactly 1.0 is bit-identical fp32,
    # so likelihood parity is preserved).
    return (mix * w_c.float().unsqueeze(2)).sum(dim=(-1, -2)).float()  # [B,ng,T_v]


_ELL_COMPILED = None  # lazily-built torch.compile wrapper of _ell_chunk_math


def _ell_chunk_fn(device: torch.device):
    """Fused (torch.compile) emission-chunk kernel on CUDA; the eager
    reference math elsewhere. The eager path is memory-bound on dozens of
    unfused fp32 elementwise kernel passes per chunk; fusing them is a >3x
    end-to-end step win. Same math, same dtypes — kill switch:
    SPRIG_COMPILE=0."""
    global _ELL_COMPILED
    if device.type != "cuda" or os.environ.get("SPRIG_COMPILE", "1") == "0":
        return _ell_chunk_math
    if _ELL_COMPILED is None:
        try:
            # One graph per (group h/w, chunk rows, batch) shape; a process
            # that probes several batch sizes exceeds dynamo's default limit
            # of 8 and would silently fall back to slow eager for the rest.
            import torch._dynamo.config as _dcfg
            if getattr(_dcfg, "recompile_limit", 0) < 64:
                _dcfg.recompile_limit = 64
            if getattr(_dcfg, "cache_size_limit", 0) < 64:
                _dcfg.cache_size_limit = 64
            _ELL_COMPILED = torch.compile(_ell_chunk_math, dynamic=False)
        except Exception:
            _ELL_COMPILED = _ell_chunk_math
    return _ELL_COMPILED


def film_scale_shift(phi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """phi [..., 8] -> (scale [..., 3], shift [..., 3]) per RGB mean group."""
    return 1.0 + phi[..., 0:3], phi[..., 3:6]


def phi_at_leaf_centers(
    Phi: torch.Tensor, rects: torch.Tensor, canvas_px: int
) -> torch.Tensor:
    """Bilinearly sample Phi [B, 8, 16, 16] at leaf-center canvas positions.

    rects int [n, 4] (x0, y0, x1, y1 in px) -> phi [B, n, 8].
    """
    B = Phi.shape[0]
    cx = (rects[:, 0] + rects[:, 2]).float() / (2.0 * canvas_px) * 2.0 - 1.0
    cy = (rects[:, 1] + rects[:, 3]).float() / (2.0 * canvas_px) * 2.0 - 1.0
    grid = torch.stack([cx, cy], dim=-1).to(Phi.device)          # [n, 2] (x, y)
    grid = grid.view(1, -1, 1, 2).expand(B, -1, -1, -1)
    out = F.grid_sample(Phi.float(), grid, mode="bilinear", align_corners=False)
    return out.squeeze(-1).permute(0, 2, 1)                      # [B, n, 8]


class TexelAtlas(nn.Module):
    """cfg is duck-typed: needs T_v, d, caption_dim (optional: atlas_heads,
    leaf_chunk)."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        T_v, d = cfg.T_v, cfg.d
        heads = getattr(cfg, "atlas_heads", 4)
        self.leaf_chunk = int(getattr(cfg, "leaf_chunk", 16))

        self.E_T = nn.Parameter(torch.randn(T_v, d) * 0.02)
        self.q_proj = nn.Linear(d, ATLAS_WIDTH)
        self.cap_proj = nn.Linear(cfg.caption_dim, ATLAS_WIDTH)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(ATLAS_WIDTH, heads) for _ in range(2)]
        )
        self.ln_out = nn.LayerNorm(ATLAS_WIDTH)
        self.seed = nn.Linear(ATLAS_WIDTH, ATLAS_WIDTH * 4 * 4)
        self.conv1 = nn.Conv2d(ATLAS_WIDTH, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, dl.N_CH, kernel_size=3, padding=1)
        # Resurrection-writable per-texel bias grid.
        self.bias_grid = nn.Parameter(torch.zeros(T_v, dl.N_CH, ATLAS_RES, ATLAS_RES))

    def render(self, emb: torch.Tensor, emb_len: torch.Tensor) -> torch.Tensor:
        """emb [B, L, 768], emb_len [B] -> atlas [B, T_v, 40, 16, 16]."""
        B, L, _ = emb.shape
        T_v = self.E_T.shape[0]
        kv = self.cap_proj(emb.float())
        kpm = caption_padding_mask(emb_len, L, emb.device)

        h = self.q_proj(self.E_T).unsqueeze(0).expand(B, -1, -1)  # [B, T_v, 256]
        for blk in self.blocks:
            h = blk(h, kv, kpm)
        h = self.ln_out(h)

        z = self.seed(h).reshape(B * T_v, ATLAS_WIDTH, 4, 4)
        z = F.gelu(self.conv1(z))
        z = F.interpolate(z, scale_factor=2, mode="nearest")      # [.., 128, 8, 8]
        z = self.conv2(z)
        z = F.interpolate(z, scale_factor=2, mode="nearest")      # [.., 40, 16, 16]
        atlas = z.reshape(B, T_v, dl.N_CH, ATLAS_RES, ATLAS_RES)
        return atlas + self.bias_grid.unsqueeze(0)

    def score_leaves(
        self,
        atlas: torch.Tensor,
        images: torch.Tensor,
        lattice,
        Phi: torch.Tensor,
        chunk: Optional[int] = None,
        pix_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Emission log-likelihoods for every (leaf region, texel) pair.

        atlas [B, T_v, 40, 16, 16], images u8 [B, C, C, 3] (C = canvas),
        Phi [B, 8, 16, 16] -> ell fp32 [B, n_leaf_regions, T_v], summed over
        the region's pixels, in lattice.leaf_ids slot order. Vectorized per
        leaf shape group, chunked over regions to bound memory.

        pix_weight (optional) fp32 [B, C, C]: per-pixel importance weights on
        the log-likelihood (object-pixel up-weighting during training). None
        means all-ones (exact NLL) — used by log_marginal/eval/parsing.
        """
        B, T_v = atlas.shape[0], atlas.shape[1]
        canvas = lattice.canvas_px
        step = int(chunk) if chunk is not None else self.leaf_chunk

        x = dl.u8_to_unit(images).permute(0, 3, 1, 2).contiguous()   # [B,3,C,C]
        x_flat = x.reshape(B, 3, canvas * canvas)
        if pix_weight is None:
            w_flat = torch.ones(B, canvas * canvas, dtype=torch.float32,
                                device=atlas.device)
        else:
            w_flat = pix_weight.to(atlas.device, torch.float32).reshape(
                B, canvas * canvas)

        n_leaf = lattice.n_leaf_regions
        ell = torch.zeros(B, n_leaf, T_v, dtype=torch.float32, device=atlas.device)

        leaf_rects = lattice.regions[lattice.leaf_ids]
        phi_leaf = phi_at_leaf_centers(Phi, leaf_rects, canvas)      # [B,n_leaf,8]
        scale, shift = film_scale_shift(phi_leaf)                    # [B,n_leaf,3]

        use_ckpt = torch.is_grad_enabled() and (
            atlas.requires_grad or Phi.requires_grad)
        chunk_fn = _ell_chunk_fn(atlas.device)

        groups = getattr(lattice, "shape_groups", None)
        if groups is None:  # foreign/stub lattice: derive on the fly
            groups = []
            for (h, w), slots in lattice.leaf_shape_groups().items():
                slots = slots.to(atlas.device)
                rects = lattice.regions[lattice.leaf_ids[slots]]     # [n_g,4]
                dy = torch.arange(h, device=atlas.device)
                dx = torch.arange(w, device=atlas.device)
                yy = rects[:, 1].view(-1, 1, 1).to(atlas.device) + dy.view(1, -1, 1)
                xx = rects[:, 0].view(-1, 1, 1).to(atlas.device) + dx.view(1, 1, -1)
                groups.append((h, w, slots, (yy * canvas + xx).reshape(-1)))

        for h, w, slots, pix in groups:
            if (h, w) == (ATLAS_RES, ATLAS_RES):
                pooled = atlas                                       # identity pool
            else:
                pooled = F.adaptive_avg_pool2d(
                    atlas.reshape(B * T_v, dl.N_CH, ATLAS_RES, ATLAS_RES), (h, w)
                ).reshape(B, T_v, dl.N_CH, h, w)

            n_g = slots.shape[0]
            crops = x_flat[:, :, pix].reshape(B, 3, n_g, h, w).permute(0, 2, 1, 3, 4)
            w_crops = w_flat[:, pix].reshape(B, n_g, h, w)

            sc = scale[:, slots]                                     # [B,n_g,3]
            sh = shift[:, slots]

            for i0 in range(0, n_g, step):
                i1 = min(n_g, i0 + step)
                args = (pooled, crops[:, i0:i1].contiguous(),
                        w_crops[:, i0:i1].contiguous(),
                        sc[:, i0:i1].contiguous(), sh[:, i0:i1].contiguous())
                if use_ckpt:
                    tot = torch.utils.checkpoint.checkpoint(
                        chunk_fn, *args, use_reentrant=False)
                else:
                    tot = chunk_fn(*args)
                ell[:, slots[i0:i1], :] = tot

        return ell
