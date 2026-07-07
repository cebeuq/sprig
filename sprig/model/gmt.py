"""Grammar Modulation Transformer (DESIGN.md section 4, gmt.py).

Symbol embeddings are queries; caption tokens are keys/values. There is no
symbol-symbol self-attention. Outputs the caption-conditioned quantities the
grammar needs: H, U (p(k|A,c) logits), cut-type logits, illumination field Phi,
plus the static tables V, W, P_T and the factorized termination head.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CUT_TYPES = 14


class CrossAttnBlock(nn.Module):
    """Pre-LN {cross-attention(queries -> memory), FFN width*4} block."""

    def __init__(self, width: int, n_heads: int) -> None:
        super().__init__()
        self.ln_q = nn.LayerNorm(width)
        self.ln_kv = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True)
        self.ln_f = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        kvn = self.ln_kv(kv)
        a, _ = self.attn(
            self.ln_q(q), kvn, kvn,
            key_padding_mask=key_padding_mask, need_weights=False,
        )
        h = q + a
        return h + self.ffn(self.ln_f(h))


def caption_padding_mask(emb_len: torch.Tensor, L: int, device: torch.device) -> torch.Tensor:
    """bool [B, L], True = padding (ignored by attention). Position 0 is always
    kept valid as a guard against fully-masked rows (NaN attention)."""
    idx = torch.arange(L, device=device)
    mask = idx.unsqueeze(0) >= emb_len.to(device).long().unsqueeze(1)
    mask[:, 0] = False
    return mask


class GMTOut(NamedTuple):
    H: torch.Tensor            # [B, S, d]
    U: torch.Tensor            # [B, S, R]   p(k|A,c) logits
    cut_logits: torch.Tensor   # [B, R, 14]  cut-type logits per component
    Phi: torch.Tensor          # [B, 8, 16, 16] illumination field


class GrammarModulationTransformer(nn.Module):
    """cfg is duck-typed: needs S, R, T_v, d, n_heads, d_t, caption_dim, n_geom."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        S, R, T_v, d = cfg.S, cfg.R, cfg.T_v, cfg.d
        if d % cfg.n_heads != 0:
            raise ValueError("d must be divisible by n_heads")

        self.E_N = nn.Parameter(torch.randn(S, d) * 0.02)
        self.cap_proj = nn.Linear(cfg.caption_dim, d)
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(d, cfg.n_heads) for _ in range(4)]
        )
        self.ln_out = nn.LayerNorm(d)

        # Heads.
        self.W_u = nn.Linear(d, R)
        self.mlp_h = nn.Sequential(nn.Linear(d, cfg.d_t), nn.GELU(), nn.Linear(cfg.d_t, cfg.d_t))
        self.mlp_g = nn.Sequential(nn.Linear(cfg.n_geom, cfg.d_t), nn.GELU(), nn.Linear(cfg.d_t, cfg.d_t))
        self.term_bias = nn.Parameter(torch.zeros(S))

        # Cut-type head: component embeddings cross-attend once to the caption.
        self.e_k = nn.Parameter(torch.randn(R, d) * 0.02)
        self.cut_block = CrossAttnBlock(d, cfg.n_heads)
        self.cut_ln = nn.LayerNorm(d)
        self.cut_out = nn.Linear(d, N_CUT_TYPES)

        # Static grammar tables (caption-independent).
        self.P_T = nn.Parameter(torch.randn(R, T_v) * 0.02)
        self.V = nn.Parameter(torch.randn(R, S) * 0.02)
        self.W = nn.Parameter(torch.randn(R, S) * 0.02)

        # Illumination field: masked-mean-pooled caption -> MLP -> 2x deconvs.
        self.phi_mlp = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 32 * 4 * 4))
        self.phi_deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.phi_deconv2 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        # Zero-init the final deconv so Phi == 0 at init (FiLM starts at identity).
        nn.init.zeros_(self.phi_deconv2.weight)
        nn.init.zeros_(self.phi_deconv2.bias)

    def forward(self, emb: torch.Tensor, emb_len: torch.Tensor) -> GMTOut:
        """emb [B, L, 768] (any float dtype), emb_len [B] int -> GMTOut."""
        B, L, _ = emb.shape
        kv = self.cap_proj(emb.float())                       # [B, L, d]
        kpm = caption_padding_mask(emb_len, L, emb.device)

        h = self.E_N.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            h = blk(h, kv, kpm)
        H = self.ln_out(h)                                    # [B, S, d]

        U = self.W_u(H)                                       # [B, S, R]

        q_k = self.e_k.unsqueeze(0).expand(B, -1, -1)
        hk = self.cut_block(q_k, kv, kpm)
        cut_logits = self.cut_out(self.cut_ln(hk))            # [B, R, 14]

        keep = (~kpm).float().unsqueeze(-1)                   # [B, L, 1]
        pooled = (kv * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        z = self.phi_mlp(pooled).reshape(B, 32, 4, 4)
        z = F.gelu(self.phi_deconv1(z))
        Phi = self.phi_deconv2(z)                             # [B, 8, 16, 16]

        return GMTOut(H=H, U=U, cut_logits=cut_logits, Phi=Phi)

    def termination_logits(self, H: torch.Tensor, phi_geom: torch.Tensor) -> torch.Tensor:
        """Factorized termination head:
        term_logit[b, r, A] = MLP_h(H)[b, A, :] . MLP_g(phi_geom)[r, :] + bias_A.

        H [B, S, d], phi_geom [N_reg, n_geom] -> logits [B, N_reg, S] fp32.
        """
        hs = self.mlp_h(H)                                    # [B, S, d_t]
        gs = self.mlp_g(phi_geom.float())                     # [N_reg, d_t]
        term = torch.einsum("bsd,nd->bns", hs.float(), gs)
        return term + self.term_bias.view(1, 1, -1)
