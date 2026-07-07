from __future__ import annotations

import torch

from sprig.model.gmt import GrammarModulationTransformer
from sprig.model.sprig import SPRIGConfig

CFG = SPRIGConfig(S=8, R=4, T_v=4, d=32, n_heads=4, d_t=16, leaf_chunk=64)


def _make():
    torch.manual_seed(0)
    return GrammarModulationTransformer(CFG)


def test_output_shapes():
    gmt = _make()
    B, L = 2, 7
    emb = torch.randn(B, L, 768).half()
    emb_len = torch.tensor([7, 3], dtype=torch.int32)
    out = gmt(emb, emb_len)
    assert out.H.shape == (B, CFG.S, CFG.d)
    assert out.U.shape == (B, CFG.S, CFG.R)
    assert out.cut_logits.shape == (B, CFG.R, 14)
    assert out.Phi.shape == (B, 8, 16, 16)
    assert gmt.P_T.shape == (CFG.R, CFG.T_v)
    assert gmt.V.shape == (CFG.R, CFG.S)
    assert gmt.W.shape == (CFG.R, CFG.S)
    phi_geom = torch.randn(11, CFG.n_geom)
    term = gmt.termination_logits(out.H, phi_geom)
    assert term.shape == (B, 11, CFG.S)
    assert term.dtype == torch.float32


def test_padding_tokens_are_ignored():
    gmt = _make()
    emb = torch.randn(2, 6, 768)
    emb_len = torch.tensor([4, 6], dtype=torch.int32)
    out1 = gmt(emb, emb_len)
    emb2 = emb.clone()
    emb2[0, 4:] = 123.0  # garbage in padded positions of row 0
    out2 = gmt(emb2, emb_len)
    for a, b in zip(out1, out2):
        assert torch.allclose(a, b, atol=1e-5)


def test_phi_zero_at_init():
    gmt = _make()
    emb = torch.randn(3, 5, 768)
    emb_len = torch.tensor([5, 5, 2], dtype=torch.int32)
    out = gmt(emb, emb_len)
    assert torch.allclose(out.Phi, torch.zeros_like(out.Phi))


def test_grads_flow_to_heads():
    gmt = _make()
    emb = torch.randn(2, 5, 768)
    emb_len = torch.tensor([5, 4], dtype=torch.int32)
    out = gmt(emb, emb_len)
    term = gmt.termination_logits(out.H, torch.randn(9, CFG.n_geom))
    total = out.U.sum() + out.cut_logits.sum() + out.Phi.sum() + term.sum()
    total.backward()
    for name in ["E_N", "e_k", "term_bias"]:
        p = getattr(gmt, name)
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert gmt.W_u.weight.grad is not None
    assert gmt.mlp_g[0].weight.grad is not None
    assert gmt.phi_deconv2.weight.grad is not None
