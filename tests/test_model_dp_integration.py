"""Informational integration probe: SPRIGModel against the real sprig.dp.inside.

Skips while the DP agent's module is absent; xfail (non-strict) so mid-flight
DP changes never block the model suite. The binding cross-check happens at G0.
"""
from __future__ import annotations

import pytest
import torch

import tests.fixtures_dp_stub as dp_stub
from sprig.model.sprig import SPRIGConfig, SPRIGModel


@pytest.mark.xfail(strict=False, reason="integration with DP-agent module; informational")
def test_real_inside_matches_stub_on_tiny_lattice():
    try:
        from sprig.dp import inside as real_dp
    except Exception:
        pytest.skip("sprig/dp/inside.py not present yet")
    if not (hasattr(real_dp, "inside_logZ") or hasattr(real_dp, "inside")):
        pytest.skip("sprig.dp.inside lacks inside_logZ/inside")

    cfg = SPRIGConfig(
        S=3, R=2, T_v=2, d=32, canvas=16, grid=8, leaf_max=16,
        n_heads=4, d_t=8, leaf_chunk=64,
    )
    torch.manual_seed(5)
    model = SPRIGModel(cfg)
    image = torch.randint(0, 256, (2, 16, 16, 3), dtype=torch.uint8)
    emb = torch.randn(2, 4, 768)
    emb_len = torch.tensor([4, 2], dtype=torch.int32)

    with torch.no_grad():
        cond = model._conditionals(emb, emb_len, images=image)
    kappa = model._kappa(0.0, image.device)
    _beta, stub_logZ = dp_stub.inside_logZ(**model._dp_kwargs(cond, kappa))

    logZ = model.log_marginal(image, emb, emb_len, report_mode=True)
    assert logZ.shape == (2,)
    assert torch.isfinite(logZ).all()
    assert torch.allclose(logZ.detach(), stub_logZ.detach(), atol=1e-3)
