from __future__ import annotations

import torch
import torch.nn.functional as F

from sprig.dp.lattice import get_lattice
from sprig.model import dl
from sprig.model.atlas import TexelAtlas, film_scale_shift, phi_at_leaf_centers
from sprig.model.sprig import SPRIGConfig

CFG = SPRIGConfig(S=8, R=4, T_v=4, d=32, n_heads=4, d_t=16, leaf_chunk=16)


def _setup():
    torch.manual_seed(0)
    atlas_mod = TexelAtlas(CFG)
    lat = get_lattice(64, 8, 16)
    emb = torch.randn(2, 6, 768)
    emb_len = torch.tensor([6, 3], dtype=torch.int32)
    return atlas_mod, lat, emb, emb_len


def test_render_shape_and_bias_grid():
    atlas_mod, _lat, emb, emb_len = _setup()
    a1 = atlas_mod.render(emb, emb_len)
    assert a1.shape == (2, CFG.T_v, 40, 16, 16)
    with torch.no_grad():
        atlas_mod.bias_grid[1] += 1.0
    a2 = atlas_mod.render(emb, emb_len)
    diff = (a2 - a1).detach().abs().amax(dim=(0, 2, 3, 4))  # per texel
    assert float(diff[1]) > 0.5
    mask = torch.ones(CFG.T_v, dtype=torch.bool)
    mask[1] = False
    assert float(diff[mask].max()) < 1e-6


def test_score_leaves_shape_and_reference():
    atlas_mod, lat, emb, emb_len = _setup()
    atlas = atlas_mod.render(emb, emb_len)
    Phi = torch.randn(2, 8, 16, 16) * 0.1
    images = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    ell = atlas_mod.score_leaves(atlas, images, lat, Phi)
    assert ell.shape == (2, lat.n_leaf_regions, CFG.T_v)
    assert ell.dtype == torch.float32
    assert torch.isfinite(ell).all()

    # Naive reference for a few (batch, slot, texel) triples.
    for slot in [0, 100, lat.n_leaf_regions - 1]:
        b, t = 1, 2
        rid = int(lat.leaf_ids[slot])
        x0, y0, x1, y1 = lat.regions[rid].tolist()
        h, w = y1 - y0, x1 - x0
        pooled = F.adaptive_avg_pool2d(atlas[b], (h, w))[t]  # [40,h,w]
        phi = phi_at_leaf_centers(Phi[b : b + 1], lat.regions[rid].view(1, 4), 64)[0, 0]
        scale, shift = film_scale_shift(phi)
        p = pooled.clone()
        for c in range(3):
            idx = [10 * j + 1 + c for j in range(4)]
            p[idx] = p[idx] * scale[c] + shift[c]
        crop = dl.u8_to_unit(images[b, y0:y1, x0:x1]).permute(2, 0, 1)
        ref = dl.dl_logprob(p.unsqueeze(0), crop.unsqueeze(0)).sum()
        assert torch.allclose(ell[b, slot, t], ref, atol=1e-3), slot


def test_score_leaves_chunk_invariance():
    atlas_mod, lat, emb, emb_len = _setup()
    atlas = atlas_mod.render(emb, emb_len)
    Phi = torch.randn(2, 8, 16, 16) * 0.1
    images = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    e1 = atlas_mod.score_leaves(atlas, images, lat, Phi, chunk=3)
    e2 = atlas_mod.score_leaves(atlas, images, lat, Phi, chunk=10_000)
    assert torch.allclose(e1, e2, atol=1e-5)


def test_film_changes_scores():
    atlas_mod, lat, emb, emb_len = _setup()
    atlas = atlas_mod.render(emb, emb_len)
    images = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    e0 = atlas_mod.score_leaves(atlas, images, lat, torch.zeros(2, 8, 16, 16))
    e1 = atlas_mod.score_leaves(atlas, images, lat, torch.ones(2, 8, 16, 16) * 0.3)
    assert not torch.allclose(e0, e1)


def test_score_leaves_grad_flows():
    atlas_mod, lat, emb, emb_len = _setup()
    atlas = atlas_mod.render(emb, emb_len)
    Phi = (torch.randn(2, 8, 16, 16) * 0.1).requires_grad_(True)
    images = torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8)
    ell = atlas_mod.score_leaves(atlas, images, lat, Phi)
    ell.sum().backward()
    assert atlas_mod.bias_grid.grad is not None
    assert torch.isfinite(atlas_mod.bias_grid.grad).all()
    assert atlas_mod.E_T.grad is not None
    assert Phi.grad is not None and torch.isfinite(Phi.grad).all()
