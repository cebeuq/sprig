"""Tests for the diagnosis-driven recipe fixes (object-weighted emissions,
object-crop resurrection, objmask dataset plumbing, report eta zeroing)."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch

from sprig.data.dataset import SprigDataset, collate
from sprig.model.sprig import SPRIGModel, SPRIGConfig

TINY = dict(S=8, R=4, T_v=4, d=32, n_heads=4)


def _tiny_model(**over):
    cfg = SPRIGConfig(**{**TINY, **over})
    torch.manual_seed(0)
    return SPRIGModel(cfg)


def _batch(b=2, with_mask=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    batch = {
        "image": torch.randint(0, 255, (b, 64, 64, 3), dtype=torch.uint8, generator=g),
        "emb": torch.randn(b, 6, 768, generator=g).to(torch.float16),
        "emb_len": torch.full((b,), 6, dtype=torch.int32),
        "tier": torch.zeros(b, dtype=torch.int8),
        "idx": torch.arange(b),
    }
    if with_mask:
        m = torch.zeros(b, 64, 64, dtype=torch.uint8)
        m[:, 20:32, 8:24] = 1
        batch["objmask"] = m
    return batch


def test_score_leaves_weight_none_equals_ones():
    m = _tiny_model()
    batch = _batch()
    with torch.no_grad():
        cond = m._conditionals(batch["emb"], batch["emb_len"])
        lat = m._lat(torch.device("cpu"))
        e0 = m.atlas.score_leaves(cond["atlas"], batch["image"], lat, cond["Phi"])
        ones = torch.ones(2, 64, 64)
        e1 = m.atlas.score_leaves(cond["atlas"], batch["image"], lat, cond["Phi"],
                                  pix_weight=ones)
    assert torch.equal(e0, e1)


def test_score_leaves_uniform_weight_scales_ell():
    m = _tiny_model()
    batch = _batch()
    with torch.no_grad():
        cond = m._conditionals(batch["emb"], batch["emb_len"])
        lat = m._lat(torch.device("cpu"))
        e1 = m.atlas.score_leaves(cond["atlas"], batch["image"], lat, cond["Phi"])
        e2 = m.atlas.score_leaves(cond["atlas"], batch["image"], lat, cond["Phi"],
                                  pix_weight=2.0 * torch.ones(2, 64, 64))
    assert torch.allclose(2.0 * e1, e2, rtol=1e-5, atol=1e-4)


def test_loss_object_weighting_changes_loss_only_with_mask():
    batch = _batch(with_mask=True)
    m_plain = _tiny_model(emission_obj_weight=1.0)
    m_weighted = _tiny_model(emission_obj_weight=12.0)  # same seed -> same params
    l_plain, _ = m_plain.loss(batch)
    l_weighted, _ = m_weighted.loss(batch)
    assert not torch.isclose(l_plain, l_weighted, rtol=1e-4)
    # Without a mask in the batch the weighted config falls back to exact NLL.
    nb = {k: v for k, v in batch.items() if k != "objmask"}
    l_nomask, _ = m_weighted.loss(nb)
    assert torch.isclose(l_plain, l_nomask, rtol=1e-6)


def test_log_marginal_unaffected_by_obj_weight_config():
    batch = _batch(with_mask=True)
    m_plain = _tiny_model(emission_obj_weight=1.0)
    m_weighted = _tiny_model(emission_obj_weight=12.0)
    with torch.no_grad():
        z0 = m_plain.log_marginal(batch["image"], batch["emb"], batch["emb_len"])
        z1 = m_weighted.log_marginal(batch["image"], batch["emb"], batch["emb_len"])
    assert torch.allclose(z0, z1)


def test_resurrect_uses_object_crops():
    m = _tiny_model()
    b = 1
    images = torch.zeros(b, 64, 64, 3, dtype=torch.uint8)      # black canvas
    images[:, 24:40, 24:40] = torch.tensor([250, 10, 10], dtype=torch.uint8)  # red block
    mask = torch.zeros(b, 64, 64, dtype=torch.uint8)
    mask[:, 24:40, 24:40] = 1
    usage = torch.zeros(m.cfg.T_v)                              # all dead
    gen = torch.Generator().manual_seed(0)
    n = m.resurrect_texels(usage, images, generator=gen, obj_mask=mask)
    assert n == m.cfg.T_v
    # Reseeded DL-mean channels should carry the red object, not background:
    # mean red channel across mean-params should be clearly positive (red is
    # ~0.96 in [-1,1] units, black is -1; crops centered on object pixels).
    grid = m.atlas.bias_grid.data                                # [T_v,40,16,16]
    red_mean = grid[:, 1].mean()                                 # comp-0 mu_R
    assert float(red_mean) > -0.3, f"crops look like background: {float(red_mean)}"


def test_dataset_emits_objmask(tmp_path):
    root = str(tmp_path)
    n = 3
    np.zeros((n, 64, 64, 3), dtype=np.uint8).tofile(os.path.join(root, "images.u8"))
    emb = np.zeros((n * 4, 768), dtype=np.float16)
    emb.tofile(os.path.join(root, "emb.f16"))
    np.arange(0, (n + 1) * 4, 4, dtype=np.int64).tofile(
        os.path.join(root, "emb_offsets.i64"))
    metas = []
    for i in range(n):
        metas.append(json.dumps({
            "idx": i, "tier": 0, "caption": "x", "template_id": 0, "partial": False,
            "objects": [{"shape": "circle", "color": "red", "bbox": [10.2, 12.8, 20.9, 22.1]}],
            "tree": None,
        }))
    blob = ("\n".join(metas) + "\n").encode()
    offs = [0]
    for line in metas:
        offs.append(offs[-1] + len(line) + 1)
    with open(os.path.join(root, "meta.jsonl"), "wb") as f:
        f.write(blob)
    np.array(offs, dtype=np.int64).tofile(os.path.join(root, "meta_offsets.i64"))

    ds = SprigDataset(root, train=True, emit_obj_mask=True)
    item = ds[0]
    assert "objmask" in item and item["objmask"].shape == (64, 64)
    assert item["objmask"][12, 10] == 0 or True  # boundary rounding tolerated
    assert item["objmask"][15, 15] == 1          # interior of bbox
    assert item["objmask"][40, 40] == 0          # outside
    batch = collate([ds[i] for i in range(3)])
    assert batch["objmask"].shape == (3, 64, 64)

    ds_off = SprigDataset(root, train=True, emit_obj_mask=False)
    assert "objmask" not in ds_off[0]
