from __future__ import annotations

import math
from typing import List, Tuple

import pytest
import torch

import sprig.model.sprig as sprig_mod
import tests.fixtures_dp_stub as dp_stub
from sprig.model.sprig import ParseNode, SPRIGConfig, SPRIGModel

CFG = SPRIGConfig(
    S=8, R=4, T_v=4, d=32, canvas=64, grid=8, leaf_max=16,
    n_heads=4, d_t=16, leaf_chunk=64,
)


@pytest.fixture(autouse=True)
def use_stub(monkeypatch):
    monkeypatch.setattr(sprig_mod, "_DP_MODULE", dp_stub)


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return SPRIGModel(CFG)


@pytest.fixture()
def batch():
    torch.manual_seed(1)
    return {
        "image": torch.randint(0, 256, (2, 64, 64, 3), dtype=torch.uint8),
        "emb": torch.randn(2, 6, 768).half(),
        "emb_len": torch.tensor([6, 4], dtype=torch.int32),
    }


# --------------------------------------------------------------------- C2

def test_log_marginal_and_report_mode(model, batch):
    logZ = model.log_marginal(batch["image"], batch["emb"], batch["emb_len"])
    assert logZ.shape == (2,)
    assert torch.isfinite(logZ).all()
    # eta == 0: report_mode is a no-op.
    lz_rep = model.log_marginal(batch["image"], batch["emb"], batch["emb_len"], report_mode=True)
    assert torch.allclose(logZ, lz_rep, atol=1e-4)
    # eta > 0: tempered != reported; reported equals the eta=0 value.
    with torch.no_grad():
        model.eta.fill_(0.7)
    lz_t = model.log_marginal(batch["image"], batch["emb"], batch["emb_len"])
    lz_r = model.log_marginal(batch["image"], batch["emb"], batch["emb_len"], report_mode=True)
    assert not torch.allclose(lz_t, lz_r, atol=1e-3)
    assert torch.allclose(lz_r, logZ, atol=1e-4)


# --------------------------------------------------------------------- loss

def test_loss_backward_finite_and_all_params_grad(model, batch):
    model.train()
    loss, metrics = model.loss(batch)
    assert torch.isfinite(loss)
    for k in ["loss", "nll", "bpd", "logZ_mean", "texel_hinge", "symbol_hinge",
              "texel_alive_frac", "symbol_eff", "mean_leaves"]:
        assert k in metrics and math.isfinite(metrics[k]), k
    loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, "no grad for %s" % name
        assert torch.isfinite(p.grad).all(), "non-finite grad for %s" % name


# --------------------------------------------------------------------- C3

def test_posterior_usage(model, batch):
    u = model.posterior_usage(batch["image"], batch["emb"], batch["emb_len"])
    assert u["symbol_usage"].shape == (CFG.S,)
    assert u["texel_usage"].shape == (CFG.T_v,)
    assert abs(float(u["symbol_usage"].sum()) - 1.0) < 1e-4
    assert abs(float(u["texel_usage"].sum()) - 1.0) < 1e-4
    assert u["node_entropy"] >= 0.0 and math.isfinite(u["node_entropy"])
    assert math.isfinite(u["emit_mag"]) and math.isfinite(u["rule_mag"])
    # 64x64 canvas with 16px max leaves: at least 16 leaves.
    assert u["mean_leaves"] >= 15.9
    assert u["mean_depth"] > 0.0


def _leaf_rects(node: ParseNode, out: List[Tuple[int, int, int, int]]) -> None:
    if not node.children:
        out.append(node.rect)
        assert node.texel is not None
    else:
        assert node.texel is None
        assert node.axis in (0, 1) and node.cut_px is not None
        x0, y0, x1, y1 = node.rect
        lo, hi = node.children
        if node.axis == 0:  # vertical cut splits x
            assert lo.rect == (x0, y0, node.cut_px, y1)
            assert hi.rect == (node.cut_px, y0, x1, y1)
        else:
            assert lo.rect == (x0, y0, x1, node.cut_px)
            assert hi.rect == (x0, node.cut_px, x1, y1)
        for ch in node.children:
            _leaf_rects(ch, out)


def _assert_tiles(rects: List[Tuple[int, int, int, int]], canvas: int) -> None:
    cover = torch.zeros(canvas, canvas, dtype=torch.int32)
    for x0, y0, x1, y1 in rects:
        assert max(x1 - x0, y1 - y0) <= 16
        cover[y0:y1, x0:x1] += 1
    assert torch.equal(cover, torch.ones(canvas, canvas, dtype=torch.int32))


def test_map_parse_valid_trees(model, batch):
    parses = model.map_parse(batch["image"], batch["emb"], batch["emb_len"])
    assert len(parses) == 2
    for root in parses:
        assert root.rect == (0, 0, 64, 64)
        assert root.symbol == CFG.axiom
        rects: List[Tuple[int, int, int, int]] = []
        _leaf_rects(root, rects)
        _assert_tiles(rects, 64)


# --------------------------------------------------------------------- C4

def _tree_struct(node: ParseNode):
    return (
        node.rect, node.symbol, node.axis, node.cut_px,
        tuple(_tree_struct(c) for c in node.children),
    )


def _tree_full(node: ParseNode):
    return (
        node.rect, node.symbol, node.axis, node.cut_px, node.texel,
        tuple(_tree_full(c) for c in node.children),
    )


def test_sample_shapes_seeds_and_tiling(model, batch):
    emb1 = batch["emb"][0].float()
    el1 = batch["emb_len"][:1]
    imgs, trees = model.sample(emb1, el1, seed_struct=123, seed_material=456, n=2)
    assert imgs.dtype == torch.uint8 and imgs.shape == (2, 64, 64, 3)
    assert len(trees) == 2
    for root in trees:
        rects: List[Tuple[int, int, int, int]] = []
        _leaf_rects(root, rects)
        _assert_tiles(rects, 64)
    # Bit-identical reproduction with the same seeds.
    imgs2, trees2 = model.sample(emb1, el1, seed_struct=123, seed_material=456, n=2)
    assert torch.equal(imgs, imgs2)
    assert [_tree_full(t) for t in trees] == [_tree_full(t) for t in trees2]


def test_sample_struct_vs_material_streams(model, batch):
    emb1 = batch["emb"][0].float()
    el1 = batch["emb_len"][:1]
    imgs_a, trees_a = model.sample(emb1, el1, seed_struct=7, seed_material=100, n=2)
    imgs_b, trees_b = model.sample(emb1, el1, seed_struct=7, seed_material=2024, n=2)
    # Identical structure (rects, symbols, cuts)...
    assert [_tree_struct(t) for t in trees_a] == [_tree_struct(t) for t in trees_b]
    # ... but different materials, hence different pixels.
    assert not torch.equal(imgs_a, imgs_b)


def test_sample_bestof_consistency(model, batch):
    emb1 = batch["emb"][0].float()
    el1 = batch["emb_len"][:1]
    K, seed = 4, 11
    best_imgs, best_trees = model.sample_bestof(emb1, el1, K=K, seed=seed)
    assert best_imgs.shape == (1, 64, 64, 3)
    assert len(best_trees) == 1
    imgs, trees, scores = model._sample_scored(
        emb1, el1, seed, seed + sprig_mod._MATERIAL_SEED_OFFSET, K
    )
    best = int(torch.tensor(scores).argmax())
    assert torch.equal(best_imgs[0], imgs[best])
    assert _tree_full(best_trees[0]) == _tree_full(trees[best])


# ------------------------------------------------------------- resurrection

def test_resurrect_overwrites_only_underused(model, batch):
    usage = torch.full((CFG.T_v,), 1.0 / CFG.T_v)
    usage[2] = 0.0
    usage = usage / usage.sum()
    bias_before = model.atlas.bias_grid.detach().clone()
    et_before = model.atlas.E_T.detach().clone()
    gen = torch.Generator().manual_seed(0)
    n = model.resurrect_texels(usage, batch["image"], generator=gen)
    assert n == 1
    bias_after = model.atlas.bias_grid.detach()
    et_after = model.atlas.E_T.detach()
    assert not torch.allclose(bias_after[2], bias_before[2])
    assert not torch.allclose(et_after[2], et_before[2])
    for t in range(CFG.T_v):
        if t == 2:
            continue
        assert torch.equal(bias_after[t], bias_before[t])
        assert torch.equal(et_after[t], et_before[t])
    # Mean channels of the resurrected row hold a valid [-1,1] crop.
    for j in range(4):
        blk = bias_after[2, 10 * j + 1 : 10 * j + 4]
        assert blk.min() >= -1.0 and blk.max() <= 1.0


# -------------------------------------------------- real-config shape sanity

def test_real_config_conditional_shapes():
    torch.manual_seed(0)
    cfg = SPRIGConfig()  # main64 values: S=1024, R=64, T_v=256, d=384
    m = SPRIGModel(cfg)
    emb = torch.randn(1, 8, 768).half()
    emb_len = torch.tensor([8], dtype=torch.int32)
    img = torch.randint(0, 256, (1, 64, 64, 3), dtype=torch.uint8)
    with torch.no_grad():
        cond = m._conditionals(emb, emb_len, images=img)
    lat = m.lattice
    assert cond["U_logmix"].shape == (1, 1024, 64)
    assert cond["term_logits"].shape == (1, lat.n_regions, 1024)
    assert cond["cut_logits"].shape == (1, 64, 14)
    assert cond["log_PT"].shape == (64, 256)
    assert cond["logV"].shape == (64, 1024)
    assert cond["logW"].shape == (64, 1024)
    assert cond["atlas"].shape == (1, 256, 40, 16, 16)
    assert cond["Phi"].shape == (1, 8, 16, 16)
    assert cond["ell"].shape == (1, lat.n_leaf_regions, 256)
    assert cond["ell"].dtype == torch.float32
    # Normalized log-distributions actually normalize.
    assert torch.allclose(cond["U_logmix"].exp().sum(-1), torch.ones(1, 1024), atol=1e-4)
    assert torch.allclose(cond["logV"].exp().sum(-1), torch.ones(64), atol=1e-4)


# ------------------------------------- stub cross-check vs brute force (16px)

def _brute_force_logZ(model: SPRIGModel, cond, kappa: torch.Tensor) -> float:
    lat = model.lattice
    S, R = model.cfg.S, model.cfg.R
    U = cond["U_logmix"][0].tolist()
    logPT = cond["log_PT"].tolist()
    logV = cond["logV"].tolist()
    logW = cond["logW"].tolist()
    ell = cond["ell"][0].tolist()
    term = cond["term_logits"][0].tolist()
    cutl = cond["cut_logits"][0].tolist()
    kap = kappa.tolist()

    def lse(vals):
        m = max(vals)
        if m == float("-inf"):
            return m
        return m + math.log(sum(math.exp(v - m) for v in vals))

    prior = [
        [lse([U[a][k] + logPT[k][t] for k in range(R)]) for t in range(len(logPT[0]))]
        for a in range(S)
    ]
    # masked per-region per-k cut-type log-softmax
    tls = {}
    for rid in range(lat.n_regions):
        present = lat.type_present[rid].tolist()
        if not any(present):
            continue
        per_k = []
        for k in range(R):
            logits = [cutl[k][t] if present[t] else float("-inf") for t in range(14)]
            z = lse([v for v in logits if v != float("-inf")])
            per_k.append([v - z if v != float("-inf") else float("-inf") for v in logits])
        tls[rid] = per_k

    memo = {}

    def beta(rid: int, a: int) -> float:
        key = (rid, a)
        if key in memo:
            return memo[key]
        acc = []
        log_cont = 0.0
        if bool(lat.leaf_mask[rid]):
            slot = int(lat.leaf_index_of_region[rid])
            mix = lse([prior[a][t] + ell[slot][t] / kap[slot] for t in range(len(ell[slot]))])
            if bool(lat.must_terminate[rid]):
                memo[key] = mix
                return mix
            lt = term[rid][a]
            acc.append(-math.log1p(math.exp(-lt)) + mix)     # logsigmoid(lt) + mix
            log_cont = -math.log1p(math.exp(lt))             # logsigmoid(-lt)
        terms = []
        for (axis, px, lo, hi, ct, logcnt) in lat.cuts_of_region[rid]:
            for k in range(R):
                base = U[a][k] + tls[rid][k][ct] - logcnt
                for bs in range(S):
                    for cs in range(S):
                        terms.append(base + logV[k][bs] + logW[k][cs] + beta(lo, bs) + beta(hi, cs))
        acc.append(log_cont + lse(terms))
        out = lse(acc)
        memo[key] = out
        return out

    return beta(lat.root_id, model.cfg.axiom)


def test_stub_dp_matches_bruteforce_16px():
    cfg = SPRIGConfig(
        S=3, R=2, T_v=2, d=32, canvas=16, grid=8, leaf_max=16,
        n_heads=4, d_t=8, leaf_chunk=64,
    )
    torch.manual_seed(3)
    model = SPRIGModel(cfg)
    with torch.no_grad():
        model.eta.fill_(0.3)  # exercise tempering in the cross-check
    image = torch.randint(0, 256, (1, 16, 16, 3), dtype=torch.uint8)
    emb = torch.randn(1, 4, 768)
    emb_len = torch.tensor([4], dtype=torch.int32)

    with torch.no_grad():
        cond = model._conditionals(emb, emb_len, images=image)
    kappa = model._kappa(0.3, image.device)
    beta, logZ = dp_stub.inside_logZ(**model._dp_kwargs(cond, kappa))
    assert beta.shape == (1, model.lattice.n_regions, cfg.S)

    ref = _brute_force_logZ(model, cond, kappa)
    assert abs(float(logZ[0]) - ref) < 1e-3

    v_score, trees = dp_stub.viterbi(**model._dp_kwargs(cond, kappa))
    assert float(v_score[0]) <= float(logZ[0]) + 1e-5

    marg = dp_stub.posterior_marginals(**model._dp_kwargs(cond, kappa))
    # Root is occupied exactly once; leaf-texel mass equals termination mass.
    root_occ = float(marg["node"][0, model.lattice.root_id, :].sum())
    assert abs(root_occ - 1.0) < 1e-4
    assert abs(float(marg["texel"].sum()) - float(marg["term"].sum())) < 1e-3
    # Node counts from the rule marginal agree with the marker-based counts.
    assert abs(float(marg["rule"].sum()) - float(marg["node"].sum())) < 1e-3
