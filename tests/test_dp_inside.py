"""DP-agent tests for sprig/dp/inside.py (DESIGN.md section 5).

The blocking correctness test: on the tiny lattice (16x16 canvas, 8px grid
-> 9 regions) with a tiny config (S=3, R=2, T_v=2, random weights),
brute-force enumerate ALL derivation trees (pure-python recursion over
cuts/symbols/texels, no memoization) and compare the summed joint
probability to the inside logZ within 1e-4. Also: the Viterbi tree's joint
prob equals the returned score and is <= logZ, and the autograd posterior
marginals satisfy the expected-count identities.

Plus: exact-fallback parity, parity (values and first-order grads) against
the independently brute-force-validated reference in
tests/fixtures_dp_stub.py on the full 64x64 lattice (which exercises
must_expand regions), and double-backward through the DP (the
create_graph=True hinge path in SPRIGModel.loss).
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn.functional as F

import tests.fixtures_dp_stub as dp_stub
from sprig.dp import inside as dp
from sprig.dp.lattice import Lattice, get_lattice

ARG_ORDER = (
    "ell_leaf", "term_logits", "cut_logits", "U_logmix", "logV", "logW",
    "lattice", "temper_kappa", "log_PT",
)


def make_inputs(
    lattice: Lattice, B: int = 2, S: int = 3, R: int = 2, T_v: int = 2,
    seed: int = 0, eta: float = 0.3,
) -> Dict:
    g = torch.Generator().manual_seed(seed)
    n_leaf = lattice.n_leaf_regions
    N = lattice.n_regions
    area = lattice.area_px[lattice.leaf_ids].float()
    kappa = torch.clamp(area ** eta, min=1.0) if eta > 0 else torch.ones_like(area)
    return {
        "ell_leaf": torch.randn(B, n_leaf, T_v, generator=g) * 2.0,
        "term_logits": torch.randn(B, N, S, generator=g),
        "cut_logits": torch.randn(B, R, 14, generator=g),
        "U_logmix": F.log_softmax(torch.randn(B, S, R, generator=g), dim=-1),
        "logV": F.log_softmax(torch.randn(R, S, generator=g), dim=-1),
        "logW": F.log_softmax(torch.randn(R, S, generator=g), dim=-1),
        "lattice": lattice,
        "temper_kappa": kappa,
        "log_PT": F.log_softmax(torch.randn(R, T_v, generator=g), dim=-1),
    }


# --------------------------------------------------------------- brute force

def _lse(vals: List[float]) -> float:
    m = max(vals)
    if m == float("-inf"):
        return m
    return m + math.log(sum(math.exp(v - m) for v in vals))


def _logsig(x: float) -> float:
    # log sigmoid(x), overflow-safe for the |x| ~ few range used here
    return -math.log1p(math.exp(-x)) if x >= 0 else x - math.log1p(math.exp(x))


class _BruteForce:
    """Pure-python enumeration of every derivation tree for one batch item."""

    def __init__(self, inputs: Dict, b: int) -> None:
        lat = inputs["lattice"]
        self.lat = lat
        self.S = inputs["U_logmix"].shape[1]
        self.R = inputs["U_logmix"].shape[2]
        self.T_v = inputs["log_PT"].shape[1]
        self.ell = inputs["ell_leaf"][b].double().tolist()
        self.term = inputs["term_logits"][b].double().tolist()
        self.U = inputs["U_logmix"][b].double().tolist()
        self.logV = inputs["logV"].double().tolist()
        self.logW = inputs["logW"].double().tolist()
        logPT = inputs["log_PT"].double().tolist()
        self.kappa = inputs["temper_kappa"].double().tolist()
        cutl = inputs["cut_logits"][b].double().tolist()

        # texel prior: log p(T|A) = lse_k(U[A][k] + log_PT[k][T])
        self.prior = [
            [_lse([self.U[a][k] + logPT[k][t] for k in range(self.R)])
             for t in range(self.T_v)]
            for a in range(self.S)
        ]
        # masked per-region cut-type log-softmax per component
        self.cut_lp: Dict[int, List[List[float]]] = {}
        for rid in range(lat.n_regions):
            present = lat.type_present[rid].tolist()
            if not any(present):
                continue
            per_k = []
            for k in range(self.R):
                vals = [cutl[k][t] for t in range(14) if present[t]]
                z = _lse(vals)
                per_k.append([cutl[k][t] - z if present[t] else float("-inf")
                              for t in range(14)])
            self.cut_lp[rid] = per_k

    def leaf_mix(self, rid: int, a: int) -> float:
        slot = int(self.lat.leaf_index_of_region[rid])
        return _lse([self.prior[a][t] + self.ell[slot][t] / self.kappa[slot]
                     for t in range(self.T_v)])

    def beta(self, rid: int, a: int) -> float:
        acc: List[float] = []
        cont = 0.0
        if bool(self.lat.leaf_mask[rid]):
            mix = self.leaf_mix(rid, a)
            if bool(self.lat.must_terminate[rid]):
                return mix
            lt = self.term[rid][a]
            acc.append(_logsig(lt) + mix)
            cont = _logsig(-lt)
        terms: List[float] = []
        for (_axis, _px, lo, hi, ct, logcnt) in self.lat.cuts_of_region[rid]:
            for k in range(self.R):
                base = self.U[a][k] + self.cut_lp[rid][k][ct] - logcnt
                for bs in range(self.S):
                    lo_v = self.beta(lo, bs)
                    for cs in range(self.S):
                        terms.append(base + self.logV[k][bs] + self.logW[k][cs]
                                     + lo_v + self.beta(hi, cs))
        acc.append(cont + _lse(terms))
        return _lse(acc)

    def tree_logp(self, node: Tuple) -> float:
        """Joint log-prob of a Viterbi tree tuple (max over the k component at
        each internal node reproduces the max-semiring score exactly)."""
        rid, sym, tex, axis, px, children = node
        if not children:
            slot = int(self.lat.leaf_index_of_region[rid])
            lp = self.prior[sym][tex] + self.ell[slot][tex] / self.kappa[slot]
            if not bool(self.lat.must_terminate[rid]):
                lp += _logsig(self.term[rid][sym])
            return lp
        lp = _logsig(-self.term[rid][sym]) if bool(self.lat.leaf_mask[rid]) else 0.0
        match = [c for c in self.lat.cuts_of_region[rid]
                 if c[0] == axis and c[1] == px]
        assert len(match) == 1, "viterbi cut (axis=%s, px=%s) not found in region %d" % (axis, px, rid)
        _axis, _px, lo, hi, ct, logcnt = match[0]
        b_sym, c_sym = children[0][1], children[1][1]
        assert children[0][0] == lo and children[1][0] == hi
        lp += max(self.U[sym][k] + self.cut_lp[rid][k][ct] - logcnt
                  + self.logV[k][b_sym] + self.logW[k][c_sym]
                  for k in range(self.R))
        return lp + self.tree_logp(children[0]) + self.tree_logp(children[1])


# ------------------------------------------------- blocking correctness test

@pytest.fixture(scope="module")
def tiny():
    lat = get_lattice(16, 8, 16)
    assert lat.n_regions == 9
    return make_inputs(lat, B=2, S=3, R=2, T_v=2, seed=7, eta=0.3)


def test_inside_matches_bruteforce_tiny(tiny):
    beta, logZ = dp.inside(**tiny)
    assert beta.shape == (2, 9, 3)
    assert torch.isfinite(beta).all()
    for b in range(2):
        bf = _BruteForce(tiny, b)
        ref = bf.beta(tiny["lattice"].root_id, 0)
        assert abs(float(logZ[b]) - ref) < 1e-4, (float(logZ[b]), ref)
        # every beta entry matches, not just the root
        for rid in range(9):
            for a in range(3):
                assert abs(float(beta[b, rid, a]) - bf.beta(rid, a)) < 1e-4


def test_inside_exact_fallback_matches_fast_path(tiny):
    _, logZ_fast = dp.inside(**tiny)
    _, logZ_exact = dp.inside(**tiny, exact=True)
    assert torch.allclose(logZ_fast, logZ_exact, atol=1e-4)


def test_viterbi_score_and_tree_tiny(tiny):
    _, logZ = dp.inside(**tiny)
    score, trees = dp.viterbi(**tiny)
    assert score.shape == (2,) and len(trees) == 2
    for b in range(2):
        assert float(score[b]) <= float(logZ[b]) + 1e-5
        bf = _BruteForce(tiny, b)
        # the returned tree's joint log-prob equals the viterbi score
        assert abs(bf.tree_logp(trees[b]) - float(score[b])) < 1e-4


def test_posterior_marginals_identities_tiny(tiny):
    marg = dp.posterior_marginals(**tiny)
    lat = tiny["lattice"]
    _, logZ = dp.inside(**tiny)
    assert torch.allclose(marg["logZ"], logZ.detach(), atol=1e-4)
    node, term, expand = marg["node"], marg["term"], marg["expand"]
    assert torch.allclose(node, term + expand, atol=1e-5)
    assert (node >= -1e-5).all() and (marg["cut"] >= -1e-5).all()
    for b in range(2):
        # root occupied exactly once
        assert abs(float(node[b, lat.root_id, :].sum()) - 1.0) < 1e-4
        # every node either terminates or picks exactly one concrete cut
        assert abs(float(marg["cut"][b].sum()) - float(expand[b].sum())) < 1e-4
        # texel mass == termination mass; rule mass == node mass
        assert abs(float(marg["texel"][b].sum()) - float(term[b].sum())) < 1e-4
        assert abs(float(marg["rule"][b].sum()) - float(node[b].sum())) < 1e-4


# ----------------------------------------------- parity vs reference (64x64)

def _run_with_grads(fn, inputs: Dict, exact: bool = False):
    grads_of = ("ell_leaf", "term_logits", "cut_logits", "U_logmix", "logV", "logW")
    kw = dict(inputs)
    leaves = []
    for k in grads_of:
        kw[k] = inputs[k].detach().clone().requires_grad_(True)
        leaves.append(kw[k])
    if exact:
        kw["exact"] = True
    _, logZ = fn(**kw)
    grads = torch.autograd.grad(logZ.sum(), leaves)
    return logZ.detach(), dict(zip(grads_of, grads))


@pytest.fixture(scope="module")
def full64():
    lat = get_lattice(64, 8, 16)
    assert bool(lat.must_expand.any())  # exercises the forced-expand branch
    return make_inputs(lat, B=2, S=4, R=2, T_v=3, seed=11, eta=0.25)


def test_inside_matches_reference_stub_64(full64):
    logZ, grads = _run_with_grads(dp.inside, full64)
    logZ_ref, grads_ref = _run_with_grads(dp_stub.inside_logZ, full64)
    assert torch.isfinite(logZ).all()
    assert torch.allclose(logZ, logZ_ref, atol=2e-3), (logZ, logZ_ref)
    for k in grads:
        assert torch.allclose(grads[k], grads_ref[k], atol=1e-4), k


def test_inside_exact_matches_reference_stub_64(full64):
    logZ, _ = _run_with_grads(dp.inside, full64, exact=True)
    logZ_ref, _ = _run_with_grads(dp_stub.inside_logZ, full64)
    assert torch.allclose(logZ, logZ_ref, atol=2e-3)


def test_viterbi_matches_reference_stub_64(full64):
    score, trees = dp.viterbi(**full64)
    score_ref, _ = dp_stub.viterbi(**full64)
    assert torch.allclose(score, score_ref, atol=1e-3)
    _, logZ = dp.inside(**full64)
    assert (score <= logZ + 1e-4).all()
    # trees tile the canvas with valid child rects
    lat = full64["lattice"]

    def check(node, rect):
        rid, _sym, tex, axis, px, children = node
        assert tuple(lat.regions[rid].tolist()) == rect
        if not children:
            assert tex is not None
            return [rect]
        x0, y0, x1, y1 = rect
        lo_rect = (x0, y0, px, y1) if axis == 0 else (x0, y0, x1, px)
        hi_rect = (px, y0, x1, y1) if axis == 0 else (x0, px, x1, y1)
        return check(children[0], lo_rect) + check(children[1], hi_rect)

    for tree in trees:
        rects = check(tree, (0, 0, 64, 64))
        cover = torch.zeros(64, 64, dtype=torch.int32)
        for x0, y0, x1, y1 in rects:
            cover[y0:y1, x0:x1] += 1
        assert torch.equal(cover, torch.ones(64, 64, dtype=torch.int32))


def test_posterior_marginals_match_reference_stub_64(full64):
    marg = dp.posterior_marginals(**full64)
    marg_ref = dp_stub.posterior_marginals(**full64)
    for k in ("node", "term", "expand", "cut", "texel", "rule"):
        assert torch.allclose(marg[k], marg_ref[k], atol=1e-4), k
    # must_expand regions never terminate
    lat = full64["lattice"]
    assert float(marg["term"][:, lat.must_expand, :].abs().max()) < 1e-8


# ------------------------------------------------------------- differentiability

def test_double_backward_hinge_path(tiny):
    kw = dict(tiny)
    u_raw = torch.randn(2, 3, 2, requires_grad=True)
    ell = tiny["ell_leaf"].detach().clone().requires_grad_(True)
    kw["U_logmix"] = F.log_softmax(u_raw, dim=-1)
    kw["ell_leaf"] = ell
    _, logZ = dp.inside(**kw)
    (g_ell,) = torch.autograd.grad(logZ.sum(), [ell], create_graph=True, retain_graph=True)
    usage = g_ell.sum(dim=(0, 1))
    usage = usage / usage.sum().clamp(min=1e-12)
    hinge = F.relu(0.5 - usage).sum()
    loss = -logZ.mean() + hinge
    loss.backward()
    assert u_raw.grad is not None and torch.isfinite(u_raw.grad).all()
    assert ell.grad is not None and torch.isfinite(ell.grad).all()


def test_logbmm_matches_exact():
    g = torch.Generator().manual_seed(3)
    x = torch.randn(2, 5, 7, generator=g) * 5.0
    w = torch.randn(7, 4, generator=g) * 5.0
    assert torch.allclose(dp.logbmm(x, w), dp.logbmm(x, w, exact=True), atol=1e-5)
    wb = torch.randn(2, 7, 4, generator=g) * 5.0
    assert torch.allclose(
        dp.logbmm_batched(x, wb), dp.logbmm_batched(x, wb, exact=True), atol=1e-5
    )
