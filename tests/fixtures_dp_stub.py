"""Reference inside-DP for MODEL-agent unit tests ONLY.

Implements the DESIGN.md section 5 signatures (inside_logZ / viterbi /
posterior_marginals) against sprig.dp.lattice.Lattice. The production DP is
sprig/dp/inside.py (owned by the DP agent); tests install this module via
sprig.model.sprig._DP_MODULE. Correct but unoptimized; fully differentiable
(including double backward) so SPRIGModel.loss can be exercised end-to-end.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

NEG = -1e30


def _leaf_terms(
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    log_PT: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    term_mark: Optional[torch.Tensor] = None,
    semiring: str = "sum",
):
    """Returns (term_score [B,n_leaf,S], log_cont [B,n_leaf,S], argT or None).

    term_score = log p_term + logsumexp_T(log p(T|A,c) + ell/kappa) with
    must_terminate handled as log p_term = 0, log(1 - p_term) = -inf.
    In the max semiring the T-mix is a max (texel prior itself stays an exact
    sum over k).
    """
    prior = torch.logsumexp(
        U_logmix.unsqueeze(-1) + log_PT.unsqueeze(0).unsqueeze(0), dim=2
    )  # [B,S,T_v]
    ell_t = ell_leaf.float() / temper_kappa.view(1, -1, 1)
    scores = prior.unsqueeze(1) + ell_t.unsqueeze(2)  # [B,n_leaf,S,T_v]
    if semiring == "sum":
        mix = torch.logsumexp(scores, dim=-1)
        argT = None
    else:
        mix, argT = scores.max(dim=-1)
    lt = term_logits[:, lattice.leaf_ids, :]
    log_term = F.logsigmoid(lt)
    log_cont = F.logsigmoid(-lt)
    mt = lattice.must_terminate[lattice.leaf_ids].view(1, -1, 1)
    log_term = torch.where(mt, torch.zeros_like(log_term), log_term)
    log_cont = torch.where(mt, torch.full_like(log_cont, NEG), log_cont)
    term_score = log_term + mix
    if term_mark is not None:
        term_score = term_score + term_mark[:, lattice.leaf_ids, :]
    return term_score, log_cont, argT


def _row_cut_logp(cut_logits: torch.Tensor, lattice, lv) -> torch.Tensor:
    """Masked, renormalized cut-type log-prob per concrete cut row -> [B,M,R].
    Same-type mass split uniformly across concrete cuts (log_cnt correction)."""
    B, R = cut_logits.shape[0], cut_logits.shape[1]
    M = lv.parent_ids.shape[0]
    pres = lattice.type_present[lv.parents]  # [P,14]
    tl = cut_logits.unsqueeze(1).masked_fill(
        ~pres.view(1, -1, 1, pres.shape[-1]), float("-inf")
    )  # [B,P,R,14]
    tls = F.log_softmax(tl, dim=-1)
    sel = tls[:, lv.parent_index, :, :]  # [B,M,R,14]
    idx = lv.cut_type.view(1, -1, 1, 1).expand(B, M, R, 1)
    return sel.gather(-1, idx).squeeze(-1) - lv.log_cnt.view(1, -1, 1)


def _sweep(
    ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
    temper_kappa, log_PT,
    term_mark=None, expand_mark=None, cut_mark=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _N, S = term_logits.shape
    device = ell_leaf.device
    term_score, log_cont, _ = _leaf_terms(
        ell_leaf, term_logits, U_logmix, log_PT, lattice, temper_kappa,
        term_mark=term_mark, semiring="sum",
    )
    beta = torch.full((B, lattice.n_regions, S), NEG, device=device)
    leaf_slot = lattice.leaf_index_of_region
    rid_nc = torch.nonzero(lattice.must_terminate, as_tuple=False).reshape(-1)
    beta[:, rid_nc, :] = term_score[:, leaf_slot[rid_nc], :]

    row_offset = 0
    for lv in lattice.levels:
        M = lv.parent_ids.shape[0]
        P = lv.parents.shape[0]
        lo_b = beta[:, lv.child_lo, :]
        hi_b = beta[:, lv.child_hi, :]
        Bhat = torch.logsumexp(lo_b.unsqueeze(2) + logV.view(1, 1, *logV.shape), dim=-1)
        Chat = torch.logsumexp(hi_b.unsqueeze(2) + logW.view(1, 1, *logW.shape), dim=-1)
        contrib = _row_cut_logp(cut_logits, lattice, lv) + Bhat + Chat  # [B,M,R]
        if cut_mark is not None:
            contrib = contrib + cut_mark[:, row_offset : row_offset + M].unsqueeze(-1)
        R = contrib.shape[-1]
        pidx = lv.parent_index.view(1, -1, 1).expand(B, M, R)
        mx = torch.full((B, P, R), NEG, device=device).scatter_reduce(
            1, pidx, contrib.detach(), reduce="amax", include_self=True
        )
        ex = torch.exp(contrib - mx[:, lv.parent_index, :])
        ssum = torch.zeros(B, P, R, device=device).index_add_(1, lv.parent_index, ex)
        comb = mx + torch.log(ssum.clamp(min=1e-38))  # [B,P,R]
        expand = torch.logsumexp(comb.unsqueeze(2) + U_logmix.unsqueeze(1), dim=-1)
        if expand_mark is not None:
            expand = expand + expand_mark[:, lv.parents, :]

        # Combine branches. Only finite values may enter the log-add: with the
        # -1e30 sentinel (or log-domain gaps > ~88 nats once emissions
        # sharpen) torch.logaddexp's backward creates exp(+huge)=inf nodes
        # whose double-backward (create_graph=True in SPRIGModel.loss) is NaN,
        # so use a detached-max shift (every exp argument <= 0).
        rid = lv.parents
        is_leaf = lattice.leaf_mask[rid]
        val = torch.zeros(B, P, S, device=device)
        not_leaf = ~is_leaf
        if bool(not_leaf.any()):
            val[:, not_leaf, :] = expand[:, not_leaf, :]
        if bool(is_leaf.any()):
            lidx = leaf_slot[rid[is_leaf]]
            a = term_score[:, lidx, :]
            b = log_cont[:, lidx, :] + expand[:, is_leaf, :]
            mshift = torch.maximum(a, b).detach()
            val[:, is_leaf, :] = mshift + torch.log(
                torch.exp(a - mshift) + torch.exp(b - mshift)
            )
        beta[:, rid, :] = val
        row_offset += M

    logZ = beta[:, lattice.root_id, 0]
    return beta, logZ


def inside_logZ(
    ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
    temper_kappa, log_PT,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DESIGN section 5: -> (beta [B,N_reg,S] fp32, logZ [B])."""
    return _sweep(
        ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
        temper_kappa, log_PT,
    )


def posterior_marginals(
    ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
    temper_kappa, log_PT,
) -> Dict[str, torch.Tensor]:
    """Expected-count marginals via the autograd identity (zero markers on the
    terminate/expand branches and per-cut rows). Returns detached tensors:
    node/term/expand [B,N_reg,S], cut [B,M_total] (levels concatenated),
    texel [B,n_leaf,T_v], rule [B,S,R], logZ [B]."""
    B, N, S = term_logits.shape
    device = ell_leaf.device
    M_total = sum(int(lv.parent_ids.shape[0]) for lv in lattice.levels)
    ell_d = ell_leaf.detach().requires_grad_(True)
    U_d = U_logmix.detach().requires_grad_(True)
    tm = torch.zeros(B, N, S, device=device, requires_grad=True)
    em = torch.zeros(B, N, S, device=device, requires_grad=True)
    cm = torch.zeros(B, M_total, device=device, requires_grad=True)
    with torch.enable_grad():
        _, logZ = _sweep(
            ell_d, term_logits.detach(), cut_logits.detach(), U_d,
            logV.detach(), logW.detach(), lattice,
            temper_kappa.detach(), log_PT.detach(),
            term_mark=tm, expand_mark=em, cut_mark=cm,
        )
        g_tm, g_em, g_cm, g_ell, g_u = torch.autograd.grad(
            logZ.sum(), [tm, em, cm, ell_d, U_d]
        )
    texel = g_ell * temper_kappa.detach().view(1, -1, 1)
    return {
        "logZ": logZ.detach(),
        "node": (g_tm + g_em).detach(),
        "term": g_tm.detach(),
        "expand": g_em.detach(),
        "cut": g_cm.detach(),
        "texel": texel.detach(),
        "rule": g_u.detach(),
    }


def viterbi(
    ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
    temper_kappa, log_PT,
) -> Tuple[torch.Tensor, List[Tuple]]:
    """Max-semiring sweep + backtrace.

    Returns (score [B], trees) where trees[b] is a nested tuple
    (region_id, symbol, texel_or_None, axis_or_None, cut_px_or_None,
    (children...)) rooted at (root region, symbol 0).
    """
    with torch.no_grad():
        B, N, S = term_logits.shape
        device = ell_leaf.device
        term_score, log_cont, argT = _leaf_terms(
            ell_leaf, term_logits, U_logmix, log_PT, lattice, temper_kappa,
            semiring="max",
        )
        beta = torch.full((B, N, S), NEG, device=device)
        term_choice = torch.zeros(B, N, S, dtype=torch.bool, device=device)
        leaf_slot = lattice.leaf_index_of_region
        rid_nc = torch.nonzero(lattice.must_terminate, as_tuple=False).reshape(-1)
        beta[:, rid_nc, :] = term_score[:, leaf_slot[rid_nc], :]
        term_choice[:, rid_nc, :] = True

        bp: List[Dict[str, torch.Tensor]] = []
        parent_pos: Dict[int, Tuple[int, int]] = {}
        for li, lv in enumerate(lattice.levels):
            for p_i, r in enumerate(lv.parents.tolist()):
                parent_pos[r] = (li, p_i)
            M = lv.parent_ids.shape[0]
            P = lv.parents.shape[0]
            lo_b = beta[:, lv.child_lo, :]
            hi_b = beta[:, lv.child_hi, :]
            Bv, argB = (lo_b.unsqueeze(2) + logV.view(1, 1, *logV.shape)).max(dim=-1)
            Cv, argC = (hi_b.unsqueeze(2) + logW.view(1, 1, *logW.shape)).max(dim=-1)
            contrib = _row_cut_logp(cut_logits, lattice, lv) + Bv + Cv  # [B,M,R]
            t = contrib.unsqueeze(2) + U_logmix.unsqueeze(1)  # [B,M,S,R]
            tmax, argk = t.max(dim=-1)  # [B,M,S]
            pidx = lv.parent_index.view(1, -1, 1).expand(B, M, S)
            pmax = torch.full((B, P, S), NEG, device=device).scatter_reduce(
                1, pidx, tmax, reduce="amax", include_self=True
            )
            rowidx = torch.arange(M, device=device).view(1, -1, 1).expand(B, M, S)
            cand = torch.where(
                tmax >= pmax[:, lv.parent_index, :],
                rowidx,
                torch.full_like(rowidx, M),
            )
            argm = torch.full((B, P, S), M, dtype=torch.int64, device=device).scatter_reduce(
                1, pidx, cand, reduce="amin", include_self=True
            )
            rid = lv.parents
            is_leaf = lattice.leaf_mask[rid]
            cont = torch.zeros(B, P, S, device=device)
            tb = torch.full((B, P, S), NEG, device=device)
            if bool(is_leaf.any()):
                lidx = leaf_slot[rid[is_leaf]]
                cont[:, is_leaf, :] = log_cont[:, lidx, :]
                tb[:, is_leaf, :] = term_score[:, lidx, :]
            exp_v = cont + pmax
            beta[:, rid, :] = torch.maximum(tb, exp_v)
            term_choice[:, rid, :] = tb >= exp_v
            bp.append({"argB": argB, "argC": argC, "argk": argk, "argm": argm})

        score = beta[:, lattice.root_id, 0]

        def extract(b: int, rid: int, sym: int) -> Tuple:
            if bool(term_choice[b, rid, sym]):
                slot = int(leaf_slot[rid])
                tex = int(argT[b, slot, sym])
                return (rid, sym, tex, None, None, ())
            li, p_i = parent_pos[rid]
            lv = lattice.levels[li]
            m = int(bp[li]["argm"][b, p_i, sym])
            k = int(bp[li]["argk"][b, m, sym])
            b_sym = int(bp[li]["argB"][b, m, k])
            c_sym = int(bp[li]["argC"][b, m, k])
            lo, hi = int(lv.child_lo[m]), int(lv.child_hi[m])
            axis, px = int(lv.cut_axis[m]), int(lv.cut_px[m])
            return (
                rid, sym, None, axis, px,
                (extract(b, lo, b_sym), extract(b, hi, c_sym)),
            )

        trees = [extract(b, lattice.root_id, 0) for b in range(B)]
    return score, trees
