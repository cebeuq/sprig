"""Inside DP over the finite region lattice (DESIGN.md section 5).

Exposes the three DP entry points SPRIGModel calls through its adapter
(sprig.model.sprig._call_dp), all keyed by the DESIGN section 5 argument
names plus ``log_PT`` (the [R, T_v] log texel prior table the leaf T-mix
needs; p(T|A,c) = logsumexp_k(U_logmix[..,A,k] + log_PT[k,T])):

    inside(ell_leaf, term_logits, cut_logits, U_logmix, logV, logW,
           lattice, temper_kappa, log_PT)      -> (beta [B,N_reg,S] fp32, logZ [B])
    inside_logZ(...)                           -> alias of ``inside``
    viterbi(...)                               -> (score [B], trees)
    posterior_marginals(...)                   -> dict of expected-count tensors

Semantics (matching tests/fixtures_dp_stub.py, the brute-force-validated
reference):
  * beta(r, A) for leaf-eligible r combines
        log p_term(r,A) + logsumexp_T(log p(T|A,c) + ell(r,T)/kappa(r))
    with
        log(1 - p_term(r,A)) + expand(r,A),
    where must_terminate regions force p_term = 1 and must_expand regions
    (any side > leaf_max) expand with probability 1 (no p_term factor).
  * expand(r, A) is computed by a level-synchronous sweep in cell-area
    ascending order using the lattice's flattened per-level cut tables;
    cut-type probability mass is split uniformly across same-type concrete
    cuts (the precomputed -log(count) correction).
  * logZ[b] = beta[b, root, 0] (axiom nonterminal = symbol 0).

Numerics: all log-domain accumulation in fp32; every logsumexp/logbmm is
max-shifted with *detached* shifts. -1e30 sentinels are never fed through
branches on the differentiable path (they only seed the beta table and the
scatter-max identity); leaf/expand branches are combined by mask + index so
the module is safe under double backward (SPRIGModel.loss uses
create_graph=True through the DP for the under-use hinges).

Every function accepts ``exact=True`` to replace the max-shifted
exp/matmul/log fast path with a direct fp32 logsumexp — the exact fallback
DESIGN section 5 asks for (used by the parity tests).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

NEG = -1e30
# Clamp floor for post-shift exp-sums before the final log. It must be large
# enough that log's DOUBLE backward (-grad/x^2) stays finite in fp32
# (x^2 >= 1e-30 > fp32 tiny), because SPRIGModel.loss differentiates the
# under-use hinges through the DP with create_graph=True. Floor 1e-15 caps a
# reduction at ~34.5 nats below the factored max-shifts, which only rounds up
# paths of negligible posterior mass. (The segment-logsumexp sums are >= 1 by
# construction and never hit the clamp.)
_TINY = 1e-15
# Element budget for the [B, chunk, S, R] / [B, chunk, S, T_v] blocks the
# max-semiring (argmax) passes materialize.
_MAX_SEMIRING_BUDGET = 1 << 24


def logbmm(x_log: torch.Tensor, w_log: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """out[..., m] = logsumexp_k(x_log[..., k] + w_log[k, m]).

    x_log [..., K] and w_log [K, M], both finite fp32. Max-shifted (detached
    shifts), exp, matmul, log, shifts added back. ``exact=True`` is the direct
    fp32 logsumexp fallback for tests.
    """
    x_log = x_log.float()
    w_log = w_log.float()
    if exact:
        expand_shape = (1,) * (x_log.dim() - 1) + tuple(w_log.shape)
        return torch.logsumexp(x_log.unsqueeze(-1) + w_log.view(expand_shape), dim=-2)
    sx = x_log.max(dim=-1, keepdim=True).values.detach()          # [..., 1]
    sw = w_log.max(dim=0, keepdim=True).values.detach()           # [1, M]
    out = torch.matmul(torch.exp(x_log - sx), torch.exp(w_log - sw))
    return torch.log(out.clamp(min=_TINY)) + sx + sw


def logbmm_batched(x_log: torch.Tensor, w_log: torch.Tensor, exact: bool = False) -> torch.Tensor:
    """out[b, p, m] = logsumexp_k(x_log[b, p, k] + w_log[b, k, m]).

    Batched-weight variant of :func:`logbmm` (x [B, P, K], w [B, K, M])."""
    x_log = x_log.float()
    w_log = w_log.float()
    if exact:
        return torch.logsumexp(x_log.unsqueeze(-1) + w_log.unsqueeze(1), dim=-2)
    sx = x_log.max(dim=-1, keepdim=True).values.detach()          # [B, P, 1]
    sw = w_log.max(dim=1, keepdim=True).values.detach()           # [B, 1, M]
    out = torch.bmm(torch.exp(x_log - sx), torch.exp(w_log - sw))
    return torch.log(out.clamp(min=_TINY)) + sx + sw


def _segment_logsumexp(
    x: torch.Tensor, seg_index: torch.Tensor, n_seg: int
) -> torch.Tensor:
    """Group-by-segment logsumexp along dim 1: x [B, M, R], seg_index [M]
    -> [B, n_seg, R]. Max-shift per segment (detached), scatter-add of exp,
    log."""
    B, M, R = x.shape
    idx = seg_index.view(1, -1, 1).expand(B, M, R)
    mx = torch.full((B, n_seg, R), NEG, device=x.device, dtype=x.dtype).scatter_reduce(
        1, idx, x.detach(), reduce="amax", include_self=True
    )
    ex = torch.exp(x - mx[:, seg_index, :])
    ssum = torch.zeros(B, n_seg, R, device=x.device, dtype=x.dtype).index_add_(
        1, seg_index, ex
    )
    return mx + torch.log(ssum.clamp(min=_TINY))


def _row_cut_logp(cut_logits: torch.Tensor, lattice, lv, exact: bool = False) -> torch.Tensor:
    """Per concrete cut row: masked + renormalized cut-type log-prob with the
    uniform same-type mass split -> [B, M, R].

    cut_logits [B, R, 14]; validity mask per unique parent from
    lattice.type_present; log-softmax over the masked set; -log(count)
    correction per row (lv.log_cnt). (Fallback path — the sweep normally uses
    :func:`_cut_logp_flat` + one gather per level.)"""
    del exact  # the masked log-softmax is already exact fp32
    B, R = cut_logits.shape[0], cut_logits.shape[1]
    M = lv.parent_ids.shape[0]
    pres = lattice.type_present[lv.parents]                       # [P, 14]
    tl = cut_logits.float().unsqueeze(1).masked_fill(
        ~pres.view(1, -1, 1, pres.shape[-1]), float("-inf")
    )                                                             # [B, P, R, 14]
    tls = F.log_softmax(tl, dim=-1)
    sel = tls[:, lv.parent_index, :, :]                           # [B, M, R, 14]
    idx = lv.cut_type.view(1, -1, 1, 1).expand(B, M, R, 1)
    return sel.gather(-1, idx).squeeze(-1) - lv.log_cnt.view(1, -1, 1)


def _cut_logp_flat(cut_logits: torch.Tensor, lattice) -> Optional[torch.Tensor]:
    """Masked + renormalized cut-type log-probs for every DISTINCT
    type-presence pattern, flattened for per-level gathering:

        out [B, n_pat * 14, R];  row(pattern p, type t) = out[:, p*14 + t, :].

    One log-softmax per sweep instead of one per level; identical values (the
    per-parent mask in :func:`_row_cut_logp` only depends on the parent's
    type-presence pattern). Returns None when the lattice does not carry the
    precomputed pattern tables (foreign/stub lattices -> per-level fallback).
    """
    pats = getattr(lattice, "type_patterns", None)
    if pats is None:
        return None
    B, R = cut_logits.shape[0], cut_logits.shape[1]
    n_pat, T = pats.shape
    tl = cut_logits.float().unsqueeze(1).masked_fill(
        ~pats.view(1, n_pat, 1, T), float("-inf")
    )                                                             # [B, n_pat, R, 14]
    tls = F.log_softmax(tl, dim=-1)
    return tls.permute(0, 1, 3, 2).reshape(B, n_pat * T, R)


def _level_cut_logp(
    tls_flat: Optional[torch.Tensor], cut_logits: torch.Tensor, lattice, lv,
) -> torch.Tensor:
    """Per-row cut log-prob [B, M, R] for one level (gather from the per-sweep
    flat table when available, else the per-level fallback)."""
    if tls_flat is not None and getattr(lv, "flat_type_idx", None) is not None:
        return tls_flat[:, lv.flat_type_idx, :] - lv.log_cnt.view(1, -1, 1)
    return _row_cut_logp(cut_logits, lattice, lv)


def _mt_meta(lattice) -> Tuple[torch.Tensor, torch.Tensor]:
    """(must-terminate region ids, their leaf slots) — precomputed on the
    lattice when available (no per-call ``nonzero``)."""
    rid_mt = getattr(lattice, "mt_ids", None)
    if rid_mt is None:
        rid_mt = torch.nonzero(lattice.must_terminate, as_tuple=False).reshape(-1)
    slot_mt = getattr(lattice, "mt_slots", None)
    if slot_mt is None:
        slot_mt = lattice.leaf_index_of_region[rid_mt]
    return rid_mt, slot_mt


def _sweep_meta(lattice) -> List[Tuple]:
    """Per-level (has_leaf, has_nonleaf, leaf_pos, nonleaf_pos, leaf_rids,
    nonleaf_rids, leaf_slots). Reads the tensors precomputed at lattice build;
    falls back to computing them (with syncs) for foreign/stub lattices."""
    metas: List[Tuple] = []
    for lv in lattice.levels:
        if getattr(lv, "leaf_pos", None) is not None:
            metas.append((lv.has_leaf, lv.has_nonleaf, lv.leaf_pos,
                          lv.nonleaf_pos, lv.leaf_rids, lv.nonleaf_rids,
                          lv.leaf_slots))
        else:
            is_leaf = lattice.leaf_mask[lv.parents]
            leaf_pos = torch.nonzero(is_leaf, as_tuple=False).reshape(-1)
            nonleaf_pos = torch.nonzero(~is_leaf, as_tuple=False).reshape(-1)
            leaf_rids = lv.parents[leaf_pos]
            nonleaf_rids = lv.parents[nonleaf_pos]
            leaf_slots = lattice.leaf_index_of_region[leaf_rids]
            metas.append((bool(leaf_pos.numel() > 0), bool(nonleaf_pos.numel() > 0),
                          leaf_pos, nonleaf_pos, leaf_rids, nonleaf_rids, leaf_slots))
    return metas


def _leaf_terms(
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    log_PT: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    term_mark: Optional[torch.Tensor] = None,
    semiring: str = "sum",
    exact: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Termination scores for every leaf-eligible region.

    Returns (term_score [B, n_leaf, S], log_cont [B, n_leaf, S], argT).
    term_score = log p_term + mix_T(log p(T|A,c) + ell/kappa) where mix is
    logsumexp (sum semiring) or max (max semiring; argT returned).
    must_terminate: log p_term = 0, log(1 - p_term) = NEG sentinel (never fed
    through logaddexp — the caller indexes around it).
    """
    prior = logbmm(U_logmix.float(), log_PT.float(), exact=exact)  # [B, S, T_v]
    ell_t = ell_leaf.float() / temper_kappa.float().view(1, -1, 1)  # [B, n_leaf, T_v]
    if semiring == "sum":
        mix = logbmm_batched(ell_t, prior.transpose(1, 2), exact=exact)  # [B, n_leaf, S]
        argT = None
    else:
        B, n_leaf, T_v = ell_t.shape
        S = prior.shape[1]
        step = max(1, _MAX_SEMIRING_BUDGET // max(1, B * S * T_v))
        mixes, args = [], []
        for i0 in range(0, n_leaf, step):
            scores = prior.unsqueeze(1) + ell_t[:, i0 : i0 + step].unsqueeze(2)
            m, a = scores.max(dim=-1)                             # [B, chunk, S]
            mixes.append(m)
            args.append(a)
        mix = torch.cat(mixes, dim=1)
        argT = torch.cat(args, dim=1)
    lt = term_logits.float()[:, lattice.leaf_ids, :]
    log_term = F.logsigmoid(lt)
    log_cont = F.logsigmoid(-lt)
    mt = lattice.must_terminate[lattice.leaf_ids].view(1, -1, 1)
    log_term = torch.where(mt, torch.zeros_like(log_term), log_term)
    log_cont = torch.where(mt, torch.full_like(log_cont, NEG), log_cont)
    term_score = log_term + mix
    if term_mark is not None:
        term_score = term_score + term_mark[:, lattice.leaf_ids, :]
    return term_score, log_cont, argT


def _sweep(
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    logV: torch.Tensor,
    logW: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    log_PT: torch.Tensor,
    term_mark: Optional[torch.Tensor] = None,
    expand_mark: Optional[torch.Tensor] = None,
    cut_mark: Optional[torch.Tensor] = None,
    exact: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Level-synchronous inside sweep -> (beta [B, N_reg, S] fp32, logZ [B]).

    All shape-dependent decisions (leaf/non-leaf membership per level,
    must-terminate ids, cut-type masks) come from lattice-precomputed index
    tensors and python bools so the loop issues no CPU-GPU syncs (no
    ``bool(tensor)``, no bool-mask advanced indexing, no ``nonzero``).

    Two equivalent implementations: the fast sweep (contiguous-id lattices,
    i.e. every Lattice built by sprig.dp.lattice) keeps per-level beta blocks
    in a list and maintains the R-compressed child tables
        Blo[b,r,k] = logsumexp_B(beta[b,r,B] + logV[k,B])   (Bhi with logW)
    so the per-level child gathers and writes touch [B,N,R] instead of
    [B,N,S] — identical values (the compression is exactly the logbmm the
    reference applies per cut row, hoisted per region) but ~S/R times less
    index/index_put traffic, which dominates the DP's backward. The reference
    sweep is kept for foreign/stub lattices."""
    B, _N, S = term_logits.shape
    device = ell_leaf.device
    logV = logV.float()
    logW = logW.float()
    U_log = U_logmix.float()

    term_score, log_cont, _ = _leaf_terms(
        ell_leaf, term_logits, U_log, log_PT, lattice, temper_kappa,
        term_mark=term_mark, semiring="sum", exact=exact,
    )
    tls_flat = _cut_logp_flat(cut_logits, lattice)
    U_log_t = U_log.transpose(1, 2)
    rid_mt, slot_mt = _mt_meta(lattice)

    if getattr(lattice, "contiguous_ids", False):
        return _sweep_fast(
            term_score, log_cont, cut_logits, U_log_t, logV, logW, lattice,
            tls_flat, slot_mt, expand_mark, cut_mark, exact,
        )

    beta = torch.full((B, lattice.n_regions, S), NEG, device=device)
    beta[:, rid_mt, :] = term_score[:, slot_mt, :]

    row_offset = 0
    for lv, meta in zip(lattice.levels, _sweep_meta(lattice)):
        has_leaf, has_nonleaf, leaf_pos, nonleaf_pos, leaf_rids, nonleaf_rids, leaf_slots = meta
        M = lv.parent_ids.shape[0]
        P = lv.parents.shape[0]
        # Bhat[b,m,k] = logsumexp_B(beta[b, child_lo[m], B] + logV[k, B]); Chat same with logW.
        Bhat = logbmm(beta[:, lv.child_lo, :], logV.t(), exact=exact)   # [B, M, R]
        Chat = logbmm(beta[:, lv.child_hi, :], logW.t(), exact=exact)
        contrib = _level_cut_logp(tls_flat, cut_logits, lattice, lv) + Bhat + Chat
        if cut_mark is not None:
            contrib = contrib + cut_mark[:, row_offset : row_offset + M].unsqueeze(-1)
        comb = _segment_logsumexp(contrib, lv.parent_index, P)          # [B, P, R]
        # expand[b,p,A] = logsumexp_k(comb[b,p,k] + U_log[b,A,k])
        expand = logbmm_batched(comb, U_log_t, exact=exact)             # [B, P, S]
        if expand_mark is not None:
            expand = expand + expand_mark[:, lv.parents, :]

        # Combine the terminate/expand branches. Only finite values may enter
        # the log-add: with the -1e30 sentinel (and equally with legitimate
        # log-domain gaps > ~88 nats once emissions sharpen) torch.logaddexp's
        # first-order backward materializes exp(b - a) = inf nodes whose
        # double-backward (create_graph=True in SPRIGModel.loss) is NaN — so
        # mask/index the branches and use a detached-max shift, which keeps
        # every exp argument <= 0 and the log argument in [1, 2].
        if has_nonleaf:
            beta[:, nonleaf_rids, :] = expand[:, nonleaf_pos, :]
        if has_leaf:
            a = term_score[:, leaf_slots, :]
            b = log_cont[:, leaf_slots, :] + expand[:, leaf_pos, :]
            m = torch.maximum(a, b).detach()
            beta[:, leaf_rids, :] = m + torch.log(torch.exp(a - m) + torch.exp(b - m))
        row_offset += M

    logZ = beta[:, lattice.root_id, 0]
    return beta, logZ


def _combine_branches(
    expand: torch.Tensor,
    term_score: torch.Tensor,
    log_cont: torch.Tensor,
    lv,
) -> torch.Tensor:
    """Terminate/expand combination for one level -> beta block [B, P, S] in
    parent order. Same guarded log-add as the reference sweep (see the long
    comment there); rows are assembled with a precomputed permutation gather
    instead of index_put (cheaper backward, no clone of the block)."""
    if not lv.has_leaf:
        return expand
    a = term_score[:, lv.leaf_slots, :]
    b = log_cont[:, lv.leaf_slots, :] + expand[:, lv.leaf_pos, :]
    m = torch.maximum(a, b).detach()
    leaf_val = m + torch.log(torch.exp(a - m) + torch.exp(b - m))
    if not lv.has_nonleaf:
        return leaf_val
    both = torch.cat([expand[:, lv.nonleaf_pos, :], leaf_val], dim=1)
    return both[:, lv.reorder, :]


def _sweep_fast(
    term_score: torch.Tensor,
    log_cont: torch.Tensor,
    cut_logits: torch.Tensor,
    U_log_t: torch.Tensor,
    logV: torch.Tensor,
    logW: torch.Tensor,
    lattice,
    tls_flat: Optional[torch.Tensor],
    slot_mt: torch.Tensor,
    expand_mark: Optional[torch.Tensor],
    cut_mark: Optional[torch.Tensor],
    exact: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast inside sweep over contiguous-id lattices (see _sweep docstring)."""
    B, S = term_score.shape[0], term_score.shape[-1]
    R = logV.shape[0]
    N = lattice.n_regions
    device = term_score.device

    base = term_score[:, slot_mt, :]                       # [B, n_mt, S] == ids [0, n_mt)
    parts = [base]
    Blo = torch.full((B, N, R), NEG, device=device)
    Bhi = torch.full((B, N, R), NEG, device=device)
    n_mt = base.shape[1]
    Blo[:, :n_mt, :] = logbmm(base, logV.t(), exact=exact)
    Bhi[:, :n_mt, :] = logbmm(base, logW.t(), exact=exact)

    row_offset = 0
    for lv in lattice.levels:
        M = lv.parent_ids.shape[0]
        P = lv.parents.shape[0]
        Bhat = Blo[:, lv.child_lo, :]                      # [B, M, R]
        Chat = Bhi[:, lv.child_hi, :]
        contrib = _level_cut_logp(tls_flat, cut_logits, lattice, lv) + Bhat + Chat
        if cut_mark is not None:
            contrib = contrib + cut_mark[:, row_offset : row_offset + M].unsqueeze(-1)
        comb = _segment_logsumexp(contrib, lv.parent_index, P)          # [B, P, R]
        expand = logbmm_batched(comb, U_log_t, exact=exact)             # [B, P, S]
        if expand_mark is not None:
            expand = expand + expand_mark[:, lv.parents, :]
        val = _combine_branches(expand, term_score, log_cont, lv)       # [B, P, S]
        parts.append(val)
        Blo[:, lv.id_lo : lv.id_hi, :] = logbmm(val, logV.t(), exact=exact)
        Bhi[:, lv.id_lo : lv.id_hi, :] = logbmm(val, logW.t(), exact=exact)
        row_offset += M

    beta = torch.cat(parts, dim=1)                         # region-id order
    logZ = beta[:, lattice.root_id, 0]
    return beta, logZ


def inside(
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    logV: torch.Tensor,
    logW: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    log_PT: torch.Tensor,
    exact: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DESIGN section 5 inside DP -> (beta [B, N_reg, S] fp32, logZ [B])."""
    return _sweep(
        ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice,
        temper_kappa, log_PT, exact=exact,
    )


# DESIGN.md section 5 names the function `inside`; SPRIGModel's adapter looks
# for `inside_logZ` first — keep both bound to the same implementation.
inside_logZ = inside


def posterior_marginals(
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    logV: torch.Tensor,
    logW: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    log_PT: torch.Tensor,
    exact: bool = False,
) -> Dict[str, torch.Tensor]:
    """Expected-count posterior marginals via the autograd identity.

    Zero-valued marker potentials are added on the terminate branch, the
    expand branch, and every concrete cut row; grad of logZ w.r.t. a marker
    equals the posterior expected count of that event. Returns detached
    tensors: node/term/expand [B, N_reg, S], cut [B, M_total] (levels
    concatenated in lattice.levels row order), texel [B, n_leaf, T_v]
    (grad w.r.t. ell rescaled by kappa), rule [B, S, R] (grad w.r.t.
    U_logmix), logZ [B].
    """
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
            term_mark=tm, expand_mark=em, cut_mark=cm, exact=exact,
        )
        g_tm, g_em, g_cm, g_ell, g_u = torch.autograd.grad(
            logZ.sum(), [tm, em, cm, ell_d, U_d]
        )
    texel = g_ell * temper_kappa.detach().float().view(1, -1, 1)
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
    ell_leaf: torch.Tensor,
    term_logits: torch.Tensor,
    cut_logits: torch.Tensor,
    U_logmix: torch.Tensor,
    logV: torch.Tensor,
    logW: torch.Tensor,
    lattice,
    temper_kappa: torch.Tensor,
    log_PT: torch.Tensor,
    exact: bool = False,
) -> Tuple[torch.Tensor, List[Tuple]]:
    """Max-semiring sweep + argmax backtrace.

    Returns (score [B], trees): trees[b] is a nested tuple
    (region_id, symbol, texel_or_None, axis_or_None, cut_px_or_None,
    (children...)) rooted at (lattice.root_id, symbol 0). The texel T-mix is
    a max; the texel prior p(T|A,c) itself stays an exact sum over k.
    """
    with torch.no_grad():
        B, N, S = term_logits.shape
        R = U_logmix.shape[-1]
        device = ell_leaf.device
        U_log = U_logmix.float()
        logV_f = logV.float()
        logW_f = logW.float()
        term_score, log_cont, argT = _leaf_terms(
            ell_leaf, term_logits, U_log, log_PT, lattice, temper_kappa,
            semiring="max", exact=exact,
        )
        beta = torch.full((B, N, S), NEG, device=device)
        term_choice = torch.zeros(B, N, S, dtype=torch.bool, device=device)
        leaf_slot = lattice.leaf_index_of_region
        rid_mt, slot_mt = _mt_meta(lattice)
        beta[:, rid_mt, :] = term_score[:, slot_mt, :]
        term_choice[:, rid_mt, :] = True

        tls_flat = _cut_logp_flat(cut_logits, lattice)
        metas = _sweep_meta(lattice)
        bp: List[Dict[str, torch.Tensor]] = []
        parent_pos: Dict[int, Tuple[int, int]] = {}
        for li, lv in enumerate(lattice.levels):
            for p_i, r in enumerate(lv.parents.tolist()):
                parent_pos[r] = (li, p_i)
            M = lv.parent_ids.shape[0]
            P = lv.parents.shape[0]

            # Per-row maxima over child symbols and over components, chunked
            # over rows to bound the [B, chunk, R, S] / [B, chunk, S, R]
            # materializations.
            step = max(1, _MAX_SEMIRING_BUDGET // max(1, B * S * R))
            cutlp = _level_cut_logp(tls_flat, cut_logits, lattice, lv)   # [B, M, R]
            argB = torch.empty(B, M, R, dtype=torch.int64, device=device)
            argC = torch.empty(B, M, R, dtype=torch.int64, device=device)
            argk = torch.empty(B, M, S, dtype=torch.int64, device=device)
            tmax = torch.empty(B, M, S, device=device)
            for m0 in range(0, M, step):
                m1 = min(M, m0 + step)
                lo_b = beta[:, lv.child_lo[m0:m1], :]                   # [B, c, S]
                hi_b = beta[:, lv.child_hi[m0:m1], :]
                Bv, aB = (lo_b.unsqueeze(2) + logV_f.view(1, 1, R, S)).max(dim=-1)
                Cv, aC = (hi_b.unsqueeze(2) + logW_f.view(1, 1, R, S)).max(dim=-1)
                argB[:, m0:m1] = aB
                argC[:, m0:m1] = aC
                contrib = cutlp[:, m0:m1] + Bv + Cv                     # [B, c, R]
                t = contrib.unsqueeze(2) + U_log.unsqueeze(1)           # [B, c, S, R]
                tm_c, ak = t.max(dim=-1)
                tmax[:, m0:m1] = tm_c
                argk[:, m0:m1] = ak

            # Group-max over the rows of each parent; ties -> lowest row index.
            pidx = lv.parent_index.view(1, -1, 1).expand(B, M, S)
            pmax = torch.full((B, P, S), NEG, device=device).scatter_reduce(
                1, pidx, tmax, reduce="amax", include_self=True
            )
            rowidx = torch.arange(M, device=device).view(1, -1, 1).expand(B, M, S)
            cand = torch.where(
                tmax >= pmax[:, lv.parent_index, :], rowidx,
                torch.full_like(rowidx, M),
            )
            argm = torch.full((B, P, S), M, dtype=torch.int64, device=device).scatter_reduce(
                1, pidx, cand, reduce="amin", include_self=True
            )

            rid = lv.parents
            has_leaf, _hn, leaf_pos, _np, _lr, _nr, leaf_slots = metas[li]
            cont = torch.zeros(B, P, S, device=device)
            tb = torch.full((B, P, S), NEG, device=device)
            if has_leaf:
                cont[:, leaf_pos, :] = log_cont[:, leaf_slots, :]
                tb[:, leaf_pos, :] = term_score[:, leaf_slots, :]
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
