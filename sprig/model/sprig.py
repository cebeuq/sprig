"""SPRIGModel — wires GMT + TexelAtlas + inside DP (contracts C2/C3/C4).

DP access: all calls to sprig/dp/inside.py go through the small adapters at the
bottom of this file (`_dp`, `_call_dp`), keyed by the DESIGN.md section 5
argument names (ell_leaf, term_logits, cut_logits, U_logmix, logV, logW,
lattice, temper_kappa; plus log_PT for the texel prior). Tests may override the
DP module via `_DP_MODULE`.

Conventions:
- Axiom (root) nonterminal is symbol 0.
- Rule-logit temperature `tau` (buffer) divides all rule/termination logits
  (U, cut-type, V, W, P_T, termination) per DESIGN.md section 2.
- Emission tempering `eta` (buffer): kappa(r) = max(1, area_px(r)^eta);
  `log_marginal(..., report_mode=True)` forces eta = 0 (C2).
"""
from __future__ import annotations

import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from sprig.dp.lattice import Lattice, get_lattice
from sprig.model import dl
from sprig.model.atlas import ATLAS_RES, TexelAtlas, film_scale_shift, phi_at_leaf_centers
from sprig.model.gmt import GrammarModulationTransformer

_MATERIAL_SEED_OFFSET = 0x9E3779B9


@dataclass
class SPRIGConfig:
    S: int = 1024
    R: int = 64
    T_v: int = 256
    d: int = 384
    canvas: int = 64
    grid: int = 8
    leaf_max: int = 16
    n_heads: int = 6
    d_t: int = 64
    caption_dim: int = 768
    n_geom: int = 64
    atlas_heads: int = 4
    leaf_chunk: int = 16
    axiom: int = 0
    texel_hinge_weight: float = 1.0
    symbol_hinge_weight: float = 0.5
    # Object-pixel importance weight on the emission log-likelihood during
    # training (diagnosis fix 1: objects are <10% of pixels, so unweighted NLL
    # lets background fidelity dominate and no object texels ever form).
    # Requires the dataloader to provide batch["objmask"] u8 [B,C,C].
    # 1.0 = exact NLL (eval/log_marginal always use exact NLL regardless).
    emission_obj_weight: float = 1.0
    # Second-order grad through the DP so the under-use hinges actually train
    # (usage comes from autograd.grad of logZ; without create_graph the hinge
    # would be constant w.r.t. parameters).
    hinge_create_graph: bool = True
    resurrect_threshold_frac: float = 0.1  # of uniform usage 1/T_v

    def __post_init__(self) -> None:
        if self.d % self.n_heads != 0:
            raise ValueError("d must be divisible by n_heads")


@dataclass
class ParseNode:
    rect: Tuple[int, int, int, int]
    axis: Optional[int]          # None for leaves; 0 = vertical cut, 1 = horizontal
    cut_px: Optional[int]
    symbol: int
    texel: Optional[int]         # None for internal nodes
    children: List["ParseNode"] = field(default_factory=list)


class SPRIGModel(nn.Module):
    def __init__(self, cfg: SPRIGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.gmt = GrammarModulationTransformer(cfg)
        self.atlas = TexelAtlas(cfg)
        self.lattice: Lattice = get_lattice(cfg.canvas, cfg.grid, cfg.leaf_max)
        self.register_buffer("tau", torch.tensor(1.0))
        self.register_buffer("eta", torch.tensor(0.0))

    # ------------------------------------------------------------------ utils

    def _lat(self, device: torch.device) -> Lattice:
        self.lattice.to(device)
        return self.lattice

    def _kappa(self, eta: float, device: torch.device) -> torch.Tensor:
        """kappa(r) = max(1, area_px(r)^eta) over leaf slots -> fp32 [n_leaf]."""
        lat = self._lat(device)
        area = lat.area_px[lat.leaf_ids].float()
        if eta <= 0.0:
            return torch.ones_like(area)
        return torch.clamp(area ** eta, min=1.0)

    def _conditionals(
        self,
        emb: torch.Tensor,
        emb_len: torch.Tensor,
        images: Optional[torch.Tensor] = None,
        pix_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """All caption-conditioned quantities, temperature already applied."""
        lat = self._lat(emb.device)
        out = self.gmt(emb, emb_len)
        tau = self.tau.clamp(min=1e-3)
        cond: Dict[str, torch.Tensor] = {}
        cond["H"] = out.H
        cond["Phi"] = out.Phi
        cond["U_logmix"] = F.log_softmax(out.U / tau, dim=-1)          # [B,S,R]
        cond["log_PT"] = F.log_softmax(self.gmt.P_T / tau, dim=-1)     # [R,T_v]
        cond["logV"] = F.log_softmax(self.gmt.V / tau, dim=-1)         # [R,S]
        cond["logW"] = F.log_softmax(self.gmt.W / tau, dim=-1)         # [R,S]
        cond["term_logits"] = self.gmt.termination_logits(out.H, lat.phi_geom) / tau
        cond["cut_logits"] = out.cut_logits / tau                      # [B,R,14]
        cond["atlas"] = self.atlas.render(emb, emb_len)
        if images is not None:
            cond["ell"] = self.atlas.score_leaves(
                cond["atlas"], images, lat, out.Phi, pix_weight=pix_weight)
        return cond

    def _texel_prior_log(self, cond: Dict[str, torch.Tensor]) -> torch.Tensor:
        """log p(T|A,c) = logsumexp_k(log p(k|A,c) + log p(T|k)) -> [B,S,T_v]."""
        return torch.logsumexp(
            cond["U_logmix"].unsqueeze(-1) + cond["log_PT"].unsqueeze(0).unsqueeze(0),
            dim=2,
        )

    def _dp_kwargs(
        self, cond: Dict[str, torch.Tensor], kappa: torch.Tensor
    ) -> Dict[str, Any]:
        return dict(
            ell_leaf=cond["ell"],
            term_logits=cond["term_logits"],
            cut_logits=cond["cut_logits"],
            U_logmix=cond["U_logmix"],
            logV=cond["logV"],
            logW=cond["logW"],
            lattice=self.lattice,
            temper_kappa=kappa,
            log_PT=cond["log_PT"],
        )

    def _inside(self, cond: Dict[str, torch.Tensor], kappa: torch.Tensor) -> torch.Tensor:
        mod = _dp()
        fn = getattr(mod, "inside_logZ", None) or getattr(mod, "inside")
        res = _call_dp(fn, self._dp_kwargs(cond, kappa))
        if isinstance(res, (tuple, list)):
            return res[-1]  # (beta, logZ) per DESIGN section 5
        return res

    # -------------------------------------------------------------- contracts

    def log_marginal(
        self,
        image: torch.Tensor,
        emb: torch.Tensor,
        emb_len: torch.Tensor,
        report_mode: bool = False,
    ) -> torch.Tensor:
        """C2: exact log p(x|c) [B]; report_mode forces eta = 0."""
        eta = 0.0 if report_mode else float(self.eta)
        cond = self._conditionals(emb, emb_len, images=image)
        kappa = self._kappa(eta, emb.device)
        return self._inside(cond, kappa)

    def loss(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """DESIGN section 6: tempered NLL (nats/subpixel) + under-use hinges.

        Usage vectors are posterior expected counts obtained from
        torch.autograd.grad(logZ.sum(), [ell, U_logmix], retain_graph=True):
        grad wrt ell (x kappa) = expected (leaf, texel) counts; grad wrt
        U_logmix summed over k = expected node counts per symbol (the texel
        prior is mixed from U_logmix, so termination counts are included).
        """
        images = batch["image"]
        emb = batch["emb"]
        emb_len = batch["emb_len"]
        cfg = self.cfg
        pix_weight = None
        if cfg.emission_obj_weight != 1.0 and batch.get("objmask") is not None:
            pix_weight = 1.0 + (cfg.emission_obj_weight - 1.0) * batch[
                "objmask"].to(images.device, torch.float32)
        cond = self._conditionals(emb, emb_len, images=images,
                                  pix_weight=pix_weight)
        kappa = self._kappa(float(self.eta), emb.device)
        logZ = self._inside(cond, kappa)

        n_subpix = 3 * cfg.canvas * cfg.canvas
        nll = (-logZ / n_subpix).mean()

        create = cfg.hinge_create_graph and torch.is_grad_enabled()
        g_ell, g_u = torch.autograd.grad(
            logZ.sum(),
            [cond["ell"], cond["U_logmix"]],
            retain_graph=True,
            create_graph=create,
        )
        texel_counts = (g_ell * kappa.view(1, -1, 1)).sum(dim=(0, 1))   # [T_v]
        symbol_counts = g_u.sum(dim=(0, 2))                             # [S]
        texel_usage = texel_counts / texel_counts.sum().clamp(min=1e-12)
        symbol_usage = symbol_counts / symbol_counts.sum().clamp(min=1e-12)

        texel_hinge = F.relu(1.0 / (4.0 * cfg.T_v) - texel_usage).sum()
        symbol_hinge = F.relu(1.0 / (4.0 * cfg.S) - symbol_usage).sum()
        total = (
            nll
            + cfg.texel_hinge_weight * texel_hinge
            + cfg.symbol_hinge_weight * symbol_hinge
        )

        with torch.no_grad():
            tu = texel_usage.detach()
            su = symbol_usage.detach()
            metrics = {
                "loss": float(total),
                "nll": float(nll),
                "bpd": float(nll) / math.log(2.0),
                "logZ_mean": float(logZ.mean()),
                "texel_hinge": float(texel_hinge),
                "symbol_hinge": float(symbol_hinge),
                "texel_alive_frac": float((tu >= 1.0 / (4.0 * cfg.T_v)).float().mean()),
                "symbol_eff": float(torch.exp(-(su.clamp(min=1e-12) * su.clamp(min=1e-12).log()).sum())),
                "mean_leaves": float((g_ell.detach() * kappa.view(1, -1, 1)).sum() / images.shape[0]),
            }
        return total, metrics

    def posterior_usage(
        self, image: torch.Tensor, emb: torch.Tensor, emb_len: torch.Tensor
    ) -> Dict[str, Any]:
        """C3: expected-count diagnostics from posterior marginals.

        node_entropy = occupancy-weighted entropy of the conditional
        split posterior at each region (choices: terminate + each concrete
        cut). mean_depth is the area-based proxy E[log2(canvas_area /
        leaf_area)] over leaf marginals.
        """
        cfg = self.cfg
        with torch.enable_grad():
            cond = self._conditionals(emb, emb_len, images=image)
            kappa = self._kappa(float(self.eta), emb.device)
            marg = _call_dp(_dp().posterior_marginals, self._dp_kwargs(cond, kappa))

        lat = self._lat(emb.device)
        node = marg["node"].float()          # [B, N_reg, S]
        term = marg["term"].float()          # [B, N_reg, S]
        cut = marg["cut"].float()            # [B, M_total] concatenated level rows
        texel = marg["texel"].float()        # [B, n_leaf, T_v]
        rule = marg["rule"].float()          # [B, S, R]

        symbol_counts = node.sum(dim=(0, 1))
        texel_counts = texel.sum(dim=(0, 1))
        symbol_usage = symbol_counts / symbol_counts.sum().clamp(min=1e-12)
        texel_usage = texel_counts / texel_counts.sum().clamp(min=1e-12)

        # Occupancy-weighted entropy of conditional split posteriors.
        B = node.shape[0]
        occ = node.sum(dim=-1)               # [B, N_reg]
        term_tot = term.sum(dim=-1)          # [B, N_reg]
        parent_of_row = torch.cat([lv.parent_ids for lv in lat.levels]).to(emb.device)
        eps = 1e-12
        occ_safe = occ.clamp(min=eps)
        q_term = (term_tot / occ_safe).clamp(min=0.0)
        plogp_term = torch.where(
            q_term > eps, q_term * q_term.clamp(min=eps).log(), torch.zeros_like(q_term)
        )
        q_cut = cut / occ_safe[:, parent_of_row]
        plogp_cut_rows = torch.where(
            q_cut > eps, q_cut * q_cut.clamp(min=eps).log(), torch.zeros_like(q_cut)
        )
        plogp_cut = torch.zeros_like(occ).index_add_(
            1, parent_of_row, plogp_cut_rows
        )
        ent_r = -(plogp_term + plogp_cut)    # [B, N_reg]
        w = occ.clamp(min=0.0)
        node_entropy = float((w * ent_r).sum() / w.sum().clamp(min=eps))

        # Magnitude monitors.
        ell = cond["ell"].detach().float()
        emit_mag = float(
            ((texel * ell).sum() / texel.sum().clamp(min=eps)).abs()
        )
        rule_mag = float(
            ((rule * cond["U_logmix"].detach()).sum() / rule.sum().clamp(min=eps)).abs()
        )

        leaf_area = lat.area_px[lat.leaf_ids].float()
        depth_proxy = torch.log2(float(cfg.canvas * cfg.canvas) / leaf_area)
        leaf_marg = texel.sum(dim=-1)        # [B, n_leaf]
        mean_depth = float(
            (leaf_marg * depth_proxy.unsqueeze(0)).sum() / leaf_marg.sum().clamp(min=eps)
        )
        mean_leaves = float(leaf_marg.sum() / B)

        return {
            "symbol_usage": symbol_usage.detach(),
            "texel_usage": texel_usage.detach(),
            "node_entropy": node_entropy,
            "emit_mag": emit_mag,
            "rule_mag": rule_mag,
            "mean_depth": mean_depth,
            "mean_leaves": mean_leaves,
        }

    def map_parse(
        self, image: torch.Tensor, emb: torch.Tensor, emb_len: torch.Tensor
    ) -> List[ParseNode]:
        """C3: Viterbi (max-semiring) parse per batch element."""
        with torch.no_grad():
            cond = self._conditionals(emb, emb_len, images=image)
            kappa = self._kappa(float(self.eta), emb.device)
            res = _call_dp(_dp().viterbi, self._dp_kwargs(cond, kappa))
        _score, trees = res
        return [self._to_parse_node(t) for t in trees]

    def _to_parse_node(self, node: Tuple) -> ParseNode:
        """Convert the DP viterbi tree tuple (region_id, symbol, texel, axis,
        cut_px, children) into a ParseNode."""
        rid, sym, texel, axis, cut_px, children = node
        rect = tuple(int(v) for v in self.lattice.regions[rid].tolist())
        return ParseNode(
            rect=rect,
            axis=None if axis is None else int(axis),
            cut_px=None if cut_px is None else int(cut_px),
            symbol=int(sym),
            texel=None if texel is None else int(texel),
            children=[self._to_parse_node(ch) for ch in children],
        )

    # --------------------------------------------------------------- sampling

    @torch.no_grad()
    def sample(
        self,
        emb: torch.Tensor,
        emb_len: torch.Tensor,
        seed_struct: int,
        seed_material: int,
        n: int,
    ) -> Tuple[torch.Tensor, List[ParseNode]]:
        """C4: ancestral sampling, breadth-parallel over the frontier.

        Two RNG streams: structural draws (termination, k, cut, B, C) from
        seed_struct; material draws (texel choice) from seed_material. Pixels
        are rendered as DL means (no pixel noise). Returns
        (images u8 [n, canvas, canvas, 3] on CPU, list of n ParseNode roots).
        """
        images, trees, _scores = self._sample_scored(emb, emb_len, seed_struct, seed_material, n)
        return images, trees

    @torch.no_grad()
    def sample_bestof(
        self, emb: torch.Tensor, emb_len: torch.Tensor, K: int, seed: int
    ) -> Tuple[torch.Tensor, List[ParseNode]]:
        """C4: sample K derivations, rerank by joint log p(tree, rendered
        image | c), return the best as (images u8 [1, canvas, canvas, 3], [tree])."""
        images, trees, scores = self._sample_scored(
            emb, emb_len, int(seed), int(seed) + _MATERIAL_SEED_OFFSET, K
        )
        best = int(torch.tensor(scores).argmax())
        return images[best : best + 1], [trees[best]]

    @torch.no_grad()
    def _sample_scored(
        self,
        emb: torch.Tensor,
        emb_len: torch.Tensor,
        seed_struct: int,
        seed_material: int,
        n: int,
    ) -> Tuple[torch.Tensor, List[ParseNode], List[float]]:
        cfg = self.cfg
        device = next(self.parameters()).device
        if emb.dim() == 2:
            emb = emb.unsqueeze(0)
        if not torch.is_tensor(emb_len):
            emb_len = torch.tensor([emb_len])
        emb_len = emb_len.reshape(-1)[:1]
        emb = emb[:1].to(device)

        cond = self._conditionals(emb, emb_len)
        lat = self._lat(device)

        # Single-caption conditionals on CPU for generator-driven draws.
        u_logmix = cond["U_logmix"][0].float().cpu()             # [S,R]
        term_logits = cond["term_logits"][0].float().cpu()       # [N,S]
        cut_logits = cond["cut_logits"][0].float().cpu()         # [R,14]
        logv = cond["logV"].float().cpu()                        # [R,S]
        logw = cond["logW"].float().cpu()                        # [R,S]
        texel_prior = self._texel_prior_log(cond)[0].float().cpu()  # [S,T_v]
        atlas0 = cond["atlas"][0].float().cpu()                  # [T_v,40,16,16]
        phi0 = cond["Phi"][:1].float().cpu()                     # [1,8,16,16]

        pooled_cache: Dict[Tuple[int, int], torch.Tensor] = {}
        for (h, w), _slots in lat.leaf_shape_groups().items():
            pooled_cache[(h, w)] = F.adaptive_avg_pool2d(atlas0, (h, w))

        gen_s = torch.Generator().manual_seed(int(seed_struct) & 0x7FFFFFFFFFFFFFFF)
        gen_m = torch.Generator().manual_seed(int(seed_material) & 0x7FFFFFFFFFFFFFFF)

        regions_cpu = lat.regions.cpu()
        leaf_mask = lat.leaf_mask.cpu()
        must_term = lat.must_terminate.cpu()
        type_present = lat.type_present.cpu()

        images_out: List[torch.Tensor] = []
        trees: List[ParseNode] = []
        scores: List[float] = []
        for _ in range(n):
            root = ParseNode(
                rect=tuple(int(v) for v in regions_cpu[lat.root_id].tolist()),
                axis=None, cut_px=None, symbol=cfg.axiom, texel=None,
            )
            frontier: List[Tuple[int, int, ParseNode]] = [(lat.root_id, cfg.axiom, root)]
            leaves: List[Tuple[Tuple[int, int, int, int], int]] = []
            logp = 0.0
            while frontier:
                nxt: List[Tuple[int, int, ParseNode]] = []
                for rid, sym, pn in frontier:
                    can_term = bool(leaf_mask[rid])
                    forced_term = bool(must_term[rid])
                    if forced_term:
                        terminate = True
                    elif not can_term:
                        terminate = False
                    else:
                        p_term = torch.sigmoid(term_logits[rid, sym])
                        u = torch.rand((), generator=gen_s)
                        terminate = bool(u < p_term)
                        logp += float(torch.log(p_term if terminate else 1.0 - p_term))
                    if terminate:
                        probs = torch.exp(texel_prior[sym])
                        t = int(torch.multinomial(probs, 1, generator=gen_m))
                        logp += float(texel_prior[sym, t])
                        pn.texel = t
                        leaves.append((pn.rect, t))
                    else:
                        k = int(torch.multinomial(torch.exp(u_logmix[sym]), 1, generator=gen_s))
                        logp += float(u_logmix[sym, k])
                        # Concrete cut: masked cut-type softmax, mass split
                        # uniformly across same-type concrete cuts.
                        cuts = lat.cuts_of_region[rid]
                        tl = cut_logits[k].masked_fill(~type_present[rid], float("-inf"))
                        tls = F.log_softmax(tl, dim=-1)
                        row_logp = torch.tensor(
                            [float(tls[c[4]]) - c[5] for c in cuts]
                        )
                        ci = int(torch.multinomial(torch.exp(row_logp), 1, generator=gen_s))
                        logp += float(row_logp[ci])
                        axis, px, lo, hi, _t, _lc = cuts[ci]
                        b_sym = int(torch.multinomial(torch.exp(logv[k]), 1, generator=gen_s))
                        c_sym = int(torch.multinomial(torch.exp(logw[k]), 1, generator=gen_s))
                        logp += float(logv[k, b_sym]) + float(logw[k, c_sym])
                        pn.axis, pn.cut_px = int(axis), int(px)
                        ch_lo = ParseNode(
                            rect=tuple(int(v) for v in regions_cpu[lo].tolist()),
                            axis=None, cut_px=None, symbol=b_sym, texel=None,
                        )
                        ch_hi = ParseNode(
                            rect=tuple(int(v) for v in regions_cpu[hi].tolist()),
                            axis=None, cut_px=None, symbol=c_sym, texel=None,
                        )
                        pn.children = [ch_lo, ch_hi]
                        nxt.append((lo, b_sym, ch_lo))
                        nxt.append((hi, c_sym, ch_hi))
                frontier = nxt

            img, emit_ll = self._render_leaves(leaves, pooled_cache, phi0)
            images_out.append(img)
            trees.append(root)
            scores.append(logp + emit_ll)

        return torch.stack(images_out, dim=0), trees, scores

    def _render_leaves(
        self,
        leaves: List[Tuple[Tuple[int, int, int, int], int]],
        pooled_cache: Dict[Tuple[int, int], torch.Tensor],
        phi0: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Paint DL-mean pixels for each leaf; also return the total emission
        log-likelihood of the rendered (quantized) pixels — the emission part
        of the joint score used by sample_bestof."""
        cfg = self.cfg
        canvas = torch.zeros(3, cfg.canvas, cfg.canvas)
        emit_ll = 0.0
        for (x0, y0, x1, y1), texel in leaves:
            h, w = y1 - y0, x1 - x0
            p = pooled_cache[(h, w)][texel].clone()              # [40,h,w]
            rect_t = torch.tensor([[x0, y0, x1, y1]])
            phi = phi_at_leaf_centers(phi0, rect_t, cfg.canvas)[0, 0]  # [8]
            scale, shift = film_scale_shift(phi)
            for c in range(3):
                idx = [10 * j + 1 + c for j in range(dl.N_COMP)]
                p[idx] = p[idx] * scale[c] + shift[c]
            pix = dl.dl_mean_pixels(p.unsqueeze(0))[0]           # [3,h,w]
            pix_u8 = dl.unit_to_u8(pix)
            canvas[:, y0:y1, x0:x1] = dl.u8_to_unit(pix_u8)
            emit_ll += float(
                dl.dl_logprob(p.unsqueeze(0), dl.u8_to_unit(pix_u8).unsqueeze(0)).sum()
            )
        img_u8 = dl.unit_to_u8(canvas).permute(1, 2, 0).contiguous()  # [C,C,3] u8
        return img_u8, emit_ll

    # ----------------------------------------------------------- resurrection

    @torch.no_grad()
    def resurrect_texels(
        self,
        usage: torch.Tensor,
        images: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        threshold: Optional[float] = None,
        noise_std: float = 0.01,
        obj_mask: Optional[torch.Tensor] = None,
    ) -> int:
        """F3.3: overwrite the bias grid of under-used texels with a
        training-image 16x16 crop converted to DL-mean params (small noise on
        the other channels) and perturb the E_T row. Returns #resurrected.

        usage: [T_v] normalized posterior usage; images: u8 [B, canvas, canvas, 3].
        obj_mask (optional) [B, canvas, canvas] bool/u8: when given, crops are
        centered on object pixels (diagnosis fix 3 — random crops are ~90%
        background, so resurrection used to reseed dead texels with yet more
        background material).
        """
        cfg = self.cfg
        thr = threshold if threshold is not None else cfg.resurrect_threshold_frac / cfg.T_v
        dead = torch.nonzero(usage.cpu() < thr, as_tuple=False).reshape(-1)
        if dead.numel() == 0:
            return 0
        gen = generator
        B = images.shape[0]
        hi_y = cfg.canvas - ATLAS_RES
        dev = self.atlas.bias_grid.device
        obj_pix: List[Tuple[int, torch.Tensor]] = []
        if obj_mask is not None:
            m = obj_mask.detach().cpu()
            for b in range(B):
                nz = torch.nonzero(m[b] > 0, as_tuple=False)   # [k, 2] (y, x)
                if nz.numel():
                    obj_pix.append((b, nz))
        for t in dead.tolist():
            if obj_pix:
                b, nz = obj_pix[int(torch.randint(0, len(obj_pix), (1,), generator=gen))]
                cy, cx = nz[int(torch.randint(0, nz.shape[0], (1,), generator=gen))].tolist()
                y0 = min(max(cy - ATLAS_RES // 2, 0), hi_y)
                x0 = min(max(cx - ATLAS_RES // 2, 0), hi_y)
            else:
                b = int(torch.randint(0, B, (1,), generator=gen))
                y0 = int(torch.randint(0, hi_y + 1, (1,), generator=gen))
                x0 = int(torch.randint(0, hi_y + 1, (1,), generator=gen))
            crop = images[b, y0 : y0 + ATLAS_RES, x0 : x0 + ATLAS_RES].cpu()
            crop_unit = dl.u8_to_unit(crop).permute(2, 0, 1)     # [3,16,16]
            bias = noise_std * torch.randn(dl.N_CH, ATLAS_RES, ATLAS_RES, generator=gen)
            for j in range(dl.N_COMP):
                bias[10 * j + 1 : 10 * j + 4] = crop_unit
            self.atlas.bias_grid.data[t] = bias.to(dev)
            self.atlas.E_T.data[t] += noise_std * torch.randn(
                cfg.d, generator=gen
            ).to(self.atlas.E_T.device)
        return int(dead.numel())


# ------------------------------------------------------------------ DP access

_DP_MODULE: Optional[Any] = None  # test hook: module-like override


def _dp() -> Any:
    if _DP_MODULE is not None:
        return _DP_MODULE
    from sprig.dp import inside
    return inside


def _call_dp(fn: Any, kwargs: Dict[str, Any]) -> Any:
    """Call a DP function by DESIGN section 5 keyword names, dropping any
    kwargs the target does not accept (e.g. log_PT if the DP mixes the texel
    prior itself)."""
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(**kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return fn(**accepted)
