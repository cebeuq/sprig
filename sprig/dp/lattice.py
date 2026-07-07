"""Finite region lattice for the SPRIG inside DP.

Regions are axis-aligned rectangles whose corners lie on a fixed pixel grid.
Every cell-interval pair is a region; every interior grid line of a region is
a valid cut producing two child regions. See DESIGN.md section 3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

N_CUT_TYPES = 14  # 2 axes x 7 relative-offset buckets {1/8 .. 7/8}
_OFFSET_BUCKETS = [i / 8.0 for i in range(1, 8)]


def _bucket(rel: float) -> int:
    return min(range(7), key=lambda i: abs(_OFFSET_BUCKETS[i] - rel))


def cut_type_id(axis: int, rel: float) -> int:
    """axis: 0 = vertical cut (splits x), 1 = horizontal cut (splits y)."""
    return axis * 7 + _bucket(rel)


@dataclass
class LatticeLevel:
    """Flattened cut tables for all parent regions of one cell-area level."""
    parent_ids: torch.Tensor    # int64 [M]
    cut_type: torch.Tensor      # int64 [M]  global cut-type id
    log_cnt: torch.Tensor       # fp32  [M]  log(#same-type cuts in this region)
    child_lo: torch.Tensor      # int64 [M]  lesser-coordinate child region id
    child_hi: torch.Tensor      # int64 [M]
    cut_axis: torch.Tensor      # int64 [M]  0=v, 1=h
    cut_px: torch.Tensor        # int64 [M]  absolute cut coordinate in px
    parents: torch.Tensor       # int64 [P]  unique parent ids at this level
    parent_index: torch.Tensor  # int64 [M]  index into `parents` for scatter
    # Precomputed sweep metadata (pure functions of the lattice; filled in by
    # Lattice.__post_init__ so the inside DP never branches on device tensors
    # or bool-mask-indexes inside the per-level loop — those force CPU-GPU
    # syncs). Optional so externally-built levels stay constructible;
    # sprig.dp.inside falls back to computing them on the fly if absent.
    leaf_pos: Optional[torch.Tensor] = None      # int64 [Pl]  positions in `parents` that are leaf-eligible
    nonleaf_pos: Optional[torch.Tensor] = None   # int64 [Pn]
    leaf_rids: Optional[torch.Tensor] = None     # int64 [Pl]  = parents[leaf_pos]
    nonleaf_rids: Optional[torch.Tensor] = None  # int64 [Pn]
    leaf_slots: Optional[torch.Tensor] = None    # int64 [Pl]  leaf_index_of_region[leaf_rids]
    flat_type_idx: Optional[torch.Tensor] = None  # int64 [M] pattern_of_region[parent]*14 + cut_type
    has_leaf: bool = False
    has_nonleaf: bool = False
    reorder: Optional[torch.Tensor] = None       # int64 [P]: cat(nonleaf, leaf) -> parent order
    id_lo: int = 0                               # parents == arange(id_lo, id_hi) when the
    id_hi: int = 0                               # lattice has contiguous_ids (see Lattice)


@dataclass
class Lattice:
    canvas_px: int
    grid_px: int
    leaf_max_px: int

    regions: torch.Tensor = field(init=False)         # int64 [N,4] x0,y0,x1,y1 (px)
    region_id: Dict[Tuple[int, int, int, int], int] = field(init=False)
    leaf_mask: torch.Tensor = field(init=False)        # bool [N]
    must_terminate: torch.Tensor = field(init=False)   # bool [N]
    must_expand: torch.Tensor = field(init=False)      # bool [N]
    area_px: torch.Tensor = field(init=False)          # fp32 [N]
    phi_geom: torch.Tensor = field(init=False)         # fp32 [N,64]
    type_present: torch.Tensor = field(init=False)     # bool [N,14]
    levels: List[LatticeLevel] = field(init=False)     # area-ascending, only levels with cuts
    root_id: int = field(init=False)
    leaf_ids: torch.Tensor = field(init=False)         # int64 [NL] leaf-eligible region ids
    leaf_index_of_region: torch.Tensor = field(init=False)  # int64 [N] (-1 if not leaf)
    cuts_of_region: Dict[int, List[Tuple[int, int, int, int, int, float]]] = field(init=False)
    # cuts_of_region[r] = list of (axis, px, child_lo, child_hi, type_id, log_cnt)
    # Precomputed sweep/emission metadata (see __post_init__).
    mt_ids: torch.Tensor = field(init=False)           # int64 [n_mt] must-terminate region ids
    mt_slots: torch.Tensor = field(init=False)         # int64 [n_mt] their leaf slots
    type_patterns: torch.Tensor = field(init=False)    # bool [n_pat, 14] distinct type_present rows
    pattern_of_region: torch.Tensor = field(init=False)  # int64 [N] (-1 if region has no cuts)
    shape_groups: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = field(init=False)
    contiguous_ids: bool = field(init=False)           # id blocks are (mt, level0, level1, ...)

    def __post_init__(self) -> None:
        g, c = self.grid_px, self.canvas_px
        assert c % g == 0
        n_cells = c // g
        lines = [i * g for i in range(n_cells + 1)]

        rects: List[Tuple[int, int, int, int]] = []
        for x0 in range(n_cells):
            for x1 in range(x0 + 1, n_cells + 1):
                for y0 in range(n_cells):
                    for y1 in range(y0 + 1, n_cells + 1):
                        rects.append((lines[x0], lines[y0], lines[x1], lines[y1]))
        rects.sort(key=lambda r: ((r[2] - r[0]) * (r[3] - r[1]), r))
        self.region_id = {r: i for i, r in enumerate(rects)}
        self.regions = torch.tensor(rects, dtype=torch.int64)
        N = len(rects)

        w = self.regions[:, 2] - self.regions[:, 0]
        h = self.regions[:, 3] - self.regions[:, 1]
        self.leaf_mask = (w <= self.leaf_max_px) & (h <= self.leaf_max_px)
        self.must_terminate = (w == g) & (h == g)
        self.must_expand = (w > self.leaf_max_px) | (h > self.leaf_max_px)
        self.area_px = (w * h).to(torch.float32)
        self.root_id = self.region_id[(0, 0, c, c)]

        self.leaf_ids = torch.nonzero(self.leaf_mask, as_tuple=False).squeeze(1)
        self.leaf_index_of_region = torch.full((N,), -1, dtype=torch.int64)
        self.leaf_index_of_region[self.leaf_ids] = torch.arange(len(self.leaf_ids))

        # Fourier geometry features (log-area, log-aspect, center x/y) -> 64 dims.
        la = torch.log(self.area_px / (c * c))
        lasp = torch.log(w.float() / h.float())
        cx = (self.regions[:, 0] + self.regions[:, 2]).float() / (2 * c)
        cy = (self.regions[:, 1] + self.regions[:, 3]).float() / (2 * c)
        base = torch.stack([la / 8.0, lasp / 4.0, cx, cy], dim=1)  # [N,4]
        freqs = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 3.0, 5.0])  # 8 freqs
        ang = base.unsqueeze(2) * freqs.view(1, 1, 8) * math.pi  # [N,4,8]
        self.phi_geom = torch.cat([torch.sin(ang), torch.cos(ang)], dim=2).reshape(N, 64)

        # Enumerate cuts.
        self.cuts_of_region = {}
        self.type_present = torch.zeros(N, N_CUT_TYPES, dtype=torch.bool)
        per_level: Dict[int, List[Tuple[int, int, int, float, int, int, int]]] = {}
        for rid, (x0, y0, x1, y1) in enumerate(rects):
            cuts: List[Tuple[int, int, int, int, int, float]] = []
            type_counts: Dict[int, int] = {}
            cand: List[Tuple[int, int, int, int, int]] = []  # axis, px, lo, hi, type
            for px in range(x0 + g, x1, g):        # vertical cuts
                lo = self.region_id[(x0, y0, px, y1)]
                hi = self.region_id[(px, y0, x1, y1)]
                t = cut_type_id(0, (px - x0) / (x1 - x0))
                cand.append((0, px, lo, hi, t))
                type_counts[t] = type_counts.get(t, 0) + 1
            for py in range(y0 + g, y1, g):        # horizontal cuts
                lo = self.region_id[(x0, y0, x1, py)]
                hi = self.region_id[(x0, py, x1, y1)]
                t = cut_type_id(1, (py - y0) / (y1 - y0))
                cand.append((1, py, lo, hi, t))
                type_counts[t] = type_counts.get(t, 0) + 1
            for axis, px, lo, hi, t in cand:
                lc = math.log(type_counts[t])
                cuts.append((axis, px, lo, hi, t, lc))
                self.type_present[rid, t] = True
            self.cuts_of_region[rid] = cuts
            if cuts and not bool(self.must_terminate[rid]):
                area_cells = int(self.area_px[rid].item()) // (g * g)
                per_level.setdefault(area_cells, [])
                for axis, px, lo, hi, t, lc in cuts:
                    per_level[area_cells].append((rid, t, lo, hi, lc, axis, px))

        self.levels = []
        for area_cells in sorted(per_level.keys()):
            rows = per_level[area_cells]
            parent_ids = torch.tensor([r[0] for r in rows], dtype=torch.int64)
            parents, parent_index = torch.unique(parent_ids, return_inverse=True)
            self.levels.append(LatticeLevel(
                parent_ids=parent_ids,
                cut_type=torch.tensor([r[1] for r in rows], dtype=torch.int64),
                log_cnt=torch.tensor([r[4] for r in rows], dtype=torch.float32),
                child_lo=torch.tensor([r[2] for r in rows], dtype=torch.int64),
                child_hi=torch.tensor([r[3] for r in rows], dtype=torch.int64),
                cut_axis=torch.tensor([r[5] for r in rows], dtype=torch.int64),
                cut_px=torch.tensor([r[6] for r in rows], dtype=torch.int64),
                parents=parents,
                parent_index=parent_index,
            ))

        # ---- precomputed sweep metadata (everything shape-related the DP
        # would otherwise derive from device tensors inside the level loop).
        self.mt_ids = torch.nonzero(self.must_terminate, as_tuple=False).reshape(-1)
        self.mt_slots = self.leaf_index_of_region[self.mt_ids]

        # Distinct cut-type-presence patterns over regions that actually have
        # cuts (every level parent). Lets the DP do ONE masked log-softmax
        # over [B, n_pat, R, 14] per sweep instead of one per level.
        pat_of: Dict[Tuple[bool, ...], int] = {}
        pats: List[torch.Tensor] = []
        self.pattern_of_region = torch.full((N,), -1, dtype=torch.int64)
        for lv in self.levels:
            for rid in lv.parents.tolist():
                key = tuple(self.type_present[rid].tolist())
                pid = pat_of.setdefault(key, len(pats))
                if pid == len(pats):
                    pats.append(self.type_present[rid].clone())
                self.pattern_of_region[rid] = pid
        self.type_patterns = torch.stack(pats, dim=0)          # bool [n_pat, 14]

        for lv in self.levels:
            is_leaf = self.leaf_mask[lv.parents]
            lv.leaf_pos = torch.nonzero(is_leaf, as_tuple=False).reshape(-1)
            lv.nonleaf_pos = torch.nonzero(~is_leaf, as_tuple=False).reshape(-1)
            lv.leaf_rids = lv.parents[lv.leaf_pos]
            lv.nonleaf_rids = lv.parents[lv.nonleaf_pos]
            lv.leaf_slots = self.leaf_index_of_region[lv.leaf_rids]
            lv.flat_type_idx = (
                self.pattern_of_region[lv.parent_ids] * N_CUT_TYPES + lv.cut_type
            )
            lv.has_leaf = bool(lv.leaf_pos.numel() > 0)
            lv.has_nonleaf = bool(lv.nonleaf_pos.numel() > 0)
            lv.reorder = torch.argsort(torch.cat([lv.nonleaf_pos, lv.leaf_pos]))

        # Region ids are assigned area-ascending (rects sorted by pixel area),
        # so the must-terminate block and each level's parent block are
        # contiguous, consecutive id ranges covering all N regions. The DP's
        # fast sweep relies on this to assemble beta with one cat instead of
        # per-level index_put on the [B, N, S] table. Verified here; any
        # violation falls back to the reference sweep.
        self.contiguous_ids = bool(
            self.mt_ids.equal(torch.arange(self.mt_ids.numel()))
        )
        off = int(self.mt_ids.numel())
        for lv in self.levels:
            P = int(lv.parents.numel())
            if not bool(lv.parents.equal(torch.arange(off, off + P))):
                self.contiguous_ids = False
                break
            lv.id_lo, lv.id_hi = off, off + P
            off += P
        if off != N:
            self.contiguous_ids = False

        # Leaf shape groups with flattened pixel-gather indices, cached for
        # emission scoring (atlas.score_leaves): (h, w, slots [n_g],
        # pix [n_g*h*w] into a [canvas*canvas]-flattened image).
        self.shape_groups = []
        for (h, w), slots in self.leaf_shape_groups().items():
            rects = self.regions[self.leaf_ids[slots]]
            yy = rects[:, 1].view(-1, 1, 1) + torch.arange(h).view(1, -1, 1)
            xx = rects[:, 0].view(-1, 1, 1) + torch.arange(w).view(1, 1, -1)
            pix = (yy * c + xx).reshape(-1)
            self.shape_groups.append((h, w, slots, pix))

    @property
    def n_regions(self) -> int:
        return int(self.regions.shape[0])

    @property
    def n_leaf_regions(self) -> int:
        return int(self.leaf_ids.shape[0])

    def leaf_shape_groups(self) -> Dict[Tuple[int, int], torch.Tensor]:
        """Map (h, w) -> leaf-slot indices (into the leaf_ids ordering)."""
        groups: Dict[Tuple[int, int], List[int]] = {}
        for slot, rid in enumerate(self.leaf_ids.tolist()):
            x0, y0, x1, y1 = self.regions[rid].tolist()
            groups.setdefault((y1 - y0, x1 - x0), []).append(slot)
        return {k: torch.tensor(v, dtype=torch.int64) for k, v in groups.items()}

    def to(self, device: torch.device) -> "Lattice":
        device = torch.device(device)
        if getattr(self, "_device", None) == device:
            return self
        for name in ("regions", "leaf_mask", "must_terminate", "must_expand",
                     "area_px", "phi_geom", "type_present", "leaf_ids",
                     "leaf_index_of_region", "mt_ids", "mt_slots",
                     "type_patterns", "pattern_of_region"):
            setattr(self, name, getattr(self, name).to(device))
        for lv in self.levels:
            for name in ("parent_ids", "cut_type", "log_cnt", "child_lo", "child_hi",
                         "cut_axis", "cut_px", "parents", "parent_index",
                         "leaf_pos", "nonleaf_pos", "leaf_rids", "nonleaf_rids",
                         "leaf_slots", "flat_type_idx", "reorder"):
                t = getattr(lv, name)
                if t is not None:
                    setattr(lv, name, t.to(device))
        self.shape_groups = [
            (h, w, slots.to(device), pix.to(device))
            for h, w, slots, pix in self.shape_groups
        ]
        self._device = device
        return self


_CACHE: Dict[Tuple[int, int, int], Lattice] = {}


def get_lattice(canvas_px: int, grid_px: int, leaf_max_px: int) -> Lattice:
    key = (canvas_px, grid_px, leaf_max_px)
    if key not in _CACHE:
        _CACHE[key] = Lattice(canvas_px, grid_px, leaf_max_px)
    return _CACHE[key]
