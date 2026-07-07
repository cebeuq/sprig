"""Parse metrics vs ground-truth region trees (identifiability-aware).

Node schema (duck-typed; both are accepted everywhere a tree is expected):
- GT JSON node (meta.jsonl "tree", written by sprig/data/procgen/sampler.py):
  internal {"rect": [x0,y0,x1,y1] px (x1/y1 exclusive), "axis": "V"|"H",
  "cut": px, "children": [lo, hi]}; leaf {"rect", "leaf": true,
  "obj": int|null, "fill": ...} — a leaf is an object-role leaf iff "obj"
  is not null (an inline "object" dict or "role" == "object" also works).
  "axis"/"cut" fields are ignored: cut geometry is derived from the children
  rects, which makes the metrics robust to axis naming conventions.
- Model ParseNode (sprig/model/sprig.py): attributes rect, children (list or
  None), axis, cut_px, symbol, texel.

`parse` arguments may be a single root node or the list returned by
`model.map_parse` (all nodes are collected either way).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Rect = Tuple[int, int, int, int]
CutSeg = Tuple[str, float, float, float]  # (axis "V"|"H", position, ext_lo, ext_hi)


# ---------------------------------------------------------------- accessors

def _rect(node) -> Rect:
    r = node["rect"] if isinstance(node, dict) else node.rect
    x0, y0, x1, y1 = (int(v) for v in r)
    return (x0, y0, x1, y1)


def _children(node) -> List:
    if isinstance(node, dict):
        ch = node.get("children")
    else:
        ch = getattr(node, "children", None)
    return list(ch) if ch else []


def _object(node) -> Optional[dict]:
    """Object payload of an object-role leaf, else None.

    Accepts the sampler schema (leaf {"obj": <int index>|null}), an inline
    {"object": {...}} dict, or a "role": "object" marker."""
    if isinstance(node, dict):
        if node.get("object") is not None:
            return node["object"]
        if node.get("obj") is not None:
            return {"index": node["obj"]}
        if node.get("role") == "object":
            return {}
        return None
    return getattr(node, "object", None)


def collect_nodes(tree_or_nodes) -> List:
    """Flatten a root node / list of nodes into a deduped list of all nodes."""
    roots = tree_or_nodes if isinstance(tree_or_nodes, (list, tuple)) else [tree_or_nodes]
    out: List = []
    seen = set()
    stack = list(roots)
    while stack:
        n = stack.pop()
        if id(n) in seen:
            continue
        seen.add(id(n))
        out.append(n)
        stack.extend(_children(n))
    return out


def leaves(tree_or_nodes) -> List:
    return [n for n in collect_nodes(tree_or_nodes) if not _children(n)]


def cut_segments(tree_or_nodes) -> List[CutSeg]:
    """All cut segments, derived from each internal node's children rects."""
    segs: List[CutSeg] = []
    for n in collect_nodes(tree_or_nodes):
        ch = _children(n)
        if len(ch) != 2:
            continue
        (ax0, ay0, ax1, ay1), (bx0, by0, bx1, by1) = _rect(ch[0]), _rect(ch[1])
        if ax1 == bx0 and ay0 == by0 and ay1 == by1:  # vertical line at x=ax1
            segs.append(("V", float(ax1), float(ay0), float(ay1)))
        elif bx1 == ax0 and ay0 == by0 and ay1 == by1:
            segs.append(("V", float(bx1), float(ay0), float(ay1)))
        elif ay1 == by0 and ax0 == bx0 and ax1 == bx1:  # horizontal line at y=ay1
            segs.append(("H", float(ay1), float(ax0), float(ax1)))
        elif by1 == ay0 and ax0 == bx0 and ax1 == bx1:
            segs.append(("H", float(by1), float(ax0), float(ax1)))
    return segs


# ------------------------------------------------------------------ metrics

def _iou(a: Rect, b: Rect) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def object_cell_recall(parse, gt, iou_thresh: float = 0.8) -> float:
    """Fraction of GT object-role leaves whose rect is matched (IoU >= thresh)
    by ANY node rect of the parse (internal or leaf). Vacuously 1.0 when the
    GT tree has no object leaves."""
    gt_obj_rects = [_rect(n) for n in leaves(gt) if _object(n) is not None]
    if not gt_obj_rects:
        return 1.0
    parse_rects = [_rect(n) for n in collect_nodes(parse)]
    hit = 0
    for gr in gt_obj_rects:
        if any(_iou(gr, pr) >= iou_thresh for pr in parse_rects):
            hit += 1
    return hit / float(len(gt_obj_rects))


def _cut_contrast(image: np.ndarray, seg: CutSeg, band: int = 2) -> float:
    """Max-channel abs difference of mean colors on the two sides of a cut."""
    img = np.asarray(image, dtype=np.float64)
    axis, pos, lo, hi = seg
    p, a, b = int(round(pos)), int(round(lo)), int(round(hi))
    if axis == "V":
        left = img[a:b, max(0, p - band):p, :]
        right = img[a:b, p:min(img.shape[1], p + band), :]
    else:
        left = img[max(0, p - band):p, a:b, :]
        right = img[p:min(img.shape[0], p + band), a:b, :]
    if left.size == 0 or right.size == 0:
        return 0.0
    return float(
        np.abs(left.reshape(-1, 3).mean(axis=0) - right.reshape(-1, 3).mean(axis=0)).max()
    )


def _visible(segs: Sequence[CutSeg], image: np.ndarray, contrast_thresh: float) -> List[CutSeg]:
    return [s for s in segs if _cut_contrast(image, s) > contrast_thresh]


def _seg_match(a: CutSeg, b: CutSeg, tol_px: float) -> bool:
    if a[0] != b[0] or abs(a[1] - b[1]) > tol_px:
        return False
    overlap = min(a[3], b[3]) - max(a[2], b[2])
    shorter = min(a[3] - a[2], b[3] - b[2])
    return shorter > 0 and overlap >= 0.5 * shorter


def visible_cut_f1(
    parse,
    gt,
    image: np.ndarray,
    contrast_thresh: float = 20.0,
    tol_px: float = 1.5,
) -> float:
    """Boundary F1 over VISIBLE cut segments.

    Both GT and parse cut segments are filtered by actual cross-cut mean-color
    contrast in `image` (> contrast_thresh, max over RGB channels); visible
    parse segments are greedily one-to-one matched to visible GT segments
    (same axis, position within tol_px, >=50% extent overlap).
    """
    gt_segs = _visible(cut_segments(gt), image, contrast_thresh)
    pr_segs = _visible(cut_segments(parse), image, contrast_thresh)
    if not gt_segs and not pr_segs:
        return 1.0
    if not gt_segs or not pr_segs:
        return 0.0
    used = [False] * len(gt_segs)
    tp = 0
    for ps in pr_segs:
        cands = [
            (abs(ps[1] - gs[1]), j)
            for j, gs in enumerate(gt_segs)
            if not used[j] and _seg_match(ps, gs, tol_px)
        ]
        if cands:
            used[min(cands)[1]] = True
            tp += 1
    prec = tp / float(len(pr_segs))
    rec = tp / float(len(gt_segs))
    return 0.0 if tp == 0 else 2 * prec * rec / (prec + rec)


def leaf_assignment_map(tree_or_nodes, canvas: int = 64) -> np.ndarray:
    """[canvas,canvas] int32 map: pixel -> index of the leaf covering it."""
    out = np.full((canvas, canvas), -1, dtype=np.int32)
    for i, leaf in enumerate(leaves(tree_or_nodes)):
        x0, y0, x1, y1 = _rect(leaf)
        out[y0:y1, x0:x1] = i
    return out


def adjusted_rand_index(a: np.ndarray, b: np.ndarray) -> float:
    """ARI between two integer label maps of identical shape (numpy only)."""
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    na, nb = ai.max() + 1, bi.max() + 1
    cont = np.bincount(ai * nb + bi, minlength=na * nb).reshape(na, nb).astype(np.float64)
    n = cont.sum()

    def _comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2.0

    sum_ij = _comb2(cont).sum()
    sum_a = _comb2(cont.sum(axis=1)).sum()
    sum_b = _comb2(cont.sum(axis=0)).sum()
    total = _comb2(np.array(n))
    expected = sum_a * sum_b / total if total > 0 else 0.0
    max_index = 0.5 * (sum_a + sum_b)
    denom = max_index - expected
    if denom == 0:
        return 1.0 if sum_ij == max_index else 0.0
    return float((sum_ij - expected) / denom)


def leaf_ari(parse, gt, canvas: int = 64) -> float:
    """Adjusted Rand index between parse and GT pixel->leaf assignment maps."""
    return adjusted_rand_index(
        leaf_assignment_map(parse, canvas), leaf_assignment_map(gt, canvas)
    )
