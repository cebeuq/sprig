"""Sampler invariants: grid alignment, offsets, tiling, margins, tier semantics."""
from __future__ import annotations

import json

import numpy as np
import pytest

from sprig.data.procgen.sampler import (
    Scene,
    iter_leaves,
    sample_scene,
)
from sprig.data.procgen.vocab import (
    CANVAS,
    GRID,
    HOLDOUT_COMBOS,
    MAX_LEAF,
    MIN_MARGIN,
    OFFSET_HI,
    OFFSET_LO,
)

SEED = 20260702
EPS = 1e-6


def _scenes(n: int, seed: int = SEED, tier=None):
    return [sample_scene(seed, i, tier=tier) for i in range(n)]


def _walk(node, fn):
    fn(node)
    if not node.get("leaf"):
        for c in node["children"]:
            _walk(c, fn)


def _check_node(node):
    x0, y0, x1, y1 = node["rect"]
    assert x0 < x1 and y0 < y1
    for v in (x0, y0, x1, y1):
        assert v % GRID == 0, "region corner off the 8-px grid"
    if node.get("leaf"):
        w, h = x1 - x0, y1 - y0
        assert GRID <= w <= MAX_LEAF and GRID <= h <= MAX_LEAF
        assert node["fill"] is not None and len(node["fill"]) == 3
        return
    cut = node["cut"]
    assert cut % GRID == 0, "cut off the 8-px grid"
    lo, hi = node["children"]
    if node["axis"] == "V":
        t = (cut - x0) / (x1 - x0)
        assert lo["rect"] == [x0, y0, cut, y1]
        assert hi["rect"] == [cut, y0, x1, y1]
    else:
        assert node["axis"] == "H"
        t = (cut - y0) / (y1 - y0)
        assert lo["rect"] == [x0, y0, x1, cut]
        assert hi["rect"] == [x0, cut, x1, y1]
    assert OFFSET_LO - EPS <= t <= OFFSET_HI + EPS, "cut offset outside [0.3,0.7]"


def test_grid_alignment_offsets_and_children():
    for scene in _scenes(200):
        assert scene.tree["rect"] == [0, 0, CANVAS, CANVAS]
        _walk(scene.tree, _check_node)


def test_leaves_exactly_tile_canvas():
    for scene in _scenes(100):
        paint = np.zeros((CANVAS, CANVAS), dtype=np.int32)
        for leaf in iter_leaves(scene.tree):
            x0, y0, x1, y1 = leaf["rect"]
            paint[y0:y1, x0:x1] += 1
        assert (paint == 1).all(), "leaf rects must tile the canvas exactly"


def test_objects_inside_cells_with_margin_no_overlap():
    for scene in _scenes(200):
        obj_leaves = [l for l in iter_leaves(scene.tree) if l["obj"] is not None]
        assert sorted(l["obj"] for l in obj_leaves) == list(range(len(scene.objects)))
        for leaf in obj_leaves:
            obj = scene.objects[leaf["obj"]]
            assert obj["cell"] == leaf["rect"]
            cx0, cy0, cx1, cy1 = leaf["rect"]
            bx0, by0, bx1, by1 = obj["bbox"]
            assert bx0 >= cx0 + MIN_MARGIN - EPS
            assert by0 >= cy0 + MIN_MARGIN - EPS
            assert bx1 <= cx1 - MIN_MARGIN + EPS
            assert by1 <= cy1 - MIN_MARGIN + EPS
        # one object per cell => cells disjoint => no overlap/occlusion
        cells = [tuple(l["rect"]) for l in obj_leaves]
        assert len(set(cells)) == len(cells)


def test_tier_semantics():
    n_per = 40
    for i in range(n_per):
        s0 = sample_scene(SEED, i, tier=0)
        assert s0.tier == 0 and len(s0.objects) == 1

        s1 = sample_scene(SEED, i, tier=1)
        assert s1.tier == 1 and len(s1.objects) == 2 and s1.relation is not None
        cut = s1.tree["cut"]
        a, b = s1.objects[s1.relation["a"]], s1.objects[s1.relation["b"]]
        if s1.relation["type"] == "left":
            assert s1.tree["axis"] == "V"
            assert a["bbox"][2] <= cut <= b["bbox"][0]
        else:
            assert s1.relation["type"] == "above" and s1.tree["axis"] == "H"
            assert a["bbox"][3] <= cut <= b["bbox"][1]

        s2 = sample_scene(SEED, i, tier=2)
        assert s2.tier == 2 and 3 <= len(s2.objects) <= 5
        if s2.background == "sky|sand":
            assert s2.tree["axis"] == "H"

        s3 = sample_scene(SEED, i, tier=3)
        assert s3.tier == 3 and len(s3.objects) == 1 and s3.frame is not None
        fx0, fy0, fx1, fy1 = s3.frame["rect"]
        ix0, iy0, ix1, iy1 = s3.frame["inner"]
        assert fx0 < ix0 and fy0 < iy0 and ix1 < fx1 and iy1 < fy1
        bx0, by0, bx1, by1 = s3.objects[0]["bbox"]
        assert ix0 <= bx0 and iy0 <= by0 and bx1 <= ix1 and by1 <= iy1
        # frame leaves fully surround the inner cell with the frame fill
        frame_leaves = [l for l in iter_leaves(s3.tree) if l.get("frame")]
        assert frame_leaves
        for l in frame_leaves:
            assert l["fill"] == s3.frame["rgb"]


def test_tier_mix_draw_and_determinism():
    scenes_a = _scenes(50)
    scenes_b = _scenes(50)
    for a, b in zip(scenes_a, scenes_b):
        assert a.tier == b.tier
        assert json.dumps(a.__dict__, sort_keys=True) == json.dumps(
            b.__dict__, sort_keys=True
        )
    assert len({s.tier for s in _scenes(200)}) == 4  # all tiers occur


def test_scene_is_json_able():
    for scene in _scenes(20):
        blob = json.dumps(
            {"tier": scene.tier, "objects": scene.objects, "tree": scene.tree}
        )
        rec = json.loads(blob)
        assert rec["tree"]["rect"] == [0, 0, CANVAS, CANVAS]


def test_holdout_never_sampled():
    for scene in _scenes(500):
        for obj in scene.objects:
            assert (obj["color"], obj["shape"]) not in HOLDOUT_COMBOS
