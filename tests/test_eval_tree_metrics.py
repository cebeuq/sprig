from __future__ import annotations

import numpy as np

from sprig.eval import tree_metrics as tm


def leaf(rect, obj=None):
    node = {"rect": list(rect)}
    if obj is not None:
        node["object"] = obj
    return node


def split(rect, children):
    return {"rect": list(rect), "children": children}


def make_gt_tree():
    """64x64: V-cut at 32; left side H-cut at 24; three leaves, two objects."""
    l_top = leaf((0, 0, 32, 24), obj={"shape": "circle", "color": "red"})
    l_bot = leaf((0, 24, 32, 64), obj={"shape": "square", "color": "blue"})
    right = leaf((32, 0, 64, 64))  # background cell
    left = split((0, 0, 32, 64), [l_top, l_bot])
    return split((0, 0, 64, 64), [left, right])


def render_leaves(tree, color_by_rect=None):
    """Fill each leaf rect with a flat color (distinct by default)."""
    palette = [(200, 40, 40), (40, 60, 200), (230, 230, 230), (40, 200, 90)]
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    ordered = sorted(tm.leaves(tree), key=lambda lf: tuple(lf["rect"]))
    for i, lf in enumerate(ordered):
        x0, y0, x1, y1 = lf["rect"]
        if color_by_rect is not None:
            img[y0:y1, x0:x1] = color_by_rect[(x0, y0, x1, y1)]
        else:
            img[y0:y1, x0:x1] = palette[i % len(palette)]
    return img


def test_self_test_gt_vs_gt_perfect():
    gt = make_gt_tree()
    img = render_leaves(gt)
    assert tm.object_cell_recall(gt, gt) == 1.0
    assert tm.visible_cut_f1(gt, gt, img) == 1.0
    assert tm.leaf_ari(gt, gt) == 1.0


def test_object_cell_recall_partial_and_internal_nodes():
    gt = make_gt_tree()
    # parse missing the left H-cut: only 2 leaves
    parse = split((0, 0, 64, 64), [leaf((0, 0, 32, 64)), leaf((32, 0, 64, 64))])
    # neither GT object leaf is matched at IoU 0.8 by any parse rect
    assert tm.object_cell_recall(parse, gt) == 0.0
    # but matching against ANY node counts internal nodes too: the GT left
    # internal rect (0,0,32,64) exists in the parse as a leaf; make a GT where
    # the object leaf IS that rect
    gt2 = split(
        (0, 0, 64, 64),
        [leaf((0, 0, 32, 64), obj={"shape": "circle", "color": "red"}), leaf((32, 0, 64, 64))],
    )
    assert tm.object_cell_recall(parse, gt2) == 1.0
    # near-miss rect below IoU threshold
    parse_shift = split(
        (0, 0, 64, 64), [leaf((0, 0, 16, 64)), leaf((16, 0, 64, 64))]
    )
    assert tm.object_cell_recall(parse_shift, gt2) == 0.0


def test_visible_cut_filtering_is_symmetric():
    gt = make_gt_tree()
    # render the two LEFT leaves with the SAME color -> the H-cut at y=24 is
    # invisible; a parse without that cut must still get F1 = 1.0
    img = render_leaves(
        gt,
        color_by_rect={
            (0, 0, 32, 24): (200, 40, 40),
            (0, 24, 32, 64): (200, 40, 40),
            (32, 0, 64, 64): (30, 30, 30),
        },
    )
    parse = split((0, 0, 64, 64), [leaf((0, 0, 32, 64)), leaf((32, 0, 64, 64))])
    assert tm.visible_cut_f1(parse, gt, img) == 1.0
    # gt vs gt also perfect (both sides filter the invisible cut)
    assert tm.visible_cut_f1(gt, gt, img) == 1.0


def test_visible_cut_position_tolerance():
    gt = split((0, 0, 64, 64), [leaf((0, 0, 32, 64)), leaf((32, 0, 64, 64))])
    img = render_leaves(
        gt,
        color_by_rect={(0, 0, 32, 64): (255, 255, 255), (32, 0, 64, 64): (0, 0, 0)},
    )
    near = split((0, 0, 64, 64), [leaf((0, 0, 33, 64)), leaf((33, 0, 64, 64))])
    far = split((0, 0, 64, 64), [leaf((0, 0, 40, 64)), leaf((40, 0, 64, 64))])
    assert tm.visible_cut_f1(near, gt, img, tol_px=1.5) == 1.0
    assert tm.visible_cut_f1(far, gt, img, tol_px=1.5) == 0.0


def test_no_cuts_single_leaf():
    gt = leaf((0, 0, 64, 64), obj={"shape": "circle", "color": "red"})
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    assert tm.visible_cut_f1(gt, gt, img) == 1.0
    assert tm.leaf_ari(gt, gt) == 1.0


def test_ari_identical_and_random():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 8, size=(64, 64))
    assert tm.adjusted_rand_index(a, a) == 1.0
    b = rng.integers(0, 8, size=(64, 64))
    assert abs(tm.adjusted_rand_index(a, b)) < 0.05
    # permuted labels are the same partition
    assert tm.adjusted_rand_index(a, (a + 3) % 8) == 1.0


def test_leaf_ari_detects_disagreement():
    gt = make_gt_tree()
    parse_same = make_gt_tree()
    assert tm.leaf_ari(parse_same, gt) == 1.0
    parse_other = split(
        (0, 0, 64, 64), [leaf((0, 0, 64, 32)), leaf((0, 32, 64, 64))]
    )
    assert tm.leaf_ari(parse_other, gt) < 0.5


def test_sampler_gt_schema():
    """Exact procgen sampler.py node schema: leaf {"leaf", "obj", "fill"}."""
    gt = {
        "rect": [0, 0, 64, 64],
        "axis": "V",
        "cut": 32,
        "children": [
            {"rect": [0, 0, 32, 64], "leaf": True, "obj": 0, "fill": None},
            {"rect": [32, 0, 64, 64], "leaf": True, "obj": None, "fill": None},
        ],
    }
    assert tm.object_cell_recall(gt, gt) == 1.0
    # "obj": 0 must count as an object leaf (falsy int is still an object)
    bad_parse = split((0, 0, 64, 64), [leaf((0, 0, 64, 32)), leaf((0, 32, 64, 64))])
    assert tm.object_cell_recall(bad_parse, gt) == 0.0
    img = render_leaves(gt)
    assert tm.visible_cut_f1(gt, gt, img) == 1.0
    assert tm.leaf_ari(gt, gt) == 1.0


def test_parse_node_duck_typing():
    """Attribute-style nodes (like model ParseNode) work everywhere."""

    class Node:
        def __init__(self, rect, children=None):
            self.rect = rect
            self.children = children
            self.axis = None
            self.cut_px = None

    gt = make_gt_tree()
    root = Node(
        (0, 0, 64, 64),
        [
            Node((0, 0, 32, 64), [Node((0, 0, 32, 24)), Node((0, 24, 32, 64))]),
            Node((32, 0, 64, 64)),
        ],
    )
    img = render_leaves(gt)
    assert tm.object_cell_recall(root, gt) == 1.0
    assert tm.visible_cut_f1(root, gt, img) == 1.0
    assert tm.leaf_ari(root, gt) == 1.0
    # list-of-nodes input (map_parse returns a list)
    assert tm.object_cell_recall([root], gt) == 1.0
