from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from sprig.eval import color_checks as cc


def test_rgb_to_lab_reference_points():
    lab_white = cc.rgb_to_lab(np.array([255.0, 255.0, 255.0]))
    assert abs(lab_white[0] - 100.0) < 0.5
    assert abs(lab_white[1]) < 0.5 and abs(lab_white[2]) < 0.5
    lab_black = cc.rgb_to_lab(np.array([0.0, 0.0, 0.0]))
    assert abs(lab_black[0]) < 1e-6
    # red has positive a*
    lab_red = cc.rgb_to_lab(np.array([255.0, 0.0, 0.0]))
    assert lab_red[1] > 40


def test_nearest_color_on_anchors_and_jitter():
    rng = np.random.default_rng(0)
    for name, rgb in cc.COLOR_ANCHORS.items():
        assert cc.nearest_color(rgb) == name
        jittered = np.clip(np.array(rgb, dtype=np.float64) + rng.integers(-12, 13, 3), 0, 255)
        assert cc.nearest_color(jittered) == name


def test_connected_components_4conn():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:3, 1:3] = True
    mask[5:7, 5:7] = True
    mask[0, 7] = True  # diagonal from nothing: own component
    labels, n = cc.connected_components(mask)
    assert n == 3
    assert labels[1, 1] == labels[2, 2]
    assert labels[1, 1] != labels[5, 5]
    # diagonal-only touching pixels are separate under 4-connectivity
    diag = np.zeros((4, 4), dtype=bool)
    diag[0, 0] = diag[1, 1] = True
    _, n2 = cc.connected_components(diag)
    assert n2 == 2


def _draw_scene(bg_rgb, obj_a, obj_b, layout):
    """Two-object PIL scene. obj = (color_name, rgb). layout 'lr' or 'ab'."""
    img = Image.new("RGB", (64, 64), tuple(int(v) for v in bg_rgb))
    d = ImageDraw.Draw(img)
    if layout == "lr":
        ca, cb = (16, 32), (48, 32)
    else:
        ca, cb = (32, 16), (32, 48)
    for (name, rgb), (cx, cy), shape in [
        (obj_a, ca, "ellipse"),
        (obj_b, cb, "rect"),
    ]:
        r = 9
        box = (cx - r, cy - r, cx + r, cy + r)
        if shape == "ellipse":
            d.ellipse(box, fill=tuple(int(v) for v in rgb))
        else:
            d.rectangle(box, fill=tuple(int(v) for v in rgb))
    return np.asarray(img, dtype=np.uint8)


def test_calibration_50_synthetic_scenes():
    """>95% color classification and relation extraction on clean renders."""
    rng = np.random.default_rng(7)
    backgrounds = [(0, 0, 0), (100, 100, 100), (25, 25, 55)]
    names = list(cc.COLOR_ANCHORS.keys())
    color_total = color_ok = 0
    rel_total = rel_ok = 0
    for i in range(50):
        na, nb = rng.choice(len(names), size=2, replace=False)
        name_a, name_b = names[na], names[nb]
        jit = lambda rgb: np.clip(  # noqa: E731
            np.array(rgb, dtype=np.float64) + rng.integers(-10, 11, 3), 0, 255
        )
        bg = backgrounds[i % len(backgrounds)]
        layout = "lr" if i % 2 == 0 else "ab"
        img = _draw_scene(
            bg,
            (name_a, jit(cc.COLOR_ANCHORS[name_a])),
            (name_b, jit(cc.COLOR_ANCHORS[name_b])),
            layout,
        )
        ext = cc.extract(img)
        found = {o["color"] for o in ext["objects"]}
        color_total += 2
        color_ok += int(name_a in found) + int(name_b in found)
        relation = "left of" if layout == "lr" else "above"
        r = cc.relation_holds(ext, name_a, name_b, relation)
        rel_total += 1
        rel_ok += int(bool(r))
    assert color_ok / color_total > 0.95, (color_ok, color_total)
    assert rel_ok / rel_total > 0.95, (rel_ok, rel_total)


def test_extract_background_and_fields():
    img = _draw_scene(
        (0, 0, 0),
        ("red", cc.COLOR_ANCHORS["red"]),
        ("cyan", cc.COLOR_ANCHORS["cyan"]),
        "lr",
    )
    ext = cc.extract(img)
    assert len(ext["objects"]) == 2
    obj = ext["objects"][0]
    assert set(obj.keys()) == {"color", "rgb", "centroid", "area", "bbox"}
    x0, y0, x1, y1 = obj["bbox"]
    assert 0 <= x0 < x1 <= 64 and 0 <= y0 < y1 <= 64
    assert obj["area"] > 50


def test_relation_holds_directions():
    img = _draw_scene(
        (0, 0, 0),
        ("red", cc.COLOR_ANCHORS["red"]),
        ("blue", cc.COLOR_ANCHORS["blue"]),
        "lr",
    )
    ext = cc.extract(img)
    assert cc.relation_holds(ext, "red", "blue", "left of") is True
    assert cc.relation_holds(ext, "red", "blue", "right of") is False
    assert cc.relation_holds(ext, "blue", "red", "right of") is True
    assert cc.relation_holds(ext, "green", "red", "left of") is None
