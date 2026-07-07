"""Deterministic rasterizer for procedural scenes.

Draws at 256x256 (4x supersample) with numpy + PIL, then BOX-downsamples to
64x64. `render_scene` is a pure function of the Scene, and the Scene is a pure
function of (global_seed, idx) via SeedSequence([global_seed, idx]) — so the
same idx always yields bit-identical pixels (regression-tested by hash).
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from .sampler import DEFAULT_TIER_MIX, Scene, iter_leaves, sample_scene
from .vocab import CANVAS, SUPERSAMPLE

_S = SUPERSAMPLE
_HI = CANVAS * _S
_BOX = getattr(getattr(Image, "Resampling", Image), "BOX")

_texture_cache: Dict[str, np.ndarray] = {}


def _texture_mask(name: str) -> np.ndarray:
    """Boolean [256,256] pattern mask in canvas-aligned hi-res coordinates."""
    if name in _texture_cache:
        return _texture_cache[name]
    yy, xx = np.mgrid[0:_HI, 0:_HI]
    if name == "solid":
        m = np.ones((_HI, _HI), dtype=bool)
    elif name == "striped":  # diagonal stripes, 2 canvas px on / 2 off
        m = ((xx + yy) // (2 * _S)) % 2 == 0
    elif name == "checker":  # 2x2 canvas px checkerboard
        m = ((xx // (2 * _S)) + (yy // (2 * _S))) % 2 == 0
    elif name == "dotted":  # dot grid, period 4 canvas px, dot radius ~1.3 px
        p = 4 * _S
        dx = (xx % p) - p / 2 + 0.5
        dy = (yy % p) - p / 2 + 0.5
        m = dx * dx + dy * dy <= (1.3 * _S) ** 2
    else:
        raise ValueError("unknown texture: {}".format(name))
    _texture_cache[name] = m
    return m


def _shape_mask(shape: str, bbox: Sequence[float]) -> np.ndarray:
    """Boolean [256,256] mask of `shape` drawn inside hi-res bbox."""
    x0, y0, x1, y1 = (v * _S for v in bbox)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    im = Image.new("L", (_HI, _HI), 0)
    d = ImageDraw.Draw(im)
    if shape == "circle":
        d.ellipse([x0, y0, x1, y1], fill=255)
    elif shape in ("square", "rectangle"):
        d.rectangle([x0, y0, x1, y1], fill=255)
    elif shape == "triangle":
        d.polygon([(cx, y0), (x1, y1), (x0, y1)], fill=255)
    elif shape == "diamond":
        d.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=255)
    elif shape == "star":
        pts: List[Tuple[float, float]] = []
        for k in range(10):
            ang = -math.pi / 2 + k * math.pi / 5
            r = 1.0 if k % 2 == 0 else 0.42
            pts.append((cx + math.cos(ang) * r * w / 2, cy + math.sin(ang) * r * h / 2))
        d.polygon(pts, fill=255)
    elif shape == "cross":
        tx, ty = w * 0.34, h * 0.34
        d.rectangle([cx - tx / 2, y0, cx + tx / 2, y1], fill=255)
        d.rectangle([x0, cy - ty / 2, x1, cy + ty / 2], fill=255)
    elif shape == "ring":
        d.ellipse([x0, y0, x1, y1], fill=255)
        inset_x, inset_y = w * 0.275, h * 0.275
        d.ellipse([x0 + inset_x, y0 + inset_y, x1 - inset_x, y1 - inset_y], fill=0)
    else:
        raise ValueError("unknown shape: {}".format(shape))
    return np.asarray(im) > 0


def render_scene(scene: Scene) -> np.ndarray:
    """Render a Scene to a [64,64,3] uint8 array."""
    hi = np.empty((_HI, _HI, 3), dtype=np.uint8)
    for leaf in iter_leaves(scene.tree):
        x0, y0, x1, y1 = leaf["rect"]
        hi[y0 * _S : y1 * _S, x0 * _S : x1 * _S] = np.asarray(
            leaf["fill"], dtype=np.uint8
        )
    for obj in scene.objects:
        mask = _shape_mask(obj["shape"], obj["bbox"])
        tex = _texture_mask(obj["texture"])
        rgb = np.asarray(obj["rgb"], dtype=np.float64)
        main = rgb.astype(np.uint8)
        dark = (rgb * 0.45).astype(np.uint8)  # texture gaps: darker shade
        hi[mask & tex] = main
        hi[mask & ~tex] = dark
    lo = Image.fromarray(hi).resize((CANVAS, CANVAS), resample=_BOX)
    return np.asarray(lo, dtype=np.uint8)


def generate(
    global_seed: int,
    idx: int,
    tier: Optional[int] = None,
    tier_mix: Sequence[float] = DEFAULT_TIER_MIX,
) -> Tuple[Scene, np.ndarray]:
    """Convenience: sample scene `idx` and render it. Deterministic per idx."""
    scene = sample_scene(global_seed, idx, tier=tier, tier_mix=tier_mix)
    return scene, render_scene(scene)
