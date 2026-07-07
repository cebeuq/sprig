"""GT-free faithfulness checks on generated images.

Pipeline (pure numpy, no scipy):
background = modal border color; foreground = pixels whose max-channel
deviation from the background exceeds `thresh`; objects = 4-connected
components of the foreground; object color = median RGB mapped to the
nearest vocabulary color anchor in CIE Lab; relations compared by centroid.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Vocabulary color anchors (RGB). Single source of truth is
# sprig/data/procgen/vocab.py (COLORS dict); these literals are the fallback.
_DEFAULT_ANCHORS: Dict[str, Tuple[int, int, int]] = {
    "red": (210, 45, 45),
    "green": (55, 170, 60),
    "blue": (50, 90, 215),
    "yellow": (235, 215, 55),
    "orange": (240, 145, 40),
    "purple": (140, 60, 185),
    "cyan": (60, 205, 215),
    "magenta": (220, 70, 175),
}


def _load_anchors() -> Dict[str, Tuple[int, int, int]]:
    try:
        from sprig.data.procgen import vocab as _vocab  # type: ignore

        for name in ("COLOR_ANCHORS", "COLORS", "COLOR_RGB"):
            anchors = getattr(_vocab, name, None)
            if isinstance(anchors, dict) and anchors:
                first = next(iter(anchors.values()))
                if hasattr(first, "__len__") and len(first) == 3:
                    return {str(k): tuple(int(v) for v in rgb) for k, rgb in anchors.items()}
    except Exception:
        pass
    return dict(_DEFAULT_ANCHORS)


COLOR_ANCHORS: Dict[str, Tuple[int, int, int]] = _load_anchors()


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB (0..255, any leading shape [..,3]) -> CIE Lab (D65), numpy only."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = lin @ m.T
    xyz = xyz / np.array([0.95047, 1.0, 1.08883])
    d = 6.0 / 29.0
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4.0 / 29.0)
    lab = np.empty_like(f)
    lab[..., 0] = 116.0 * f[..., 1] - 16.0
    lab[..., 1] = 500.0 * (f[..., 0] - f[..., 1])
    lab[..., 2] = 200.0 * (f[..., 1] - f[..., 2])
    return lab


_ANCHOR_NAMES = list(COLOR_ANCHORS.keys())
_ANCHOR_LAB = rgb_to_lab(np.array([COLOR_ANCHORS[n] for n in _ANCHOR_NAMES], dtype=np.float64))


def nearest_color(rgb) -> str:
    """Nearest vocabulary anchor to an RGB triple, in Lab space."""
    lab = rgb_to_lab(np.asarray(rgb, dtype=np.float64))
    d = np.linalg.norm(_ANCHOR_LAB - lab[None, :], axis=1)
    return _ANCHOR_NAMES[int(np.argmin(d))]


def _modal_border_color(img: np.ndarray) -> np.ndarray:
    border = np.concatenate([img[0], img[-1], img[1:-1, 0], img[1:-1, -1]], axis=0)
    codes = (
        border[:, 0].astype(np.int64) * 65536
        + border[:, 1].astype(np.int64) * 256
        + border[:, 2].astype(np.int64)
    )
    vals, counts = np.unique(codes, return_counts=True)
    code = int(vals[np.argmax(counts)])
    return np.array([code // 65536, (code // 256) % 256, code % 256], dtype=np.float64)


def connected_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """4-connectivity components of a boolean mask via label propagation.

    Returns (labels [H,W] int32 with 0 = background, 1..n components, n).
    """
    h, w = mask.shape
    labels = np.where(mask, np.arange(1, h * w + 1, dtype=np.int64).reshape(h, w), 0)
    while True:
        prev = labels
        up = np.zeros_like(labels)
        up[1:, :] = labels[:-1, :]
        down = np.zeros_like(labels)
        down[:-1, :] = labels[1:, :]
        left = np.zeros_like(labels)
        left[:, 1:] = labels[:, :-1]
        right = np.zeros_like(labels)
        right[:, :-1] = labels[:, 1:]
        stacked = np.stack([labels, up, down, left, right], axis=0)
        stacked = np.where(stacked == 0, np.iinfo(np.int64).max, stacked)
        labels = np.where(mask, stacked.min(axis=0), 0)
        if np.array_equal(labels, prev):
            break
    vals = np.unique(labels)
    vals = vals[vals > 0]
    out = np.zeros_like(labels, dtype=np.int32)
    for i, v in enumerate(vals):
        out[labels == v] = i + 1
    return out, len(vals)


def extract(image: np.ndarray, thresh: int = 40, min_area: int = 12) -> Dict[str, object]:
    """Extract background + object list from a rendered/generated 64x64 image.

    Returns {"background": color_name, "background_rgb": (r,g,b),
             "objects": [{"color", "rgb", "centroid": (x,y), "area",
                          "bbox": (x0,y0,x1,y1)}]} sorted by area descending.
    """
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bg = _modal_border_color(img)
    dev = np.abs(img.astype(np.float64) - bg[None, None, :]).max(axis=-1)
    mask = dev > float(thresh)
    labels, n = connected_components(mask)
    objects: List[Dict[str, object]] = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labels == i)
        if ys.size < min_area:
            continue
        med = np.median(img[ys, xs].astype(np.float64), axis=0)
        objects.append(
            {
                "color": nearest_color(med),
                "rgb": tuple(float(v) for v in med),
                "centroid": (float(xs.mean()), float(ys.mean())),
                "area": int(ys.size),
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            }
        )
    objects.sort(key=lambda o: -int(o["area"]))
    return {
        "background": nearest_color(bg),
        "background_rgb": tuple(float(v) for v in bg),
        "objects": objects,
    }


def _find_by_color(objects: List[Dict[str, object]], color: str) -> Optional[Dict[str, object]]:
    for o in objects:
        if o["color"] == color:
            return o
    return None


def relation_holds(
    extraction: Dict[str, object], color_a: str, color_b: str, relation: str
) -> Optional[bool]:
    """Does 'the color_a object is <relation> the color_b object' hold by centroids?

    relation in {"left of", "right of", "above", "below"}.
    Returns None when either object is missing (unscoreable).
    """
    objs = extraction["objects"]  # type: ignore[index]
    a = _find_by_color(objs, color_a)  # type: ignore[arg-type]
    b = _find_by_color(objs, color_b)  # type: ignore[arg-type]
    if a is None or b is None:
        return None
    ax, ay = a["centroid"]  # type: ignore[misc]
    bx, by = b["centroid"]  # type: ignore[misc]
    if relation == "left of":
        return bool(ax < bx)
    if relation == "right of":
        return bool(ax > bx)
    if relation == "above":
        return bool(ay < by)
    if relation == "below":
        return bool(ay > by)
    raise ValueError("unknown relation: {}".format(relation))
