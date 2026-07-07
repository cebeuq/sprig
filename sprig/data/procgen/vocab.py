"""Vocabulary and geometry constants for the procedural 2D scene generator.

Everything here is a plain constant so the sampler / renderer / captioner and
the eval agents (color anchor classification) share one source of truth.
"""
from __future__ import annotations

from typing import Dict, Tuple

# --- geometry (must match the model lattice in DESIGN.md §2/§3) -------------
CANVAS: int = 64          # canvas side, px
GRID: int = 8             # model lattice stride, px — ALL cuts land on this grid
MAX_LEAF: int = 16        # model support: leaf regions have both sides <= 16 px
SUPERSAMPLE: int = 4      # render at 256x256, BOX-downsample to 64x64
OFFSET_LO: float = 0.3    # relative cut offset range (inside model support)
OFFSET_HI: float = 0.7
MIN_MARGIN: float = 2.0   # min px margin between a shape bbox and its cell walls

# --- object vocabulary -------------------------------------------------------
# 8 colors: name -> RGB anchor (renderer jitters around these; eval classifies
# back to nearest anchor in Lab space).
COLORS: Dict[str, Tuple[int, int, int]] = {
    "red": (210, 45, 45),
    "green": (55, 170, 60),
    "blue": (50, 90, 215),
    "yellow": (235, 215, 55),
    "orange": (240, 145, 40),
    "purple": (140, 60, 185),
    "cyan": (60, 205, 215),
    "magenta": (220, 70, 175),
}
COLOR_NAMES: Tuple[str, ...] = tuple(COLORS.keys())

SHAPES: Tuple[str, ...] = (
    "circle", "square", "triangle", "rectangle", "diamond", "star", "cross", "ring",
)

# 2 sizes: name -> shape bbox side in px. Objects live in 16x16 leaf cells with
# >= MIN_MARGIN px of clearance: large 11 -> 2.5 px margin, small 6 -> 5 px.
SIZES: Dict[str, float] = {"small": 6.0, "large": 11.0}
SIZE_NAMES: Tuple[str, ...] = tuple(SIZES.keys())

TEXTURES: Tuple[str, ...] = ("solid", "striped", "dotted", "checker")
# caption adjective per texture ("" = no adjective for solid fills)
TEXTURE_ADJ: Dict[str, str] = {
    "solid": "",
    "striped": "striped",
    "dotted": "dotted",
    "checker": "checkered",
}

# 5 backgrounds: name -> RGB anchor. "sky"/"sand" are also used as the fixed
# pair for the tier-2 sky/ground root split.
BACKGROUNDS: Dict[str, Tuple[int, int, int]] = {
    "white": (245, 245, 245),
    "black": (28, 28, 28),
    "gray": (128, 128, 128),
    "sky": (170, 210, 240),
    "sand": (205, 180, 130),
}
BACKGROUND_NAMES: Tuple[str, ...] = tuple(BACKGROUNDS.keys())

# per-channel uniform jitter amplitude applied to anchors at scene-sample time
COLOR_JITTER: int = 12
BG_JITTER: int = 8

# --- compositional holdout ---------------------------------------------------
# These (color, shape) combos NEVER appear in training scenes or captions
# (enforced by rejection-resampling in the sampler; scanned in tests/CI).
HOLDOUT_COMBOS: frozenset = frozenset(
    {("blue", "triangle"), ("red", "ring"), ("green", "star"), ("yellow", "cross")}
)
