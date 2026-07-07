"""Fixed 32-prompt evaluation bank, minimal pairs, and held-out combos.

The bank composition is pinned by the project plan (Part 3):
8 seen single-object, 8 relations, 3 counts, 1 containment, 2 sizes,
2 backgrounds, 4 held-out combos, 2 partial, 2 CLEVR-style = 32 prompts.

Prompt phrasings follow the training caption templates in
sprig/data/procgen/captions.py (T1/T2/T3/T5/T6/T7/T9 realizations; the
attribute-dropped forms are valid partial captions), and the CLEVR-style
prompts follow sprig/data/clevr/prep.py's synthesized caption grammar.

The four compositional holdout combos {blue triangle, red ring, green star,
yellow cross} never appear in training data; they appear here (only) as the
held-out generalization probes.
"""
from __future__ import annotations

from typing import List, Tuple

# Canonical attribute vocabularies — single source of truth is
# sprig/data/procgen/vocab.py; literals kept as fallback for lean imports.
try:
    from sprig.data.procgen.vocab import COLOR_NAMES as _COLOR_NAMES
    from sprig.data.procgen.vocab import SHAPES as _SHAPES

    COLORS: List[str] = list(_COLOR_NAMES)
    SHAPES: List[str] = list(_SHAPES)
except Exception:  # pragma: no cover
    COLORS = ["red", "green", "blue", "yellow", "orange", "purple", "cyan", "magenta"]
    SHAPES = ["circle", "square", "triangle", "rectangle", "diamond", "star", "cross", "ring"]

SIZES: List[str] = ["small", "large"]
RELATIONS: List[str] = ["to the left of", "to the right of", "above", "below"]

# (color, shape) pairs excluded from all training scenes/captions.
HELDOUT_COMBOS: List[Tuple[str, str]] = [
    ("blue", "triangle"),
    ("red", "ring"),
    ("green", "star"),
    ("yellow", "cross"),
]

_SINGLE_OBJECT = [
    "a red circle",
    "a blue square",
    "a green triangle",
    "a yellow star",
    "a purple diamond",
    "an orange cross",
    "a cyan ring",
    "a magenta rectangle",
]

_RELATIONS = [
    "a red circle to the left of a blue square",
    "a green diamond to the right of a yellow circle",
    "a purple square above an orange circle",
    "a cyan triangle below a magenta square",
    "a yellow rectangle to the left of a purple star",
    "an orange square to the right of a cyan diamond",
    "a magenta circle above a red square",
    "a blue diamond below a green rectangle",
]

_COUNTS = [
    "a scene with two shapes on a white background",
    "three shapes: a red circle, a blue square, and a green diamond",
    "a scene with four shapes",
]

_CONTAINMENT = [
    "a yellow circle inside a purple frame",
]

_SIZES = [
    "a large orange diamond",
    "a small cyan square",
]

_BACKGROUNDS = [
    "a purple circle on a black background",
    "a red square on a gray background",
]

_HELDOUT = ["a {} {}".format(c, s) for (c, s) in HELDOUT_COMBOS]

_PARTIAL = [
    "a circle",
    "a striped square",
]

_CLEVR_STYLE = [
    "a large red rubber cube to the left of a small blue metal sphere",
    "a scene with three objects, including a small green metal cylinder",
]

PROMPTS: List[str] = (
    _SINGLE_OBJECT
    + _RELATIONS
    + _COUNTS
    + _CONTAINMENT
    + _SIZES
    + _BACKGROUNDS
    + _HELDOUT
    + _PARTIAL
    + _CLEVR_STYLE
)

# Named groups for reporting / grid row labels.
PROMPT_GROUPS = {
    "single_object": _SINGLE_OBJECT,
    "relations": _RELATIONS,
    "counts": _COUNTS,
    "containment": _CONTAINMENT,
    "sizes": _SIZES,
    "backgrounds": _BACKGROUNDS,
    "heldout_combos": _HELDOUT,
    "partial": _PARTIAL,
    "clevr_style": _CLEVR_STYLE,
}

# (prompt_a, prompt_b, changed_attribute); attribute in
# {"color", "shape", "relation", "size"}.
MINIMAL_PAIRS: List[Tuple[str, str, str]] = [
    ("a red circle", "a blue circle", "color"),
    ("a green square", "a purple square", "color"),
    ("a yellow diamond", "a cyan diamond", "color"),
    ("an orange ring", "a magenta ring", "color"),
    (
        "a red circle to the left of a blue square",
        "a red circle to the right of a blue square",
        "relation",
    ),
    (
        "a green circle above a yellow square",
        "a green circle below a yellow square",
        "relation",
    ),
    ("a small purple square", "a large purple square", "size"),
    ("a red circle", "a red square", "shape"),
]
