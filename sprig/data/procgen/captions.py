"""Templated dense captions for procedural scenes.

10 training templates (T1..T10) + 2 held-out eval templates (E1, E2 — never
sampled in training mode). 15% of training captions are partial (attribute or
object dropping) and flagged `partial=True` to teach marginalization.

Template inventory (tiers they apply to):
  T1  attribute                        "a small striped red circle"          (0)
  T2  relation, subject-first          "a X to the left of a Y"              (1)
  T3  attribute + background           "a X on a gray background"            (0)
  T4  enumerative                      "a scene with a X, a Y, and a Z"      (1,2)
  T5  relation, INVERTED mention order "a Y to the right of a X"             (1)
  T6  containment                      "a X inside a blue frame"             (3)
  T7  count + background               "a scene with three shapes on ..."    (1,2)
  T8  background-first relation        "on a gray background, a X above a Y" (1)
  T9  count + enumeration              "three shapes: a X, a Y, and a Z"     (2)
  T10 containment, frame-first         "a blue frame containing a X"         (3)
  E1  eval rephrase                    "the image shows ..."                 (all)
  E2  eval rephrase                    "there is ... in the picture"         (all)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .sampler import Scene
from .vocab import TEXTURE_ADJ

TRAIN_TEMPLATE_IDS: Tuple[str, ...] = (
    "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10",
)
EVAL_TEMPLATE_IDS: Tuple[str, ...] = ("E1", "E2")

_TIER_TEMPLATES: Dict[int, Tuple[str, ...]] = {
    0: ("T1", "T3"),
    1: ("T2", "T5", "T4", "T7", "T8"),
    2: ("T4", "T7", "T9"),
    3: ("T6", "T10"),
}

_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
# relation type -> (forward phrase for A rel B, inverted phrase for B rel A)
_REL = {"left": ("to the left of", "to the right of"), "above": ("above", "below")}

PARTIAL_RATE = 0.15


@dataclass
class Caption:
    text: str
    template_id: str
    partial: bool


def _article(phrase: str) -> str:
    return "an" if phrase and phrase[0] in "aeiou" else "a"


def _obj_phrase(obj: Dict[str, Any], drops: Sequence[str] = ()) -> str:
    words: List[str] = []
    if "size" not in drops:
        words.append(obj["size"])
    adj = TEXTURE_ADJ[obj["texture"]]
    if adj and "texture" not in drops:
        words.append(adj)
    if "color" not in drops:
        words.append(obj["color"])
    words.append(obj["shape"])
    core = " ".join(words)
    return "{} {}".format(_article(core), core)


def _list_phrase(phrases: Sequence[str]) -> str:
    if len(phrases) == 1:
        return phrases[0]
    if len(phrases) == 2:
        return "{} and {}".format(phrases[0], phrases[1])
    return "{}, and {}".format(", ".join(phrases[:-1]), phrases[-1])


def _bg_phrase(scene: Scene) -> str:
    names = scene.background.split("|")
    if len(names) == 2:
        return "a {} and {} background".format(names[0], names[1])
    return "a {} background".format(names[0])


def _relation_parts(
    scene: Scene, drops: Sequence[str]
) -> Tuple[str, str, str, str]:
    rel = scene.relation
    assert rel is not None, "relation template on a scene without a relation"
    fwd, inv = _REL[rel["type"]]
    pa = _obj_phrase(scene.objects[rel["a"]], drops)
    pb = _obj_phrase(scene.objects[rel["b"]], drops)
    return pa, pb, fwd, inv


def render_template(
    scene: Scene,
    template_id: str,
    drops: Sequence[str] = (),
    obj_order: Optional[Sequence[int]] = None,
) -> str:
    """Realize `template_id` for `scene`.

    `drops`: subset of {"size","texture","color","bg"} removed from the text
    (partial captions). `obj_order`: object index order/subset for the
    enumerative templates T4/T9/E1/E2.
    """
    objs = scene.objects
    order = list(obj_order) if obj_order is not None else list(range(len(objs)))
    phrases = [_obj_phrase(objs[i], drops) for i in order]

    if template_id == "T1":
        return phrases[0]
    if template_id == "T2":
        pa, pb, fwd, _ = _relation_parts(scene, drops)
        return "{} {} {}".format(pa, fwd, pb)
    if template_id == "T3":
        if "bg" in drops:
            return phrases[0]
        return "{} on {}".format(phrases[0], _bg_phrase(scene))
    if template_id == "T4":
        return "a scene with {}".format(_list_phrase(phrases))
    if template_id == "T5":
        pa, pb, _, inv = _relation_parts(scene, drops)
        return "{} {} {}".format(pb, inv, pa)
    if template_id == "T6":
        assert scene.frame is not None
        fc = scene.frame["color"]
        return "{} inside {} {} frame".format(phrases[0], _article(fc), fc)
    if template_id == "T7":
        n = _NUM_WORDS[len(objs)]
        noun = "shape" if len(objs) == 1 else "shapes"
        if "bg" in drops:
            return "a scene with {} {}".format(n, noun)
        return "a scene with {} {} on {}".format(n, noun, _bg_phrase(scene))
    if template_id == "T8":
        pa, pb, fwd, _ = _relation_parts(scene, drops)
        if "bg" in drops:
            return "{} {} {}".format(pa, fwd, pb)
        return "on {}, {} {} {}".format(_bg_phrase(scene), pa, fwd, pb)
    if template_id == "T9":
        n = _NUM_WORDS[len(order)]
        noun = "shape" if len(order) == 1 else "shapes"
        return "{} {}: {}".format(n, noun, _list_phrase(phrases))
    if template_id == "T10":
        assert scene.frame is not None
        fc = scene.frame["color"]
        return "{} {} frame containing {}".format(_article(fc), fc, phrases[0])
    if template_id in ("E1", "E2"):
        content = _eval_content(scene, phrases)
        if template_id == "E1":
            return "the image shows {}".format(content)
        verb = "are" if len(objs) > 1 else "is"
        return "there {} {} in the picture".format(verb, content)
    raise ValueError("unknown template id: {}".format(template_id))


def _eval_content(scene: Scene, phrases: Sequence[str]) -> str:
    if scene.tier == 1 and scene.relation is not None:
        pa, pb, fwd, _ = _relation_parts(scene, ())
        return "{} {} {}".format(pa, fwd, pb)
    if scene.tier == 3 and scene.frame is not None:
        fc = scene.frame["color"]
        return "{} inside {} {} frame".format(phrases[0], _article(fc), fc)
    return _list_phrase(phrases)


def _drop_candidates(scene: Scene, template_id: str) -> List[str]:
    cands: List[str] = []
    if template_id != "T7":  # T7 mentions no object attributes
        cands.extend(["size", "color"])
        if any(TEXTURE_ADJ[o["texture"]] for o in scene.objects):
            cands.append("texture")
    if template_id in ("T3", "T7", "T8"):
        cands.append("bg")
    return cands


def sample_caption(
    scene: Scene, rng: np.random.Generator, mode: str = "train"
) -> Caption:
    """Sample a caption for `scene`.

    mode="train": tier-appropriate training templates, 15% partial captions.
    mode="eval": held-out E1/E2 templates only, never partial.
    """
    if mode == "eval":
        tid = EVAL_TEMPLATE_IDS[int(rng.integers(len(EVAL_TEMPLATE_IDS)))]
        return Caption(render_template(scene, tid), tid, False)
    assert mode == "train", "mode must be 'train' or 'eval'"

    pool = _TIER_TEMPLATES[scene.tier]
    tid = pool[int(rng.integers(len(pool)))]

    obj_order: Optional[List[int]] = None
    if tid in ("T4", "T9"):
        obj_order = [int(i) for i in rng.permutation(len(scene.objects))]

    partial = bool(rng.random() < PARTIAL_RATE)
    drops: Tuple[str, ...] = ()
    if partial:
        if tid == "T4" and len(scene.objects) >= 2 and rng.random() < 0.5:
            # object dropping: keep a strict nonempty subset
            keep = 1 + int(rng.integers(len(scene.objects) - 1))
            assert obj_order is not None
            obj_order = obj_order[:keep]
        else:
            cands = _drop_candidates(scene, tid)
            k = 1 if (len(cands) == 1 or rng.random() < 0.5) else 2
            picked = rng.permutation(len(cands))[:k]
            drops = tuple(cands[int(i)] for i in picked)

    text = render_template(scene, tid, drops=drops, obj_order=obj_order)
    return Caption(text, tid, partial)
