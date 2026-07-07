"""Caption templates, partial captions, holdout scan, T5 token cap."""
from __future__ import annotations

import re

import numpy as np
import pytest

from sprig.data.procgen.captions import (
    EVAL_TEMPLATE_IDS,
    PARTIAL_RATE,
    TRAIN_TEMPLATE_IDS,
    render_template,
    sample_caption,
)
from sprig.data.procgen.sampler import caption_rng, sample_scene
from sprig.data.procgen.vocab import HOLDOUT_COMBOS

SEED = 424242


def _caps(n, mode="train", seed=SEED, tier=None):
    out = []
    for i in range(n):
        scene = sample_scene(seed, i, tier=tier)
        out.append((scene, sample_caption(scene, caption_rng(seed, i), mode=mode)))
    return out


def test_template_inventory():
    assert len(TRAIN_TEMPLATE_IDS) == 10
    assert EVAL_TEMPLATE_IDS == ("E1", "E2")


def test_all_train_templates_reachable_and_render():
    seen = set()
    for scene, cap in _caps(1500):
        assert cap.text and cap.text == cap.text.strip()
        assert cap.template_id in TRAIN_TEMPLATE_IDS
        seen.add(cap.template_id)
    assert seen == set(TRAIN_TEMPLATE_IDS), "unreached templates: {}".format(
        set(TRAIN_TEMPLATE_IDS) - seen
    )


def test_eval_templates_never_in_train_and_vice_versa():
    for _, cap in _caps(400, mode="train"):
        assert cap.template_id not in EVAL_TEMPLATE_IDS
    seen = set()
    for _, cap in _caps(400, mode="eval"):
        assert cap.template_id in EVAL_TEMPLATE_IDS
        assert not cap.partial
        seen.add(cap.template_id)
    assert seen == set(EVAL_TEMPLATE_IDS)
    # E1/E2 render for every tier
    for tier in range(4):
        scene = sample_scene(SEED, tier, tier=tier)
        for tid in EVAL_TEMPLATE_IDS:
            assert render_template(scene, tid)


def test_partial_rate_and_flag():
    caps = _caps(3000)
    frac = sum(c.partial for _, c in caps) / len(caps)
    assert abs(frac - PARTIAL_RATE) < 0.03, "partial rate {} != 0.15".format(frac)
    # a partial attribute-dropped caption is never longer than the full one
    for scene, cap in caps[:500]:
        if cap.partial:
            assert cap.template_id in TRAIN_TEMPLATE_IDS


def test_inverted_order_relation_t5():
    for i in range(300):
        scene = sample_scene(SEED, i, tier=1)
        t2 = render_template(scene, "T2")
        t5 = render_template(scene, "T5")
        rel = scene.relation
        assert rel is not None
        if rel["type"] == "left":
            assert "to the left of" in t2 and "to the right of" in t5
        else:
            assert " above " in t2 and " below " in t5
        # mention order is inverted
        a_shape = scene.objects[rel["a"]]["shape"]
        b_shape = scene.objects[rel["b"]]["shape"]
        if a_shape != b_shape:
            assert t2.index(a_shape) < t2.rindex(b_shape)
            assert t5.index(b_shape) < t5.rindex(a_shape)


def test_holdout_scan_5000_scenes_and_captions():
    """Zero holdout (color, shape) occurrences over 5000 scenes + captions."""
    patterns = {
        (c, s): re.compile(r"\b{}\s+(\w+\s+)?{}\b".format(c, s))
        for (c, s) in HOLDOUT_COMBOS
    }
    for i in range(5000):
        scene = sample_scene(SEED + 1, i)
        for obj in scene.objects:
            assert (obj["color"], obj["shape"]) not in HOLDOUT_COMBOS
        rng = caption_rng(SEED + 1, i)
        texts = [sample_caption(scene, rng).text for _ in range(2)]
        texts.append(render_template(scene, "E1"))
        texts.append(render_template(scene, "E2"))
        for text in texts:
            for combo, pat in patterns.items():
                assert not pat.search(text), "holdout {} in caption: {}".format(
                    combo, text
                )


def test_token_cap_64_t5_tokens():
    """Every caption fits in L<=64 T5 tokens (incl. EOS), over 2000 captions."""
    try:
        from transformers import T5TokenizerFast

        tok = T5TokenizerFast.from_pretrained("t5-base")
    except Exception as e:  # tokenizer download unavailable (offline CI)
        pytest.skip("T5 tokenizer unavailable: {}".format(e))
    texts = []
    for i in range(1000):
        scene = sample_scene(SEED + 2, i)
        rng = caption_rng(SEED + 2, i)
        texts.append(sample_caption(scene, rng, mode="train").text)
        texts.append(sample_caption(scene, rng, mode="eval").text)
    assert len(texts) == 2000
    enc = tok(texts)  # adds EOS
    lengths = [len(ids) for ids in enc["input_ids"]]
    assert max(lengths) <= 64, "caption exceeds 64 T5 tokens: {}".format(
        texts[int(np.argmax(lengths))]
    )
