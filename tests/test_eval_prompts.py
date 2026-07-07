from __future__ import annotations

from sprig.eval.prompts import (
    COLORS,
    HELDOUT_COMBOS,
    MINIMAL_PAIRS,
    PROMPT_GROUPS,
    PROMPTS,
    SHAPES,
)


def test_exactly_32_prompts():
    assert len(PROMPTS) == 32
    assert len(set(PROMPTS)) == 32  # no duplicates


def test_group_composition():
    sizes = {name: len(g) for name, g in PROMPT_GROUPS.items()}
    assert sizes == {
        "single_object": 8,
        "relations": 8,
        "counts": 3,
        "containment": 1,
        "sizes": 2,
        "backgrounds": 2,
        "heldout_combos": 4,
        "partial": 2,
        "clevr_style": 2,
    }
    flat = [p for g in PROMPT_GROUPS.values() for p in g]
    assert sorted(flat) == sorted(PROMPTS)


def test_heldout_combos_present_as_prompts():
    assert len(HELDOUT_COMBOS) == 4
    assert set(HELDOUT_COMBOS) == {
        ("blue", "triangle"),
        ("red", "ring"),
        ("green", "star"),
        ("yellow", "cross"),
    }
    for color, shape in HELDOUT_COMBOS:
        assert "a {} {}".format(color, shape) in PROMPTS


def test_heldout_combos_do_not_leak_into_other_prompts():
    non_heldout = [p for p in PROMPTS if p not in PROMPT_GROUPS["heldout_combos"]]
    for prompt in non_heldout:
        for color, shape in HELDOUT_COMBOS:
            assert "{} {}".format(color, shape) not in prompt, prompt


def test_minimal_pairs():
    assert len(MINIMAL_PAIRS) == 8
    attrs = [attr for _, _, attr in MINIMAL_PAIRS]
    assert set(attrs) <= {"color", "shape", "relation", "size"}
    assert attrs.count("relation") == 2
    for a, b, _ in MINIMAL_PAIRS:
        assert a != b


def test_vocab_sizes():
    assert len(COLORS) == 8
    assert len(SHAPES) == 8
