"""gen_data.py split bookkeeping: sizes, disjoint seed ranges, probe rules."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import train

REPO = Path(train.__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "gen_data", str(REPO / "scripts" / "gen_data.py"))
gen_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen_data)


def test_split_sizes():
    specs = gen_data.split_specs()
    assert specs["train"]["n"] == 2_000_000
    assert specs["val"]["n"] == 20_000
    assert specs["test"]["n"] == 20_000
    assert specs["parse_eval"]["n"] == 2_000
    assert specs["val_fast"]["n"] == 512
    assert specs["probe"]["n"] == 100_000
    for name, spec in specs.items():
        assert spec["seed_end"] - spec["seed_start"] == spec["n"], name


def test_seed_ranges_disjoint():
    specs = gen_data.split_specs()
    ranges = sorted((s["seed_start"], s["seed_end"], name)
                    for name, s in specs.items())
    for (a0, a1, an), (b0, b1, bn) in zip(ranges, ranges[1:]):
        assert a1 <= b0, "overlap between %s and %s" % (an, bn)


def test_parse_eval_tier_balanced_and_probe_holdout():
    specs = gen_data.split_specs()
    assert specs["parse_eval"]["tier_weights"] == [0.25, 0.25, 0.25, 0.25]
    # probe: single-object (tier 0 only) and INCLUDES holdout combos
    assert specs["probe"]["tier_weights"] == [1.0, 0.0, 0.0, 0.0]
    assert specs["probe"]["include_holdout"] is True
    # all proc2d splits exclude holdout combos
    for name in ("train", "val", "test", "parse_eval", "val_fast"):
        assert specs[name]["include_holdout"] is False


def test_scale_keeps_ranges_disjoint_and_nonempty():
    specs = gen_data.split_specs(scale=0.001)
    for name, s in specs.items():
        assert s["n"] >= 1, name
    ranges = sorted((s["seed_start"], s["seed_end"], n)
                    for n, s in specs.items())
    for (a0, a1, an), (b0, b1, bn) in zip(ranges, ranges[1:]):
        assert a1 <= b0, "overlap between %s and %s" % (an, bn)


def test_holdout_fallback_list():
    assert set(gen_data.FALLBACK_HOLDOUT) == {
        ("blue", "triangle"), ("red", "ring"),
        ("green", "star"), ("yellow", "cross")}
