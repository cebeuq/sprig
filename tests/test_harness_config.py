"""Config load + schema validation for all three harness YAMLs.

Validation is schema-only: data paths intentionally do not need to exist.
"""
from __future__ import annotations

import copy
from pathlib import Path

import pytest

import train

REPO = Path(train.__file__).resolve().parent
CONFIG_NAMES = ["main64.yaml", "smoke.yaml", "clevr_ft.yaml"]


def _load(name):
    return train.load_config(str(REPO / "configs" / name))


@pytest.mark.parametrize("name", CONFIG_NAMES)
def test_load_and_validate(name):
    cfg = _load(name)
    train.validate_config(cfg)  # must not raise


def test_main64_matches_design():
    cfg = _load("main64.yaml")
    m = cfg["model"]
    assert (m["S"], m["R"], m["T_v"], m["d"]) == (1024, 64, 256, 384)
    assert (m["canvas"], m["grid"]) == (64, 8)
    t = cfg["train"]
    assert t["batch_size"] == 256
    assert t["total_steps"] == 80000  # shortened from DESIGN 200k to fit measured 2.6s/step budget
    assert t["betas"] == [0.9, 0.95]
    assert abs(t["lr_tables"] - 3e-4) < 1e-12
    assert abs(t["lr_networks"] - 1e-4) < 1e-12
    assert t["warmup_steps"] == 2000
    assert abs(t["ema"] - 0.9999) < 1e-12
    assert t["grad_clip"] == 1.0
    assert (t["tau_start"], t["tau_end"], t["tau_steps"]) == (2.0, 1.0, 50000)
    assert t["eta_update_every"] == 2000
    assert t["eta_band"] == [0.5, 3.0]
    assert t["eta_max"] == 1.5
    assert t["eta_final_anneal_steps"] == 20000
    sched = t["tier_schedule"]
    assert len(sched) == 3
    assert sched[0] == {"until": 20000, "weights": [0.55, 0.35, 0.10, 0.00]}
    assert sched[1] == {"until": 60000, "weights": [0.25, 0.35, 0.30, 0.10]}
    assert sched[2]["until"] is None
    assert sched[2]["weights"] == [0.10, 0.30, 0.40, 0.20]
    for ent in sched:
        assert abs(sum(ent["weights"]) - 1.0) < 1e-9
    assert cfg["checkpoint"]["every_steps"] == 25000
    assert cfg["checkpoint"]["every_minutes"] == 30
    assert "proc2d" in cfg["data"]["train_dir"]
    e = cfg["eval"]
    assert (e["scalars_every"], e["val_fast_every"]) == (500, 2000)
    assert (e["parse_every"], e["full_every"]) == (10000, 25000)


def test_smoke_matches_design():
    cfg = _load("smoke.yaml")
    m = cfg["model"]
    assert (m["S"], m["R"], m["T_v"], m["d"]) == (128, 16, 32, 128)
    assert cfg["train"]["batch_size"] == 32
    assert cfg["train"]["total_steps"] == 200
    assert cfg["eval"]["val_fast_n"] == 64
    assert cfg["device"] == "auto"  # CPU on Mac, GPU on klaus-1


def test_clevr_ft_matches_design():
    cfg = _load("clevr_ft.yaml")
    t = cfg["train"]
    assert isinstance(t["init_from"], str) and t["init_from"]
    assert abs(t["lr_tables"] - 0.3 * 3e-4) < 1e-12
    assert abs(t["lr_networks"] - 0.3 * 1e-4) < 1e-12
    assert t["total_steps"] == 40000
    d = cfg["data"]
    assert abs(d["replay_frac"] - 0.20) < 1e-12
    assert "proc2d" in d["replay_dir"]
    assert "clevr" in d["train_dir"]
    assert "tier_schedule" not in t  # CLEVR is untiered


def test_validation_missing_key():
    cfg = copy.deepcopy(_load("main64.yaml"))
    del cfg["train"]["batch_size"]
    with pytest.raises(ValueError, match="train.batch_size"):
        train.validate_config(cfg)


def test_validation_bad_tier_weights():
    cfg = copy.deepcopy(_load("main64.yaml"))
    cfg["train"]["tier_schedule"][0]["weights"] = [0.5, 0.5, 0.5, 0.5]
    with pytest.raises(ValueError, match="tier_schedule"):
        train.validate_config(cfg)


def test_validation_tier_schedule_last_until_must_be_null():
    cfg = copy.deepcopy(_load("main64.yaml"))
    cfg["train"]["tier_schedule"][-1]["until"] = 999999
    with pytest.raises(ValueError, match="until"):
        train.validate_config(cfg)


def test_validation_replay_needs_dir():
    cfg = copy.deepcopy(_load("clevr_ft.yaml"))
    del cfg["data"]["replay_dir"]
    with pytest.raises(ValueError, match="replay_dir"):
        train.validate_config(cfg)


def test_validation_bad_types():
    cfg = copy.deepcopy(_load("main64.yaml"))
    cfg["train"]["ema"] = 1.5
    with pytest.raises(ValueError, match="ema"):
        train.validate_config(cfg)


def test_overrides():
    cfg = copy.deepcopy(_load("smoke.yaml"))
    train.apply_overrides(cfg, ["train.total_steps=7", "data.num_workers=0"])
    assert cfg["train"]["total_steps"] == 7
    assert cfg["data"]["num_workers"] == 0
    train.validate_config(cfg)
