"""Checkpoint save/resume roundtrip, atomicity, and rolling behavior."""
from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

import train


def _make_model_opt(seed: int):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Linear(8, 4))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(3):  # populate optimizer state
        loss = model(torch.randn(2, 4)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, opt


def test_roundtrip_exact(tmp_path):
    model, opt = _make_model_opt(seed=1)

    torch.manual_seed(123)
    np.random.seed(9)
    random.seed(7)
    torch.rand(5)  # advance all three RNG streams past their seed state
    np.random.rand(5)
    random.random()

    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "step": 42,
        "rng": train.capture_rng(),
        "eta": 0.35,
        "tau": 1.5,
    }
    train.save_checkpoint(tmp_path / "last.pt", payload)

    # the draws we expect to reproduce after restore
    expect_t = torch.rand(3)
    expect_n = np.random.rand(3)
    expect_p = random.random()

    # scramble everything
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    model2, opt2 = _make_model_opt(seed=999)

    ck = train.load_checkpoint(tmp_path / "last.pt")
    assert ck["step"] == 42
    assert ck["eta"] == 0.35
    model2.load_state_dict(ck["model"])
    opt2.load_state_dict(ck["optimizer"])
    train.restore_rng(ck["rng"])

    # next random draws are identical
    assert torch.equal(torch.rand(3), expect_t)
    assert np.array_equal(np.random.rand(3), expect_n)
    assert random.random() == expect_p

    # weights and optimizer state restored exactly
    for a, b in zip(model.parameters(), model2.parameters()):
        assert torch.equal(a, b)
    s1 = opt.state_dict()["state"]
    s2 = opt2.state_dict()["state"]
    assert set(s1.keys()) == set(s2.keys())
    for k in s1:
        for field in ("exp_avg", "exp_avg_sq"):
            assert torch.equal(s1[k][field], s2[k][field])
        assert float(s1[k]["step"]) == float(s2[k]["step"])


def test_atomic_crash_leaves_no_partial_file(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"

    def boom(src, dst):
        raise RuntimeError("simulated crash between tmp write and rename")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        train.save_checkpoint(target, {"v": 2})
    assert not target.exists()  # no partial target file


def test_atomic_crash_preserves_previous_checkpoint(tmp_path, monkeypatch):
    target = tmp_path / "last.pt"
    train.save_checkpoint(target, {"v": 1})

    def boom(src, dst):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(RuntimeError):
        train.save_checkpoint(target, {"v": 2})
    monkeypatch.undo()
    assert train.load_checkpoint(target)["v"] == 1  # old content intact


def test_rolling_last_prev_and_permanent(tmp_path):
    train.save_rolling(tmp_path, {"v": 1})
    assert train.load_checkpoint(tmp_path / "last.pt")["v"] == 1
    assert not (tmp_path / "prev.pt").exists()

    train.save_rolling(tmp_path, {"v": 2})
    assert train.load_checkpoint(tmp_path / "last.pt")["v"] == 2
    assert train.load_checkpoint(tmp_path / "prev.pt")["v"] == 1

    train.save_rolling(tmp_path, {"v": 3}, permanent_step=25000)
    assert train.load_checkpoint(tmp_path / "last.pt")["v"] == 3
    assert train.load_checkpoint(tmp_path / "prev.pt")["v"] == 2
    assert train.load_checkpoint(tmp_path / "step_0025000.pt")["v"] == 3
    assert not list(tmp_path.glob("*.tmp"))  # no stray tmp files
