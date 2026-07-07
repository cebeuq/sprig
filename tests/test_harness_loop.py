"""train_loop behavior with a dummy model/dataset: STOP file, checkpoint +
exact resume, scalars.jsonl output, decreasing loss, param-group split."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import torch
from torch import nn

import train

REPO = Path(train.__file__).resolve().parent


class DummyModel(nn.Module):
    """Minimal model exposing .loss(batch) -> (loss, metrics) per DESIGN."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(8))       # "networks" group
        self.E_T = nn.Parameter(torch.zeros(4, 8))  # "tables" group

    def loss(self, batch):
        x = batch["image"].float().mean() / 255.0
        loss = ((self.w - 1.0) ** 2).mean() + 0.0 * x + 0.0 * self.E_T.sum()
        return loss, {"bpd": float(loss.detach()) / math.log(2.0)}


class DummyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, i):
        return {
            "image": torch.zeros(64, 64, 3, dtype=torch.uint8),
            "emb": torch.zeros(4, 768, dtype=torch.float16),
            "emb_len": torch.tensor(4, dtype=torch.int32),
            "tier": torch.tensor(0, dtype=torch.int8),
            "idx": torch.tensor(i, dtype=torch.int64),
        }


def _cfg():
    cfg = train.load_config(str(REPO / "configs" / "smoke.yaml"))
    cfg = copy.deepcopy(cfg)
    big = 10 ** 9
    cfg["train"].update({
        "total_steps": 5, "batch_size": 2, "warmup_steps": 1,
        "eta_update_every": big, "tau_steps": 2,
        "lr_networks": 0.05,  # so the dummy loss visibly decreases in 3 steps
    })
    cfg["train"].pop("tier_schedule", None)  # force fallback uniform sampler
    cfg["data"]["num_workers"] = 0
    cfg["eval"].update({"scalars_every": 1, "val_fast_every": big,
                        "parse_every": big, "full_every": big})
    cfg["checkpoint"].update({"every_steps": 2, "every_minutes": big})
    return cfg


def test_stop_file_halts_loop_immediately(tmp_path):
    cfg = _cfg()
    (tmp_path / "STOP").write_text("")
    res = train.train_loop(cfg, DummyDataset(), model=DummyModel(),
                           run_dir=tmp_path, device="cpu")
    assert res["stopped"] is True
    assert res["step"] == 0
    assert (tmp_path / "last.pt").exists()  # final checkpoint still written


def test_short_run_scalars_and_checkpoints(tmp_path):
    cfg = _cfg()
    res = train.train_loop(cfg, DummyDataset(), steps=3, model=DummyModel(),
                           run_dir=tmp_path, device="cpu")
    assert res["stopped"] is False
    assert res["step"] == 3
    assert len(res["bpd_history"]) == 3

    # loss/bpd decreasing on the dummy quadratic objective
    assert res["bpd_history"][-1][1] < res["bpd_history"][0][1]

    # scalars.jsonl: one JSON per step (scalars_every=1)
    lines = (tmp_path / "scalars.jsonl").read_text().strip().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert [r["step"] for r in recs] == [1, 2, 3]
    assert all("loss" in r and "bpd" in r for r in recs)
    assert "lr_tables" in recs[0] and "lr_networks" in recs[0]

    # rolling checkpoint from every_steps=2 plus the final save
    assert (tmp_path / "last.pt").exists()
    ck = train.load_checkpoint(tmp_path / "last.pt")
    assert ck["step"] == 3
    assert "rng" in ck and "ema" in ck and "config" in ck
    assert "eta" in ck and "tau" in ck


def test_resume_auto_continues_from_last(tmp_path):
    cfg = _cfg()
    res1 = train.train_loop(cfg, DummyDataset(), steps=3, model=DummyModel(),
                            run_dir=tmp_path, device="cpu")
    w_end = res1["model"].w.detach().clone()

    res2 = train.train_loop(cfg, DummyDataset(), steps=5, model=DummyModel(),
                            run_dir=tmp_path, device="cpu", resume="auto")
    assert res2["start_step"] == 3
    assert res2["step"] == 5
    # weights were restored (training continued from res1, not from scratch)
    hist2 = dict(res2["bpd_history"])
    assert 4 in hist2 and 5 in hist2
    first_resumed_bpd = res2["bpd_history"][0][1]
    expected = float(((w_end - 1.0) ** 2).mean())  # loss at restored weights
    # first resumed step was computed from the restored weights
    assert abs(first_resumed_bpd * math.log(2.0) - expected) < expected * 0.5


def test_param_group_split():
    model = DummyModel()
    tables, nets = train.split_param_groups(model)
    assert len(tables) == 1 and tables[0] is model.E_T
    assert len(nets) == 1 and nets[0] is model.w
    cfg = _cfg()
    opt = train.build_optimizer(model, cfg)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert set(by_name) == {"tables", "networks"}
    assert by_name["tables"]["lr"] == cfg["train"]["lr_tables"]
    assert by_name["networks"]["lr"] == cfg["train"]["lr_networks"]


def test_lr_schedule_warmup_and_cosine_floor():
    cfg = _cfg()
    cfg["train"].update({"warmup_steps": 10, "lr_decay_steps": 100,
                         "lr_min_ratio": 0.1, "lr_networks": 1.0})
    model = DummyModel()
    opt = train.build_optimizer(model, cfg)
    sched = train.build_scheduler(opt, cfg)
    base = {g["name"]: g["initial_lr"] for g in opt.param_groups}
    lrs = []
    for _ in range(100):
        lrs.append({g["name"]: g["lr"] for g in opt.param_groups})
        opt.step()
        sched.step()
    # warmup ramps up
    assert lrs[0]["networks"] < lrs[5]["networks"] <= base["networks"]
    # cosine floor at 10% of peak
    final = {g["name"]: g["lr"] for g in opt.param_groups}
    assert abs(final["networks"] - 0.1 * base["networks"]) < 1e-9
    assert abs(final["tables"] - 0.1 * base["tables"]) < 1e-9


def test_tau_and_eta_schedules():
    cfg = _cfg()
    cfg["train"].update({"tau_start": 2.0, "tau_end": 1.0, "tau_steps": 100,
                         "eta_final_anneal_steps": 20})
    assert train.tau_at(cfg, 0) == 2.0
    assert abs(train.tau_at(cfg, 50) - 1.5) < 1e-9
    assert train.tau_at(cfg, 100) == 1.0
    assert train.tau_at(cfg, 10 ** 6) == 1.0
    total = 100
    assert train.eta_anneal_factor(cfg, 0, total) == 1.0
    assert train.eta_anneal_factor(cfg, 79, total) == 1.0
    assert abs(train.eta_anneal_factor(cfg, 90, total) - 0.5) < 1e-9
    assert train.eta_anneal_factor(cfg, 100, total) == 0.0


def test_pi_controller_matches_design():
    # H below band: eta += 0.05 + 0.1*(0.5 - H)
    assert abs(train.pi_update_eta(0.0, 0.1) - (0.05 + 0.1 * 0.4)) < 1e-12
    # inside band: unchanged
    assert train.pi_update_eta(0.7, 1.0) == 0.7
    # above band: eta -= 0.05
    assert abs(train.pi_update_eta(0.7, 3.5) - 0.65) < 1e-12
    # clamped to [0, 1.5]
    assert train.pi_update_eta(0.01, 5.0) == 0.0
    assert train.pi_update_eta(1.49, 0.0) == 1.5
