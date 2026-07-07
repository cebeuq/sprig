from __future__ import annotations

import json
import os

import numpy as np
import torch

from sprig.eval import report
from sprig.eval.prompts import MINIMAL_PAIRS, PROMPTS

S_STUB, TV_STUB = 128, 32

# exact sampler.py GT-tree schema
GT_TREE = {
    "rect": [0, 0, 64, 64],
    "axis": "V",
    "cut": 32,
    "children": [
        {"rect": [0, 0, 32, 64], "leaf": True, "obj": 0, "fill": None},
        {"rect": [32, 0, 64, 64], "leaf": True, "obj": 1, "fill": None},
    ],
}


def render_gt_image():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:, :32] = (200, 40, 40)
    img[:, 32:] = (45, 75, 220)
    return img


class StubModel:
    """Implements just enough of contracts C2-C4 for run_report wiring."""

    def log_marginal(self, image, emb, emb_len):
        img_term = image.to(torch.float32).mean(dim=(1, 2, 3))
        emb_term = emb.to(torch.float32).abs().mean(dim=(1, 2))
        return -(3.0 * 64 * 64) * (2.0 + 0.001 * img_term + 0.01 * emb_term)

    def sample(self, emb, emb_len, seed_struct, seed_material, n):
        img = np.zeros((n, 64, 64, 3), dtype=np.uint8)
        img[:, 20:44, 6:26] = (220, 40, 40)
        img[:, 20:44, 38:58] = (45, 75, 220)
        return torch.from_numpy(img), [None] * n

    def map_parse(self, image, emb, emb_len):
        return [GT_TREE]

    def posterior_usage(self, image, emb, emb_len):
        return {
            "symbol_usage": torch.ones(S_STUB),
            "texel_usage": torch.ones(TV_STUB),
            "node_entropy": 1.2,
            "emit_mag": 3.0,
            "rule_mag": 1.0,
            "mean_depth": 2.0,
            "mean_leaves": 3.0,
        }


def _write_split(split_dir, images, tiers, trees=None):
    os.makedirs(split_dir, exist_ok=True)
    n = images.shape[0]
    images.tofile(os.path.join(split_dir, "images.u8"))
    rng = np.random.default_rng(0)
    lens = [4] * n
    flat = rng.standard_normal((sum(lens), 768)).astype(np.float16)
    flat.tofile(os.path.join(split_dir, "emb.f16"))
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    offsets.tofile(os.path.join(split_dir, "emb_offsets.i64"))
    with open(os.path.join(split_dir, "meta.jsonl"), "w") as f:
        for i in range(n):
            meta = {"caption": "stub", "tier": int(tiers[i])}
            if trees is not None:
                meta["tree"] = trees[i]
            f.write(json.dumps(meta) + "\n")


def _write_npz(t5_dir, name, n_rows, L=3, seed=1):
    rng = np.random.default_rng(seed)
    emb = rng.standard_normal((n_rows, L, 768)).astype(np.float16)
    lens = np.full(n_rows, L, dtype=np.int32)
    np.savez(os.path.join(t5_dir, name + ".npz"), emb=emb, len=lens)


def make_data_dir(root):
    rng = np.random.default_rng(0)
    val_imgs = rng.integers(0, 256, size=(16, 64, 64, 3)).astype(np.uint8)
    _write_split(os.path.join(root, "val"), val_imgs, [i % 3 for i in range(16)])
    parse_imgs = np.stack([render_gt_image() for _ in range(4)])
    _write_split(
        os.path.join(root, "parse_eval"), parse_imgs, [1, 1, 2, 2], trees=[GT_TREE] * 4
    )
    t5_dir = os.path.join(root, "t5")
    os.makedirs(t5_dir, exist_ok=True)
    rng.standard_normal((2, 768)).astype(np.float16).tofile(os.path.join(t5_dir, "null.f16"))
    _write_npz(t5_dir, "promptbank", len(PROMPTS))
    _write_npz(t5_dir, "minimal_pairs", 2 * len(MINIMAL_PAIRS))
    return root


def test_run_report_end_to_end(tmp_path):
    data_dir = make_data_dir(str(tmp_path / "data"))
    out_dir = str(tmp_path / "out")
    metrics = report.run_report(
        ckpt_path=None,
        data_dir=data_dir,
        out_dir=out_dir,
        device="cpu",
        model=StubModel(),
        n_bank_seeds=2,
        n_pair_seeds=2,
        max_bpd_images=16,
        max_parse_images=4,
        b0_steps=5,
    )
    for fname in (
        "metrics.json",
        "final_report.md",
        "prompt_bank_grid.jpg",
        "layout_material_grid.jpg",
    ):
        assert os.path.exists(os.path.join(out_dir, fname)), fname

    assert np.isfinite(metrics["bpd_val"])
    assert set(metrics["bpd_per_tier"].keys()) == {0, 1, 2}
    assert np.isfinite(metrics["delta_c"])
    assert 0.0 <= metrics["caption_swap_win_frac"] <= 1.0
    assert np.isfinite(metrics["b0_bpd"])
    # stub parses == GT -> perfect tree metrics
    assert metrics["tree"]["recall_tier1"] == 1.0
    assert metrics["tree"]["recall_tier2"] == 1.0
    assert metrics["tree"]["visible_cut_f1"] == 1.0
    assert metrics["tree"]["leaf_ari"] == 1.0
    assert metrics["prompt_swap"]["attribute_move"] is not None
    assert metrics["holdout_probe_acc"] is None  # no probe ckpt
    assert metrics["health"]["S"] == S_STUB
    assert abs(metrics["health"]["s_eff"] - S_STUB) < 1e-6  # uniform usage

    blob = json.load(open(os.path.join(out_dir, "metrics.json")))
    assert set(blob["gates"].keys()) == {
        "1_likelihood", "2_parses", "3_prompt_control", "4_compositional", "5_health",
    }
    assert blob["gates"]["4_compositional"]["status"] == "SKIPPED"
    assert blob["gates"]["2_parses"]["status"] == "PASS"
    assert blob["gates"]["5_health"]["status"] == "PASS"


def _full_metrics():
    return {
        "bpd_tier_ge1": 2.0,
        "b0_bpd": 2.3,
        "delta_c": 0.10,
        "tree": {"recall_tier1": 0.8, "recall_tier2": 0.6, "visible_cut_f1": 0.7},
        "prompt_swap": {"attribute_move": 0.9, "relation_accuracy": 0.8},
        "holdout_probe_acc": 0.7,
        "health": {"s_eff": 400.0, "S": 1024, "alive_texel_frac": 0.6},
    }


def test_evaluate_gates_all_pass():
    gates = report.evaluate_gates(_full_metrics())
    assert all(g["status"] == "PASS" for g in gates.values())


def test_evaluate_gates_failures():
    m = _full_metrics()
    m["b0_bpd"] = 2.05  # margin 0.05 < 0.15
    m["tree"]["visible_cut_f1"] = 0.5
    m["prompt_swap"]["relation_accuracy"] = 0.5
    m["holdout_probe_acc"] = 0.1
    m["health"]["s_eff"] = 100.0  # < 0.25*1024
    gates = report.evaluate_gates(m)
    assert all(g["status"] == "FAIL" for g in gates.values())


def test_evaluate_gates_skipped_on_missing():
    gates = report.evaluate_gates({})
    assert all(g["status"] == "SKIPPED" for g in gates.values())


def test_image_grid_shape():
    rows = [[np.zeros((64, 64, 3), dtype=np.uint8)] * 3] * 2
    grid = report.image_grid(rows, pad=2)
    assert grid.size == (3 * 66 + 2, 2 * 66 + 2)  # PIL size is (W, H)


def test_bpd_from_logz():
    # logZ = -3*64*64*ln2 nats -> exactly 1 bit/dim
    logz = np.array([-3.0 * 64 * 64 * np.log(2.0)])
    assert abs(report.bpd_from_logz(logz)[0] - 1.0) < 1e-12
