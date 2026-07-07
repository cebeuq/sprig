"""Writer roundtrip: memmaps, meta.jsonl + offsets, per-tier index arrays."""
from __future__ import annotations

import json
import os

import numpy as np

from sprig.data.procgen.captions import TRAIN_TEMPLATE_IDS
from sprig.data.procgen.render import render_scene
from sprig.data.procgen.sampler import sample_scene
from sprig.data.procgen.writer import IMG_SHAPE, main, write_dataset

SEED = 99
N = 64
TIER_MIX = (0.1, 0.3, 0.4, 0.2)


def _read_all(out_dir):
    images = np.memmap(
        os.path.join(out_dir, "images.u8"), dtype=np.uint8, mode="r",
        shape=(N,) + IMG_SHAPE,
    )
    offsets = np.fromfile(os.path.join(out_dir, "meta_offsets.i64"), dtype=np.int64)
    metas = []
    with open(os.path.join(out_dir, "meta.jsonl"), "rb") as f:
        for off in offsets:
            f.seek(int(off))
            metas.append(json.loads(f.readline().decode("utf-8")))
    return images, offsets, metas


def test_writer_roundtrip_multiprocess(tmp_path):
    out_dir = str(tmp_path / "proc2d")
    write_dataset(out_dir, N, seed=SEED, tier_mix=TIER_MIX, workers=2)

    images, offsets, metas = _read_all(out_dir)
    assert offsets.shape == (N,) and offsets[0] == 0
    assert (np.diff(offsets) > 0).all()

    for i, rec in enumerate(metas):
        assert rec["idx"] == i
        assert rec["tier"] in (0, 1, 2, 3)
        assert rec["caption"]
        assert rec["template_id"] in TRAIN_TEMPLATE_IDS
        assert isinstance(rec["partial"], bool)
        assert rec["tree"]["rect"] == [0, 0, 64, 64]
        assert len(rec["objects"]) >= 1

    # images match an independent regeneration (worker-count independent)
    for idx in (0, 5, 31, 32, 63):
        scene = sample_scene(SEED, idx, tier_mix=TIER_MIX)
        assert (images[idx] == render_scene(scene)).all()
        assert metas[idx]["tier"] == scene.tier

    # tier index files partition [0, N) and agree with meta
    tiers = np.array([rec["tier"] for rec in metas], dtype=np.int64)
    all_idx = []
    for t in range(4):
        p = os.path.join(out_dir, "tier_idx", "tier{}.i64".format(t))
        idxs = np.fromfile(p, dtype=np.int64)
        assert (tiers[idxs] == t).all()
        all_idx.append(idxs)
    cat = np.sort(np.concatenate(all_idx))
    assert (cat == np.arange(N)).all()

    # no leftover shard files
    assert not [f for f in os.listdir(out_dir) if ".part" in f]


def test_writer_single_worker_identical(tmp_path):
    d1 = str(tmp_path / "w1")
    d2 = str(tmp_path / "w2")
    write_dataset(d1, 16, seed=SEED, tier_mix=TIER_MIX, workers=1)
    write_dataset(d2, 16, seed=SEED, tier_mix=TIER_MIX, workers=3)
    for name in ("images.u8", "meta.jsonl", "meta_offsets.i64"):
        with open(os.path.join(d1, name), "rb") as a, open(
            os.path.join(d2, name), "rb"
        ) as b:
            assert a.read() == b.read(), "{} differs across worker counts".format(name)


def test_cli_main(tmp_path):
    out_dir = str(tmp_path / "cli")
    main(
        [
            "--out", out_dir, "--n", "8", "--seed", "3",
            "--tier-mix", "0.25,0.25,0.25,0.25", "--workers", "1",
        ]
    )
    assert os.path.getsize(os.path.join(out_dir, "images.u8")) == 8 * 64 * 64 * 3
    with open(os.path.join(out_dir, "meta.jsonl")) as f:
        assert len(f.readlines()) == 8
