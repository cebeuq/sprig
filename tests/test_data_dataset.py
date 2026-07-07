"""Tests for sprig/data/dataset.py: roundtrip, collate (C1), null substitution,
curriculum sampler, emb offset alignment, multi-caption variants."""

from __future__ import annotations

import itertools
import json
import os

import numpy as np
import pytest
import torch

from sprig.data.dataset import (
    SprigDataset,
    TierCurriculumSampler,
    collate,
    load_tier_indices,
    read_meta,
)

EMB_DIM = 768


def make_dataset_dir(root, n=12, n_variants=1, tiers=None, seed=0):
    """Write a tiny synthetic dataset dir directly (no procgen dependency).

    Sample i, variant v gets length (i % 5) + 1 + v tokens; every token row of
    (i, v) is filled with the value 100*v + i so content is identifiable.
    """
    os.makedirs(root, exist_ok=True)
    rng = np.random.default_rng(seed)

    images = rng.integers(0, 256, size=(n, 64, 64, 3), dtype=np.uint8)
    images.tofile(os.path.join(root, "images.u8"))

    lens_by_variant = []
    for v in range(n_variants):
        lens = np.array([(i % 5) + 1 + v for i in range(n)], dtype=np.int64)
        offsets = np.zeros(n + 1, dtype=np.int64)
        offsets[1:] = np.cumsum(lens)
        emb = np.zeros((int(offsets[-1]), EMB_DIM), dtype=np.float16)
        for i in range(n):
            emb[offsets[i] : offsets[i + 1]] = np.float16(100 * v + i)
        prefix = "emb" if n_variants == 1 else "emb%d" % v
        emb.tofile(os.path.join(root, prefix + ".f16"))
        offsets.tofile(os.path.join(root, prefix + "_offsets.i64"))
        lens_by_variant.append(lens)

    if tiers is None:
        tiers = np.zeros(n, dtype=np.int64)
    tiers = np.asarray(tiers, dtype=np.int64)
    tier_dir = os.path.join(root, "tier_idx")
    os.makedirs(tier_dir, exist_ok=True)
    for t in range(int(tiers.max()) + 1):
        np.where(tiers == t)[0].astype(np.int64).tofile(
            os.path.join(tier_dir, "tier%d.i64" % t)
        )

    offsets_bytes = [0]
    with open(os.path.join(root, "meta.jsonl"), "wb") as f:
        for i in range(n):
            if n_variants == 1:
                rec = {"idx": i, "tier": int(tiers[i]), "caption": "cap %d" % i}
            else:
                rec = {
                    "idx": i,
                    "tier": int(tiers[i]),
                    "captions": ["cap %d v%d" % (i, v) for v in range(n_variants)],
                }
            line = (json.dumps(rec) + "\n").encode()
            f.write(line)
            offsets_bytes.append(offsets_bytes[-1] + len(line))
    np.asarray(offsets_bytes, dtype=np.int64).tofile(
        os.path.join(root, "meta_offsets.i64")
    )
    return images, lens_by_variant


def make_null(path, l0=2, value=7.0):
    np.full((l0, EMB_DIM), value, dtype=np.float16).tofile(str(path))


def test_roundtrip_and_offset_alignment(tmp_path):
    root = str(tmp_path / "ds")
    images, (lens,) = make_dataset_dir(root, n=12)
    ds = SprigDataset(root, train=False)
    assert len(ds) == 12
    for i in [0, 3, 11]:
        item = ds[i]
        assert item["image"].dtype == torch.uint8
        assert item["image"].shape == (64, 64, 3)
        assert torch.equal(item["image"], torch.from_numpy(images[i]))
        assert int(item["emb_len"]) == lens[i]
        assert item["emb"].shape == (lens[i], EMB_DIM)
        assert item["emb"].dtype == torch.float16
        # Content matches the packed slice for this index.
        assert torch.all(item["emb"] == float(i))
        assert int(item["idx"]) == i
    meta = read_meta(root, 5)
    assert meta["caption"] == "cap 5"


def test_collate_c1_shapes_dtypes_padding(tmp_path):
    root = str(tmp_path / "ds")
    make_dataset_dir(root, n=8, tiers=[0, 1, 2, 3, 0, 1, 2, 3])
    ds = SprigDataset(root, train=False)
    batch = collate([ds[i] for i in range(6)])

    assert batch["image"].dtype == torch.uint8
    assert batch["image"].shape == (6, 64, 64, 3)
    assert batch["emb"].dtype == torch.float16
    lens = [(i % 5) + 1 for i in range(6)]
    lmax = max(lens)
    assert batch["emb"].shape == (6, lmax, EMB_DIM)
    assert batch["emb_len"].dtype == torch.int32
    assert batch["emb_len"].tolist() == lens
    assert batch["tier"].dtype == torch.int8
    assert batch["tier"].tolist() == [0, 1, 2, 3, 0, 1]
    assert batch["idx"].dtype == torch.int64
    assert batch["idx"].tolist() == list(range(6))
    # Zero padding beyond emb_len; valid region intact.
    for i in range(6):
        assert torch.all(batch["emb"][i, : lens[i]] == float(i))
        assert torch.all(batch["emb"][i, lens[i] :] == 0.0)


def test_null_substitution_rate(tmp_path):
    root = str(tmp_path / "ds")
    make_dataset_dir(root, n=10)
    null_path = tmp_path / "null.f16"
    make_null(null_path, l0=2, value=7.0)

    ds = SprigDataset(root, p_null=0.1, null_emb_path=str(null_path), train=True, seed=3)
    n_null = 0
    for k in range(2000):
        item = ds[k % 10]
        if int(item["emb_len"]) == 2 and float(item["emb"][0, 0]) == 7.0:
            n_null += 1
    rate = n_null / 2000.0
    assert 0.07 <= rate <= 0.13, "null rate %.3f outside [0.07, 0.13]" % rate


def test_null_never_in_eval_mode(tmp_path):
    root = str(tmp_path / "ds")
    make_dataset_dir(root, n=10)
    null_path = tmp_path / "null.f16"
    make_null(null_path)
    ds = SprigDataset(root, p_null=0.5, null_emb_path=str(null_path), train=False)
    for k in range(200):
        assert float(ds[k % 10]["emb"][0, 0]) == float(k % 10)


def test_multi_caption_variants(tmp_path):
    root = str(tmp_path / "ds")
    make_dataset_dir(root, n=9, n_variants=3)
    ds = SprigDataset(root, train=False)
    assert ds.n_variants == 3
    # Eval: deterministic (idx + epoch) % 3; value encodes 100*v + i.
    for i in range(9):
        v = i % 3
        item = ds[i]
        assert float(item["emb"][0, 0]) == float(100 * v + i)
        assert int(item["emb_len"]) == (i % 5) + 1 + v
    ds.set_epoch(1)
    for i in range(9):
        v = (i + 1) % 3
        assert float(ds[i]["emb"][0, 0]) == float(100 * v + i)
    # Train: all three variants show up for a fixed idx.
    ds_train = SprigDataset(root, train=True, seed=1)
    seen = set()
    for _ in range(60):
        seen.add(int(float(ds_train[4]["emb"][0, 0])) // 100)
    assert seen == {0, 1, 2}


def test_dataloader_end_to_end(tmp_path):
    root = str(tmp_path / "ds")
    make_dataset_dir(root, n=16, tiers=[i % 2 for i in range(16)])
    ds = SprigDataset(root, train=True, seed=0)
    sampler = TierCurriculumSampler(
        load_tier_indices(root), [(0, [0.5, 0.5])], batch_size=4, seed=0
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=4, sampler=sampler, collate_fn=collate, num_workers=0
    )
    it = iter(loader)
    for _ in range(3):
        batch = next(it)
        assert set(batch.keys()) == {"image", "emb", "emb_len", "tier", "idx"}
        assert batch["image"].shape == (4, 64, 64, 3)


def test_sampler_respects_weights(tmp_path):
    tier_indices = [
        np.arange(0, 100, dtype=np.int64),
        np.arange(100, 200, dtype=np.int64),
        np.arange(200, 300, dtype=np.int64),
        np.arange(300, 400, dtype=np.int64),
    ]
    schedule = [(0, [1.0, 0.0, 0.0, 0.0]), (10, [0.3, 0.7, 0.0, 0.0])]
    s = TierCurriculumSampler(tier_indices, schedule, batch_size=1, seed=0)
    it = iter(s)
    first = [next(it) for _ in range(10)]
    assert all(i < 100 for i in first), "phase 1 must draw only tier 0"
    rest = [next(it) for _ in range(4000)]
    assert all(i < 200 for i in rest), "phase 2 must draw only tiers 0/1"
    frac1 = sum(1 for i in rest if i >= 100) / len(rest)
    assert 0.65 <= frac1 <= 0.75, "tier-1 fraction %.3f not ~0.7" % frac1


def test_sampler_zero_weight_empty_tier():
    # An empty tier with nonzero weight is renormalized away, not sampled.
    tier_indices = [np.arange(10, dtype=np.int64), np.zeros(0, dtype=np.int64)]
    s = TierCurriculumSampler(tier_indices, [(0, [0.5, 0.5])], seed=0)
    draws = list(itertools.islice(iter(s), 100))
    assert all(0 <= i < 10 for i in draws)


def test_sampler_state_dict_roundtrip():
    tier_indices = [np.arange(0, 50, dtype=np.int64), np.arange(50, 100, dtype=np.int64)]
    schedule = [(0, [0.8, 0.2]), (30, [0.2, 0.8])]
    s1 = TierCurriculumSampler(tier_indices, schedule, batch_size=2, seed=7)
    it1 = iter(s1)
    _ = [next(it1) for _ in range(37)]
    state = s1.state_dict()
    seq_a = [next(it1) for _ in range(50)]

    s2 = TierCurriculumSampler(tier_indices, schedule, batch_size=2, seed=7)
    s2.load_state_dict(state)
    seq_b = list(itertools.islice(iter(s2), 50))
    assert seq_a == seq_b


def test_load_tier_indices_fallback(tmp_path):
    root = str(tmp_path / "flat")
    os.makedirs(root)
    tiers = load_tier_indices(root, n=5)
    assert len(tiers) == 1
    assert tiers[0].tolist() == [0, 1, 2, 3, 4]
