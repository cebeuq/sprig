"""Tests for sprig/data/clevr/prep.py: crop/drop logic, caption synthesis from
a hand-written 3-object fixture scene, and end-to-end prep on a tiny fake
CLEVR directory."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from PIL import Image

from sprig.data.clevr.prep import (
    CROP_X0,
    CROP_X1,
    crop_resize,
    obj_phrase,
    prep_split,
    synth_captions,
    visible_objects,
)


def _obj(shape, color, size, material, x, y, depth=10.0):
    return {
        "shape": shape,
        "color": color,
        "size": size,
        "material": material,
        "pixel_coords": [x, y, depth],
    }


# Hand-written fixture: 3 objects, one of which (the cylinder) is outside the
# x in [80, 400) crop and must be dropped.
FIXTURE_SCENE = {
    "image_index": 0,
    "image_filename": "CLEVR_train_000000.png",
    "objects": [
        _obj("cube", "red", "large", "rubber", 120.0, 200.0),
        _obj("sphere", "blue", "small", "metal", 340.0, 150.0),
        _obj("cylinder", "green", "large", "metal", 50.0, 180.0),  # cropped out
    ],
    "relationships": {},
}


def _rng(seed=0):
    return np.random.Generator(np.random.PCG64(seed))


def test_visible_objects_drops_outside_crop():
    kept, dropped = visible_objects(FIXTURE_SCENE)
    assert dropped == 1
    assert len(kept) == 2
    assert {o["shape"] for o in kept} == {"cube", "sphere"}
    # Boundary: x == 80 kept, x == 400 dropped.
    edge = {"objects": [_obj("cube", "red", "large", "rubber", 80.0, 10.0),
                        _obj("cube", "red", "large", "rubber", 400.0, 10.0)]}
    kept, dropped = visible_objects(edge)
    assert len(kept) == 1 and dropped == 1


def test_obj_phrase():
    assert obj_phrase(FIXTURE_SCENE["objects"][0]) == "a large red rubber cube"


def test_relation_caption_direction():
    kept, _ = visible_objects(FIXTURE_SCENE)  # cube at x=120, sphere at x=340
    cube = "a large red rubber cube"
    sphere = "a small blue metal sphere"
    for seed in range(10):
        cap = synth_captions(kept, _rng(seed))[0]
        if "to the left of" in cap:
            assert cap == "%s to the left of %s" % (cube, sphere)
        else:
            assert cap == "%s to the right of %s" % (sphere, cube)


def test_relation_caption_depth_axis():
    # Nearly equal x -> front/behind from pixel y (larger y = in front).
    objs = [
        _obj("cube", "red", "large", "rubber", 200.0, 250.0),
        _obj("sphere", "blue", "small", "metal", 210.0, 100.0),
    ]
    cap = synth_captions(objs, _rng(0))[0]
    assert ("in front of" in cap) or ("behind" in cap)
    if "in front of" in cap:
        assert cap.startswith("a large red rubber cube")
    else:
        assert cap.startswith("a small blue metal sphere")


def test_enumeration_caption_capped_at_4():
    objs = [
        _obj("cube", c, "small", "rubber", 100.0 + 10 * i, 100.0)
        for i, c in enumerate(["red", "blue", "green", "gray", "cyan", "brown"])
    ]
    cap = synth_captions(objs, _rng(1))[1]
    assert cap.startswith("a scene with ")
    assert cap.count("a small") == 4  # exactly 4 objects listed
    assert " and " in cap


def test_count_caption():
    kept, _ = visible_objects(FIXTURE_SCENE)
    cap = synth_captions(kept, _rng(2))[2]
    assert "two objects" in cap
    assert "including a" in cap
    single = synth_captions(kept[:1], _rng(2))[2]
    assert "one object," in single


def test_captions_deterministic_and_three_variants():
    kept, _ = visible_objects(FIXTURE_SCENE)
    a = synth_captions(kept, _rng(5))
    b = synth_captions(kept, _rng(5))
    assert a == b
    assert len(a) == 3
    empty = synth_captions([], _rng(0))
    assert empty == ["an empty scene"] * 3


def test_captions_under_64_t5_tokens_heuristic():
    # Word-count proxy for the <=64-token budget (T5 ~1.3 tokens/word here).
    objs = [
        _obj("cylinder", "yellow", "large", "rubber", 100.0 + i, 100.0)
        for i in range(10)
    ]
    for cap in synth_captions(objs, _rng(3)):
        assert len(cap.split()) <= 40


def test_crop_resize():
    img = Image.new("RGB", (480, 320), (0, 0, 0))
    # Paint the crop region white; the result must be all white.
    px = img.load()
    for x in range(CROP_X0, CROP_X1):
        for y in range(0, 320, 8):
            for yy in range(y, min(y + 8, 320)):
                px[x, yy] = (255, 255, 255)
    out = crop_resize(img)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
    assert int(out.min()) >= 250


def _make_fake_clevr(root, n=3):
    os.makedirs(os.path.join(root, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "scenes"), exist_ok=True)
    scenes = []
    rng = np.random.default_rng(0)
    for i in range(n):
        fname = "CLEVR_train_%06d.png" % i
        arr = rng.integers(0, 256, size=(320, 480, 3), dtype=np.uint8)
        Image.fromarray(arr).save(os.path.join(root, "images", "train", fname))
        scene = dict(FIXTURE_SCENE)
        scene["image_index"] = i
        scene["image_filename"] = fname
        scenes.append(scene)
    with open(os.path.join(root, "scenes", "CLEVR_train_scenes.json"), "w") as f:
        json.dump({"scenes": scenes}, f)


def test_prep_split_end_to_end(tmp_path):
    clevr_root = str(tmp_path / "CLEVR_v1.0")
    out = str(tmp_path / "out")
    _make_fake_clevr(clevr_root, n=3)

    stats = prep_split(clevr_root, "train", out, seed=0)
    assert stats["n_images"] == 3
    assert stats["total_dropped"] == 3  # one per fixture scene
    assert abs(stats["drop_frac"] - 1.0 / 3.0) < 1e-9

    images = np.fromfile(os.path.join(out, "images.u8"), dtype=np.uint8)
    assert images.size == 3 * 64 * 64 * 3

    off = np.fromfile(os.path.join(out, "meta_offsets.i64"), dtype=np.int64)
    assert off.shape == (4,)
    assert off[-1] == os.path.getsize(os.path.join(out, "meta.jsonl"))
    with open(os.path.join(out, "meta.jsonl"), "rb") as f:
        for i in range(3):
            f.seek(off[i])
            rec = json.loads(f.readline())
            assert rec["idx"] == i
            assert len(rec["captions"]) == 3
            assert rec["n_objects"] == 2
            assert rec["n_dropped"] == 1
            assert rec["tier"] == 0

    tier0 = np.fromfile(os.path.join(out, "tier_idx", "tier0.i64"), dtype=np.int64)
    assert tier0.tolist() == [0, 1, 2]

    # Determinism: same seed -> identical captions across runs.
    out2 = str(tmp_path / "out2")
    prep_split(clevr_root, "train", out2, seed=0)
    with open(os.path.join(out, "meta.jsonl")) as f1, open(
        os.path.join(out2, "meta.jsonl")
    ) as f2:
        assert f1.read() == f2.read()


def test_prep_limit(tmp_path):
    clevr_root = str(tmp_path / "CLEVR_v1.0")
    out = str(tmp_path / "out")
    _make_fake_clevr(clevr_root, n=3)
    stats = prep_split(clevr_root, "train", out, limit=2, seed=0)
    assert stats["n_images"] == 2
    images = np.fromfile(os.path.join(out, "images.u8"), dtype=np.uint8)
    assert images.size == 2 * 64 * 64 * 3


def test_prep_output_loads_in_dataset(tmp_path):
    """Prep output + fake emb0/1/2 files must load through SprigDataset."""
    from sprig.data.dataset import SprigDataset, collate

    clevr_root = str(tmp_path / "CLEVR_v1.0")
    out = str(tmp_path / "out")
    _make_fake_clevr(clevr_root, n=3)
    prep_split(clevr_root, "train", out, seed=0)

    for v in range(3):
        lens = np.array([v + 2] * 3, dtype=np.int64)
        offsets = np.zeros(4, dtype=np.int64)
        offsets[1:] = np.cumsum(lens)
        emb = np.full((int(offsets[-1]), 768), float(v), dtype=np.float16)
        emb.tofile(os.path.join(out, "emb%d.f16" % v))
        offsets.tofile(os.path.join(out, "emb%d_offsets.i64" % v))

    ds = SprigDataset(out, train=False)
    assert len(ds) == 3
    assert ds.n_variants == 3
    batch = collate([ds[i] for i in range(3)])
    assert batch["image"].shape == (3, 64, 64, 3)
    assert batch["emb_len"].tolist() == [2, 3, 4]  # variant (idx+0)%3
