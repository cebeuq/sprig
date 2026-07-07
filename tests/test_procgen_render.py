"""Renderer determinism (per-idx hash regression) and basic pixel semantics."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from sprig.data.procgen.render import generate, render_scene
from sprig.data.procgen.sampler import iter_leaves, sample_scene

SEED = 777


def _hash(img: np.ndarray) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()


def test_output_shape_dtype():
    _, img = generate(SEED, 0)
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8


def test_per_idx_determinism_bit_identical():
    """Same (global_seed, idx) => identical scene JSON and identical pixels."""
    for idx in range(24):
        scene_a, img_a = generate(SEED, idx)
        scene_b, img_b = generate(SEED, idx)
        assert json.dumps(scene_a.__dict__, sort_keys=True) == json.dumps(
            scene_b.__dict__, sort_keys=True
        )
        assert _hash(img_a) == _hash(img_b)
        assert (img_a == img_b).all()


def test_different_idx_different_image():
    hashes = {_hash(generate(SEED, i)[1]) for i in range(16)}
    assert len(hashes) == 16


def test_render_is_pure_function_of_scene():
    scene = sample_scene(SEED, 3)
    assert _hash(render_scene(scene)) == _hash(render_scene(scene))


def test_background_pixels_match_leaf_fill():
    # Corner pixels sit >=2px away from any object bbox, so after BOX
    # downsampling of a uniform region they equal the leaf fill exactly.
    for idx in range(20):
        scene, img = generate(SEED, idx, tier=0)
        for (px, py) in ((0, 0), (63, 0), (0, 63), (63, 63)):
            leaf = next(
                l
                for l in iter_leaves(scene.tree)
                if l["rect"][0] <= px < l["rect"][2]
                and l["rect"][1] <= py < l["rect"][3]
            )
            assert img[py, px].tolist() == leaf["fill"]


def test_object_color_present():
    # some downsampled pixel inside the object's cell must be close to the
    # object's jittered rgb (or its darker texture-gap shade).
    for idx in range(20):
        scene, img = generate(SEED, idx, tier=0)
        obj = scene.objects[0]
        x0, y0, x1, y1 = obj["cell"]
        patch = img[y0:y1, x0:x1].reshape(-1, 3).astype(np.int64)
        rgb = np.asarray(obj["rgb"], dtype=np.int64)
        dark = (rgb * 0.45).astype(np.int64)
        d_main = np.abs(patch - rgb).sum(axis=1).min()
        d_dark = np.abs(patch - dark).sum(axis=1).min()
        assert min(d_main, d_dark) <= 90, (
            "object color not found in cell (idx {}, shape {})".format(
                idx, obj["shape"]
            )
        )
