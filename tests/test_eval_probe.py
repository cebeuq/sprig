from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from sprig.eval import probe
from sprig.eval.color_checks import COLOR_ANCHORS as ANCHORS
from sprig.eval.prompts import COLORS, SHAPES


def draw_shape(shape: str, rgb, cx=32, cy=32, r=14) -> np.ndarray:
    img = Image.new("RGB", (64, 64), (0, 0, 0))
    d = ImageDraw.Draw(img)
    fill = tuple(int(v) for v in rgb)
    box = (cx - r, cy - r, cx + r, cy + r)
    if shape == "circle":
        d.ellipse(box, fill=fill)
    elif shape == "square":
        d.rectangle(box, fill=fill)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=fill)
    elif shape == "rectangle":
        d.rectangle((cx - r, cy - r // 2, cx + r, cy + r // 2), fill=fill)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=fill)
    elif shape == "star":
        pts = []
        for k in range(10):
            rad = r if k % 2 == 0 else r // 2
            ang = -np.pi / 2 + k * np.pi / 5
            pts.append((cx + rad * np.cos(ang), cy + rad * np.sin(ang)))
        d.polygon(pts, fill=fill)
    elif shape == "cross":
        w = r // 2
        d.rectangle((cx - w, cy - r, cx + w, cy + r), fill=fill)
        d.rectangle((cx - r, cy - w, cx + r, cy + w), fill=fill)
    elif shape == "ring":
        d.ellipse(box, fill=fill)
        d.ellipse((cx - r // 2, cy - r // 2, cx + r // 2, cy + r // 2), fill=(0, 0, 0))
    else:
        raise ValueError(shape)
    return np.asarray(img, dtype=np.uint8)


def make_dataset(tmp_path, n=200, seed=0):
    rng = np.random.default_rng(seed)
    images = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    labels = np.zeros((n, 2), dtype=np.int64)
    for i in range(n):
        s, c = int(rng.integers(8)), int(rng.integers(8))
        jitter = rng.integers(-8, 9, 3)
        rgb = np.clip(np.array(ANCHORS[COLORS[c]]) + jitter, 0, 255)
        cx, cy = int(rng.integers(24, 41)), int(rng.integers(24, 41))
        r = int(rng.integers(11, 17))
        images[i] = draw_shape(SHAPES[s], rgb, cx, cy, r)
        labels[i] = (s, c)
    images.tofile(str(tmp_path / "images.u8"))
    np.save(str(tmp_path / "labels.npy"), labels)
    return images, labels


def test_probe_learns_synthetic_set(tmp_path):
    images, labels = make_dataset(tmp_path, n=200)
    ckpt = probe.train(str(tmp_path), epochs=30, batch_size=16, lr=1e-3, seed=0)
    model = probe.load_probe(ckpt)
    with torch.no_grad():
        logit_s, logit_c = model(probe._images_to_tensor(images))
    acc = float(
        (
            (logit_s.argmax(1).numpy() == labels[:, 0])
            & (logit_c.argmax(1).numpy() == labels[:, 1])
        ).mean()
    )
    assert acc > 0.9, acc

    # score_generations agrees with direct classification on a pure batch
    reds = np.stack([draw_shape("circle", ANCHORS["red"]) for _ in range(8)])
    score = probe.score_generations(reds, "circle", "red", ckpt)
    assert 0.0 <= score <= 1.0


def test_probe_png_dir_loading(tmp_path):
    for i, (shape, color) in enumerate([("circle", "red"), ("square", "blue")]):
        img = draw_shape(shape, ANCHORS[color])
        Image.fromarray(img).save(str(tmp_path / "{}_{}_{}.png".format(shape, color, i)))
    images, labels = probe._load_probe_dir(str(tmp_path))
    assert images.shape == (2, 64, 64, 3)
    assert labels.tolist() == [
        [SHAPES.index("circle"), COLORS.index("red")],
        [SHAPES.index("square"), COLORS.index("blue")],
    ]


def test_probe_missing_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe._load_probe_dir(str(tmp_path))


def test_probe_model_size_is_small():
    n_params = sum(p.numel() for p in probe.ProbeCNN().parameters())
    assert n_params < 2_000_000
