"""CLEVR v1.0 -> SPRIG memmap preprocessing.

CLI:

    python -m sprig.data.clevr.prep --clevr-root ~/data/CLEVR_v1.0 \
        --split train --out ~/data/sprig/clevr/train [--limit N] [--seed 0]

Reads `<clevr-root>/scenes/CLEVR_<split>_scenes.json` and
`<clevr-root>/images/<split>/<image_filename>` (480x320 renders). Each image
is center-cropped to x in [80, 400) (full height) -> 320x320, LANCZOS-resized
to 64x64, and written to the standard SPRIG memmap layout (see
`sprig/data/dataset.py`). Objects whose `pixel_coords` x-center falls outside
the crop are dropped from the caption pool; the dropped fraction is logged.

Three caption variants are synthesized per image from the scene graph
(deterministic per (seed, idx)):

  0. pairwise relation read off pixel_coords of two sampled visible objects
     ("a large red rubber cube to the left of a small blue metal sphere";
     left/right from pixel x, in front of/behind from pixel y when the
     x-separation is small);
  1. partial enumeration capped at 4 objects, order shuffled
     ("a scene with a ..., a ..., a ... and a ...");
  2. count + existence ("a scene with five objects, including a gray cube").

Output files: images.u8 [N,64,64,3], meta.jsonl (records with
"captions": [c0, c1, c2], visible objects, drop counts), meta_offsets.i64
[N+1] byte offsets, tier_idx/tier0.i64 (CLEVR has no tiers; everything is
tier 0). Embeddings are computed afterwards by
`python -m sprig.data.embed_t5 --data-dir <out>`, which detects the
multi-caption meta and writes emb0/1/2.f16 with their own offsets; the
dataset then picks one variant per visit (uniform in training).
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

CROP_X0 = 80
CROP_X1 = 400
IMG_SIZE = 64
N_CAPTIONS = 3
# Min horizontal pixel separation (in original 480-px coords) for a
# left/right relation; below this we use depth-axis (pixel y) instead.
X_REL_THRESH = 24.0

NUM_WORDS = [
    "zero", "one", "two", "three", "four", "five",
    "six", "seven", "eight", "nine", "ten",
]


def num_word(n: int) -> str:
    return NUM_WORDS[n] if 0 <= n < len(NUM_WORDS) else str(n)


def obj_phrase(o: Dict) -> str:
    return "a %s %s %s %s" % (o["size"], o["color"], o["material"], o["shape"])


def crop_resize(img: Image.Image) -> np.ndarray:
    """480x320 CLEVR render -> center crop x in [80,400) -> 64x64 u8 RGB."""
    img = img.convert("RGB").crop((CROP_X0, 0, CROP_X1, img.height))
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def visible_objects(scene: Dict) -> Tuple[List[Dict], int]:
    """Objects whose pixel-x center lies inside the crop; also #dropped."""
    kept: List[Dict] = []
    dropped = 0
    for o in scene["objects"]:
        x = float(o["pixel_coords"][0])
        if CROP_X0 <= x < CROP_X1:
            kept.append(o)
        else:
            dropped += 1
    return kept, dropped


def _relation_caption(objects: List[Dict], rng: np.random.Generator) -> str:
    if not objects:
        return "an empty scene"
    if len(objects) == 1:
        return obj_phrase(objects[0])
    i, j = rng.choice(len(objects), size=2, replace=False)
    a, b = objects[int(i)], objects[int(j)]
    ax, ay = float(a["pixel_coords"][0]), float(a["pixel_coords"][1])
    bx, by = float(b["pixel_coords"][0]), float(b["pixel_coords"][1])
    if abs(ax - bx) >= X_REL_THRESH:
        rel = "to the left of" if ax < bx else "to the right of"
    else:
        # Larger pixel y = lower in frame = closer to the camera.
        rel = "in front of" if ay > by else "behind"
    return "%s %s %s" % (obj_phrase(a), rel, obj_phrase(b))


def _enumeration_caption(objects: List[Dict], rng: np.random.Generator) -> str:
    if not objects:
        return "an empty scene"
    order = rng.permutation(len(objects))[:4]
    phrases = [obj_phrase(objects[int(k)]) for k in order]
    if len(phrases) == 1:
        listed = phrases[0]
    else:
        listed = ", ".join(phrases[:-1]) + " and " + phrases[-1]
    return "a scene with " + listed


def _count_caption(objects: List[Dict], rng: np.random.Generator) -> str:
    n = len(objects)
    if n == 0:
        return "an empty scene"
    o = objects[int(rng.integers(n))]
    plural = "object" if n == 1 else "objects"
    return "a scene with %s %s, including %s" % (num_word(n), plural, obj_phrase(o))


def synth_captions(objects: List[Dict], rng: np.random.Generator) -> List[str]:
    """The three caption variants (relation, enumeration, count/existence)."""
    return [
        _relation_caption(objects, rng),
        _enumeration_caption(objects, rng),
        _count_caption(objects, rng),
    ]


def prep_split(
    clevr_root: str,
    split: str,
    out_dir: str,
    limit: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, float]:
    scenes_path = os.path.join(clevr_root, "scenes", "CLEVR_%s_scenes.json" % split)
    with open(scenes_path, "r") as f:
        scenes = json.load(f)["scenes"]
    if limit is not None:
        scenes = scenes[: int(limit)]
    n = len(scenes)
    if n == 0:
        raise ValueError("no scenes found in %s" % scenes_path)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tier_idx"), exist_ok=True)

    images = np.memmap(
        os.path.join(out_dir, "images.u8"),
        dtype=np.uint8,
        mode="w+",
        shape=(n, IMG_SIZE, IMG_SIZE, 3),
    )
    meta_offsets = np.zeros(n + 1, dtype=np.int64)
    total_objects = 0
    total_dropped = 0

    with open(os.path.join(out_dir, "meta.jsonl"), "wb") as meta_f:
        for idx, scene in enumerate(scenes):
            img_path = os.path.join(clevr_root, "images", split, scene["image_filename"])
            with Image.open(img_path) as img:
                images[idx] = crop_resize(img)

            kept, dropped = visible_objects(scene)
            total_objects += len(scene["objects"])
            total_dropped += dropped
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, idx])))
            captions = synth_captions(kept, rng)

            record = {
                "idx": idx,
                "image_filename": scene["image_filename"],
                "tier": 0,
                "captions": captions,
                "n_objects": len(kept),
                "n_dropped": dropped,
                "objects": [
                    {
                        "shape": o["shape"],
                        "color": o["color"],
                        "size": o["size"],
                        "material": o["material"],
                        "pixel_coords": o["pixel_coords"],
                    }
                    for o in kept
                ],
            }
            line = (json.dumps(record) + "\n").encode("utf-8")
            meta_f.write(line)
            meta_offsets[idx + 1] = meta_offsets[idx] + len(line)

    images.flush()
    del images
    meta_offsets.tofile(os.path.join(out_dir, "meta_offsets.i64"))
    np.arange(n, dtype=np.int64).tofile(os.path.join(out_dir, "tier_idx", "tier0.i64"))

    drop_frac = total_dropped / max(1, total_objects)
    stats = {
        "n_images": float(n),
        "total_objects": float(total_objects),
        "total_dropped": float(total_dropped),
        "drop_frac": drop_frac,
    }
    print(
        "prep %s: %d images; dropped %d/%d objects outside crop (%.2f%%)"
        % (split, n, total_dropped, total_objects, 100.0 * drop_frac),
        flush=True,
    )
    return stats


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clevr-root", required=True, help="CLEVR_v1.0 directory")
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--out", required=True, help="output dataset directory")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    prep_split(args.clevr_root, args.split, args.out, args.limit, args.seed)


if __name__ == "__main__":
    main()
