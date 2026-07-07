"""Parallel pregeneration of the procedural dataset to raw memmaps.

Outputs in --out DIR:
  images.u8            uint8 memmap [N,64,64,3]
  meta.jsonl           one JSON object per line:
                       {idx, tier, caption, template_id, partial, objects, tree}
  meta_offsets.i64     int64 [N]: byte offset of the START of line i
  tier_idx/tier{0..3}.i64  int64 sorted idx arrays per tier (curriculum)

Embeddings are NOT written here — sprig/data/embed_t5.py does that later.

CLI:
  python -m sprig.data.procgen.writer --out DIR --n N --seed SEED \
      --tier-mix "0.1,0.3,0.4,0.2" --workers K

Workers own disjoint contiguous row ranges; each writes its rows of the image
memmap plus a meta/tier shard, and the parent concatenates shards in order.
Every sample is a pure function of (seed, idx), so the output is independent
of the worker count.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .captions import sample_caption
from .render import render_scene
from .sampler import DEFAULT_TIER_MIX, caption_rng, sample_scene

IMG_SHAPE = (64, 64, 3)


def _meta_part(out_dir: str, wid: int) -> str:
    return os.path.join(out_dir, "meta.part{:03d}.jsonl".format(wid))


def _tier_part(out_dir: str, wid: int) -> str:
    return os.path.join(out_dir, "tiers.part{:03d}.i64".format(wid))


def _generate_range(
    out_dir: str,
    n: int,
    seed: int,
    tier_mix: Sequence[float],
    lo: int,
    hi: int,
    wid: int,
    seed_start: int = 0,
) -> None:
    """Worker: generate rows [lo, hi) into the shared image memmap + shards.

    Row idx is generated from scene index seed_start + idx, so different
    splits get disjoint scene streams from disjoint seed ranges."""
    images = np.memmap(
        os.path.join(out_dir, "images.u8"), dtype=np.uint8, mode="r+",
        shape=(n,) + IMG_SHAPE,
    )
    tiers = np.empty(hi - lo, dtype=np.int64)
    with open(_meta_part(out_dir, wid), "w", encoding="utf-8") as f:
        for idx in range(lo, hi):
            sidx = seed_start + idx
            scene = sample_scene(seed, sidx, tier_mix=tier_mix)
            images[idx] = render_scene(scene)
            cap = sample_caption(scene, caption_rng(seed, sidx), mode="train")
            rec = {
                "idx": idx,
                "tier": scene.tier,
                "caption": cap.text,
                "template_id": cap.template_id,
                "partial": cap.partial,
                "objects": scene.objects,
                "tree": scene.tree,
            }
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
            tiers[idx - lo] = scene.tier
    images.flush()
    del images
    tiers.tofile(_tier_part(out_dir, wid))


def write_dataset(
    out_dir: str,
    n: int,
    seed: int = 0,
    tier_mix: Sequence[float] = DEFAULT_TIER_MIX,
    workers: int = 1,
    seed_start: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tier_idx"), exist_ok=True)
    workers = max(1, min(int(workers), n))

    # preallocate the image memmap
    images = np.memmap(
        os.path.join(out_dir, "images.u8"), dtype=np.uint8, mode="w+",
        shape=(n,) + IMG_SHAPE,
    )
    images.flush()
    del images

    # disjoint contiguous row ranges
    bounds = np.linspace(0, n, workers + 1).astype(np.int64)
    ranges: List[Tuple[int, int]] = [
        (int(bounds[w]), int(bounds[w + 1])) for w in range(workers)
    ]

    t0 = time.time()
    if workers == 1:
        _generate_range(out_dir, n, seed, tier_mix, 0, n, 0, seed_start)
    else:
        ctx = mp.get_context("spawn")
        procs = []
        for wid, (lo, hi) in enumerate(ranges):
            p = ctx.Process(
                target=_generate_range,
                args=(out_dir, n, seed, tuple(tier_mix), lo, hi, wid, seed_start),
            )
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        failed = [p.exitcode for p in procs if p.exitcode != 0]
        if failed:
            raise RuntimeError("worker(s) failed with exit codes {}".format(failed))

    # concatenate meta shards in order, recording line-start byte offsets
    offsets = np.empty(n, dtype=np.int64)
    tiers = np.empty(n, dtype=np.int64)
    pos = 0
    row = 0
    with open(os.path.join(out_dir, "meta.jsonl"), "wb") as out:
        for wid in range(workers):
            with open(_meta_part(out_dir, wid), "rb") as part:
                for line in part:
                    offsets[row] = pos
                    out.write(line)
                    pos += len(line)
                    row += 1
            tiers_part = np.fromfile(_tier_part(out_dir, wid), dtype=np.int64)
            lo, hi = ranges[wid]
            tiers[lo:hi] = tiers_part
            os.remove(_meta_part(out_dir, wid))
            os.remove(_tier_part(out_dir, wid))
    assert row == n, "meta line count {} != n {}".format(row, n)
    offsets.tofile(os.path.join(out_dir, "meta_offsets.i64"))
    for t in range(4):
        idxs = np.nonzero(tiers == t)[0].astype(np.int64)
        idxs.tofile(os.path.join(out_dir, "tier_idx", "tier{}.i64".format(t)))

    counts = [int((tiers == t).sum()) for t in range(4)]
    print(
        "wrote {} samples to {} in {:.1f}s (tier counts: {})".format(
            n, out_dir, time.time() - t0, counts
        )
    )


def _parse_tier_mix(s: str) -> Tuple[float, ...]:
    parts = tuple(float(v) for v in s.split(","))
    if len(parts) != 4 or min(parts) < 0 or sum(parts) <= 0:
        raise argparse.ArgumentTypeError(
            "--tier-mix must be 4 nonnegative comma-separated floats"
        )
    return parts


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Pregenerate the proc2d dataset.")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, required=True, help="number of samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--tier-mix", type=_parse_tier_mix,
        default=DEFAULT_TIER_MIX, help='e.g. "0.1,0.3,0.4,0.2"',
    )
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="scene-index offset (disjoint ranges per split)")
    args = ap.parse_args(argv)
    write_dataset(args.out, args.n, args.seed, args.tier_mix, args.workers,
                  seed_start=args.seed_start)


if __name__ == "__main__":
    main(sys.argv[1:])
