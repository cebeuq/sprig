#!/usr/bin/env python
"""Orchestrates procedural dataset generation (plan Part 1).

Splits (disjoint seed ranges, all derived from GLOBAL_SEED):
  train 2M / val 20k / test 20k / parse_eval 2k (tier-balanced) / val_fast 512
plus the probe dataset: 100k labeled single-object images that INCLUDE the
compositional holdout combos (leakage-safe by directory separation:
<out_root>/probe, never <out_root>/proc2d). Probe labels live in meta.jsonl.

The heavy lifting is done by the PROCGEN agent's writer
(sprig.data.procgen.writer); this script adapts to its keyword names via
signature inspection and fails loudly if no compatible entry point exists.

Usage:
  python scripts/gen_data.py --out-root ~/data/sprig --workers 30
  python scripts/gen_data.py --splits val_fast,probe --scale 0.001   # tiny dev
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

GLOBAL_SEED = 0
PROBE_SEED_START = 10_000_000  # far away from every proc2d split

# {blue triangle, red ring, green star, yellow cross} — fallback if the
# procgen vocab does not export its own holdout list.
FALLBACK_HOLDOUT = [("blue", "triangle"), ("red", "ring"),
                    ("green", "star"), ("yellow", "cross")]


def split_specs(scale: float = 1.0) -> Dict[str, Dict[str, Any]]:
    """Split sizes and disjoint seed ranges. `scale` shrinks everything
    proportionally for dev-sized datasets (each split keeps >= 1 sample)."""
    base: List[Tuple[str, int, Optional[List[float]]]] = [
        ("train", 2_000_000, None),
        ("val", 20_000, None),
        ("test", 20_000, None),
        ("parse_eval", 2_000, [0.25, 0.25, 0.25, 0.25]),  # tier-balanced
        ("val_fast", 512, None),
    ]
    specs: Dict[str, Dict[str, Any]] = {}
    start = 0
    for name, n, tier_weights in base:
        n = max(1, int(round(n * scale)))
        specs[name] = {
            "n": n,
            "seed_start": start,
            "seed_end": start + n,
            "tier_weights": tier_weights,
            "include_holdout": False,
        }
        start += n
    pn = max(1, int(round(100_000 * scale)))
    specs["probe"] = {
        "n": pn,
        "seed_start": PROBE_SEED_START,
        "seed_end": PROBE_SEED_START + pn,
        "tier_weights": [1.0, 0.0, 0.0, 0.0],  # single-object images
        "include_holdout": True,
    }
    return specs


def _accepted_kwargs(fn: Callable) -> Tuple[set, bool]:
    sig = inspect.signature(fn)
    names = set()
    has_var = False
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            has_var = True
        elif p.name != "self":
            names.add(p.name)
    return names, has_var


def _writer_fn() -> Callable:
    from sprig.data.procgen import writer
    for name in ("write_split", "write_dataset", "write", "generate_split"):
        fn = getattr(writer, name, None)
        if callable(fn):
            return fn
    raise RuntimeError(
        "sprig.data.procgen.writer exposes none of "
        "write_split/write_dataset/write/generate_split")


def run_split(name: str, spec: Dict[str, Any], out_root: Path,
              workers: int) -> None:
    fn = _writer_fn()
    out_dir = out_root / ("probe" if name == "probe" else "proc2d/" + name)
    out_dir.mkdir(parents=True, exist_ok=True)
    names, has_var = _accepted_kwargs(fn)
    cand: Dict[str, Any] = {
        "out_dir": str(out_dir), "output_dir": str(out_dir), "dst": str(out_dir),
        "n": spec["n"], "num_images": spec["n"], "n_images": spec["n"],
        "num_samples": spec["n"],
        "seed_start": spec["seed_start"],
        "seed_range": (spec["seed_start"], spec["seed_end"]),
        "global_seed": GLOBAL_SEED, "seed": GLOBAL_SEED,
        "num_workers": workers, "workers": workers, "n_workers": workers,
        "split": name,
        "tier_weights": spec["tier_weights"],
        "tier_mix": spec["tier_weights"],
        "include_holdout": spec["include_holdout"],
        "allow_holdout": spec["include_holdout"],
        "enforce_holdout": not spec["include_holdout"],
        "single_object": name == "probe",
    }
    kw = {k: v for k, v in cand.items()
          if (has_var or k in names) and v is not None}
    if not any(k in kw for k in ("out_dir", "output_dir", "dst")):
        raise RuntimeError(
            "writer entry point %s accepts no recognized output-dir kwarg "
            "(looked for out_dir/output_dir/dst; signature: %s)"
            % (fn.__name__, sorted(names)))
    print("[gen_data] %s: n=%d seeds=[%d,%d) -> %s"
          % (name, spec["n"], spec["seed_start"], spec["seed_end"], out_dir))
    fn(**kw)


# ---------------------------------------------------------------------------
# Probe generation. The proc2d writer/sampler enforce the compositional
# holdout, so the probe (which must INCLUDE the holdout combos; leakage-safe
# by directory separation) is generated here: tier-0-style single-object
# scenes with uniformly sampled attributes, written as images.u8 +
# labels.npy int64 [N, 2] (shape_idx, color_idx in canonical vocab order —
# the sprig/eval/probe.py dataset format) + meta.jsonl.
# ---------------------------------------------------------------------------

def gen_probe_local(out_dir: Path, n: int, seed_start: int) -> None:
    import numpy as np

    from sprig.data.procgen import sampler as S
    from sprig.data.procgen.render import render_scene
    from sprig.data.procgen.vocab import (
        COLOR_NAMES, HOLDOUT_COMBOS, SHAPES, SIZE_NAMES,
    )

    canvas = S.CANVAS
    out_dir.mkdir(parents=True, exist_ok=True)
    img_mm = np.memmap(str(out_dir / "images.u8"), dtype=np.uint8, mode="w+",
                       shape=(n, canvas, canvas, 3))
    labels = np.zeros((n, 2), dtype=np.int64)
    holdout = {tuple(h) for h in HOLDOUT_COMBOS}

    with open(out_dir / "meta.jsonl", "w") as mf:
        for i in range(n):
            rng = S.scene_rng(GLOBAL_SEED, seed_start + i)
            # Uniform attributes INCLUDING the holdout combos.
            color = COLOR_NAMES[int(rng.integers(len(COLOR_NAMES)))]
            shape = SHAPES[int(rng.integers(len(SHAPES)))]
            size = SIZE_NAMES[int(rng.integers(len(SIZE_NAMES)))]
            texture = "solid" if rng.random() < 0.4 else (
                ("striped", "dotted", "checker")[int(rng.integers(3))]
            )
            attrs = {"shape": shape, "color": color, "size": size,
                     "texture": texture}
            # Tier-0-style scene: grow a BSP tree, put the object in the most
            # central 16x16 cell (mirrors sampler._build_tier0 with the
            # holdout rejection removed).
            for _ in range(200):
                tree = S._grow(rng, (0, 0, canvas, canvas))
                cells = S._object_cells(tree)
                if cells:
                    break
            else:
                raise RuntimeError("probe: no object cell after retries")
            c = canvas / 2.0
            cell = min(
                cells,
                key=lambda l: ((l["rect"][0] + l["rect"][2]) / 2 - c) ** 2
                + ((l["rect"][1] + l["rect"][3]) / 2 - c) ** 2,
            )
            obj = S._place_object(rng, cell["rect"], attrs, centered=True)
            cell["obj"] = 0
            bg_name, bg_rgb = S._pick_bg(rng)
            S._apply_bg(tree, bg_rgb)
            scene = S.Scene(idx=i, tier=0, tree=tree, objects=[obj],
                            background=bg_name)
            img_mm[i] = render_scene(scene)
            labels[i, 0] = SHAPES.index(shape)
            labels[i, 1] = COLOR_NAMES.index(color)
            rec = {"idx": i, "seed": seed_start + i, "background": bg_name,
                   "holdout": (color, shape) in holdout}
            rec.update(attrs)
            mf.write(json.dumps(rec) + "\n")
    img_mm.flush()
    np.save(str(out_dir / "labels.npy"), labels)
    n_hold = sum(
        1 for i in range(n)
        if (COLOR_NAMES[labels[i, 1]], SHAPES[labels[i, 0]]) in holdout
    )
    print("[gen_data] probe: wrote %d images (%d holdout-combo) -> %s"
          % (n, n_hold, out_dir))


def run_probe(spec: Dict[str, Any], out_root: Path, workers: int) -> None:
    # The writer path cannot include holdout combos (the sampler rejects
    # them), so the probe is always generated locally.
    del workers
    gen_probe_local(out_root / "probe", spec["n"], spec["seed_start"])


def run_dev(out_dir: Path, n: int, workers: int) -> None:
    """--dev mode: one small self-contained split (images + meta + tiers) for
    local development; embeddings are added separately with embed_t5."""
    from sprig.data.procgen.writer import write_dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[gen_data] dev: n=%d -> %s" % (n, out_dir))
    write_dataset(str(out_dir), n, seed=GLOBAL_SEED, workers=workers)
    print("[gen_data] dev done; now run:\n"
          "  python -m sprig.data.embed_t5 --data-dir %s" % out_dir)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Generate SPRIG datasets")
    ap.add_argument("--out-root", default="~/data/sprig",
                    help="root data dir (default ~/data/sprig)")
    ap.add_argument("--splits",
                    default="train,val,test,parse_eval,val_fast,probe",
                    help="comma-separated subset of splits to generate")
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="scale all split sizes (e.g. 0.001 for tiny dev sets)")
    ap.add_argument("--dev", default=None, metavar="DIR",
                    help="write ONE small dev split (default 200 images) to "
                         "DIR and exit (ignores --splits/--out-root)")
    ap.add_argument("--dev-n", type=int, default=200,
                    help="number of dev images for --dev (default 200)")
    args = ap.parse_args(argv)

    if args.dev:
        run_dev(Path(args.dev).expanduser(), args.dev_n, max(1, args.workers))
        return

    out_root = Path(args.out_root).expanduser()
    specs = split_specs(args.scale)
    wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in specs]
    if unknown:
        raise SystemExit("unknown splits: %s (have %s)"
                         % (unknown, sorted(specs)))
    for name in wanted:
        if name == "probe":
            run_probe(specs[name], out_root, args.workers)
        else:
            run_split(name, specs[name], out_root, args.workers)
    print("[gen_data] done")


if __name__ == "__main__":
    main()
