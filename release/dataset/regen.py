"""Regenerate the SPRIG procedural 2D scene dataset (deterministic).

  python regen.py --out ./sprig_data --n 2000000 --workers 30
  python regen.py --out ./sample --n 10000 --workers 8        # quick browsable set

Produces the memmap layout consumed by SPRIG (images.u8 + meta.jsonl + offsets +
tier_idx/). Same --seed-start => bit-identical output. Caption embeddings are a
separate step (frozen T5-base; see the code repo's sprig/data/embed_t5.py).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sprig.data.procgen import writer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--tier-mix", default="0.10,0.30,0.40,0.20")
    args = ap.parse_args()
    mix = [float(x) for x in args.tier_mix.split(",")]
    writer.write_dataset(args.out, n=args.n, seed_start=args.seed_start,
                         workers=args.workers, tier_mix=mix)
    print("wrote", args.n, "scenes ->", args.out)


if __name__ == "__main__":
    main()
