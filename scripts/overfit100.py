#!/usr/bin/env python
"""G2 gate (DESIGN.md section 9): overfit 100 captioned images.

Trains the tiny config on 100 fixed images, prints the bpd trajectory,
caption information gain delta_c = bpd(x, null) - bpd(x, c), and the alive
texel fraction. Exits non-zero unless bpd is decreasing AND delta_c > 0 AND
texels alive > 50%.

Usage:
  python scripts/overfit100.py --data-dir ~/data/sprig/proc2d/val_fast \\
      [--config configs/smoke.yaml] [--steps 4000] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from overfit1 import check_trajectory, prepare_cfg  # noqa: E402

BPD_GATE = 2.0  # looser than G1: 100 images, tiny model
N_IMAGES = 100


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G2: overfit 100 images")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "smoke.yaml"))
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args(argv)

    import train
    cfg = prepare_cfg(args.config, args.steps, args.batch_size)
    base = train.build_dataset(cfg, args.data_dir, train=False)
    n = min(N_IMAGES, len(base))
    ds = train.FixedSubset(base, list(range(n)), virtual_len=4096)

    print("[overfit100] training %d steps on %d images from %s"
          % (args.steps, n, args.data_dir))
    res = train.train_loop(cfg, ds, steps=args.steps, run_dir=args.run_dir,
                           device=args.device)
    model = res["model"]
    device = next(model.parameters()).device.type
    ok_bpd = check_trajectory(res["bpd_history"], BPD_GATE, "G2/bpd")

    import torch

    # delta_c over the fixed 100 images (report mode, eta=0)
    ok_dc = False
    try:
        dc, margin = train.delta_c_and_margin(
            model, train.FixedSubset(base, list(range(n))), cfg, device,
            n, int(cfg["eval"].get("eval_batch_size", 16)))
        print("G2/delta_c = %.5f bits/dim (gate > 0), swap margin = %.3f"
              % (dc, margin))
        ok_dc = dc > 0.0
    except Exception as e:
        print("G2/delta_c FAILED to compute: %r" % e)
    print("G2/delta_c: %s" % ("PASS" if ok_dc else "FAIL"))

    # texel aliveness from posterior usage on one batch
    ok_tex = False
    try:
        collate = train._default_collate(ds)
        b = train.move_batch(
            collate([base[i] for i in range(min(n, 32))]), device)
        usage = model.posterior_usage(b["image"], b["emb"], b["emb_len"])
        tu = usage["texel_usage"].detach().float().cpu().flatten()
        tu = tu / tu.sum().clamp_min(1e-12)
        T_v = int(cfg["model"]["T_v"])
        thr = float(cfg["train"].get("texel_dead_frac", 0.1)) / T_v
        alive = float((tu >= thr).float().mean())
        print("G2/texels alive = %.1f%% (gate > 50%%)" % (100 * alive))
        ok_tex = alive > 0.5
    except Exception as e:
        print("G2/texel usage FAILED to compute: %r" % e)
    print("G2/texels: %s" % ("PASS" if ok_tex else "FAIL"))

    ok = ok_bpd and ok_dc and ok_tex
    print("G2 overall: %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
