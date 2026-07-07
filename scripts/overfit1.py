#!/usr/bin/env python
"""G1 gate (DESIGN.md section 9): overfit ONE image with the tiny config.

Trains 2k steps on a single fixed image, prints the bpd trajectory, and
exits non-zero unless bpd is decreasing and ends below the 1.0 gate.
Also prints a MAP-parse summary at the end (non-fatal if parsing fails).

Usage:
  python scripts/overfit1.py --data-dir ~/data/sprig/proc2d/val_fast \\
      [--config configs/smoke.yaml] [--steps 2000] [--index 0] [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BPD_GATE = 1.0


def prepare_cfg(config_path: str, steps: int, batch_size: int):
    import train
    cfg = train.load_config(config_path)
    cfg["train"].update({
        "total_steps": int(steps),
        "batch_size": int(batch_size),
        "warmup_steps": max(1, min(50, steps // 10)),
        "lr_decay_steps": int(steps),
        "tau_steps": max(1, steps // 2),
        "eta_final_anneal_steps": max(1, steps // 10),
    })
    cfg["train"].pop("tier_schedule", None)  # fixed subset: uniform sampling
    cfg["data"]["num_workers"] = 0
    big = 10 ** 9
    cfg["eval"].update({"val_fast_every": big, "parse_every": big,
                        "full_every": big, "scalars_every": 100})
    cfg["checkpoint"].update({"every_steps": big, "every_minutes": big})
    train.validate_config(cfg)
    return cfg


def check_trajectory(hist: List, gate: float, label: str) -> bool:
    steps_ = [s for s, _ in hist]
    bpds = [b for _, b in hist]
    k = max(1, len(bpds) // 20)
    first = sum(bpds[:k]) / k
    last = sum(bpds[-k:]) / k
    for s, b in zip(steps_, bpds):
        if s % 100 == 0 or s == steps_[-1]:
            print("  step %6d  bpd %.4f" % (s, b))
    print("%s: first-5%% mean bpd = %.4f, last-5%% mean bpd = %.4f, gate < %.2f"
          % (label, first, last, gate))
    ok = last < first and last < gate
    print("%s: %s" % (label, "PASS" if ok else "FAIL"))
    return ok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G1: overfit one image")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default=str(REPO / "configs" / "smoke.yaml"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--run-dir", default=None)
    args = ap.parse_args(argv)

    import train
    cfg = prepare_cfg(args.config, args.steps, args.batch_size)
    base = train.build_dataset(cfg, args.data_dir, train=False)
    ds = train.FixedSubset(base, [args.index], virtual_len=1024)

    print("[overfit1] training %d steps on image idx=%d from %s"
          % (args.steps, args.index, args.data_dir))
    res = train.train_loop(cfg, ds, steps=args.steps, run_dir=args.run_dir,
                           device=args.device)
    ok = check_trajectory(res["bpd_history"], BPD_GATE, "G1")

    # MAP-parse stability probe (informational, never fails the gate by crash)
    try:
        import torch
        model = res["model"]
        collate = train._default_collate(ds)
        with torch.no_grad(), train.report_mode(model):
            b = train.move_batch(collate([ds[0]]),
                                 next(model.parameters()).device.type)
            roots = model.map_parse(b["image"], b["emb"], b["emb_len"])
            root = roots[0] if isinstance(roots, (list, tuple)) else roots

            def count(node):
                kids = getattr(node, "children", None) or []
                return 1 + sum(count(c) for c in kids)

            print("[overfit1] final MAP parse: %d nodes, root rect %s"
                  % (count(root), tuple(root.rect)))
    except Exception as e:
        print("[overfit1] MAP parse probe failed (non-fatal): %r" % e)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
