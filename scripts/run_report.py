"""Run the full SPRIG evaluation report on a checkpoint.

Usage: python scripts/run_report.py --ckpt PATH --data-dir ~/data/sprig/proc2d \
    --probe-data ~/data/sprig/probe --out DIR [--device cuda]

Trains the probe classifier once (cached at <probe-data>/probe_cnn.pt) and
then calls sprig.eval.report.run_report.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--probe-data", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--probe-epochs", type=int, default=10)
    args = ap.parse_args()

    probe_ckpt = None
    if args.probe_data:
        probe_data = os.path.expanduser(args.probe_data)
        probe_ckpt = os.path.join(probe_data, "probe_cnn.pt")
        if not os.path.exists(probe_ckpt):
            from sprig.eval import probe
            print("[report] training probe classifier on", probe_data)
            probe_ckpt = probe.train(probe_data, epochs=args.probe_epochs,
                                     device=args.device, ckpt_path=probe_ckpt)
        print("[report] probe:", probe_ckpt)

    from sprig.eval.report import run_report
    metrics = run_report(
        ckpt_path=os.path.expanduser(args.ckpt),
        data_dir=os.path.expanduser(args.data_dir),
        out_dir=os.path.expanduser(args.out),
        device=args.device,
        probe_ckpt=probe_ckpt,
    )
    gates = {k: v for k, v in metrics.items() if k.startswith("gate_")}
    print(json.dumps(gates, indent=1, default=str))
    print("[report] done ->", args.out)


if __name__ == "__main__":
    main()
