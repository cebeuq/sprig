"""Measure peak CUDA memory + step time of loss/backward at several batch sizes.

Usage: python scripts/mem_probe.py [--batches 16,32,64,128] [--leaf-chunk 8]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sprig.model.sprig import SPRIGModel, SPRIGConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", default="16,32,64,128")
    ap.add_argument("--leaf-chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=2,
                    help="untimed steps before measuring (torch.compile of the "
                         "emission kernels happens here); peak memory stats are "
                         "reset after warmup so the report is steady-state")
    ap.add_argument("--autocast", action="store_true")
    ap.add_argument("--no-hinge-cg", action="store_true",
                    help="disable create_graph double-backward for hinges")
    args = ap.parse_args()

    cfg = SPRIGConfig(S=1024, R=64, T_v=256, d=384, leaf_chunk=args.leaf_chunk,
                      hinge_create_graph=not args.no_hinge_cg)
    model = SPRIGModel(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print("params: %.1fM" % (n_params / 1e6))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for bs in [int(x) for x in args.batches.split(",")]:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            img = torch.randint(0, 255, (bs, 64, 64, 3), dtype=torch.uint8, device="cuda")
            emb = torch.randn(bs, 24, 768, dtype=torch.float16, device="cuda")
            el = torch.full((bs,), 24, dtype=torch.int32, device="cuda")
            batch = {"image": img, "emb": emb, "emb_len": el,
                     "tier": torch.zeros(bs, dtype=torch.int8, device="cuda"),
                     "idx": torch.arange(bs, device="cuda")}

            def one_step():
                if args.autocast:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss, metrics = model.loss(batch)
                else:
                    loss, metrics = model.loss(batch)
                loss.backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
                return loss

            for _ in range(args.warmup):
                loss = one_step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(args.steps):
                torch.cuda.synchronize()
                t0 = time.time()
                loss = one_step()
                torch.cuda.synchronize()
                times.append(time.time() - t0)
            peak = torch.cuda.max_memory_allocated() / 2**30
            print("B=%4d  peak %.1f GiB  step %.2fs (last %.2fs)  loss %.3f"
                  % (bs, peak, sum(times) / len(times), times[-1], float(loss)))
        except torch.OutOfMemoryError:
            print("B=%4d  OOM" % bs)
            del batch
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
