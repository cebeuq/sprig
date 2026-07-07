"""Generate a sample grid from a live checkpoint without touching the trainer.

Usage: python scripts/sample_now.py --ckpt ~/runs/sprig/main64/last.pt \
    --bank ~/data/sprig/t5/prompt_bank.pt --out samples_now.jpg \
    [--prompts 0,8,9,20] [--seeds 4] [--device cuda] [--ema]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sprig.model.sprig import SPRIGModel, SPRIGConfig
from sprig.eval.report import save_grid_jpeg
from sprig.eval.prompts import PROMPTS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="0,3,8,9,11,20,25,17")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--ema", action="store_true", help="use EMA shadow weights")
    ap.add_argument("--tau", type=float, default=None,
                    help="override rule temperature (deployment = 1.0)")
    args = ap.parse_args()

    ckpt = torch.load(Path(args.ckpt).expanduser(), map_location="cpu", weights_only=False)
    mcfg = ckpt["config"]["model"]
    fields = set(SPRIGConfig.__dataclass_fields__.keys())
    cfg = SPRIGConfig(**{k: v for k, v in mcfg.items() if k in fields})
    model = SPRIGModel(cfg)
    state = dict(ckpt["model"])
    if args.ema and "ema" in ckpt:
        ema = ckpt["ema"]
        state.update(ema.get("shadow", ema) if isinstance(ema, dict) else ema)
    model.load_state_dict(state, strict=True)
    model.eval().to(args.device)
    if args.tau is not None:
        model.tau.fill_(args.tau)
    step = ckpt.get("step", "?")

    bank = torch.load(Path(args.bank).expanduser(), map_location="cpu", weights_only=False)
    emb_all, len_all = bank["emb"], bank["emb_len"]

    idxs = [int(i) for i in args.prompts.split(",")]
    rows = []
    with torch.no_grad():
        for pi in idxs:
            emb = emb_all[pi:pi + 1].to(args.device)
            el = len_all[pi:pi + 1].to(torch.int32).to(args.device)
            row = []
            for s in range(args.seeds):
                imgs, _trees = model.sample(emb, el, seed_struct=1000 + s,
                                            seed_material=2000 + s, n=1)
                row.append(imgs[0].cpu().numpy())
            rows.append(row)
            print("prompt %2d: %s" % (pi, PROMPTS[pi]))
    save_grid_jpeg(rows, args.out)
    print("step %s -> %s (%d prompts x %d seeds)" % (step, args.out, len(idxs), args.seeds))


if __name__ == "__main__":
    main()
