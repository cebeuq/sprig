"""Export a slim, inference-only SPRIG release checkpoint.

Takes a training checkpoint (model + optimizer + EMA + ...), overlays the EMA
shadow weights (eval-only, best for release), and writes:
  - sprig-v0.1.safetensors : model weights only (EMA-merged), fp32
  - config.json            : the SPRIGConfig fields + release metadata

Usage:
  python scripts/export_release.py --ckpt ~/runs/sprig/main64b/last.pt \
      --out-dir ~/release_model --name sprig-v0.1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="sprig-v0.1")
    args = ap.parse_args()

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    ck = torch.load(Path(args.ckpt).expanduser(), map_location="cpu", weights_only=False)

    # Merge EMA shadow onto the raw model state (EMA is eval-only / release-preferred).
    state = dict(ck["model"])
    ema = ck.get("ema")
    if isinstance(ema, dict) and isinstance(ema.get("shadow"), dict):
        state.update(ema["shadow"])
        merged = "ema"
    else:
        merged = "raw"

    # safetensors requires plain, contiguous, non-shared CPU tensors.
    clean = {}
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue
        clean[k] = v.detach().to(torch.float32).contiguous().clone()

    st_path = out / (args.name + ".safetensors")
    save_file(clean, str(st_path), metadata={"format": "pt", "weights": merged})

    cfg = ck.get("config", {})
    mcfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    meta = {
        "architecture": "SPRIG",
        "version": "0.1",
        "weights_file": args.name + ".safetensors",
        "weights_merged": merged,
        "trained_steps": int(ck.get("step", 0)),
        "canvas_px": mcfg.get("canvas", 64),
        "grid_px": mcfg.get("grid", 8),
        "model": mcfg,
        "note": ("Inference-only export: EMA-merged weights, buffers "
                 "eta/tau are re-initialised at load (tau=1.0, eta=0.0)."),
    }
    with open(out / "config.json", "w") as f:
        json.dump(meta, f, indent=2)

    n = sum(t.numel() for t in clean.values())
    print("wrote %s (%.1f MB, %d tensors, %.1fM params, %s weights)" % (
        st_path, st_path.stat().st_size / 2**20, len(clean), n / 1e6, merged))
    print("wrote", out / "config.json")


if __name__ == "__main__":
    main()
