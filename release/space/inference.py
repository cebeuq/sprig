"""Minimal inference for SPRIG v0.1 — load the released safetensors and sample.

Requires the `sprig` package (https://github.com/  -- or the code repo bundled
with this release) plus torch, safetensors, transformers (for the T5 caption
encoder). The model itself is ~16M params and runs on CPU.

    python inference.py --weights sprig-v0.1.safetensors --config config.json \
        --prompt "a red circle on a white background" --out out.png

Programmatic:

    from inference import load_sprig, sample
    model = load_sprig("sprig-v0.1.safetensors", "config.json")
    img = sample(model, "a green triangle", seed=0)   # PIL.Image, 64x64
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

_T5 = None


def _t5_embed(prompt: str, device: str = "cpu"):
    """Encode a caption with frozen T5-base -> (emb [1,L,768] f16, len [1] i32)."""
    global _T5
    if _T5 is None:
        from transformers import T5EncoderModel, T5TokenizerFast
        tok = T5TokenizerFast.from_pretrained("google-t5/t5-base")
        enc = T5EncoderModel.from_pretrained("google-t5/t5-base").eval().to(device)
        _T5 = (tok, enc)
    tok, enc = _T5
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=64).to(device)
    with torch.no_grad():
        h = enc(**ids).last_hidden_state          # [1, L, 768]
    n = int(ids["attention_mask"].sum())
    return h[:, :n].to(torch.float16), torch.tensor([n], dtype=torch.int32, device=device)


def load_sprig(weights: str, config: str, device: str = "cpu"):
    from sprig.model.sprig import SPRIGModel, SPRIGConfig
    meta = json.loads(Path(config).read_text())
    fields = set(SPRIGConfig.__dataclass_fields__)
    cfg = SPRIGConfig(**{k: v for k, v in meta.get("model", {}).items() if k in fields})
    model = SPRIGModel(cfg)
    model.load_state_dict(load_file(weights), strict=False)
    model.tau.fill_(1.0)          # deployment temperature
    model.eta.fill_(0.0)          # untempered (exact) emissions
    return model.eval().to(device)


def sample(model, prompt: str, seed: int = 0, device: str = "cpu") -> Image.Image:
    emb, ln = _t5_embed(prompt, device)
    with torch.no_grad():
        imgs, _trees = model.sample(emb, ln, seed_struct=seed, seed_material=seed, n=1)
    return Image.fromarray(imgs[0].cpu().numpy().astype("uint8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="sprig-v0.1.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--upscale", type=int, default=6)
    args = ap.parse_args()
    model = load_sprig(args.weights, args.config, args.device)
    img = sample(model, args.prompt, args.seed, args.device)
    if args.upscale > 1:
        img = img.resize((64 * args.upscale, 64 * args.upscale), Image.NEAREST)
    img.save(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
