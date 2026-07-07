"""SPRIG v0.1 interactive demo.

Two things you can do:
  1. Prompt -> image: derive a 64x64 image from a caption.
  2. Image -> parse: infer the most likely grammar derivation of an image and
     overlay its region tree (SPRIG's strongest, most novel capability).

The model is ~16M params and runs on CPU. Weights are pulled from the companion
model repo on first load.
"""
from __future__ import annotations

import os

# --- Work around a persistent gradio_client schema bug (present in gradio 4.44
# through 5.9): building the /info API schema crashes with
# "TypeError: argument of type 'bool' is not iterable" when a JSON schema node
# is a bare boolean (e.g. additionalProperties: true/false). The frontend calls
# that endpoint on load, so the crash surfaces as "No API found". We neutralise
# it by making the schema->python-type helpers tolerate non-dict nodes.
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    return _orig_get_type(schema) if isinstance(schema, dict) else "Any"
_gcu.get_type = _safe_get_type

_orig_j2pt = _gcu._json_schema_to_python_type
def _safe_j2pt(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_j2pt(schema, defs)
_gcu._json_schema_to_python_type = _safe_j2pt
# --- end workaround

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL_REPO = os.environ.get("SPRIG_MODEL_REPO", "cebeuq/sprig")
_MODEL = None


def _load():
    global _MODEL
    if _MODEL is None:
        from inference import load_sprig
        # Prefer weights bundled in the Space (self-contained, no cross-repo auth);
        # fall back to downloading from the model repo if they aren't present.
        w, c = "sprig-v0.1.safetensors", "config.json"
        if not (os.path.exists(w) and os.path.exists(c)):
            from huggingface_hub import hf_hub_download
            w = hf_hub_download(MODEL_REPO, "sprig-v0.1.safetensors")
            c = hf_hub_download(MODEL_REPO, "config.json")
        _MODEL = load_sprig(w, c, device="cpu")
    return _MODEL


def _up(img: Image.Image, k: int = 5) -> Image.Image:
    return img.resize((img.width * k, img.height * k), Image.NEAREST)


def generate(prompt: str, seed: int):
    from inference import sample
    model = _load()
    return _up(sample(model, prompt or "a red circle", int(seed)))


def parse_image(img: Image.Image):
    """Overlay the MAP derivation's region rectangles on the input image."""
    model = _load()
    im = img.convert("RGB").resize((64, 64), Image.LANCZOS)
    arr = torch.from_numpy(np.asarray(im)).unsqueeze(0)
    emb = torch.zeros(1, 1, 768, dtype=torch.float16)   # unconditional parse
    ln = torch.ones(1, dtype=torch.int32)
    with torch.no_grad():
        nodes = model.map_parse(arr, emb, ln)
    canvas = _up(im, 5).convert("RGB")
    d = ImageDraw.Draw(canvas)

    def walk(n, depth=0):
        r = getattr(n, "rect", None)
        if r is not None:
            x0, y0, x1, y1 = [v * 5 for v in r]
            col = [(255, 80, 80), (80, 160, 255), (80, 220, 120), (240, 200, 60)][depth % 4]
            d.rectangle([x0, y0, x1 - 1, y1 - 1], outline=col, width=2)
        for c in (getattr(n, "children", None) or []):
            walk(c, depth + 1)

    for n in (nodes if isinstance(nodes, list) else [nodes]):
        walk(n)
    return canvas


EXAMPLES = ["a red circle on a white background", "a green triangle",
            "a blue square", "a yellow star on a black background",
            "a purple cross", "a large orange diamond"]

with gr.Blocks(title="SPRIG v0.1") as demo:
    gr.Markdown(
        "# SPRIG v0.1 — images are *derived*, not denoised\n"
        "A text-to-image **scene grammar** (not diffusion/AR/GAN). Generation is one "
        "top-down derivation; the same grammar can also **parse** an image. "
        "Research preview at 64×64 — see the model card for the honest scorecard. "
        "Object binding is still partial; **parsing** is the strong suit.")
    with gr.Tab("Prompt → image"):
        with gr.Row():
            p = gr.Textbox(label="Prompt", value=EXAMPLES[0])
            s = gr.Slider(0, 999, value=0, step=1, label="Seed")
        go = gr.Button("Derive")
        out = gr.Image(label="64×64 (nearest-neighbor upscaled)")
        gr.Examples(EXAMPLES, inputs=p)
        go.click(generate, [p, s], out)
    with gr.Tab("Image → parse"):
        gr.Markdown("Upload an image; SPRIG infers its most likely derivation and "
                    "overlays the region tree (color = depth).")
        inp = gr.Image(type="pil", label="Input")
        pbtn = gr.Button("Parse")
        pov = gr.Image(label="MAP derivation overlay")
        pbtn.click(parse_image, inp, pov)

if __name__ == "__main__":
    # SSR (Gradio 5's experimental server-side rendering) breaks routing on HF
    # Spaces here ("No API found"); the classic SPA serve is what we want.
    demo.launch(ssr_mode=False)
