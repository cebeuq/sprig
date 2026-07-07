---
license: mit
task_categories:
  - text-to-image
tags:
  - synthetic
  - compositional
  - procedural-generation
  - scene-graph
pretty_name: SPRIG Procedural 2D Scenes
---

# SPRIG Procedural 2D Scenes (generator)

The training data for [SPRIG v0.1](https://huggingface.co/cebeuq/sprig) —
a **seeded, deterministic generator** for compositional 2D scenes. Rather than
ship ~90 GB of images, this repo ships the code: the full 2-million-image
dataset (and any size you want) regenerates bit-identically in well under an
hour on a multicore machine.

## What the scenes are

Each scene is produced by **first sampling a ground-truth binary-space-partition
region tree**, then decorating its leaves — so every image comes with an exact
structural label (the tree doubles as a parse target). Objects are simple
colored shapes with attributes and spatial relations; captions are templated
and dense.

- **Shapes** (8): circle, square, triangle, rectangle, diamond, star, cross, ring
- **Colors** (8), **sizes** (2), **textures** (4: solid/striped/dotted/checker),
  **backgrounds** (5, incl. sky/ground splits)
- **Difficulty tiers 0–3**: single object → two objects with a relation →
  3–5 objects with attributes → containment/nesting
- **Held-out combinations** (never in training): `blue triangle`, `red ring`,
  `green star`, `yellow cross` — for measuring compositional generalization
- **Dense captions**: 10 templates (attributes, relations, counts, containment,
  inverted order) + 15% partial captions

Per-sample record: the image (64×64 RGB), the caption(s), the tier, the object
list with bounding boxes, and the ground-truth region tree.

## Regenerate

```bash
pip install torch numpy pillow
# 2M train + val/test/parse-eval splits, deterministic, ~30 workers:
python regen.py --out ./sprig_data --n 2000000 --workers 30
# a quick 10k sample to browse:
python regen.py --out ./sample --n 10000 --workers 8
```

Captions are embedded with frozen **T5-base** for training (see the code repo's
`embed_t5.py`); the images/captions/trees above are encoder-independent.

## Format

Memmap layout per split: `images.u8` [N,64,64,3], `meta.jsonl` (+`meta_offsets.i64`)
with caption / tier / objects / GT tree, `tier_idx/`. Fully documented in
`sprig/data/dataset.py`.

## License

MIT. Synthetic data, no external images.
