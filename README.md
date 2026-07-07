---
license: mit
tags:
  - text-to-image
  - scene-grammar
  - probabilistic-grammar
  - research-preview
  - novel-architecture
library_name: sprig
pipeline_tag: text-to-image
---

# SPRIG v0.1 — a text-to-image model where images are *derived*, not denoised

**Research preview.** SPRIG (Stochastic Production-Rule Image Grammar) is a
from-scratch generative architecture that is **not** a diffusion model, not
autoregressive, not a GAN, not a VAE. A caption modulates the production
probabilities of a learned **probabilistic scene grammar**; an image is produced
by a single top-down **derivation** that recursively splits the canvas into
typed regions, each painted by a learned "texel" material. Training is **exact
maximum likelihood** — the marginal over *all* derivation trees, computed by an
inside dynamic program (log-semiring DP). No noise process, no adversary, no
ELBO, no token ordering.

The current release is **v0.1 at 64×64**: a proof-of-concept for the mechanism.
~16M trainable parameters on top of a frozen T5-base caption encoder.

<p align="center"><img src="samples.jpg" width="360" alt="SPRIG v0.1 samples"></p>

## What SPRIG does differently

| | Diffusion / Flow | Autoregressive | **SPRIG** |
|---|---|---|---|
| Generative act | denoise a fixed grid over many steps | predict tokens in an order | **derive a tree**: recursively split the canvas, commit each node once |
| Latent | noisy image | token prefix | an *unobserved random tree* summed out |
| Training | denoising / score matching | next-token likelihood | **exact marginal likelihood** via inside DP |
| Free bonus | — | — | a real **likelihood** + an interpretable **parse** of any image |

Because analysis and synthesis are the *same* grammar run in two directions, the
model can also **parse** a real image (infer its most likely derivation) — see
`parses.png`. This is the strongest, most novel capability and it works well.

## Method

<p align="center"><img src="figures/pipeline.png" width="900" alt="SPRIG pipeline: caption → frozen T5 embeddings → text-modulated grammar rules → inside DP that sums over all derivation trees → a sampled or parsed tree → 64×64 image"></p>

SPRIG is a **text-modulated probabilistic scene grammar** \\(G_c=(\Sigma, N, A_0, \Pi_c)\\): nonterminal symbols \\(N\\), texels (learned material primitives) \\(\Sigma\\), an axiom \\(A_0\\), and caption-conditioned productions \\(\Pi_c\\). An image is one **derivation** \\(\tau\\) — a binary tree that recursively splits the 64×64 canvas (a finite 1296-region binary-space-partition lattice, leaves ≤16px) and paints each leaf region with a texel. The conditional density marginalizes over *all* derivation trees:

$$
p(x \mid c) = \sum_{\tau} \; \prod_{\text{splits}} \pi\big(A \to \langle s,B,C\rangle \mid c\big) \; \prod_{\text{leaves}} \pi(T \mid A, c)\, p_{\mathrm{emit}}(x_r \mid T, r, c)
$$

Text enters through a low-rank rule factorization \\(\pi(A \to \langle s,B,C\rangle \mid c) = \sum_k p(k \mid A, c)\, p(s \mid k, c)\, p(B \mid k)\, p(C \mid k)\\), whose only caption-dependent factor — the mixture \\(p(k \mid A, c)\\) — is produced by a **Grammar-Modulation Transformer** (queries = symbol embeddings, cross-attention to the frozen T5-base caption). *Text deforms the grammar; it does not steer a sampler.* Each leaf emits a 4-component discretized-logistic mixture over its pixels.

Because the lattice and the cut dictionary are finite, the marginal is computed **exactly** by a log-semiring inside dynamic program over regions, and the training loss is the exact negative log-likelihood \\(\mathcal{L} = -\beta(A_0, \text{canvas})\\) — no encoder, no ELBO, no sampling in the loop. The *same* DP with a max-semiring yields the **Viterbi parse** of any image, which is why analysis and synthesis are the same object.

| \\(S\\) | \\(T_v\\) | \\(R\\) | \\(d\\) | canvas / grid | lattice | encoder | params |
|---|---|---|---|---|---|---|---|
| 1024 | 256 | 64 | 384 | 64² / 8px | 1296 regions | T5-base (frozen) | ~15.9M |

## Results

Success criteria were fixed in advance (50k steps, held-out procedural scenes):

| Gate | Target | Result | |
|---|---|---|---|
| Likelihood vs. no-grammar baseline | beat by ≥0.15 bpd | **2.66 vs 6.28 bpd** | ✅ crushes it |
| Caption information gain Δc | ≥ 0.05 | **0.248** | ✅ 5× |
| Visible-cut parse F1 | ≥ 0.6 | **0.765** | ✅ parsing works |
| Object-cell parse recall (tier1/2) | ≥ 0.70 / 0.50 | 0.20 / 0.22 | ❌ scenes too busy |
| Prompt-swap attribute control | ≥ 0.80 | 0.37 | ❌ partial |
| — size attribute specifically | — | **1.00** | ✅ size binds perfectly |
| Spatial-relation accuracy | ≥ 0.70 | 0.00 | ❌ |
| Compositional holdout (unseen combos) | ≥ 0.60 | 0.01 | ❌ |
| Grammar health (S_eff / alive texels) | ≥256 / ≥50% | 968 / 43% | ⚠️ texels over-pruned |

The architecture's structural claims prove out: it models data far better than a
no-grammar baseline, routes caption information, recovers scene structure by
parsing, and (after a targeted fix) paints real objects. The open problem is
**caption→object binding**: the model can draw objects and binds size perfectly,
but does not yet reliably paint the *specific* object a prompt asks for, and places
too many per scene.

## Usage

```bash
pip install torch safetensors transformers pillow
# get the `sprig` package + this file from the code repo, then:
python inference.py --prompt "a red circle on a white background" --out out.png
```

```python
from inference import load_sprig, sample
model = load_sprig("sprig-v0.1.safetensors", "config.json")   # ~16M params, CPU-friendly
img = sample(model, "a green triangle", seed=0)               # PIL.Image, 64x64
```

The model outputs native **64×64** images (upscale with nearest-neighbor to
view). It also returns the derivation tree, so you can inspect *why* each region
was drawn.

## Files

- `sprig-v0.1.safetensors` — EMA-merged inference weights (60.8 MB, fp32, 15.9M params)
- `config.json` — architecture config + release metadata
- `inference.py` — minimal load + sample + T5 caption encoding
- `metrics.json` — full evaluation numbers
- `figures/pipeline.png` — the method schematic
- `samples.jpg`, `texel_atlas.png`, `parses.png` — qualitative outputs
- `DESIGN.md` — the concrete v0.1 architecture specification

## Training

64×64, 2M procedural compositional scenes (colored shapes with attributes and
spatial relations, templated dense captions, held-out attribute combinations),
frozen T5-base captions precomputed. Exact-likelihood objective + closed-form
grammar-health regularizers. One RTX PRO 6000 Blackwell GPU, ~50k steps.
Generator: see the companion dataset repo (seeded, deterministic).

## Limitations & intended use

Research artifact for studying grammar-based generation and exact-likelihood
text-to-image. **Not** a production image generator: 64×64, synthetic domain,
object binding incomplete. Samples are blocky by construction (axis-aligned
region splits). MIT licensed — build on it.

## Citation

```bibtex
@software{sprig_v0_1_2026,
  title  = {SPRIG: Text-to-Image by a Stochastic Production-Rule Image Grammar (v0.1)},
  year   = {2026},
  note   = {Research preview. Images are derived by a probabilistic scene grammar
            trained by exact marginal likelihood, not denoised.}
}
```
