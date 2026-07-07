# SPRIG — images are *derived*, not denoised

**SPRIG** (Stochastic Production-Rule Image Grammar) is a from-scratch text-to-image
architecture that is **not** a diffusion model, not autoregressive, not a GAN, not a VAE.
A caption modulates the production probabilities of a learned **probabilistic scene
grammar**; an image is produced by a single top-down **derivation** that recursively
splits the canvas into typed regions, each painted by a learned "texel" material.
Training is **exact maximum likelihood** — the marginal over *all* derivation trees,
computed by an inside dynamic program (log-semiring DP). No noise process, no adversary,
no ELBO, no token ordering.

Because analysis and synthesis are the *same* grammar run in two directions, the model
can also **parse** a real image (infer its most likely derivation) — its strongest,
most novel capability.

This repo is the **v0.1** proof-of-concept at 64×64 (~16M trainable params on top of a
frozen T5-base caption encoder).

- 🤗 **Model:** [huggingface.co/cebeuq/sprig](https://huggingface.co/cebeuq/sprig)
- 🎮 **Live demo:** [huggingface.co/spaces/cebeuq/sprig-demo](https://huggingface.co/spaces/cebeuq/sprig-demo)
- 📖 **Method + config + contracts:** [`DESIGN.md`](DESIGN.md)

## How it works (one paragraph)

An image `x` under caption `c` is defined by a text-modulated probabilistic scene grammar
whose conditional density marginalizes over every derivation tree `τ`:
`p(x|c) = Σ_τ Π_splits π(A→⟨s,B,C⟩|c) Π_leaves π(T|A,c) p_emit(x_r|T,r,c)`.
The caption enters only through a low-rank rule factorization
`π(A→⟨s,B,C⟩|c) = Σ_k p(k|A,c) p(s|k,c) p(B|k) p(C|k)`, whose caption-dependent mixture
`p(k|A,c)` is produced by a Grammar-Modulation Transformer (queries = symbol embeddings,
cross-attention to the caption). Because the region lattice and cut dictionary are finite,
the marginal is computed **exactly** by a bottom-up inside DP over regions, and the loss is
the exact NLL `L = -β(A₀, canvas)`. The same DP with a max-semiring gives the Viterbi parse.

## Honest status (v0.1, 50k steps)

Passes **1 of 5** pre-registered proof-of-concept gates — and the failures are localized
and understood, not diffuse:

| Gate | Result |
|---|---|
| Likelihood vs. no-grammar baseline | **2.66 vs 6.28 bpd** ✅ |
| Caption information gain Δc | **0.248** ✅ (5×) |
| Visible-cut parse F1 | **0.765** ✅ (parsing works) |
| Object-cell parse recall | 0.20 / 0.22 ❌ |
| Prompt-swap attribute control | 0.37 ❌ (size binds perfectly: 1.00) |
| Compositional holdout | 0.01 ❌ |

The architecture's structural claims hold — it beats a no-grammar baseline decisively,
routes caption information, and recovers scene structure by parsing. The open problem is
**caption→object binding** (the model draws objects and binds *size* perfectly, but does
not yet reliably paint the *specific* object a prompt asks for).

## Quick start (CPU)

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt torch
.venv/bin/python -m pytest tests/ -x -q     # unit tests incl. brute-force inside-DP proof
.venv/bin/python scripts/overfit1.py --data-dir local_data/dev   # overfit-one-image sanity
```

Run the released model:

```bash
pip install torch safetensors transformers pillow
# download sprig-v0.1.safetensors + config.json from the HF model repo, then:
python release/model/inference.py --prompt "a red circle on a white background" --out out.png
```

## Repository layout

```
sprig/            core package
  dp/             region lattice + exact inside DP (log-semiring), Viterbi parse
  model/          grammar-modulation transformer, texel atlas, discretized-logistic emission
  data/           procedural scene generator + dataset + T5 embedding precompute
  eval/           metrics, parse diagnostics, probe classifier, report
train.py          training entrypoint (config-driven, resumable)
configs/          main64 / smoke / clevr-ft
scripts/          data gen, overfit gates, release export, perf probes
tests/            full suite (the brute-force DP-vs-enumeration test is the correctness anchor)
release/          HF model card, dataset generator, and Gradio demo app
DESIGN.md         the binding v0.1 spec: math, module contracts, config, milestone gates
```

## Training

64×64, 2M procedural compositional scenes (colored shapes with attributes and spatial
relations, dense templated captions, held-out attribute combinations), frozen T5-base
captions precomputed. Exact-likelihood objective + closed-form grammar-health
regularizers. Single modern GPU, ~50k steps.

```bash
python scripts/gen_data.py --out-root ./data          # generate scenes + splits
python -m sprig.data.embed_t5 --data-dir ./data/proc2d/train   # precompute T5 embeddings
python train.py --config configs/main64.yaml --run-dir ./runs/main64 --resume auto
```

The `infra/` scripts (rsync deploy, crash-resilient run loop, status) target a remote GPU
box via `$SPRIG_HOST`; set it to your own host.

## License

MIT — see [`LICENSE`](LICENSE).
