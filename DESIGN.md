# SPRIG v0.1 — Design & Contracts (single source of truth)

Reference docs: full architecture spec `~/workspace/SPRIG-C-architecture.md`; project plan `~/.claude/plans/ok-now-i-ltierally-calm-sifakis.md` (data/eval/infra details).
v0.1 = finite-lattice Stage A at 64x64. No continuous cuts, no morphogen, no occlusion masks, no DINO channel.

## 1. Repo layout & file ownership

```
sprig/data/procgen/{vocab,sampler,captions,render,writer}.py   # owner: PROCGEN agent
sprig/data/{dataset,embed_t5}.py, sprig/data/clevr/prep.py     # owner: DATA agent
sprig/dp/{lattice,inside}.py                                   # owner: DP agent
sprig/model/{gmt,atlas,dl,sprig}.py                            # owner: MODEL agent
sprig/eval/{baseline_pixmix,tree_metrics,color_checks,probe,prompts,report,monitors}.py  # owner: EVAL agent
train.py, configs/*.yaml, infra/*.sh                           # owner: HARNESS agent
tests/test_<module>_*.py                                       # each agent writes tests for its own modules
```
Python 3.12, torch. No deps beyond: torch, numpy, pillow, transformers, sentencepiece, protobuf, tensorboard, matplotlib, pyyaml, tqdm, einops, pytest. Everything must run CPU-only for tests (no `.cuda()` hardcoded; device from config).

## 2. Global config (configs/main64.yaml values)

- Canvas 64x64 RGB. Grid stride g=8 px → 8x8 cells.
- S=1024 nonterminal symbols, R=64 rule components, T_v=256 texels, d=384 model width.
- Caption: precomputed T5-base token embeddings [L,768], L≤64.
- Leaf-eligible regions: both sides ≤ 16 px. Regions with any side > 16 px MUST expand. 8x8 regions MUST terminate.
- Batch 256. AdamW: lr 3e-4 for embedding tables (E_N, E_T, V, W, P_T, cut-type tables), 1e-4 for GMT + atlas renderer + Φ; betas (0.9, 0.95); cosine decay to 10% over 250k steps, 2k warmup; grad clip 1.0; EMA 0.9999 (eval only).
- Rule-logit temperature τ_ann: 2.0 → 1.0 linearly over first 50k steps (divide rule/termination logits by τ_ann).
- 10% of steps: caption replaced by null embedding (dataloader-level).
- bf16 autocast for matmuls; ALL logsumexp/log-domain accumulation in fp32. Emission DL log-scales clamped [-7, 2].

## 3. Lattice (sprig/dp/lattice.py) — DP agent

Region = axis-aligned rect with corners on the 8-px grid, i.e. cell-interval pair. All (C(9,2))²=1296 rects are regions. Precompute once (pure function of canvas/grid config, cache to `.pt`):
- `regions: int32 [N_reg, 4]` (x0,y0,x1,y1 in px), `region_id` lookup dict.
- Cuts: for each region, every interior grid line on each axis is a valid cut → children are regions. Global **cut-type vocabulary** for parameter tying: (axis ∈ {H,V}) × (relative offset bucket ∈ 7 buckets, nearest of {1/8..7/8}) = 14 cut types. Per region: `cut_list` of (cut_type_id, child_lo_id, child_hi_id). child_lo = lesser-coordinate child (left for V, top for H) — this ordering is what lets the grammar express left/right, above/below.
- Level order: by cell area ascending (levels 1..64). Per level: flattened index tensors `parent_ids [M]`, `cut_type [M]`, `child_lo [M]`, `child_hi [M]` for vectorized DP.
- `leaf_mask [N_reg]` (both sides ≤16), `must_terminate [N_reg]` (8x8), `must_expand [N_reg]` (any side >16), `phi_geom [N_reg, 64]` Fourier features of (log-area, log-aspect, center-x, center-y).
- Leaf shape groups: leaf regions grouped by (h,w) ∈ {8,16}² for batched emission scoring.

## 4. Model (sprig/model/) — MODEL agent

**gmt.py — GrammarModulationTransformer.** Symbol embeddings E_N [S,d] (queries; no symbol-symbol self-attention). 4 blocks: {cross-attn to projected caption tokens (768→d, key_padding_mask from emb_len), FFN d→4d→d, pre-LN}. Output H [B,S,d]. Heads:
- `U = H @ W_u` → p(k|A,c) logits [B,S,R].
- Termination: MLP([H_A ; phi_geom(r)]) → logit per (B, region, S) — computed lazily as MLP over H then dot with a geometry projection: implement as `term_logit[b,r,A] = MLP_h(H)[b,A,:] · MLP_g(phi_geom)[r,:] + bias_A` (factorized, cheap).
- Cut-type distribution p(s|k,c): component embeddings e_k [R,d] cross-attend once to caption → logits [B,R,14]; per region, mask to valid cut types and renormalize (log-softmax over the masked set). NOTE: multiple concrete cuts in a region can share a cut type — split p(s|k,c) mass uniformly across same-type concrete cuts (add -log(count) correction, precomputed per region).
- Terminal texel prior: p(T|A,c) = Σ_k p(k|A,c) softmax(P_T[k]) with static P_T [R,T_v].
- Children: static V, W [R,S]: p(B|k)=softmax(V[k]), p(C|k)=softmax(W[k]). (v0.1: no q(B) coupling.)
- Illumination field Φ(c): mean-pooled caption → MLP → deconv to [B,8,16,16], bilinear-resampled at leaf positions, FiLM (scale,shift per DL-mean channel group) on emission means.

**atlas.py — TexelAtlas.** E_T [T_v,d]. Renderer: per texel, cross-attn(E_T row → caption, 2 blocks width 256) → seed [256,4,4] → 2× (conv + 2x nearest-upsample) → atlas [B,T_v,40,16,16]. 40 ch = 4 DL components × (1 weight + 3 means + 3 log-scales + 3 channel-coupling coeffs). Per-texel additive **bias grid** [T_v,40,16,16] (trainable table, no renderer) — the resurrection-writable parameterization (F3.3/M1.2).
- Emission scoring: for each leaf shape group (h,w): resample atlas 16x16 → (h,w) via adaptive avg pool; FiLM by Φ at leaf position; score pixels under 4-comp discretized logistic with RGB coupling (means: μ_R; μ_G+α·R; μ_B+β·R+γ·G), quantized to 256 bins → `ell [B, n_leaf_in_group, T_v]` summed over pixels. fp32 result.

**dl.py** — discretized logistic mixture log-prob (PixelCNN++ style, vectorized, fp32-safe).

**sprig.py — SPRIGModel** (implements contracts):
- `log_marginal(image_u8, emb, emb_len) -> logZ [B]` — full inside DP (tempered during training via `self.eta`; an `eta=0` flag for reported numbers).
- `loss(batch) -> (loss, metrics_dict)` — see §6.
- `map_parse(image, emb, emb_len) -> list[ParseNode]` — Viterbi (max-semiring) + backtrace. `ParseNode{rect, axis, cut_px, symbol, texel, children}` (dataclass in sprig/model/sprig.py; texel/leaf fields None for internal nodes).
- `posterior_usage(image, emb, emb_len) -> dict(symbol_usage [S], texel_usage [T_v], node_entropy, emit_mag, rule_mag, mean_depth, mean_leaves)` — expected counts via the autograd identity: make per-(region,symbol) termination potentials and per-(region,cut) potentials require grad, grad of logZ w.r.t. them = posterior marginals; node entropy = occupancy-weighted entropy of conditional split posteriors (good enough for the PI controller).
- `sample(emb, emb_len, seed_struct, seed_material, n) -> (images_u8 [n,64,64,3], trees)` — ancestral, breadth-parallel over the frontier; two `torch.Generator`s: structural draws (term/k/s/B/C) from seed_struct, material draws (texel choice + pixel sampling; pixels = DL means for v0.1 crispness, texel choice sampled) from seed_material. Best-of-K: `sample_bestof(emb, K)` reranks K derivations by joint log p(τ, x̂|c).

## 5. Inside DP (sprig/dp/inside.py) — DP agent

`inside(ell_leaf, term_logits, cut_logits, U_logmix, logV, logW, lattice, temper_kappa) -> beta [B,N_reg,S] fp32, logZ [B]`
- β(r,A) for leaf-eligible r: logaddexp( log p_term + logsumexp_T(log p(T|A,c) + ell(r,T)/κ(r)), log(1−p_term) + expand_term(r,A) ) with must_terminate/must_expand masks (∓inf).
- expand_term via level-synchronous sweep, area ascending. Per level, with index tensors:
  `Bhat[m,k] = logbmm(beta[:, child_lo[m], :], logV.T)`; same for Chat with logW; `comb[m,k] = logsumexp over cuts grouped by parent of (log p(s=cut_type|k,c) − log count_correction + Bhat + Chat)`; `expand[parent,A] = logbmm(comb, U_log.T)` where U_log = log-softmax(U/τ_ann).
  Group-by-parent logsumexp via `torch.segment_reduce` or index_put with scatter-logsumexp helper (write one: max-shift per segment, scatter_add of exp, log). All fp32.
- `logbmm(x_log [.., K], w_log [K, M])`: max-shifted: `x_log.max(-1)` shift, exp, matmul (bf16 ok), log, add shifts. Provide exact fp32 fallback for tests.
- Tempering: κ(r) = max(1, area_px(r)^η), η a module buffer. PI controller (in train.py, every 2k steps): target node entropy band [0.5, 3.0] nats; if H < 0.5: η += 0.05 + 0.1·(0.5−H); if H > 3.0: η −= 0.05; clamp [0, 1.5]. Anneal η→0 linearly over final 20k steps; all REPORTED bpd at η=0.
- Viterbi: same sweep, max instead of logsumexp, argmax backtrace tables per level.
- CORRECTNESS TEST (blocking): tiny lattice (16x16 canvas, 8px grid → 9 regions), tiny config (S=3, R=2, T_v=2, random weights), brute-force enumerate ALL derivation trees (recursive over cuts/symbols/texels), sum exact joint probs, compare to inside logZ within 1e-4. Also: Viterbi tree's joint prob ≤ logZ; posterior marginals from autograd sum to expected node counts.

## 6. Loss & regularizers

```
L = mean_b( −logZ_tempered / (3·64·64) )                      # nats/subpixel
  + 1.0 · Σ_T max(0, 1/(4·T_v) − texel_usage_T)               # texel under-use hinge (F3.2)
  + 0.5 · Σ_A max(0, 1/(4·S)  − symbol_usage_A)               # symbol under-use hinge (M1-style)
```
usage vectors = batch-mean posterior expected counts, normalized to sum 1 (from `posterior_usage` computed on the training batch — reuse the same graph's grads: compute via one extra backward-free trick or just take grads of logZ before the optimizer backward; simplest correct: compute usage with `torch.autograd.grad(logZ.sum(), potentials, retain_graph=True)`).
InfoNCE caption-contrast: DEFERRED; add only if caption-swap margin is flat by 50k steps.
Dead-texel resurrection (train.py, every 2k steps): texels with usage < 0.1/T_v → overwrite bias grid with (a random training-image 16x16 crop converted to DL-mean params, small noise on other channels), perturb E_T row ±ε. Log resurrection count.

## 7. Training curriculum (procedural main run)

- Steps 0–200k on proc2d with tier reweighting: 0–20k tiers (0:.55,1:.35,2:.10,3:0); 20k–60k (.25,.35,.30,.10); >60k (.10,.30,.40,.20) [target mix].
- η per PI controller; τ_ann 2→1 over 50k; report-eta-0 bpd from 180k; final 20k η→0 anneal.
- CLEVR fine-tune (separate config): init from main ckpt, lr×0.3, 30–50k steps, 20% proc2d replay.
- Smoke config (configs/smoke.yaml): S=128, R=16, T_v=32, d=128, batch 32, 200 steps, val_fast 64 — must run CPU-only on Mac AND on klaus-1 GPU.

## 8. Contracts (C1–C4) — binding for all agents

**C1 batch** (from `sprig/data/dataset.py` collate): `{image: u8 [B,64,64,3], emb: f16 [B,L,768], emb_len: i32 [B], tier: i8 [B], idx: i64 [B]}`. Null-caption substitution (10%, train only) in the dataset/loader using `~/data/sprig/t5/null.f16`.
**C2** `model.log_marginal(image, emb, emb_len) -> logZ [B]` (η=0 honored when `model.eval()` + `report_mode=True`).
**C3** `model.map_parse(...)`, `model.posterior_usage(...)` as §4.
**C4** `model.sample(emb, emb_len, seed_struct, seed_material, n)`, `model.sample_bestof(emb, emb_len, K, seed)`.
Data formats, eval metrics, prompt bank, run infra: per the plan file §Part 1/3/4 (memmaps, `meta.jsonl` GT-tree schema, 32-prompt bank, `scalars.jsonl`, atomic checkpoints, run_forever.sh) — copy schemas exactly from there.

## 9. Milestone gates (in order, each blocking)

- **G0**: full pytest suite green on Mac CPU (incl. DP brute-force test, holdout scan test, determinism hashes, tree-metric self-test).
- **G1**: overfit 1 image (script `scripts/overfit1.py`): 2k steps tiny config → bpd < 1.0, MAP parse stable across steps.
- **G2**: overfit 100 images with captions: Δ_c > 0 (caption info gain), texels alive > 50%.
- **G3**: klaus-1 smoke (30 min real config): >2 steps/s at batch 256 (else profile before main run), dataloader ≥95% synthetic throughput, ckpt kill/resume works, eval callback produces grids/overlays.
- **G4**: main run launch. Twice-daily monitor checks per plan alarms.
