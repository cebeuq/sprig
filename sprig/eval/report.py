"""Full evaluation report for a checkpoint: metrics.json + final_report.md.

Expected data_dir layout (plan Part 1/3 formats):
  data_dir/val/          images.u8 [N,64,64,3], emb.f16 (packed ragged),
                         emb_offsets.i64 [N+1], meta.jsonl
  data_dir/parse_eval/   same files; meta lines carry the GT tree under
                         "tree" (or "gt_tree")
  data_dir/train/images.u8   (optional; B0 is fit on val images otherwise)
  data_dir/t5/null.f16       null-caption embedding, packed f16 [L0*768]
  data_dir/t5/promptbank.npz npz {emb: f16 [P,Lmax,768] zero-padded,
                         len: i32 [P]} from sprig.data.embed_t5 --prompts-out,
                         32 prompts in sprig.eval.prompts.PROMPTS order
  data_dir/t5/minimal_pairs.npz  same npz format, 16 prompts interleaved
                         a0,b0,a1,b1,... in MINIMAL_PAIRS order (optional —
                         the prompt-control gate is SKIPPED if absent)

Stages that need a missing optional input are recorded as None and their
gate reported SKIPPED, so the report degrades gracefully while any gate that
can be evaluated is a hard PASS/FAIL.
"""
from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from sprig.eval import color_checks, monitors, tree_metrics
from sprig.eval import baseline_pixmix
from sprig.eval.prompts import COLORS, HELDOUT_COMBOS, MINIMAL_PAIRS, PROMPTS, SHAPES

LOG2 = math.log(2.0)
DIMS = 3.0 * 64 * 64

GATE_THRESH = {
    "b0_margin": 0.15,
    "delta_c": 0.05,
    "recall_tier1": 0.70,
    "recall_tier2": 0.50,
    "visible_cut_f1": 0.60,
    "attribute_move": 0.80,
    "relation_accuracy": 0.70,
    "holdout_probe": 0.60,
    "s_eff_frac": 0.25,
    "alive_texel_frac": 0.50,
}


# --------------------------------------------------------------- data access

class MemmapSplit:
    """Reader for one dataset split in the plan's raw-memmap format."""

    def __init__(self, split_dir: str):
        self.dir = split_dir
        img_path = os.path.join(split_dir, "images.u8")
        n_bytes = os.path.getsize(img_path)
        self.n = n_bytes // (64 * 64 * 3)
        self.images = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(self.n, 64, 64, 3))
        off_path = os.path.join(split_dir, "emb_offsets.i64")
        emb_path = os.path.join(split_dir, "emb.f16")
        self.emb_offsets = np.fromfile(off_path, dtype=np.int64)
        n_rows = int(self.emb_offsets[-1])
        self.emb = np.memmap(emb_path, dtype=np.float16, mode="r", shape=(n_rows, 768))
        self.meta: List[dict] = []
        meta_path = os.path.join(split_dir, "meta.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = [json.loads(line) for line in f if line.strip()]

    def __len__(self) -> int:
        return self.n

    def image(self, i: int) -> np.ndarray:
        return np.asarray(self.images[i])

    def emb_i(self, i: int) -> torch.Tensor:
        lo, hi = int(self.emb_offsets[i]), int(self.emb_offsets[i + 1])
        return torch.from_numpy(np.array(self.emb[lo:hi])).to(torch.float16)

    def tier(self, i: int) -> int:
        return int(self.meta[i].get("tier", 0)) if self.meta else 0

    def tree(self, i: int) -> Optional[dict]:
        if not self.meta:
            return None
        return self.meta[i].get("tree") or self.meta[i].get("gt_tree")


def load_prompt_npz(path: str) -> List[torch.Tensor]:
    """embed_t5 --prompts-out npz {emb [P,Lmax,768] f16, len [P] i32} ->
    list of unpadded [L_i,768] f16 tensors."""
    z = np.load(path)
    emb, lens = z["emb"], z["len"]
    return [
        torch.from_numpy(np.array(emb[i, : int(lens[i])])).to(torch.float16)
        for i in range(emb.shape[0])
    ]


def load_null_emb(t5_dir: str) -> torch.Tensor:
    flat = np.fromfile(os.path.join(t5_dir, "null.f16"), dtype=np.float16)
    return torch.from_numpy(flat.reshape(-1, 768)).to(torch.float16)


def pad_embs(embs: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ragged [L_i,768] f16 list -> (emb [B,Lmax,768] f16, emb_len [B] i32)."""
    lens = torch.tensor([e.shape[0] for e in embs], dtype=torch.int32)
    lmax = int(lens.max())
    out = torch.zeros(len(embs), lmax, 768, dtype=torch.float16)
    for i, e in enumerate(embs):
        out[i, : e.shape[0]] = e
    return out, lens


# -------------------------------------------------------------- model access

def _build_model_from_cfg(cfg):
    """Instantiate SPRIGModel from a checkpoint 'config' entry, which may be a
    SPRIGConfig, a full train-yaml dict (model section nested), or a flat dict
    of SPRIGConfig fields."""
    import dataclasses

    from sprig.model.sprig import SPRIGConfig, SPRIGModel  # lazy: heavy import

    if isinstance(cfg, SPRIGConfig):
        return SPRIGModel(cfg)
    if isinstance(cfg, dict):
        section = cfg.get("model", cfg)
        if isinstance(section, dict):
            fields = {f.name for f in dataclasses.fields(SPRIGConfig)}
            kw = {k: v for k, v in section.items() if k in fields}
            return SPRIGModel(SPRIGConfig(**kw))
    try:
        return SPRIGModel(cfg)
    except TypeError:
        return SPRIGModel(**cfg)


def load_model(ckpt_path: str, device: str = "cpu"):
    """Load a SPRIGModel from a training checkpoint (lazy model import).

    Prefers EMA weights (DESIGN: EMA is eval-only). train.py checkpoints
    store EMA as {"decay", "shadow": {param_name: tensor}} — the shadow is
    overlaid on the raw 'model' state dict (which also carries the buffers).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    else:
        cfg = ckpt.get("config") or ckpt.get("cfg") or {}
        model = _build_model_from_cfg(cfg)
        state = None
        for key in ("ema", "ema_state_dict", "model", "model_state_dict", "state_dict"):
            if key in ckpt:
                state = ckpt[key]
                break
        if state is None:
            state = ckpt
        if isinstance(state, dict) and isinstance(state.get("shadow"), dict):
            base = dict(ckpt.get("model") or {})
            base.update(state["shadow"])
            state = base
        model.load_state_dict(state)
    model = model.to(device)
    # Report numbers are untempered by definition: mid-anneal checkpoints carry
    # a nonzero eta buffer which silently deflates bpd/delta_c (eval-audit
    # finding C1 — the 75k ckpt reported bpd 0.53 instead of 3.97).
    eta = getattr(model, "eta", None)
    if isinstance(eta, torch.Tensor):
        eta.data.fill_(0.0)
    model.eval()
    if hasattr(model, "report_mode"):
        model.report_mode = True
    return model


def _as_u8_numpy(images) -> np.ndarray:
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    return np.asarray(images).astype(np.uint8)


def _log_marginal(model, images: np.ndarray, embs: Sequence[torch.Tensor], device: str) -> np.ndarray:
    emb, emb_len = pad_embs(embs)
    img = torch.from_numpy(np.array(images, dtype=np.uint8)).to(device)
    with torch.no_grad():
        logz = model.log_marginal(img, emb.to(device), emb_len.to(device))
    return logz.detach().cpu().to(torch.float64).numpy()


def bpd_from_logz(logz: np.ndarray) -> np.ndarray:
    return -np.asarray(logz, dtype=np.float64) / (DIMS * LOG2)


def _sample_images(model, emb: torch.Tensor, seed_struct: int, seed_material: int,
                   n: int, device: str) -> np.ndarray:
    e, elen = pad_embs([emb])
    with torch.no_grad():
        images, _trees = model.sample(
            e.to(device), elen.to(device), seed_struct, seed_material, n
        )
    return _as_u8_numpy(images)


def _parse_one(model, image: np.ndarray, emb: torch.Tensor, device: str):
    e, elen = pad_embs([emb])
    img = torch.from_numpy(np.array(image[None], dtype=np.uint8)).to(device)
    with torch.no_grad():
        result = model.map_parse(img, e.to(device), elen.to(device))
    if result and isinstance(result[0], (list, tuple)):  # batched return
        result = result[0]
    return list(result)


# ------------------------------------------------------------- image grids

def image_grid(rows: Sequence[Sequence[np.ndarray]], pad: int = 2) -> Image.Image:
    """[R][C] of u8 [64,64,3] arrays -> one PIL image with white padding."""
    r, c = len(rows), max(len(row) for row in rows)
    h = w = 64
    canvas = np.full((r * (h + pad) + pad, c * (w + pad) + pad, 3), 255, dtype=np.uint8)
    for i, row in enumerate(rows):
        for j, img in enumerate(row):
            y, x = pad + i * (h + pad), pad + j * (w + pad)
            canvas[y:y + h, x:x + w] = np.asarray(img, dtype=np.uint8)
    return Image.fromarray(canvas)


def save_grid_jpeg(rows: Sequence[Sequence[np.ndarray]], path: str) -> None:
    image_grid(rows).save(path, format="JPEG", quality=92)


# ------------------------------------------------------------ eval stages

def eval_bpd(model, split: MemmapSplit, device: str, max_images: int = 2000,
             batch_size: int = 32) -> Dict[str, object]:
    n = min(len(split), max_images)
    bpds = np.zeros(n)
    tiers = np.array([split.tier(i) for i in range(n)], dtype=np.int64)
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        embs = [split.emb_i(k) for k in range(i, j)]
        logz = _log_marginal(model, np.asarray(split.images[i:j]), embs, device)
        bpds[i:j] = bpd_from_logz(logz)
    per_tier = {
        int(t): float(bpds[tiers == t].mean()) for t in np.unique(tiers)
    }
    ge1 = bpds[tiers >= 1]
    return {
        "bpd_val": float(bpds.mean()),
        "bpd_per_tier": per_tier,
        "bpd_tier_ge1": float(ge1.mean()) if ge1.size else float(bpds.mean()),
        "tier_ge1_index": np.nonzero(tiers >= 1)[0],
        "n": n,
    }


def eval_delta_c(model, split: MemmapSplit, null_emb: torch.Tensor, device: str,
                 max_images: int = 512, batch_size: int = 32) -> float:
    n = min(len(split), max_images)
    total = 0.0
    for i in range(0, n, batch_size):
        j = min(i + batch_size, n)
        imgs = np.asarray(split.images[i:j])
        embs = [split.emb_i(k) for k in range(i, j)]
        logz_c = _log_marginal(model, imgs, embs, device)
        logz_0 = _log_marginal(model, imgs, [null_emb] * (j - i), device)
        total += float((bpd_from_logz(logz_0) - bpd_from_logz(logz_c)).sum())
    return total / n


def eval_caption_swap(model, split: MemmapSplit, device: str, n_groups: int = 4,
                      group: int = 8) -> float:
    """8-way in-batch caption swap: fraction of images whose own caption gives
    a higher logZ than all 7 swapped captions."""
    rng = np.random.default_rng(0)
    n = len(split)
    wins = 0
    total = 0
    for _ in range(n_groups):
        idx = rng.choice(n, size=group, replace=False)
        imgs = np.asarray(split.images[np.sort(idx)])
        embs = [split.emb_i(int(i)) for i in np.sort(idx)]
        scores = np.zeros((group, group))
        for j in range(group):
            scores[:, j] = _log_marginal(model, imgs, [embs[j]] * group, device)
        for i in range(group):
            off_diag = np.delete(scores[i], i)
            wins += int(scores[i, i] > off_diag.max())
            total += 1
    return wins / float(total)


def eval_b0(fit_images: np.ndarray, eval_images: np.ndarray, steps: int = 2000,
            device: str = "cpu") -> float:
    b0 = baseline_pixmix.fit(fit_images, steps=steps, device=device)
    return float(b0.bpd(eval_images))


def eval_tree_metrics(model, split: MemmapSplit, device: str,
                      max_images: int = 512) -> Dict[str, object]:
    n = min(len(split), max_images)
    recalls: Dict[int, List[float]] = {}
    f1s: List[float] = []
    aris: List[float] = []
    for i in range(n):
        gt = split.tree(i)
        if gt is None:
            continue
        img = split.image(i)
        parse = _parse_one(model, img, split.emb_i(i), device)
        recalls.setdefault(split.tier(i), []).append(
            tree_metrics.object_cell_recall(parse, gt)
        )
        f1s.append(tree_metrics.visible_cut_f1(parse, gt, img))
        aris.append(tree_metrics.leaf_ari(parse, gt))
    out: Dict[str, object] = {
        "visible_cut_f1": float(np.mean(f1s)) if f1s else None,
        "leaf_ari": float(np.mean(aris)) if aris else None,
        "recall_per_tier": {t: float(np.mean(v)) for t, v in recalls.items()},
    }
    out["recall_tier1"] = out["recall_per_tier"].get(1)  # type: ignore[union-attr]
    out["recall_tier2"] = out["recall_per_tier"].get(2)  # type: ignore[union-attr]
    return out


def sample_prompt_bank_grid(model, prompt_embs: Sequence[torch.Tensor], out_path: str,
                            device: str, n_seeds: int = 8, seed0: int = 0) -> None:
    rows = []
    for i, emb in enumerate(prompt_embs):
        row = [
            _sample_images(model, emb, seed0 + 1000 * s + i, seed0 + 5000 + 1000 * s + i,
                           1, device)[0]
            for s in range(n_seeds)
        ]
        rows.append(row)
    save_grid_jpeg(rows, out_path)


def layout_material_grid(model, emb: torch.Tensor, out_path: str, device: str,
                         k: int = 4) -> None:
    """k x k grid: rows = frozen structural seed, cols = rerolled material seed."""
    rows = []
    for r in range(k):
        rows.append([
            _sample_images(model, emb, 100 + r, 900 + c, 1, device)[0] for c in range(k)
        ])
    save_grid_jpeg(rows, out_path)


def _tokens(prompt: str) -> List[str]:
    return prompt.replace(",", " ").split()

def _color_in(prompt: str) -> List[str]:
    return [t for t in _tokens(prompt) if t in COLORS]

def _shape_in(prompt: str) -> List[str]:
    return [t for t in _tokens(prompt) if t in SHAPES]

def _relation_in(prompt: str) -> Optional[str]:
    for rel in ("left of", "right of"):
        if rel in prompt:
            return rel
    for rel in ("above", "below"):
        if rel in prompt.split():
            return rel
    return None


def _median_object_area(images: np.ndarray) -> float:
    areas = []
    for img in images:
        objs = color_checks.extract(img)["objects"]
        if objs:
            areas.append(objs[0]["area"])
    return float(np.median(areas)) if areas else 0.0


def eval_minimal_pairs(model, pair_embs: Sequence[torch.Tensor], device: str,
                       n_seeds: int = 64, probe_ckpt: Optional[str] = None,
                       batch: int = 16) -> Dict[str, object]:
    """pair_embs: 16 embeddings, (a_i, b_i) interleaved, MINIMAL_PAIRS order."""
    per_pair: Dict[str, float] = {}
    move_scores: List[float] = []
    rel_scores: List[float] = []
    for p, (prompt_a, prompt_b, attr) in enumerate(MINIMAL_PAIRS):
        emb_a, emb_b = pair_embs[2 * p], pair_embs[2 * p + 1]
        imgs_b = np.concatenate([
            _sample_images(model, emb_b, 10 * p + s, 77 + 10 * p + s, 1, device)
            for s in range(n_seeds)
        ])
        score: Optional[float] = None
        if attr == "color":
            new_color = [c for c in _color_in(prompt_b) if c not in _color_in(prompt_a)][0]
            hits = [
                any(o["color"] == new_color for o in color_checks.extract(im)["objects"])
                for im in imgs_b
            ]
            score = float(np.mean(hits))
        elif attr == "relation":
            colors = _color_in(prompt_b)
            rel = _relation_in(prompt_b)
            oks = []
            for im in imgs_b:
                r = color_checks.relation_holds(
                    color_checks.extract(im), colors[0], colors[1], rel
                )
                oks.append(bool(r) if r is not None else False)
            score = float(np.mean(oks))
        elif attr == "size":
            imgs_a = np.concatenate([
                _sample_images(model, emb_a, 10 * p + s, 77 + 10 * p + s, 1, device)
                for s in range(n_seeds)
            ])
            bigger_in_b = "large" in _tokens(prompt_b)
            area_a, area_b = _median_object_area(imgs_a), _median_object_area(imgs_b)
            score = float(area_b > area_a) if bigger_in_b else float(area_b < area_a)
        elif attr == "shape" and probe_ckpt is not None:
            from sprig.eval import probe

            score = probe.score_generations(
                imgs_b, _shape_in(prompt_b)[0], _color_in(prompt_b)[0], probe_ckpt, device
            )
        if score is not None:
            per_pair["{} -> {}".format(prompt_a, prompt_b)] = score
            (rel_scores if attr == "relation" else move_scores).append(score)
    return {
        "per_pair": per_pair,
        "attribute_move": float(np.mean(move_scores)) if move_scores else None,
        "relation_accuracy": float(np.mean(rel_scores)) if rel_scores else None,
    }


def eval_heldout_combos(model, prompt_embs: Sequence[torch.Tensor], probe_ckpt: str,
                        device: str, n: int = 64) -> Dict[str, object]:
    from sprig.eval import probe

    model_probe = probe.load_probe(probe_ckpt, device)
    per_combo: Dict[str, float] = {}
    for color, shape in HELDOUT_COMBOS:
        prompt = "a {} {}".format(color, shape)
        emb = prompt_embs[PROMPTS.index(prompt)]
        imgs = np.concatenate([
            _sample_images(model, emb, 313 + s, 707 + s, 1, device) for s in range(n)
        ])
        per_combo[prompt] = probe.score_generations(imgs, shape, color, model_probe, device)
    return {
        "per_combo": per_combo,
        "holdout_probe_acc": float(np.mean(list(per_combo.values()))),
    }


def eval_health(model, split: MemmapSplit, device: str, n_images: int = 64) -> Dict[str, object]:
    n = min(len(split), n_images)
    imgs = torch.from_numpy(np.array(split.images[:n], dtype=np.uint8)).to(device)
    emb, elen = pad_embs([split.emb_i(i) for i in range(n)])
    stats = model.posterior_usage(imgs, emb.to(device), elen.to(device))
    usage = stats["symbol_usage"]
    texels = stats["texel_usage"]
    s = int(usage.shape[-1]) if hasattr(usage, "shape") else len(usage)
    t_v = int(texels.shape[-1]) if hasattr(texels, "shape") else len(texels)
    return {
        "S": s,
        "T_v": t_v,
        "s_eff": monitors.s_eff(usage),
        "alive_texel_frac": monitors.alive_texels(texels, t_v) / float(t_v),
        "node_entropy": float(stats.get("node_entropy", float("nan"))),
    }


# ---------------------------------------------------------------- gates

def _gate(ok: Optional[bool], detail: str) -> Dict[str, str]:
    if ok is None:
        return {"status": "SKIPPED", "detail": detail}
    return {"status": "PASS" if ok else "FAIL", "detail": detail}


def evaluate_gates(metrics: Dict[str, object]) -> Dict[str, Dict[str, str]]:
    """The plan's 5 proof-of-concept success criteria as PASS/FAIL/SKIPPED."""
    g = GATE_THRESH
    gates: Dict[str, Dict[str, str]] = {}

    bpd = metrics.get("bpd_tier_ge1")
    b0 = metrics.get("b0_bpd")
    dc = metrics.get("delta_c")
    if bpd is None or b0 is None or dc is None:
        gates["1_likelihood"] = _gate(None, "missing bpd/b0/delta_c")
    else:
        ok = (b0 - bpd >= g["b0_margin"]) and (dc >= g["delta_c"])
        gates["1_likelihood"] = _gate(ok, "bpd(tier>=1)={:.3f} vs B0={:.3f} (need margin >= {}); delta_c={:.3f} (need >= {})".format(bpd, b0, g["b0_margin"], dc, g["delta_c"]))

    tm = metrics.get("tree") or {}
    r1, r2, f1 = tm.get("recall_tier1"), tm.get("recall_tier2"), tm.get("visible_cut_f1")
    if r1 is None or r2 is None or f1 is None:
        gates["2_parses"] = _gate(None, "missing tree metrics")
    else:
        ok = (r1 >= g["recall_tier1"]) and (r2 >= g["recall_tier2"]) and (f1 >= g["visible_cut_f1"])
        gates["2_parses"] = _gate(ok, "recall t1={:.2f} (>= {}), t2={:.2f} (>= {}), cut F1={:.2f} (>= {})".format(r1, g["recall_tier1"], r2, g["recall_tier2"], f1, g["visible_cut_f1"]))

    ps = metrics.get("prompt_swap") or {}
    move, rel = ps.get("attribute_move"), ps.get("relation_accuracy")
    if move is None or rel is None:
        gates["3_prompt_control"] = _gate(None, "missing minimal-pair scores")
    else:
        ok = (move >= g["attribute_move"]) and (rel >= g["relation_accuracy"])
        gates["3_prompt_control"] = _gate(ok, "attribute-move={:.2f} (>= {}), relation={:.2f} (>= {})".format(move, g["attribute_move"], rel, g["relation_accuracy"]))

    hp = metrics.get("holdout_probe_acc")
    if hp is None:
        gates["4_compositional"] = _gate(None, "no probe checkpoint / heldout scores")
    else:
        gates["4_compositional"] = _gate(hp >= g["holdout_probe"], "held-out combo probe acc={:.2f} (>= {})".format(hp, g["holdout_probe"]))

    health = metrics.get("health") or {}
    s_eff, s = health.get("s_eff"), health.get("S")
    alive = health.get("alive_texel_frac")
    if s_eff is None or alive is None or s is None:
        gates["5_health"] = _gate(None, "missing health stats")
    else:
        ok = (s_eff >= g["s_eff_frac"] * s) and (alive >= g["alive_texel_frac"])
        gates["5_health"] = _gate(ok, "S_eff={:.0f} (>= {:.0f}), alive texels={:.0%} (>= {:.0%})".format(s_eff, g["s_eff_frac"] * s, alive, g["alive_texel_frac"]))
    return gates


def write_report(metrics: Dict[str, object], gates: Dict[str, Dict[str, str]],
                 out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump({"metrics": metrics, "gates": gates}, f, indent=2, default=_default)
    lines = ["# SPRIG v0.1 — Final Report", "", "## Success-criteria gates", ""]
    lines.append("| gate | status | detail |")
    lines.append("|---|---|---|")
    for name, g in gates.items():
        lines.append("| {} | **{}** | {} |".format(name, g["status"], g["detail"]))
    lines += ["", "## Key metrics", "", "```json"]
    lines.append(json.dumps(metrics, indent=2, default=_default))
    lines.append("```")
    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------- orchestrator

def run_report(
    ckpt_path: Optional[str],
    data_dir: str,
    out_dir: str,
    device: str = "cpu",
    model=None,
    probe_ckpt: Optional[str] = None,
    n_bank_seeds: int = 8,
    n_pair_seeds: int = 64,
    max_bpd_images: int = 2000,
    max_parse_images: int = 512,
    b0_steps: int = 2000,
) -> Dict[str, object]:
    """Run the full evaluation suite on a checkpoint.

    Pass `model` directly to skip checkpoint loading (used by tests/harness).
    Returns the metrics dict; writes metrics.json, final_report.md and the
    sample grids into out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    if model is None:
        model = load_model(ckpt_path, device)

    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(val_dir):
        val_dir = os.path.join(data_dir, "val_fast")
    val = MemmapSplit(val_dir)
    t5_dir = os.path.join(data_dir, "t5")

    metrics: Dict[str, object] = {}

    # --- likelihood
    bpd_stats = eval_bpd(model, val, device, max_images=max_bpd_images)
    tier_ge1_index = bpd_stats.pop("tier_ge1_index")
    metrics.update(bpd_stats)

    null_path = os.path.join(t5_dir, "null.f16")
    metrics["delta_c"] = (
        eval_delta_c(model, val, load_null_emb(t5_dir), device)
        if os.path.exists(null_path) else None
    )
    metrics["caption_swap_win_frac"] = eval_caption_swap(model, val, device)

    # --- B0 baseline (fit on train images when available, else val)
    train_imgs_path = os.path.join(data_dir, "train", "images.u8")
    if os.path.exists(train_imgs_path):
        n_fit = os.path.getsize(train_imgs_path) // (64 * 64 * 3)
        fit_imgs = np.memmap(train_imgs_path, dtype=np.uint8, mode="r",
                             shape=(n_fit, 64, 64, 3))[: 4096]
    else:
        fit_imgs = np.asarray(val.images[: min(len(val), 4096)])
    eval_idx = tier_ge1_index if len(tier_ge1_index) else np.arange(min(len(val), 512))
    eval_imgs = np.asarray(val.images)[eval_idx[:512]]
    metrics["b0_bpd"] = eval_b0(np.asarray(fit_imgs), eval_imgs, steps=b0_steps, device=device)

    # --- parse metrics
    parse_dir = os.path.join(data_dir, "parse_eval")
    metrics["tree"] = (
        eval_tree_metrics(model, MemmapSplit(parse_dir), device, max_parse_images)
        if os.path.isdir(parse_dir) else None
    )

    # --- prompt bank grids
    bank_npz = os.path.join(t5_dir, "promptbank.npz")
    prompt_embs: Optional[List[torch.Tensor]] = None
    if os.path.exists(bank_npz):
        prompt_embs = load_prompt_npz(bank_npz)
        sample_prompt_bank_grid(
            model, prompt_embs, os.path.join(out_dir, "prompt_bank_grid.jpg"),
            device, n_seeds=n_bank_seeds,
        )
        layout_material_grid(
            model, prompt_embs[8], os.path.join(out_dir, "layout_material_grid.jpg"), device
        )

    # --- minimal pairs
    pairs_npz = os.path.join(t5_dir, "minimal_pairs.npz")
    if os.path.exists(pairs_npz):
        pair_embs = load_prompt_npz(pairs_npz)
        metrics["prompt_swap"] = eval_minimal_pairs(
            model, pair_embs, device, n_seeds=n_pair_seeds, probe_ckpt=probe_ckpt
        )
    else:
        metrics["prompt_swap"] = None

    # --- held-out combos via probe
    metrics["holdout_probe_acc"] = None
    if probe_ckpt is not None and prompt_embs is not None:
        heldout = eval_heldout_combos(model, prompt_embs, probe_ckpt, device)
        metrics["heldout_per_combo"] = heldout["per_combo"]
        metrics["holdout_probe_acc"] = heldout["holdout_probe_acc"]

    # --- health
    metrics["health"] = eval_health(model, val, device)

    gates = evaluate_gates(metrics)
    write_report(metrics, gates, out_dir)
    return metrics
