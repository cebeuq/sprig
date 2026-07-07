"""Decompose step cost: emission fwd/bwd, DP fwd/bwd+double-bwd, GMT, opt.

Each piece is timed on a detached sub-graph so checkpoint/compile graphs are
consumed exactly once (as in training).

Usage: python scripts/bench_decomp.py [--batch 128] [--leaf-chunk 8]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from sprig.model.sprig import SPRIGModel, SPRIGConfig


def tick():
    torch.cuda.synchronize()
    return time.time()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--leaf-chunk", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    cfg = SPRIGConfig(S=1024, R=64, T_v=256, d=384, leaf_chunk=args.leaf_chunk)
    model = SPRIGModel(cfg).cuda()
    B = args.batch
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 255, (B, 64, 64, 3), dtype=torch.uint8, generator=g).cuda()
    emb = torch.randn(B, 24, 768, generator=g).to(torch.float16).cuda()
    el = torch.full((B,), 24, dtype=torch.int32).cuda()
    batch = {"image": img, "emb": emb, "emb_len": el}

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(2):  # warmup + compile
        loss, _ = model.loss(batch)
        loss.backward()
        opt.zero_grad(set_to_none=True)

    lat = model._lat(emb.device)
    kappa = model._kappa(float(model.eta), emb.device)
    n_subpix = 3 * cfg.canvas * cfg.canvas

    for it in range(args.iters):
        rep = it == args.iters - 1
        # ---- full training step (reference)
        t0 = tick()
        loss, _ = model.loss(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        t1 = tick()
        if rep:
            print("B=%d leaf_chunk=%d   full step: %.3fs" % (B, args.leaf_chunk, t1 - t0))

        # ---- emission isolated
        t0 = tick()
        out = model.gmt(emb, el)
        cond_atlas = model.atlas.render(emb, el)
        t1 = tick()
        atlas_d = cond_atlas.detach().requires_grad_(True)
        phi_d = out.Phi.detach().requires_grad_(True)
        t2 = tick()
        ell = model.atlas.score_leaves(atlas_d, img, lat, phi_d)
        t3 = tick()
        torch.autograd.grad(ell.sum(), [atlas_d, phi_d])
        t4 = tick()
        if rep:
            print("  gmt+render fwd       %.3fs" % (t1 - t0))
            print("  score_leaves fwd     %.3fs" % (t3 - t2))
            print("  score_leaves bwd     %.3fs (recompute+bwd)" % (t4 - t3))

        # ---- DP isolated (detached inputs, incl. hinge double-backward)
        with torch.no_grad():
            cond = model._conditionals(emb, el, images=img)
        dp_in = {
            "ell": cond["ell"].detach().requires_grad_(True),
            "U_logmix": cond["U_logmix"].detach().requires_grad_(True),
            "term_logits": cond["term_logits"].detach().requires_grad_(True),
            "cut_logits": cond["cut_logits"].detach().requires_grad_(True),
            "logV": cond["logV"].detach().requires_grad_(True),
            "logW": cond["logW"].detach().requires_grad_(True),
            "log_PT": cond["log_PT"].detach().requires_grad_(True),
        }
        cond2 = dict(cond)
        cond2.update(dp_in)
        cond2["ell"] = dp_in["ell"]
        t5 = tick()
        logZ = model._inside(cond2, kappa)
        t6 = tick()
        g_ell, g_u = torch.autograd.grad(
            logZ.sum(), [dp_in["ell"], dp_in["U_logmix"]],
            retain_graph=True, create_graph=True)
        t7 = tick()
        nll = (-logZ / n_subpix).mean()
        texel_counts = (g_ell * kappa.view(1, -1, 1)).sum(dim=(0, 1))
        symbol_counts = g_u.sum(dim=(0, 2))
        texel_usage = texel_counts / texel_counts.sum().clamp(min=1e-12)
        symbol_usage = symbol_counts / symbol_counts.sum().clamp(min=1e-12)
        total = (nll + cfg.texel_hinge_weight * F.relu(1.0 / (4.0 * cfg.T_v) - texel_usage).sum()
                 + cfg.symbol_hinge_weight * F.relu(1.0 / (4.0 * cfg.S) - symbol_usage).sum())
        t8 = tick()
        total.backward(inputs=list(dp_in.values()))
        t9 = tick()
        if rep:
            print("  inside_dp fwd        %.3fs" % (t6 - t5))
            print("  hinge grad (cg)      %.3fs" % (t7 - t6))
            print("  dp bwd (+double)     %.3fs (to DP inputs)" % (t9 - t8))
        opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
