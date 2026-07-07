"""Profile one full loss+backward+opt step (phase timers + torch.profiler).

Usage: python scripts/profile_step.py [--batch 64] [--leaf-chunk 8]
       [--steps 3] [--no-profiler] [--phases]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule

from sprig.model.sprig import SPRIGModel, SPRIGConfig


def make_batch(bs: int, device: str = "cuda"):
    g = torch.Generator().manual_seed(0)
    img = torch.randint(0, 255, (bs, 64, 64, 3), dtype=torch.uint8, generator=g).to(device)
    emb = torch.randn(bs, 24, 768, generator=g).to(torch.float16).to(device)
    el = torch.full((bs,), 24, dtype=torch.int32).to(device)
    return {"image": img, "emb": emb, "emb_len": el}


def phase_step(model, batch, opt):
    """Replicates SPRIGModel.loss with cuda-sync timers per phase."""
    import torch.nn.functional as F
    times = {}

    def tick():
        torch.cuda.synchronize()
        return time.time()

    emb, el, images = batch["emb"], batch["emb_len"], batch["image"]
    cfg = model.cfg
    t0 = tick()
    lat = model._lat(emb.device)
    out = model.gmt(emb, el)
    tau = model.tau.clamp(min=1e-3)
    cond = {
        "H": out.H, "Phi": out.Phi,
        "U_logmix": F.log_softmax(out.U / tau, dim=-1),
        "log_PT": F.log_softmax(model.gmt.P_T / tau, dim=-1),
        "logV": F.log_softmax(model.gmt.V / tau, dim=-1),
        "logW": F.log_softmax(model.gmt.W / tau, dim=-1),
        "term_logits": model.gmt.termination_logits(out.H, lat.phi_geom) / tau,
        "cut_logits": out.cut_logits / tau,
    }
    t1 = tick(); times["gmt+heads"] = t1 - t0
    cond["atlas"] = model.atlas.render(emb, el)
    t2 = tick(); times["atlas_render"] = t2 - t1
    cond["ell"] = model.atlas.score_leaves(cond["atlas"], images, lat, out.Phi)
    t3 = tick(); times["score_leaves_fwd"] = t3 - t2
    kappa = model._kappa(float(model.eta), emb.device)
    logZ = model._inside(cond, kappa)
    t4 = tick(); times["inside_dp_fwd"] = t4 - t3
    n_subpix = 3 * cfg.canvas * cfg.canvas
    nll = (-logZ / n_subpix).mean()
    g_ell, g_u = torch.autograd.grad(
        logZ.sum(), [cond["ell"], cond["U_logmix"]],
        retain_graph=True, create_graph=True)
    t5 = tick(); times["hinge_grad(create_graph)"] = t5 - t4
    texel_counts = (g_ell * kappa.view(1, -1, 1)).sum(dim=(0, 1))
    symbol_counts = g_u.sum(dim=(0, 2))
    texel_usage = texel_counts / texel_counts.sum().clamp(min=1e-12)
    symbol_usage = symbol_counts / symbol_counts.sum().clamp(min=1e-12)
    total = (nll + cfg.texel_hinge_weight * F.relu(1.0 / (4.0 * cfg.T_v) - texel_usage).sum()
             + cfg.symbol_hinge_weight * F.relu(1.0 / (4.0 * cfg.S) - symbol_usage).sum())
    t6 = tick(); times["hinge_rest"] = t6 - t5
    total.backward()
    t7 = tick(); times["backward"] = t7 - t6
    opt.step(); opt.zero_grad(set_to_none=True)
    t8 = tick(); times["opt"] = t8 - t7
    times["TOTAL"] = t8 - t0
    return times


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--leaf-chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no-profiler", action="store_true")
    ap.add_argument("--phases", action="store_true")
    args = ap.parse_args()

    cfg = SPRIGConfig(S=1024, R=64, T_v=256, d=384, leaf_chunk=args.leaf_chunk)
    model = SPRIGModel(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = make_batch(args.batch)

    def one_step():
        loss, _ = model.loss(batch)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    # warmup
    for _ in range(2):
        one_step()
    torch.cuda.synchronize()

    if args.phases:
        for i in range(args.steps):
            times = phase_step(model, batch, opt)
            print("phase step %d:" % i)
            for k, v in times.items():
                print("   %-28s %7.3fs" % (k, v))

    t0 = time.time()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()
    print("plain step avg: %.3fs (B=%d, leaf_chunk=%d)"
          % ((time.time() - t0) / args.steps, args.batch, args.leaf_chunk))

    if args.no_profiler:
        return
    sched = schedule(wait=1, warmup=1, active=args.steps)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
    ) as prof:
        for _ in range(2 + args.steps):
            with record_function("full_step"):
                one_step()
            prof.step()
    ka = prof.key_averages()
    print("\n=== sort by self_cuda_time_total ===")
    print(ka.table(sort_by="self_cuda_time_total", row_limit=30))
    print("\n=== sort by self_cpu_time_total ===")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=30))
    # sync / sync-adjacent ops
    print("\n=== sync-suspect ops ===")
    for evt in ka:
        n = evt.key
        if any(s in n for s in ("Synchronize", "aten::item", "nonzero", "unique",
                                "aten::to", "Memcpy", "aten::_local_scalar_dense")):
            print("%-60s count=%6d cpu=%10.1fus cuda=%10.1fus"
                  % (n[:60], evt.count, evt.self_cpu_time_total,
                     getattr(evt, "self_device_time_total", 0.0)))


if __name__ == "__main__":
    main()
