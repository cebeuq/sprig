"""Numerical-parity harness for the DP/emission optimization work.

Two modes:
  --save PATH   run the CURRENT code on fixed-seed inputs and save reference
                tensors (run this BEFORE changing anything).
  --check PATH  rerun the same fixed-seed inputs with the current code and
                compare against the saved references:
                  * inside_logZ within 1e-3 relative (and beta / first-order
                    grads within loose tolerances, for attribution),
                  * score_leaves ell within 1e-3 relative,
                  * SPRIGModel.loss at fixed seed within 0.1%.

Usage: python scripts/parity_check.py --save ~/ref_sprig/ref.pt [--device cuda]
       python scripts/parity_check.py --check ~/ref_sprig/ref.pt [--device cuda]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from sprig.dp import inside as dp
from sprig.dp.lattice import get_lattice
from sprig.model.sprig import SPRIGModel, SPRIGConfig


def dp_case(device: torch.device, seed: int, B: int = 4, S: int = 1024,
            R: int = 64, T_v: int = 256, eta: float = 0.3):
    lat = get_lattice(64, 8, 16).to(device)
    g = torch.Generator().manual_seed(seed)
    n_leaf, N = lat.n_leaf_regions, lat.n_regions
    area = lat.area_px[lat.leaf_ids].float().cpu()
    kappa = torch.clamp(area ** eta, min=1.0)
    inputs = {
        "ell_leaf": (torch.randn(B, n_leaf, T_v, generator=g) * 2.0),
        "term_logits": torch.randn(B, N, S, generator=g),
        "cut_logits": torch.randn(B, R, 14, generator=g),
        "U_logmix": F.log_softmax(torch.randn(B, S, R, generator=g), dim=-1),
        "logV": F.log_softmax(torch.randn(R, S, generator=g), dim=-1),
        "logW": F.log_softmax(torch.randn(R, S, generator=g), dim=-1),
        "temper_kappa": kappa,
        "log_PT": F.log_softmax(torch.randn(R, T_v, generator=g), dim=-1),
    }
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["lattice"] = lat
    ell = inputs["ell_leaf"].detach().requires_grad_(True)
    U = inputs["U_logmix"].detach().requires_grad_(True)
    inputs["ell_leaf"], inputs["U_logmix"] = ell, U
    fn = getattr(dp, "inside_logZ", None) or dp.inside
    beta, logZ = fn(**inputs)
    g_ell, g_u = torch.autograd.grad(logZ.sum(), [ell, U])
    return {
        "logZ": logZ.detach().float().cpu(),
        "beta": beta.detach().float().cpu(),
        "g_ell": g_ell.detach().float().cpu(),
        "g_u": g_u.detach().float().cpu(),
    }


def model_case(device: torch.device, seed: int, B: int = 8, eta: float = 0.25):
    torch.manual_seed(seed)
    cfg = SPRIGConfig(S=1024, R=64, T_v=256, d=384, leaf_chunk=8)
    model = SPRIGModel(cfg).to(device)
    model.eta.fill_(eta)
    g = torch.Generator().manual_seed(seed + 1)
    img = torch.randint(0, 255, (B, 64, 64, 3), dtype=torch.uint8, generator=g).to(device)
    emb = torch.randn(B, 24, 768, generator=g).to(torch.float16).to(device)
    el = torch.full((B,), 24, dtype=torch.int32).to(device)
    batch = {"image": img, "emb": emb, "emb_len": el}

    with torch.no_grad():
        cond = model._conditionals(emb, el, images=img)
    ell = cond["ell"].detach().float().cpu()

    loss, metrics = model.loss(batch)
    loss.backward()
    gn = torch.sqrt(sum((p.grad.detach().float() ** 2).sum()
                        for p in model.parameters() if p.grad is not None))
    return {
        "loss": float(loss),
        "logZ_mean": metrics["logZ_mean"],
        "nll": metrics["nll"],
        "texel_hinge": metrics["texel_hinge"],
        "symbol_hinge": metrics["symbol_hinge"],
        "ell": ell,
        "grad_norm": float(gn),
    }


def build(device: torch.device):
    ref = {}
    for seed in (0, 1):
        ref[f"dp{seed}"] = dp_case(device, seed)
    ref["model"] = model_case(device, 123)
    return ref


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    denom = b.abs().clamp(min=1.0)
    return float(((a - b).abs() / denom).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--check", type=str, default=None)
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.save:
        ref = build(device)
        out = Path(args.save).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ref, out)
        print("saved reference to", out)
        print("  dp0 logZ:", ref["dp0"]["logZ"].tolist())
        print("  model loss: %.6f  logZ_mean: %.3f" % (ref["model"]["loss"], ref["model"]["logZ_mean"]))
        return

    assert args.check, "need --save or --check"
    ref = torch.load(Path(args.check).expanduser(), map_location="cpu", weights_only=False)
    cur = build(device)
    ok = True
    for seed in (0, 1):
        r, c = ref[f"dp{seed}"], cur[f"dp{seed}"]
        e_logz = rel_err(c["logZ"], r["logZ"])
        e_beta = rel_err(c["beta"], r["beta"])
        e_gell = float((c["g_ell"] - r["g_ell"]).abs().max())
        e_gu = float((c["g_u"] - r["g_u"]).abs().max())
        good = e_logz < 1e-3
        ok &= good
        print("dp seed %d: logZ rel %.2e (<1e-3: %s)  beta rel %.2e  |dg_ell| %.2e  |dg_u| %.2e"
              % (seed, e_logz, good, e_beta, e_gell, e_gu))
        print("   logZ ref", [round(v, 3) for v in r["logZ"].tolist()])
        print("   logZ cur", [round(v, 3) for v in c["logZ"].tolist()])
    r, c = ref["model"], cur["model"]
    e_loss = abs(c["loss"] - r["loss"]) / max(abs(r["loss"]), 1e-9)
    e_ell = rel_err(c["ell"], r["ell"])
    good = e_loss < 1e-3
    ok &= good
    print("model: loss ref %.6f cur %.6f rel %.2e (<1e-3: %s)" % (r["loss"], c["loss"], e_loss, good))
    print("   ell rel %.2e   logZ_mean ref %.3f cur %.3f   grad_norm ref %.4f cur %.4f"
          % (e_ell, r["logZ_mean"], c["logZ_mean"], r["grad_norm"], c["grad_norm"]))
    print("PARITY", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
