#!/usr/bin/env python
"""SPRIG v0.1 training harness (DESIGN.md sections 2, 6, 7, 9).

Entry point:
    python train.py --config configs/main64.yaml --run-dir runs/main64 --resume auto

All heavy imports (torch, sprig.*) are deferred inside functions so that
``python train.py --help`` works without torch installed. The core loop is
factored into ``train_loop(cfg, dataset, steps=...)`` which the overfit gate
scripts (scripts/overfit1.py, scripts/overfit100.py) reuse.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import math
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("config %s did not parse to a mapping" % path)
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: List[str]) -> None:
    """Apply ``--set a.b.c=value`` style dotted overrides in place."""
    import yaml
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError("override %r is not KEY=VALUE" % item)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(raw)


def _cfg_get(cfg: Dict[str, Any], dotted: str) -> Tuple[Any, bool]:
    node: Any = cfg
    for p in dotted.split("."):
        if not isinstance(node, dict) or p not in node:
            return None, False
        node = node[p]
    return node, True


def validate_config(cfg: Dict[str, Any]) -> None:
    """Schema-only validation (paths are NOT checked for existence)."""
    errs: List[str] = []

    def req(path: str, types: Any, pred: Optional[Callable[[Any], bool]] = None,
            desc: str = "") -> Any:
        v, ok = _cfg_get(cfg, path)
        if not ok:
            errs.append("missing key: %s" % path)
            return None
        if types is not None and not isinstance(v, types):
            errs.append("%s: expected %s, got %r" % (path, types, type(v).__name__))
            return None
        if pred is not None and not pred(v):
            errs.append("%s: invalid value %r %s" % (path, v, desc))
        return v

    req("run_name", str)
    req("seed", int)
    for k in ("S", "R", "T_v", "d", "canvas", "grid"):
        req("model.%s" % k, int, lambda v: v > 0, "(must be > 0)")
    req("data.train_dir", str)
    req("data.null_emb", str)
    req("data.p_null", (int, float), lambda v: 0.0 <= v <= 1.0, "(must be in [0,1])")
    req("data.num_workers", int, lambda v: v >= 0, "(must be >= 0)")
    for k in ("batch_size", "total_steps", "warmup_steps", "tau_steps",
              "eta_update_every", "eta_final_anneal_steps"):
        req("train.%s" % k, int, lambda v: v > 0, "(must be > 0)")
    for k in ("lr_tables", "lr_networks"):
        req("train.%s" % k, (int, float), lambda v: v > 0, "(must be > 0)")
    req("train.grad_clip", (int, float), lambda v: v > 0, "(must be > 0)")
    req("train.ema", (int, float), lambda v: 0.0 < v < 1.0, "(must be in (0,1))")
    req("train.betas", list,
        lambda v: len(v) == 2 and all(0.0 < b < 1.0 for b in v), "(2 floats in (0,1))")
    for k in ("tau_start", "tau_end"):
        req("train.%s" % k, (int, float), lambda v: v > 0, "(must be > 0)")
    req("train.eta_band", list,
        lambda v: len(v) == 2 and 0.0 < v[0] < v[1], "(need 0 < lo < hi)")
    req("train.eta_max", (int, float), lambda v: v >= 0, "(must be >= 0)")
    for k in ("scalars_every", "val_fast_every", "parse_every", "full_every",
              "val_fast_n"):
        req("eval.%s" % k, int, lambda v: v > 0, "(must be > 0)")
    req("checkpoint.every_steps", int, lambda v: v > 0, "(must be > 0)")
    req("checkpoint.every_minutes", (int, float), lambda v: v > 0, "(must be > 0)")

    sched, ok = _cfg_get(cfg, "train.tier_schedule")
    if ok and sched is not None:
        if not isinstance(sched, list) or not sched:
            errs.append("train.tier_schedule: expected a non-empty list")
        else:
            prev_until = -1
            for i, ent in enumerate(sched):
                if not isinstance(ent, dict) or "until" not in ent or "weights" not in ent:
                    errs.append("train.tier_schedule[%d]: need {until, weights}" % i)
                    continue
                w = ent["weights"]
                if (not isinstance(w, list) or len(w) != 4
                        or any((not isinstance(x, (int, float))) or x < 0 for x in w)
                        or abs(sum(w) - 1.0) > 1e-6):
                    errs.append("train.tier_schedule[%d].weights: need 4 non-negative "
                                "floats summing to 1" % i)
                u = ent["until"]
                last = i == len(sched) - 1
                if last:
                    if u is not None:
                        errs.append("train.tier_schedule[-1].until must be null")
                else:
                    if not isinstance(u, int) or u <= prev_until:
                        errs.append("train.tier_schedule[%d].until must be an "
                                    "increasing int" % i)
                    else:
                        prev_until = u

    rf, ok = _cfg_get(cfg, "data.replay_frac")
    if ok and rf:
        if not isinstance(rf, (int, float)) or not (0.0 < rf < 1.0):
            errs.append("data.replay_frac: must be in (0,1)")
        rd, ok2 = _cfg_get(cfg, "data.replay_dir")
        if not ok2 or not isinstance(rd, str):
            errs.append("data.replay_dir: required (string) when replay_frac is set")

    init_from, ok = _cfg_get(cfg, "train.init_from")
    if ok and init_from is not None and not isinstance(init_from, str):
        errs.append("train.init_from: must be a string path")

    if errs:
        raise ValueError("config invalid:\n  " + "\n  ".join(errs))


def resolve_device(cfg: Dict[str, Any], arg: Optional[str] = None) -> str:
    import torch
    d = arg or cfg.get("device", "auto")
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    return str(d)


# ---------------------------------------------------------------------------
# Adaptive construction helpers (other agents own the constructors; we match
# whatever subset of these keyword names their signatures accept).
# ---------------------------------------------------------------------------

def _accepted_kwargs(fn: Callable) -> Tuple[set, bool]:
    sig = inspect.signature(fn)
    names = set()
    has_var = False
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            has_var = True
        elif p.name != "self":
            names.add(p.name)
    return names, has_var


def build_model(cfg: Dict[str, Any]):
    from sprig.model.sprig import SPRIGModel
    try:
        import dataclasses
        from sprig.model.sprig import SPRIGConfig
    except ImportError:
        SPRIGConfig = None  # type: ignore[assignment]
    if SPRIGConfig is not None:
        # SPRIGModel takes a SPRIGConfig dataclass; build it from the yaml
        # `model` section, dropping keys that are not config fields (e.g.
        # emb_dim/L_max, which belong to the data pipeline).
        fields = {f.name for f in dataclasses.fields(SPRIGConfig)}
        kw = {k: v for k, v in cfg.get("model", {}).items() if k in fields}
        return SPRIGModel(SPRIGConfig(**kw))
    names, has_var = _accepted_kwargs(SPRIGModel.__init__)
    for key in ("cfg", "config"):
        if key in names:
            return SPRIGModel(**{key: cfg})
    kw = dict(cfg.get("model", {}))
    if not has_var:
        kw = {k: v for k, v in kw.items() if k in names}
    return SPRIGModel(**kw)


def build_dataset(cfg: Dict[str, Any], data_dir: str, train: bool = False):
    from sprig.data.dataset import SprigDataset
    data_dir = str(Path(str(data_dir)).expanduser())
    d = cfg.get("data", {})
    p_null = float(d.get("p_null", 0.1)) if train else 0.0
    null_emb = d.get("null_emb")
    if null_emb:
        null_emb = str(Path(str(null_emb)).expanduser())
    cand = {
        "null_emb": null_emb, "null_emb_path": null_emb, "null_path": null_emb,
        "p_null": p_null, "null_p": p_null, "null_prob": p_null,
        "train": train, "L_max": cfg.get("model", {}).get("L_max"),
        # objmask only matters for the weighted training loss / resurrection.
        "emit_obj_mask": bool(d.get("emit_obj_mask", False)) if train else False,
    }
    names, has_var = _accepted_kwargs(SprigDataset.__init__)
    kw = {k: v for k, v in cand.items() if (has_var or k in names) and v is not None}
    for pname in ("data_dir", "root", "path", "split_dir", "dir"):
        if pname in names:
            kw[pname] = data_dir
            return SprigDataset(**kw)
    return SprigDataset(data_dir, **kw)


class InfiniteRandomSampler:
    """Fallback index sampler: endless shuffled epochs from a seeded generator.

    Used when there is no tier schedule or TierCurriculumSampler cannot be
    constructed. Duck-types torch Sampler (DataLoader accepts any iterable).
    """

    def __init__(self, n: int, seed: int = 0):
        import torch
        self.n = int(n)
        self.g = torch.Generator()
        self.g.manual_seed(int(seed))

    def __iter__(self) -> Iterator[int]:
        import torch
        while True:
            for i in torch.randperm(self.n, generator=self.g).tolist():
                yield i

    def state_dict(self) -> Dict[str, Any]:
        return {"g": self.g.get_state(), "n": self.n}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.g.set_state(sd["g"])


def _tier_schedule_pairs(sched: List[Any]) -> List[Tuple[int, List[float]]]:
    """Convert config-style [{until, weights}, ...] entries (entry i active
    until step `until`, last entry has until: null) into the
    [(step_start, weights), ...] pairs TierCurriculumSampler expects."""
    pairs: List[Tuple[int, List[float]]] = []
    start = 0
    for ent in sched:
        if isinstance(ent, dict):
            pairs.append((start, list(ent["weights"])))
            u = ent.get("until")
            if u is not None:
                start = int(u)
        else:  # already (step_start, weights)-shaped
            pairs.append((int(ent[0]), list(ent[1])))
    return pairs


def _tier_indices_for(dataset) -> Optional[List[Any]]:
    """Per-tier sample-index arrays for a dataset, if discoverable."""
    ti = getattr(dataset, "tier_indices", None)
    if ti is not None and len(ti):
        return list(ti)
    root = getattr(dataset, "root", None)
    if root is not None:
        try:
            from sprig.data.dataset import load_tier_indices
            return list(load_tier_indices(str(root), len(dataset)))
        except Exception:
            pass
    tier = getattr(dataset, "tier", None)
    if tier is not None:
        import numpy as np
        t = np.asarray(tier)
        return [np.nonzero(t == k)[0].astype(np.int64)
                for k in range(int(t.max()) + 1)]
    return None


def build_sampler(cfg: Dict[str, Any], dataset, seed: int) -> Tuple[Any, bool]:
    """Returns (sampler, is_batch_sampler)."""
    sched = cfg.get("train", {}).get("tier_schedule")
    if sched:
        TCS = None
        try:
            from sprig.data.dataset import TierCurriculumSampler as TCS  # type: ignore
        except Exception:
            try:
                from sprig.data import TierCurriculumSampler as TCS  # type: ignore
            except Exception:
                TCS = None
        if TCS is not None:
            try:
                names, has_var = _accepted_kwargs(TCS.__init__)
                pairs = _tier_schedule_pairs(sched)
                tiers = _tier_indices_for(dataset)
                if tiers is not None:
                    # Pad with empty tiers so weight vectors line up (the
                    # sampler renormalizes empty tiers away).
                    import numpy as np
                    want = max(len(w) for _, w in pairs)
                    while len(tiers) < want:
                        tiers.append(np.zeros(0, dtype=np.int64))
                cand = {
                    "dataset": dataset,
                    "tiers": tiers,
                    "tier_indices": tiers,
                    "schedule": pairs, "tier_schedule": pairs,
                    "seed": seed,
                    "total_steps": cfg["train"]["total_steps"],
                    "batch_size": cfg["train"]["batch_size"],
                }
                kw = {k: v for k, v in cand.items()
                      if (has_var or k in names) and v is not None}
                sampler = TCS(**kw)
                # TierCurriculumSampler yields single sample indices (its
                # batch_size only converts draws to optimizer steps), so it is
                # a plain index sampler, never a batch sampler.
                return sampler, False
            except Exception:
                print("[train] WARNING: TierCurriculumSampler construction failed; "
                      "falling back to uniform sampling", file=sys.stderr)
                traceback.print_exc()
    return InfiniteRandomSampler(len(dataset), seed), False


def find_collate(dataset) -> Optional[Callable]:
    for attr in ("collate", "collate_fn"):
        fn = getattr(dataset, attr, None)
        if callable(fn):
            return fn
    try:
        from sprig.data import dataset as dmod
        for attr in ("collate", "collate_fn", "sprig_collate"):
            fn = getattr(dmod, attr, None)
            if callable(fn):
                return fn
    except Exception:
        pass
    return None


def build_loader(cfg: Dict[str, Any], dataset, sampler, is_batch: bool, device: str):
    import torch
    kw: Dict[str, Any] = {
        "num_workers": int(cfg.get("data", {}).get("num_workers", 0)),
        "collate_fn": find_collate(dataset),
        "pin_memory": device.startswith("cuda"),
    }
    if kw["num_workers"] > 0:
        kw["persistent_workers"] = True
    if is_batch:
        return torch.utils.data.DataLoader(dataset, batch_sampler=sampler, **kw)
    return torch.utils.data.DataLoader(
        dataset, batch_size=int(cfg["train"]["batch_size"]), sampler=sampler, **kw)


def infinite_batches(loader) -> Iterator[Any]:
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    import torch
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
    return out


class FixedSubset:
    """Repeat a fixed set of base-dataset indices to a virtual length.

    Used by the overfit gate scripts (G1/G2) to train on N fixed images.
    """

    def __init__(self, dataset, indices: List[int], virtual_len: Optional[int] = None):
        self.dataset = dataset
        self.indices = list(indices)
        self.virtual_len = int(virtual_len) if virtual_len else len(self.indices)

    def __len__(self) -> int:
        return self.virtual_len

    def __getitem__(self, i: int):
        return self.dataset[self.indices[i % len(self.indices)]]

    @property
    def collate(self):
        return find_collate(self.dataset)


# ---------------------------------------------------------------------------
# Optimizer / schedules / EMA
# ---------------------------------------------------------------------------

_TABLE_RE = re.compile(r"(^|\.)(E_N|E_T|V|W|P_T)($|\.)|cut_type|bias_grid")


def split_param_groups(model) -> Tuple[List, List]:
    """Embedding tables (E_N, E_T, V, W, P_T, cut-type tables, bias grid) vs
    networks (GMT, atlas renderer, Phi, everything else)."""
    tables, nets = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (tables if _TABLE_RE.search(name) else nets).append(p)
    return tables, nets


def build_optimizer(model, cfg: Dict[str, Any]):
    import torch
    t = cfg["train"]
    tables, nets = split_param_groups(model)
    groups = []
    if tables:
        groups.append({"params": tables, "lr": float(t["lr_tables"]),
                       "name": "tables"})
    if nets:
        groups.append({"params": nets, "lr": float(t["lr_networks"]),
                       "name": "networks"})
    if not groups:
        raise ValueError("model has no trainable parameters")
    return torch.optim.AdamW(groups, betas=tuple(t["betas"]),
                             weight_decay=float(t.get("weight_decay", 0.0)))


def build_scheduler(opt, cfg: Dict[str, Any]):
    import torch
    t = cfg["train"]
    warm = max(1, int(t["warmup_steps"]))
    decay = int(t.get("lr_decay_steps", t["total_steps"]))
    min_ratio = float(t.get("lr_min_ratio", 0.1))

    def lam(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        prog = min(1.0, (step - warm) / max(1, decay - warm))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, [lam] * len(opt.param_groups))


def tau_at(cfg: Dict[str, Any], step: int) -> float:
    t = cfg["train"]
    s0 = float(t.get("tau_start", 2.0))
    s1 = float(t.get("tau_end", 1.0))
    n = max(1, int(t.get("tau_steps", 50000)))
    frac = min(1.0, step / float(n))
    return s0 + (s1 - s0) * frac


def eta_anneal_factor(cfg: Dict[str, Any], step: int, total_steps: int) -> float:
    """Linear eta -> 0 over the final `eta_final_anneal_steps` (DESIGN section 7)."""
    ann = int(cfg["train"].get("eta_final_anneal_steps", 20000))
    if ann <= 0 or step < total_steps - ann:
        return 1.0
    return max(0.0, (total_steps - step) / float(ann))


def pi_update_eta(eta: float, node_entropy: float,
                  band: Tuple[float, float] = (0.5, 3.0), eta_max: float = 1.5,
                  step_up: float = 0.05, step_down: float = 0.05) -> float:
    """DESIGN section 5 PI controller (local fallback)."""
    lo, hi = band
    if node_entropy < lo:
        eta = eta + step_up + 0.1 * (lo - node_entropy)
    elif node_entropy > hi:
        eta = eta - step_down
    return float(min(max(eta, 0.0), eta_max))


def _external_pi() -> Optional[Callable]:
    try:
        from sprig.eval import monitors
    except Exception:
        return None
    for name in ("pi_update_eta", "update_eta", "eta_pi_update", "pi_controller"):
        fn = getattr(monitors, name, None)
        if callable(fn):
            return fn
    return None


class EMA:
    """Exponential moving average of parameters, eval-only (DESIGN section 2)."""

    def __init__(self, model, decay: float):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone().float()
                       for k, p in model.named_parameters() if p.requires_grad}

    def update(self, model) -> None:
        import torch
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in self.shadow:
                    self.shadow[k].mul_(self.decay).add_(
                        p.detach().float().to(self.shadow[k].device),
                        alpha=1.0 - self.decay)

    @contextlib.contextmanager
    def swap(self, model):
        import torch
        backup: Dict[str, Any] = {}
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k in self.shadow:
                    backup[k] = p.detach().clone()
                    p.copy_(self.shadow[k].to(dtype=p.dtype, device=p.device))
        try:
            yield model
        finally:
            with torch.no_grad():
                for k, p in model.named_parameters():
                    if k in backup:
                        p.copy_(backup[k])

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        self.decay = float(sd["decay"])
        self.shadow = {k: v.clone().float() for k, v in sd["shadow"].items()}


def set_model_scalar(model, names: Tuple[str, ...], value: float) -> bool:
    import torch
    for n in names:
        if hasattr(model, n):
            cur = getattr(model, n)
            if isinstance(cur, torch.Tensor):
                with torch.no_grad():
                    cur.fill_(float(value))
            elif isinstance(cur, (int, float)):
                setattr(model, n, float(value))
            else:
                continue
            return True
    return False


def get_model_scalar(model, names: Tuple[str, ...], default: float = 0.0) -> float:
    import torch
    for n in names:
        v = getattr(model, n, None)
        if isinstance(v, torch.Tensor) and v.numel() >= 1:
            return float(v.detach().flatten()[0])
        if isinstance(v, (int, float)):
            return float(v)
    return default


@contextlib.contextmanager
def report_mode(model):
    """model.eval() + reported-numbers mode (eta=0), restored on exit (C2)."""
    import torch
    was_training = model.training
    model.eval()
    flag = getattr(model, "report_mode", None)
    use_flag = hasattr(model, "report_mode") and not callable(flag)
    saved_eta = None
    eta_kind = None
    if use_flag:
        model.report_mode = True
    else:
        v = getattr(model, "eta", None)
        if isinstance(v, torch.Tensor):
            saved_eta = v.detach().clone()
            eta_kind = "tensor"
            with torch.no_grad():
                v.fill_(0.0)
        elif isinstance(v, (int, float)):
            saved_eta = v
            eta_kind = "scalar"
            model.eta = 0.0
    try:
        yield model
    finally:
        if use_flag:
            model.report_mode = flag if flag is not None else False
        if eta_kind == "tensor":
            with torch.no_grad():
                getattr(model, "eta").copy_(saved_eta)
        elif eta_kind == "scalar":
            model.eta = saved_eta
        if was_training:
            model.train()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class ScalarsLogger:
    """Appends one JSON object per log() to <run_dir>/scalars.jsonl and mirrors
    numeric values to TensorBoard event files under <run_dir>/tb."""

    def __init__(self, run_dir, tensorboard: bool = True):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "scalars.jsonl"
        self._fh = open(self.path, "a")
        self._want_tb = tensorboard
        self._tb: Any = None

    def _tb_writer(self):
        if self._tb is None and self._want_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception:
                self._want_tb = False
                self._tb = False
        return self._tb if self._tb else None

    def log(self, step: int, scalars: Dict[str, Any]) -> None:
        rec: Dict[str, Any] = {"step": int(step), "time": time.time()}
        numeric: Dict[str, float] = {}
        for k, v in scalars.items():
            if isinstance(v, bool):
                rec[k] = v
            elif isinstance(v, (int, float)):
                rec[k] = float(v)
                numeric[k] = float(v)
            elif hasattr(v, "__float__"):
                rec[k] = float(v)
                numeric[k] = float(v)
            else:
                rec[k] = v
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()
        tb = self._tb_writer()
        if tb is not None:
            for k, v in numeric.items():
                if math.isfinite(v):
                    tb.add_scalar(k, v, int(step))

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
        if self._tb:
            try:
                self._tb.close()
            except Exception:
                pass


def _numeric_items(d: Any, prefix: str = "") -> Dict[str, float]:
    import torch
    out: Dict[str, float] = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[prefix + str(k)] = float(v)
        elif isinstance(v, torch.Tensor) and v.numel() == 1:
            out[prefix + str(k)] = float(v.detach())
    return out


def _safe(name: str, fn: Callable[[], Any], logger: Optional[ScalarsLogger],
          step: int) -> Any:
    """Run an eval/monitor callback; a failure is logged and never fatal."""
    try:
        return fn()
    except Exception:
        print("[train] non-fatal failure in %s @ step %d\n%s"
              % (name, step, traceback.format_exc()), file=sys.stderr)
        if logger is not None:
            try:
                logger.log(step, {"eval_error/%s" % name: 1.0})
            except Exception:
                pass
        return None


# ---------------------------------------------------------------------------
# RNG + checkpointing
# ---------------------------------------------------------------------------

def capture_rng() -> Dict[str, Any]:
    import numpy as np
    import torch
    st: Dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        try:
            st["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return st


def restore_rng(st: Dict[str, Any]) -> None:
    import numpy as np
    import torch
    torch.set_rng_state(st["torch"].cpu().to(torch.uint8)
                        if hasattr(st["torch"], "cpu") else st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:
            pass


def save_checkpoint(path, payload: Dict[str, Any]) -> None:
    """Atomic write: serialize to <name>.tmp in the same dir, then rename."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, str(tmp))
    os.replace(str(tmp), str(path))


def save_rolling(run_dir, payload: Dict[str, Any],
                 permanent_step: Optional[int] = None) -> None:
    """Rolling last.pt/prev.pt plus optional permanent step_XXXXXXX.pt."""
    import torch
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    last = run_dir / "last.pt"
    prev = run_dir / "prev.pt"
    tmp = run_dir / "last.pt.tmp"
    torch.save(payload, str(tmp))
    if last.exists():
        os.replace(str(last), str(prev))
    os.replace(str(tmp), str(last))
    if permanent_step is not None:
        perm = run_dir / ("step_%07d.pt" % int(permanent_step))
        ptmp = run_dir / ("step_%07d.pt.tmp" % int(permanent_step))
        shutil.copyfile(str(last), str(ptmp))
        os.replace(str(ptmp), str(perm))


def load_checkpoint(path):
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


def make_payload(model, opt, sched, ema: EMA, step: int, sampler,
                 cfg: Dict[str, Any], eta: float, tau: float) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "ema": ema.state_dict(),
        "step": int(step),
        "rng": capture_rng(),
        "config": cfg,
        "eta": float(eta),
        "tau": float(tau),
    }
    if hasattr(sampler, "state_dict"):
        try:
            payload["sampler"] = sampler.state_dict()
        except Exception:
            pass
    return payload


# ---------------------------------------------------------------------------
# Monitors, PI controller, dead-texel resurrection (every 2k steps)
# ---------------------------------------------------------------------------

def resurrect_dead_texels(model, texel_usage, images, cfg: Dict[str, Any],
                          obj_mask=None) -> int:
    """DESIGN section 6: texels with usage < 0.1/T_v get their bias grid
    overwritten with a training-image crop (DL-mean channels) and their
    E_T row perturbed. Prefers a model/atlas-provided hook when available.
    obj_mask [B,C,C] (optional) biases reseed crops onto object pixels."""
    import torch
    T_v = int(cfg["model"]["T_v"])
    thresh = float(cfg["train"].get("texel_dead_frac", 0.1)) / T_v
    u = texel_usage.detach().float().cpu().flatten()
    u = u / max(float(u.sum()), 1e-12)
    dead = (u < thresh).nonzero().flatten().tolist()
    if not dead:
        return 0
    for host in (model, getattr(model, "atlas", None)):
        if host is None:
            continue
        for name in ("resurrect_texels", "resurrect"):
            fn = getattr(host, name, None)
            if callable(fn):
                names, has_var = _accepted_kwargs(fn)
                cand = {"dead": dead, "dead_ids": dead, "texel_ids": dead,
                        "ids": dead, "images": images, "usage": texel_usage,
                        "obj_mask": obj_mask}
                kw = {k: v for k, v in cand.items() if has_var or k in names}
                fn(**kw)
                return len(dead)
    atlas = getattr(model, "atlas", None)
    grid = None
    for name in ("bias_grid", "bias", "texel_bias"):
        g = getattr(atlas, name, None) if atlas is not None else None
        if isinstance(g, torch.Tensor) and g.dim() == 4:
            grid = g
            break
    if grid is None:
        raise RuntimeError("no resurrection hook and no atlas bias grid found; "
                           "skipping resurrection of %d texels" % len(dead))
    E_T = getattr(atlas, "E_T", None)
    if hasattr(E_T, "weight"):
        E_T = E_T.weight
    B, H, W = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    ph, pw = int(grid.shape[2]), int(grid.shape[3])
    n_comp = max(1, int(grid.shape[1]) // 10)
    with torch.no_grad():
        for t in dead:
            b = random.randrange(B)
            y = random.randrange(max(1, H - ph + 1))
            x = random.randrange(max(1, W - pw + 1))
            crop = images[b, y:y + ph, x:x + pw, :3].float() / 255.0 * 2.0 - 1.0
            crop = crop.permute(2, 0, 1).to(grid.device, grid.dtype)
            gt = grid[t]
            gt.normal_(0.0, 0.01)
            for k in range(n_comp):
                base = k * 10
                gt[base + 1:base + 4] = crop
            if isinstance(E_T, torch.Tensor) and E_T.dim() == 2:
                E_T[t].add_(torch.randn_like(E_T[t]) * 0.01)
    return len(dead)


def run_monitors(model, batch: Dict[str, Any], cfg: Dict[str, Any],
                 state: Dict[str, Any], logger: Optional[ScalarsLogger],
                 step: int) -> None:
    """posterior_usage on the current batch -> PI eta update, resurrection,
    monitor scalars (DESIGN sections 4/5/6)."""
    import torch
    usage = model.posterior_usage(batch["image"], batch["emb"], batch["emb_len"])
    node_h = float(usage["node_entropy"]) if "node_entropy" in usage else float("nan")

    t = cfg["train"]
    new_eta: Optional[float] = None
    ext = _external_pi()
    if ext is not None:
        try:
            new_eta = float(ext(state["eta"], node_h))
        except Exception:
            new_eta = None
    if new_eta is None:
        new_eta = pi_update_eta(
            state["eta"], node_h,
            band=tuple(t.get("eta_band", [0.5, 3.0])),
            eta_max=float(t.get("eta_max", 1.5)),
            step_up=float(t.get("eta_step_up", 0.05)),
            step_down=float(t.get("eta_step_down", 0.05)))
    state["eta"] = new_eta

    scalars: Dict[str, float] = {"mon/node_entropy": node_h, "mon/eta_pi": new_eta}
    su = usage.get("symbol_usage")
    if isinstance(su, torch.Tensor):
        p = su.detach().float().cpu().flatten()
        p = p / p.sum().clamp_min(1e-12)
        scalars["mon/S_eff"] = float(torch.exp(-(p * (p + 1e-12).log()).sum()))
    tu = usage.get("texel_usage")
    n_res = 0
    if isinstance(tu, torch.Tensor):
        T_v = int(cfg["model"]["T_v"])
        thr = float(t.get("texel_dead_frac", 0.1)) / T_v
        q = tu.detach().float().cpu().flatten()
        q = q / q.sum().clamp_min(1e-12)
        scalars["mon/texel_alive_frac"] = float((q >= thr).float().mean())
        res = _safe("resurrection",
                    lambda: resurrect_dead_texels(model, tu, batch["image"], cfg,
                                                  obj_mask=batch.get("objmask")),
                    logger, step)
        n_res = int(res or 0)
    scalars["mon/resurrected"] = float(n_res)
    for k in ("emit_mag", "rule_mag", "mean_depth", "mean_leaves"):
        if k in usage:
            v = usage[k]
            scalars["mon/" + k] = float(v.detach()) if isinstance(v, torch.Tensor) \
                else float(v)
    if logger is not None:
        logger.log(step, scalars)


# ---------------------------------------------------------------------------
# Eval helpers (called on cadence; every callback is wrapped by _safe)
# ---------------------------------------------------------------------------

def bpd_from_logZ(logZ, canvas: int):
    return (-logZ / (3.0 * canvas * canvas)) / math.log(2.0)


def _default_collate(dataset) -> Callable:
    fn = find_collate(dataset)
    if fn is not None:
        return fn
    import torch
    return torch.utils.data.default_collate


def dataset_bpd(model, dataset, cfg: Dict[str, Any], device: str,
                n: int, batch_size: int) -> float:
    import torch
    canvas = int(cfg["model"]["canvas"])
    collate = _default_collate(dataset)
    n = min(int(n), len(dataset))
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, n))]
            b = move_batch(collate(items), device)
            logZ = model.log_marginal(b["image"], b["emb"], b["emb_len"])
            tot += float(bpd_from_logZ(logZ, canvas).sum())
            cnt += int(logZ.shape[0])
    return tot / max(1, cnt)


def load_null_emb(cfg: Dict[str, Any]):
    import numpy as np
    import torch
    p = Path(str(cfg["data"]["null_emb"])).expanduser()
    arr = np.fromfile(str(p), dtype=np.float16).reshape(-1, 768)
    return torch.from_numpy(arr.copy())


def delta_c_and_margin(model, dataset, cfg: Dict[str, Any], device: str,
                       n: int, batch_size: int) -> Tuple[float, float]:
    """Caption info gain delta_c = bpd(x,null) - bpd(x,c), and in-batch
    caption-swap margin = fraction with logZ(correct) > logZ(swapped)."""
    import torch
    canvas = int(cfg["model"]["canvas"])
    null = load_null_emb(cfg)
    collate = _default_collate(dataset)
    n = min(int(n), len(dataset))
    d_sum, m_sum, cnt = 0.0, 0.0, 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            items = [dataset[i] for i in range(start, min(start + batch_size, n))]
            b = move_batch(collate(items), device)
            emb, emb_len = b["emb"], b["emb_len"]
            B = int(emb.shape[0])
            logZ_c = model.log_marginal(b["image"], emb, emb_len)
            nb = null.unsqueeze(0).expand(B, -1, -1).to(device=emb.device,
                                                        dtype=emb.dtype)
            nl = torch.full((B,), null.shape[0], dtype=emb_len.dtype,
                            device=emb_len.device)
            logZ_0 = model.log_marginal(b["image"], nb, nl)
            logZ_sw = model.log_marginal(
                b["image"], torch.roll(emb, 1, dims=0), torch.roll(emb_len, 1, dims=0))
            d_sum += float((bpd_from_logZ(logZ_0, canvas)
                            - bpd_from_logZ(logZ_c, canvas)).sum())
            m_sum += float((logZ_c > logZ_sw).float().sum())
            cnt += B
    cnt = max(1, cnt)
    return d_sum / cnt, m_sum / cnt


def load_prompt_bank(cfg: Dict[str, Any], device: str):
    """Loads precomputed prompt-bank embeddings -> (emb [P,L,768], emb_len [P])."""
    import torch
    try:
        from sprig.eval import prompts as pr
        for name in ("load_prompt_bank_emb", "load_bank", "load_prompt_bank"):
            fn = getattr(pr, name, None)
            if callable(fn):
                out = fn()
                if isinstance(out, tuple) and len(out) >= 2:
                    return out[0].to(device), out[1].to(device)
                if isinstance(out, dict):
                    return out["emb"].to(device), out["emb_len"].to(device)
    except Exception:
        pass
    p = cfg.get("eval", {}).get("prompt_bank_emb")
    if not p:
        raise FileNotFoundError("eval.prompt_bank_emb not configured")
    d = torch.load(str(Path(str(p)).expanduser()), map_location="cpu",
                   weights_only=False)
    if isinstance(d, dict):
        return d["emb"].to(device), d["emb_len"].to(device)
    raise ValueError("unrecognized prompt bank format at %s" % p)


def _to_u8_np(images):
    import numpy as np
    import torch
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    return np.asarray(images).astype("uint8")


def save_image_rows(rows: List[Any], path) -> None:
    """rows: list of [H,W,3] u8 arrays; stacked vertically, padded to max W."""
    import numpy as np
    from PIL import Image
    width = max(int(r.shape[1]) for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < width:
            pad = np.zeros((r.shape[0], width - r.shape[1], 3), dtype="uint8")
            r = np.concatenate([r, pad], axis=1)
        padded.append(r)
    canvas = np.concatenate(padded, axis=0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(str(path))


def draw_parse_overlay(image_u8, root, scale: int = 4):
    """Draws MAP-parse region rectangles (depth-colored) over the image."""
    import numpy as np
    from PIL import Image, ImageDraw
    img = Image.fromarray(np.asarray(image_u8, dtype="uint8")).convert("RGB")
    img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    d = ImageDraw.Draw(img)
    palette = [(255, 64, 64), (64, 255, 64), (64, 128, 255), (255, 255, 64),
               (255, 64, 255), (64, 255, 255), (255, 160, 64), (200, 200, 200)]
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node is None:
            continue
        rect = tuple(int(v) for v in node.rect)
        x0, y0, x1, y1 = [v * scale for v in rect]
        d.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)],
                    outline=palette[depth % len(palette)], width=1)
        for ch in (getattr(node, "children", None) or []):
            stack.append((ch, depth + 1))
    return img


def eval_val_fast(model, ema: EMA, eval_datasets: Dict[str, Any],
                  cfg: Dict[str, Any], device: str, logger: ScalarsLogger,
                  step: int) -> None:
    ds = eval_datasets.get("val_fast") or eval_datasets.get("val")
    if ds is None:
        print("[train] val_fast eval skipped: no val dataset", file=sys.stderr)
        return
    e = cfg.get("eval", {})
    with ema.swap(model), report_mode(model):
        bpd = dataset_bpd(model, ds, cfg, device,
                          int(e.get("val_fast_n", 512)),
                          int(e.get("eval_batch_size", 64)))
    logger.log(step, {"val_fast/bpd": bpd})


def eval_full_val(model, ema: EMA, eval_datasets: Dict[str, Any],
                  cfg: Dict[str, Any], device: str, logger: ScalarsLogger,
                  step: int) -> None:
    ds = eval_datasets.get("val") or eval_datasets.get("val_fast")
    if ds is None:
        print("[train] full eval skipped: no val dataset", file=sys.stderr)
        return
    e = cfg.get("eval", {})
    bs = int(e.get("eval_batch_size", 64))
    scal: Dict[str, float] = {}
    with ema.swap(model), report_mode(model):
        scal["val/bpd"] = dataset_bpd(model, ds, cfg, device,
                                      int(e.get("full_val_n", 20000)), bs)
        try:
            dc, margin = delta_c_and_margin(model, ds, cfg, device,
                                            min(512, len(ds)), bs)
            scal["val/delta_c"] = dc
            scal["val/swap_margin"] = margin
        except Exception:
            print("[train] delta_c/margin failed:\n%s" % traceback.format_exc(),
                  file=sys.stderr)
    logger.log(step, scal)


def eval_parse_overlays(model, ema: EMA, eval_datasets: Dict[str, Any],
                        cfg: Dict[str, Any], device: str, run_dir, step: int) -> None:
    import numpy as np
    import torch
    ds = (eval_datasets.get("parse_eval") or eval_datasets.get("val")
          or eval_datasets.get("val_fast"))
    if ds is None:
        print("[train] parse overlays skipped: no eval dataset", file=sys.stderr)
        return
    n = min(int(cfg.get("eval", {}).get("parse_images", 32)), len(ds))
    collate = _default_collate(ds)
    tiles: List[Any] = []
    with torch.no_grad(), ema.swap(model), report_mode(model):
        for start in range(0, n, 8):
            items = [ds[i] for i in range(start, min(start + 8, n))]
            b = move_batch(collate(items), device)
            roots = model.map_parse(b["image"], b["emb"], b["emb_len"])
            if not isinstance(roots, (list, tuple)):
                roots = [roots]
            for j, root in enumerate(roots):
                img = b["image"][j].detach().cpu().numpy()
                tiles.append(np.asarray(draw_parse_overlay(img, root)))
    ncol = 8
    rows = [np.concatenate(tiles[i:i + ncol], axis=1)
            for i in range(0, len(tiles), ncol)]
    save_image_rows(rows, Path(run_dir) / "eval" / ("parse_%07d.png" % step))


def eval_prompt_grid(model, ema: EMA, cfg: Dict[str, Any], device: str,
                     run_dir, step: int, n_prompts: int, per_prompt: int,
                     tag: str) -> None:
    import numpy as np
    import torch
    emb, emb_len = load_prompt_bank(cfg, device)
    rows: List[Any] = []
    with torch.no_grad(), ema.swap(model), report_mode(model):
        for i in range(min(int(n_prompts), int(emb.shape[0]))):
            imgs, _trees = model.sample(emb[i:i + 1], emb_len[i:i + 1],
                                        1000 + i, 2000 + i, int(per_prompt))
            arr = _to_u8_np(imgs)
            rows.append(np.concatenate(list(arr), axis=1))
    save_image_rows(rows, Path(run_dir) / "eval" / ("%s_%07d.png" % (tag, step)))


def eval_seed_split_grid(model, ema: EMA, cfg: Dict[str, Any], device: str,
                         run_dir, step: int, n_prompts: int = 4,
                         per_row: int = 4) -> None:
    """Layout-freeze (fixed structural seed) vs material-reroll rows (C4)."""
    import numpy as np
    import torch
    emb, emb_len = load_prompt_bank(cfg, device)
    rows: List[Any] = []
    with torch.no_grad(), ema.swap(model), report_mode(model):
        for i in range(min(int(n_prompts), int(emb.shape[0]))):
            e, l = emb[i:i + 1], emb_len[i:i + 1]
            frozen_struct = [_to_u8_np(model.sample(e, l, 7, 100 + j, 1)[0])[0]
                             for j in range(per_row)]
            frozen_material = [_to_u8_np(model.sample(e, l, 100 + j, 7, 1)[0])[0]
                               for j in range(per_row)]
            rows.append(np.concatenate(frozen_struct + frozen_material, axis=1))
    save_image_rows(rows, Path(run_dir) / "eval" / ("seedsplit_%07d.png" % step))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_loop(cfg: Dict[str, Any], dataset, steps: Optional[int] = None,
               model=None, run_dir=None, device: Optional[str] = None,
               resume: str = "none", eval_datasets: Optional[Dict[str, Any]] = None,
               replay_dataset=None) -> Dict[str, Any]:
    """Core training loop (DESIGN sections 2/6/7). Returns a result dict with
    step counts, bpd history, and the trained model/EMA."""
    import numpy as np
    import torch

    tcfg = cfg["train"]
    ecfg = cfg.get("eval", {})
    eval_datasets = eval_datasets or {}
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    random.seed(seed)

    device = device or resolve_device(cfg)
    if model is None:
        model = build_model(cfg)
    model.to(device)
    model.train()

    total = int(steps if steps is not None else tcfg["total_steps"])
    opt = build_optimizer(model, cfg)
    sched = build_scheduler(opt, cfg)
    ema = EMA(model, float(tcfg.get("ema", 0.9999)))
    state = {"eta": get_model_scalar(model, ("eta",), 0.0)}

    run_dir = Path(run_dir) if run_dir is not None else None
    logger = ScalarsLogger(run_dir) if run_dir is not None else None

    # --- resume / init_from -------------------------------------------------
    start_step = 0
    resumed = False
    sampler_state = None
    if run_dir is not None and resume and resume != "none":
        ckpt_path = None
        if resume == "auto":
            for cand in ("last.pt", "prev.pt"):
                if (run_dir / cand).exists():
                    ckpt_path = run_dir / cand
                    break
        elif Path(resume).exists():
            ckpt_path = Path(resume)
        if ckpt_path is not None:
            payload = load_checkpoint(ckpt_path)
            model.load_state_dict(payload["model"])
            opt.load_state_dict(payload["optimizer"])
            sched.load_state_dict(payload["scheduler"])
            ema.load_state_dict(payload["ema"])
            restore_rng(payload["rng"])
            start_step = int(payload["step"])
            state["eta"] = float(payload.get("eta", state["eta"]))
            sampler_state = payload.get("sampler")
            resumed = True
            print("[train] resumed from %s at step %d" % (ckpt_path, start_step))

    init_from = tcfg.get("init_from")
    if init_from and not resumed:
        payload = load_checkpoint(Path(str(init_from)).expanduser())
        sd = payload["model"] if isinstance(payload, dict) and "model" in payload \
            else payload
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print("[train] init_from: %d missing / %d unexpected keys"
                  % (len(missing), len(unexpected)), file=sys.stderr)
        ema = EMA(model, float(tcfg.get("ema", 0.9999)))
        print("[train] initialized weights from %s" % init_from)

    # --- data ----------------------------------------------------------------
    sampler, is_batch = build_sampler(cfg, dataset, seed)
    if sampler_state is not None and hasattr(sampler, "load_state_dict"):
        try:
            sampler.load_state_dict(sampler_state)
        except Exception:
            print("[train] WARNING: sampler state restore failed", file=sys.stderr)
    loader = build_loader(cfg, dataset, sampler, is_batch, device)
    batches = infinite_batches(loader)

    replay_frac = 0.0
    replay_batches = None
    if replay_dataset is not None:
        replay_frac = float(cfg.get("data", {}).get("replay_frac", 0.0) or 0.0)
    if replay_frac > 0.0:
        rs = InfiniteRandomSampler(len(replay_dataset), seed ^ 0x5EED)
        replay_batches = infinite_batches(
            build_loader(cfg, replay_dataset, rs, False, device))
    replay_rng = random.Random(seed ^ 0x0DDBA11)

    # --- schedule constants ---------------------------------------------------
    scalars_every = max(1, int(ecfg.get("scalars_every", 500)))
    val_fast_every = max(1, int(ecfg.get("val_fast_every", 2000)))
    parse_every = max(1, int(ecfg.get("parse_every", 10000)))
    full_every = max(1, int(ecfg.get("full_every", 25000)))
    eta_every = max(1, int(tcfg.get("eta_update_every", 2000)))
    ckpt_steps = max(1, int(cfg.get("checkpoint", {}).get("every_steps", 25000)))
    ckpt_secs = float(cfg.get("checkpoint", {}).get("every_minutes", 30)) * 60.0
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    use_amp = device.startswith("cuda")

    last_ckpt_time = time.time()
    t_last = time.time()
    loss_acc: List[float] = []
    bpd_history: List[Tuple[int, float]] = []
    stopped = False
    gstep = start_step
    tau = tau_at(cfg, start_step)

    for step in range(start_step, total):
        if run_dir is not None and (run_dir / "STOP").exists():
            print("[train] STOP file present; checkpointing and exiting")
            stopped = True
            break

        if hasattr(sampler, "set_step"):
            try:
                sampler.set_step(step)
            except Exception:
                pass

        tau = tau_at(cfg, step)
        set_model_scalar(model, ("tau_ann", "tau"), tau)
        eta_eff = state["eta"] * eta_anneal_factor(cfg, step, total)
        set_model_scalar(model, ("eta",), eta_eff)

        use_replay = (replay_batches is not None
                      and replay_rng.random() < replay_frac)
        batch = next(replay_batches) if use_replay else next(batches)
        batch = move_batch(batch, device)

        opt.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, metrics = model.loss(batch)
        else:
            loss, metrics = model.loss(batch)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        sched.step()
        ema.update(model)

        gstep = step + 1
        lv = float(loss.detach())
        m = metrics if isinstance(metrics, dict) else {}
        bpd = m.get("bpd")
        if bpd is None:
            bpd = lv / math.log(2.0)
        bpd = float(bpd.detach()) if hasattr(bpd, "detach") else float(bpd)
        bpd_history.append((gstep, bpd))
        loss_acc.append(lv)

        if logger is not None and gstep % scalars_every == 0:
            dt = time.time() - t_last
            t_last = time.time()
            rec = {
                "loss": sum(loss_acc) / len(loss_acc),
                "bpd": bpd,
                "grad_norm": float(gnorm),
                "tau": tau,
                "eta": eta_eff,
                "steps_per_s": (scalars_every / dt) if dt > 0 else float("nan"),
                "replay": 1.0 if use_replay else 0.0,
            }
            for g in opt.param_groups:
                rec["lr_%s" % g.get("name", "group")] = float(g["lr"])
            rec.update(_numeric_items(m, "train/"))
            logger.log(gstep, rec)
            loss_acc = []

        if gstep % eta_every == 0:
            _safe("monitors",
                  lambda b=batch, s=gstep: run_monitors(model, b, cfg, state,
                                                        logger, s),
                  logger, gstep)

        if logger is not None:
            if gstep % val_fast_every == 0:
                _safe("val_fast",
                      lambda s=gstep: eval_val_fast(model, ema, eval_datasets,
                                                    cfg, device, logger, s),
                      logger, gstep)
            if gstep % parse_every == 0:
                _safe("parse_overlays",
                      lambda s=gstep: eval_parse_overlays(
                          model, ema, eval_datasets, cfg, device, run_dir, s),
                      logger, gstep)
                _safe("prompt_grid",
                      lambda s=gstep: eval_prompt_grid(
                          model, ema, cfg, device, run_dir, s,
                          int(ecfg.get("prompt_grid_n", 8)), 4, "grid"),
                      logger, gstep)
            if gstep % full_every == 0:
                _safe("full_val",
                      lambda s=gstep: eval_full_val(model, ema, eval_datasets,
                                                    cfg, device, logger, s),
                      logger, gstep)
                _safe("bank_grid",
                      lambda s=gstep: eval_prompt_grid(
                          model, ema, cfg, device, run_dir, s, 32,
                          int(ecfg.get("bank_seeds", 8)), "bank"),
                      logger, gstep)
                _safe("seed_split",
                      lambda s=gstep: eval_seed_split_grid(model, ema, cfg,
                                                           device, run_dir, s),
                      logger, gstep)

        if run_dir is not None:
            perm = gstep if gstep % ckpt_steps == 0 else None
            if perm is not None or (time.time() - last_ckpt_time) >= ckpt_secs:
                save_rolling(run_dir,
                             make_payload(model, opt, sched, ema, gstep, sampler,
                                          cfg, state["eta"], tau),
                             permanent_step=perm)
                last_ckpt_time = time.time()

    if run_dir is not None:
        save_rolling(run_dir, make_payload(model, opt, sched, ema, gstep, sampler,
                                           cfg, state["eta"], tau))
    if logger is not None:
        logger.close()

    return {
        "step": gstep,
        "start_step": start_step,
        "stopped": stopped,
        "bpd_history": bpd_history,
        "model": model,
        "ema": ema,
        "eta": state["eta"],
        "run_dir": str(run_dir) if run_dir is not None else None,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="SPRIG v0.1 trainer")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--run-dir", default=None,
                    help="run directory (default: runs/<run_name>)")
    ap.add_argument("--resume", default="auto",
                    help="'auto' (pick up last.pt), 'none', or a checkpoint path")
    ap.add_argument("--device", default=None, help="cpu / cuda / auto")
    ap.add_argument("--steps", type=int, default=None,
                    help="override train.total_steps")
    ap.add_argument("--set", action="append", default=[], dest="overrides",
                    metavar="KEY=VAL", help="dotted config override, repeatable")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    apply_overrides(cfg, args.overrides)
    validate_config(cfg)

    run_dir = Path(args.run_dir) if args.run_dir else \
        Path("runs") / str(cfg.get("run_name", "run"))
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        dst = run_dir / "config.yaml"
        if Path(args.config).resolve() != dst.resolve():
            shutil.copyfile(args.config, str(dst))
        if Path(".sync_meta").exists():
            shutil.copyfile(".sync_meta", str(run_dir / ".sync_meta"))
    except Exception:
        pass

    device = resolve_device(cfg, args.device)
    d = cfg["data"]
    train_ds = build_dataset(cfg, d["train_dir"], train=True)

    eval_datasets: Dict[str, Any] = {}
    for key, name in (("val_fast_dir", "val_fast"), ("val_dir", "val"),
                      ("parse_eval_dir", "parse_eval")):
        p = d.get(key)
        if p and Path(str(p)).expanduser().exists():
            try:
                eval_datasets[name] = build_dataset(cfg, p, train=False)
            except Exception:
                print("[train] WARNING: could not build eval dataset %s"
                      % name, file=sys.stderr)
                traceback.print_exc()

    replay_ds = None
    if d.get("replay_dir") and d.get("replay_frac"):
        replay_ds = build_dataset(cfg, d["replay_dir"], train=True)

    train_loop(cfg, train_ds, steps=args.steps, run_dir=run_dir, device=device,
               resume=args.resume, eval_datasets=eval_datasets,
               replay_dataset=replay_ds)


if __name__ == "__main__":
    main()
