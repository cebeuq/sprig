"""Training-health monitors: pure functions over posterior-usage statistics.

Inputs come from `SPRIGModel.posterior_usage` (DESIGN §4/C3):
symbol_usage [S], texel_usage [T_v] (expected-count vectors, any positive
scale — normalized here), node_entropy (nats/node), emit_mag / rule_mag
(gradient-magnitude scalars), mean_depth, mean_leaves.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, "torch.Tensor", list]  # noqa: F821

# Alarm thresholds (plan §Verification).
S_EFF_COLLAPSE_FRAC = 0.15        # S_eff < 0.15*S -> grammar collapse
MAG_RATIO_ALARM = 100.0           # emission/rule ratio > 100 ...
NODE_ENTROPY_FLOOR = 0.1          # ... with node entropy < 0.1 nats
ALIVE_TEXEL_ALARM_FRAC = 0.25     # alive fraction below this -> texel death
QUADTREE_DEPTH = 6.0              # uniform 64px -> 8px binary-split depth
QUADTREE_LEAVES = 64.0            # 8x8 grid of 8px leaves
ALIVE_USAGE_THRESH_FRAC = 0.1     # texel alive iff usage > 0.1 / T_v


def _to_numpy(x: ArrayLike) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor without importing torch
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def s_eff(symbol_usage: ArrayLike) -> float:
    """Effective symbol count exp(H(usage)); usage is normalized internally."""
    p = _to_numpy(symbol_usage)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    nz = p[p > 0]
    return float(np.exp(-(nz * np.log(nz)).sum()))


def alive_texels(texel_usage: ArrayLike, T_v: int) -> int:
    """Number of texels with normalized usage above 0.1/T_v (resurrection thresh)."""
    p = _to_numpy(texel_usage)
    total = p.sum()
    if total <= 0:
        return 0
    p = p / total
    return int((p > ALIVE_USAGE_THRESH_FRAC / float(T_v)).sum())


def magnitude_ratio(emit_mag: float, rule_mag: float, eps: float = 1e-12) -> float:
    """Emission-vs-rule gradient magnitude ratio (tempered-DP failure detector)."""
    return float(emit_mag) / (float(rule_mag) + eps)


def pi_controller_eta(
    eta: float,
    node_entropy: float,
    band: Tuple[float, float] = (0.5, 3.0),
    eta_range: Tuple[float, float] = (0.0, 1.5),
) -> float:
    """One PI-controller update of the tempering exponent eta (DESIGN §5).

    If posterior node entropy H is below the band, raise eta by
    0.05 + 0.1*(lo - H); if above the band, lower it by 0.05; clamp to range.
    """
    lo, hi = band
    h = float(node_entropy)
    new_eta = float(eta)
    if h < lo:
        new_eta += 0.05 + 0.1 * (lo - h)
    elif h > hi:
        new_eta -= 0.05
    return float(min(max(new_eta, eta_range[0]), eta_range[1]))


def build_alarms(stats: Dict[str, object], S: int, T_v: int) -> Dict[str, bool]:
    """Alarm dict per the plan's training alarms.

    `stats` is the posterior_usage dict: keys symbol_usage, texel_usage,
    node_entropy, emit_mag, rule_mag, mean_depth, mean_leaves.
    Returns booleans: grammar_collapse, tempered_dp_failure, texel_death,
    parse_collapse, and `any`.
    """
    seff = s_eff(stats["symbol_usage"])
    ratio = magnitude_ratio(float(stats["emit_mag"]), float(stats["rule_mag"]))
    node_h = float(stats["node_entropy"])
    alive_frac = alive_texels(stats["texel_usage"], T_v) / float(T_v)
    mean_depth = float(stats.get("mean_depth", 0.0))
    mean_leaves = float(stats.get("mean_leaves", 0.0))

    alarms = {
        "grammar_collapse": seff < S_EFF_COLLAPSE_FRAC * S,
        "tempered_dp_failure": (ratio > MAG_RATIO_ALARM) and (node_h < NODE_ENTROPY_FLOOR),
        "texel_death": alive_frac < ALIVE_TEXEL_ALARM_FRAC,
        "parse_collapse": (mean_leaves >= 0.95 * QUADTREE_LEAVES)
        and (mean_depth >= 0.95 * QUADTREE_DEPTH),
    }
    alarms["any"] = bool(any(alarms.values()))
    return alarms
