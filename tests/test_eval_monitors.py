from __future__ import annotations

import numpy as np
import torch

from sprig.eval import monitors


def test_s_eff_uniform_and_onehot():
    S = 128
    assert abs(monitors.s_eff(np.ones(S)) - S) < 1e-6
    one_hot = np.zeros(S)
    one_hot[3] = 5.0
    assert abs(monitors.s_eff(one_hot) - 1.0) < 1e-9
    # torch input accepted
    assert abs(monitors.s_eff(torch.ones(S)) - S) < 1e-4


def test_alive_texels():
    T_v = 256
    usage = np.zeros(T_v)
    usage[:16] = 1.0  # 16 texels share all mass, each 1/16 >> 0.1/256
    assert monitors.alive_texels(usage, T_v) == 16
    assert monitors.alive_texels(np.ones(T_v), T_v) == T_v
    assert monitors.alive_texels(np.zeros(T_v), T_v) == 0


def test_magnitude_ratio():
    assert abs(monitors.magnitude_ratio(10.0, 2.0) - 5.0) < 1e-9
    assert monitors.magnitude_ratio(1.0, 0.0) > 1e10  # no division by zero


def test_pi_controller_directionality():
    # below band -> eta goes UP, and further below pushes harder
    up_small = monitors.pi_controller_eta(0.5, 0.4)
    up_big = monitors.pi_controller_eta(0.5, 0.0)
    assert up_small > 0.5
    assert up_big > up_small
    assert abs(up_big - (0.5 + 0.05 + 0.1 * 0.5)) < 1e-9
    # above band -> eta goes DOWN by 0.05
    assert abs(monitors.pi_controller_eta(0.5, 4.0) - 0.45) < 1e-9
    # inside band -> unchanged
    assert monitors.pi_controller_eta(0.5, 1.7) == 0.5
    # clamping
    assert monitors.pi_controller_eta(0.02, 5.0) == 0.0
    assert monitors.pi_controller_eta(1.49, 0.0) == 1.5


def _stats(symbol_usage, texel_usage, node_entropy, emit_mag, rule_mag,
           mean_depth=3.0, mean_leaves=10.0):
    return {
        "symbol_usage": symbol_usage,
        "texel_usage": texel_usage,
        "node_entropy": node_entropy,
        "emit_mag": emit_mag,
        "rule_mag": rule_mag,
        "mean_depth": mean_depth,
        "mean_leaves": mean_leaves,
    }


def test_alarms_healthy():
    S, T_v = 128, 64
    alarms = monitors.build_alarms(
        _stats(np.ones(S), np.ones(T_v), 1.5, 10.0, 5.0), S, T_v
    )
    assert not alarms["any"]


def test_alarms_fire():
    S, T_v = 128, 64
    collapsed = np.zeros(S)
    collapsed[0] = 1.0
    a = monitors.build_alarms(
        _stats(collapsed, np.ones(T_v), 1.5, 10.0, 5.0), S, T_v
    )
    assert a["grammar_collapse"] and a["any"]

    a = monitors.build_alarms(
        _stats(np.ones(S), np.ones(T_v), 0.05, 500.0, 1.0), S, T_v
    )
    assert a["tempered_dp_failure"]
    # high ratio alone (healthy entropy) does NOT fire
    a = monitors.build_alarms(
        _stats(np.ones(S), np.ones(T_v), 1.0, 500.0, 1.0), S, T_v
    )
    assert not a["tempered_dp_failure"]

    dead = np.zeros(T_v)
    dead[:3] = 1.0
    a = monitors.build_alarms(_stats(np.ones(S), dead, 1.5, 1.0, 1.0), S, T_v)
    assert a["texel_death"]

    a = monitors.build_alarms(
        _stats(np.ones(S), np.ones(T_v), 1.5, 1.0, 1.0, mean_depth=6.0, mean_leaves=64.0),
        S, T_v,
    )
    assert a["parse_collapse"]
