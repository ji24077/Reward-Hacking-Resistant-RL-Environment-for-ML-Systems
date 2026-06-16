"""
Public correctness tests — these are visible to the agent.
Hidden tests follow the same structure but use different seeds / shapes / dtypes.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_env import moe_forward_reference, generate_moe_inputs, MoEConfig
from moe_env.utils import allclose_with_report, get_tolerances
from solution.solution import moe_forward


def _check(cfg: MoEConfig, num_tokens: int, seed: int, routing: str = "uniform",
           make_contiguous: bool = True):
    inp = generate_moe_inputs(cfg, num_tokens, seed=seed, routing=routing,
                               make_contiguous=make_contiguous)

    expected = moe_forward_reference(*inp.as_tuple())
    got = moe_forward(*inp.as_tuple())

    assert got is not None, "moe_forward returned None"
    assert got.shape == expected.shape, f"shape mismatch: {got.shape} vs {expected.shape}"
    assert not torch.isnan(got).any(), "output contains NaN"
    assert not torch.isinf(got).any(), "output contains Inf"

    atol, rtol = get_tolerances(cfg.dtype)
    passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
    assert passed, (
        f"Numerical mismatch — max_abs_err={info['max_abs_err']:.2e}, "
        f"max_rel_err={info['max_rel_err']:.2e} (atol={atol}, rtol={rtol})"
    )


# ---------------------------------------------------------------------------
# Basic shapes
# ---------------------------------------------------------------------------

class TestBasicShapes:
    def test_single_token(self):
        _check(MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2),
               num_tokens=1, seed=0)

    def test_small_batch(self):
        _check(MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2),
               num_tokens=16, seed=1)

    def test_medium_batch(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=128, seed=2)

    def test_top1_routing(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=1),
               num_tokens=64, seed=3)

    def test_top2_routing(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=64, seed=4)

    def test_large_expert_count(self):
        _check(MoEConfig(num_experts=16, token_dim=64, expert_hidden_dim=256, top_k=2),
               num_tokens=64, seed=5)

    def test_top_k_equals_num_experts(self):
        _check(MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=4),
               num_tokens=32, seed=6)


# ---------------------------------------------------------------------------
# Routing distributions
# ---------------------------------------------------------------------------

class TestRoutingDistributions:
    def test_skewed_routing(self):
        """One expert receives most of the tokens."""
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=128, seed=10, routing="skewed")

    def test_sparse_routing(self):
        """Half the experts receive zero tokens."""
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=64, seed=11, routing="sparse")

    def test_uniform_routing(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=64, seed=12, routing="uniform")
