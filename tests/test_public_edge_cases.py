"""
Public edge case tests — these expose the common failure modes described in the
environment spec. Hidden tests add more unseen seeds and shapes.
"""

import pytest
import torch
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from moe_env import (moe_forward_reference, generate_moe_inputs, MoEConfig,
                     make_repeated_expert_routing)
from moe_env.types import MoEInputs
from moe_env.utils import allclose_with_report, get_tolerances, make_expert_weights
from solution.solution import moe_forward


def _check(cfg, num_tokens, seed, routing="uniform", make_contiguous=True):
    inp = generate_moe_inputs(cfg, num_tokens, seed=seed, routing=routing,
                               make_contiguous=make_contiguous)
    expected = moe_forward_reference(*inp.as_tuple())
    got = moe_forward(*inp.as_tuple())
    assert got.shape == expected.shape
    assert not torch.isnan(got).any(), "NaN in output"
    assert not torch.isinf(got).any(), "Inf in output"
    atol, rtol = get_tolerances(cfg.dtype)
    passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
    assert passed, (
        f"max_abs_err={info['max_abs_err']:.2e}, max_rel_err={info['max_rel_err']:.2e}"
    )


class TestEdgeCases:
    def test_empty_experts(self):
        """Some experts receive zero tokens — must not crash or produce NaN."""
        _check(MoEConfig(num_experts=8, token_dim=32, expert_hidden_dim=64, top_k=1),
               num_tokens=4, seed=20, routing="sparse")

    def test_non_contiguous_x(self):
        """x may be non-contiguous — view() will fail if not handled."""
        _check(MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2),
               num_tokens=32, seed=21, make_contiguous=False)

    def test_token_order_preserved(self):
        """Outputs must be in original token order, not expert-sorted order."""
        cfg = MoEConfig(num_experts=4, token_dim=16, expert_hidden_dim=32, top_k=2)
        inp = generate_moe_inputs(cfg, num_tokens=32, seed=22, routing="skewed")
        expected = moe_forward_reference(*inp.as_tuple())
        got = moe_forward(*inp.as_tuple())
        # Check full tensor equality (not just shape) — wrong order shows up here
        atol, rtol = get_tolerances(cfg.dtype)
        passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
        assert passed, f"Token order likely wrong: max_abs_err={info['max_abs_err']:.2e}"

    def test_router_weights_broadcast(self):
        """Expert weights must multiply the *output* of each expert, not x."""
        cfg = MoEConfig(num_experts=4, token_dim=8, expert_hidden_dim=16, top_k=2)
        inp = generate_moe_inputs(cfg, num_tokens=8, seed=23)

        # Manually verify one token
        expected = moe_forward_reference(*inp.as_tuple())
        got = moe_forward(*inp.as_tuple())

        # If weights are applied to x instead of the expert output, the outputs differ
        atol, rtol = get_tolerances(cfg.dtype)
        passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
        assert passed

    def test_single_expert_all_tokens(self):
        """All tokens route to the same expert (top_k=1, all select expert 0)."""
        cfg = MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=1)
        inp = generate_moe_inputs(cfg, num_tokens=32, seed=24)

        # Force all tokens to expert 0 — copy tensors to avoid mutating inp in-place
        forced = MoEInputs(
            x=inp.x,
            expert_ids=inp.expert_ids.clone().fill_(0),
            expert_weights=inp.expert_weights.clone().fill_(1.0),
            w1=inp.w1, b1=inp.b1, w2=inp.w2, b2=inp.b2,
        )

        expected = moe_forward_reference(*forced.as_tuple())
        got = moe_forward(*forced.as_tuple())
        atol, rtol = get_tolerances(cfg.dtype)
        passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
        assert passed

    def test_two_tokens(self):
        """Minimal batch — catches off-by-one bugs in scatter logic."""
        _check(MoEConfig(num_experts=4, token_dim=16, expert_hidden_dim=32, top_k=2),
               num_tokens=2, seed=25)

    def test_larger_batch(self):
        """Larger batch to stress scatter_add_ correctness."""
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2),
               num_tokens=512, seed=26)

    def test_output_dtype_preserved(self):
        """Output dtype must match input x dtype."""
        cfg = MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2)
        inp = generate_moe_inputs(cfg, num_tokens=16, seed=27)
        got = moe_forward(*inp.as_tuple())
        assert got.dtype == inp.x.dtype, (
            f"dtype mismatch: got {got.dtype}, expected {inp.x.dtype}"
        )

    def test_output_device_preserved(self):
        """Output device must match input device."""
        cfg = MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2,
                        device="cpu")
        inp = generate_moe_inputs(cfg, num_tokens=16, seed=28)
        got = moe_forward(*inp.as_tuple())
        assert got.device == inp.x.device

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
    def test_cuda_correctness(self):
        """Correctness on GPU."""
        cfg = MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                        device="cuda")
        inp = generate_moe_inputs(cfg, num_tokens=128, seed=29)
        expected = moe_forward_reference(*inp.as_tuple()).cpu()
        got = moe_forward(*inp.as_tuple()).cpu()
        atol, rtol = get_tolerances(cfg.dtype)
        passed, info = allclose_with_report(got, expected, atol=atol, rtol=rtol)
        assert passed, f"CUDA: max_abs_err={info['max_abs_err']:.2e}"

    def test_repeated_expert_ids(self):
        """A token routes to the same expert multiple times (e.g. expert_ids[t] = [2, 2]).
        The implementation must accumulate both contributions, not overwrite one."""
        T, K, E, D, H = 32, 2, 4, 32, 64
        expert_ids, expert_weights = make_repeated_expert_routing(T, K, E, seed=30)
        w1, b1, w2, b2 = make_expert_weights(E, H, D, seed=30)
        x = torch.randn(T, D)

        expected = moe_forward_reference(x, expert_ids, expert_weights, w1, b1, w2, b2)
        got = moe_forward(x, expert_ids, expert_weights, w1, b1, w2, b2)

        assert got.shape == expected.shape
        assert not torch.isnan(got).any()
        passed, info = allclose_with_report(got, expected, atol=1e-5, rtol=1e-4)
        assert passed, (
            f"Repeated expert ID accumulation wrong: "
            f"max_abs_err={info['max_abs_err']:.2e}"
        )

    def test_repeated_routing_via_generate(self):
        """Same test driven through generate_moe_inputs with routing='repeated'."""
        _check(MoEConfig(num_experts=4, token_dim=32, expert_hidden_dim=64, top_k=2),
               num_tokens=32, seed=31, routing="repeated")

    def test_fp16_correctness(self):
        """FP16 inputs use looser dtype-aware tolerances."""
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                         dtype=torch.float16),
               num_tokens=64, seed=32, routing="uniform")

    def test_fp16_skewed_routing(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                         dtype=torch.float16),
               num_tokens=64, seed=33, routing="skewed")

    def test_bfloat16_correctness(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=2,
                         dtype=torch.bfloat16),
               num_tokens=64, seed=34, routing="uniform")

    def test_bfloat16_sparse_routing(self):
        _check(MoEConfig(num_experts=8, token_dim=64, expert_hidden_dim=128, top_k=1,
                         dtype=torch.bfloat16),
               num_tokens=32, seed=35, routing="sparse")
