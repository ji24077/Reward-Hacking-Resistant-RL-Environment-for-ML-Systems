"""Helpers for generating MoE test inputs and checking numerical equivalence."""

import torch
import torch.nn.functional as F
from typing import Optional

from .types import MoEConfig, MoEInputs


def generate_moe_inputs(
    cfg: MoEConfig,
    num_tokens: int,
    seed: Optional[int] = None,
    routing: str = "uniform",
    make_contiguous: bool = True,
) -> MoEInputs:
    """
    Generate a random MoEInputs for the given config.

    Args:
        cfg:           MoEConfig
        num_tokens:    T
        seed:          RNG seed for reproducibility
        routing:       "uniform"  — each expert equally likely
                       "skewed"   — one expert gets ~80% of tokens
                       "sparse"   — some experts get zero tokens
        make_contiguous: if False, returns non-contiguous x (transpose then back)
    """
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    T = num_tokens
    E = cfg.num_experts
    D = cfg.token_dim
    H = cfg.expert_hidden_dim
    K = cfg.top_k
    dtype = cfg.dtype
    device = cfg.device

    x = torch.randn(T, D, dtype=dtype, device=device, generator=rng)
    if not make_contiguous:
        x = x.T.contiguous().T  # non-contiguous

    # Build expert_ids with the requested routing distribution
    if routing == "uniform":
        scores = torch.rand(T, E, dtype=torch.float32, device=device, generator=rng)
    elif routing == "skewed":
        scores = torch.rand(T, E, dtype=torch.float32, device=device, generator=rng)
        scores[:, 0] += 4.0  # expert 0 dominates
    elif routing == "sparse":
        scores = torch.rand(T, E, dtype=torch.float32, device=device, generator=rng)
        # zero out half the experts to create empty experts
        dead = torch.randperm(E, generator=rng)[: E // 2]
        scores[:, dead] = -1e9
    elif routing == "repeated":
        # Force each token to select the same expert twice (tests accumulation logic)
        scores = torch.rand(T, E, dtype=torch.float32, device=device, generator=rng)
    else:
        raise ValueError(f"Unknown routing: {routing}")

    _, top_ids = scores.topk(K, dim=1)  # [T, K]
    expert_ids = top_ids.long()

    # For "repeated" routing: overwrite expert_ids so all K slots pick the same expert
    if routing == "repeated" and K >= 2:
        expert_ids = expert_ids[:, :1].expand(T, K).contiguous()
        # Weights must be consistent with the overwritten expert_ids: uniform 1/K
        expert_weights = torch.full((T, K), 1.0 / K, dtype=dtype, device=device)
    else:
        selected_scores = scores.gather(1, top_ids)
        expert_weights = F.softmax(selected_scores, dim=1).to(dtype)

    # Expert weight matrices
    w1 = torch.randn(E, H, D, dtype=dtype, device=device, generator=rng) * 0.02
    b1 = torch.zeros(E, H, dtype=dtype, device=device)
    w2 = torch.randn(E, D, H, dtype=dtype, device=device, generator=rng) * 0.02
    b2 = torch.zeros(E, D, dtype=dtype, device=device)

    return MoEInputs(x=x, expert_ids=expert_ids, expert_weights=expert_weights,
                     w1=w1, b1=b1, w2=w2, b2=b2)


def make_repeated_expert_routing(
    T: int,
    K: int,
    E: int,
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Return (expert_ids, expert_weights) where every token routes to the same
    expert in all K slots.  Tests that the implementation accumulates multiple
    contributions from the same expert correctly rather than overwriting.

    Returns:
        expert_ids:     [T, K]  all entries in row t are the same expert index
        expert_weights: [T, K]  uniform 1/K per slot
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    scores = torch.rand(T, E, dtype=torch.float32, device=device, generator=rng)
    top1 = scores.argmax(dim=1, keepdim=True)  # [T, 1]
    expert_ids = top1.expand(T, K).contiguous().long()
    expert_weights = torch.full((T, K), 1.0 / K, dtype=dtype, device=device)
    return expert_ids, expert_weights


def make_expert_weights(E: int, H: int, D: int, seed: int = 0,
                        dtype=torch.float32, device="cpu"):
    """Return (w1, b1, w2, b2) with Xavier-style init."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    std = (2.0 / (D + H)) ** 0.5
    w1 = torch.randn(E, H, D, dtype=dtype, device=device, generator=rng) * std
    b1 = torch.zeros(E, H, dtype=dtype, device=device)
    w2 = torch.randn(E, D, H, dtype=dtype, device=device, generator=rng) * std
    b2 = torch.zeros(E, D, dtype=dtype, device=device)
    return w1, b1, w2, b2


def allclose_with_report(
    got: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> tuple[bool, dict]:
    """Returns (passed, info_dict) with diagnostic details on failure.

    Deliberately does NOT use torch.allclose — a monkeypatched solution
    could replace torch.allclose with lambda a, b, **kw: True.  Instead we
    implement the same |a - b| <= atol + rtol * |b| check from first principles
    using only raw tensor arithmetic, which cannot be patched at the module level.
    """
    if got.shape != expected.shape:
        return False, {"error": "shape mismatch", "got": got.shape, "expected": expected.shape}

    a = got.float()
    b = expected.float()
    abs_err = (a - b).abs()
    rel_err = abs_err / (b.abs() + 1e-8)

    # Manual allclose: all elements satisfy |a - b| <= atol + rtol * |b|
    threshold = atol + rtol * b.abs()
    passed = bool((abs_err <= threshold).all().item())

    info = {
        "passed": passed,
        "max_abs_err": abs_err.max().item(),
        "mean_abs_err": abs_err.mean().item(),
        "max_rel_err": rel_err.max().item(),
        "atol": atol,
        "rtol": rtol,
        "dtype": str(got.dtype),
        "shape": list(got.shape),
    }
    return passed, info


DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-5, 1e-4),
    torch.float16: (1e-2, 1e-2),
    torch.bfloat16: (1e-2, 1e-2),
}


def get_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    return DTYPE_TOLERANCES.get(dtype, (1e-5, 1e-4))
