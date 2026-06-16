"""
Your task: implement moe_forward() below.

The function must compute the forward pass of a top-k sparse Mixture-of-Experts
layer and match the reference oracle numerically on ALL hidden test cases.

Correctness is a hard gate — a fast but incorrect solution scores 0.

You may:
  - use any PyTorch operations
  - use torch.compile
  - use Triton kernels (if available)
  - use custom CUDA extensions (if available)

Do NOT modify:
  - src/moe_env/reference.py
  - tests/
  - benchmarks/
  - judge/

Run public tests:  uv run pytest
Run benchmark:     uv run python benchmarks/benchmark_moe.py
"""

import torch
import torch.nn.functional as F


def moe_forward(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """
    Sparse MoE forward pass.

    Args:
        x:              [T, D]  token activations
        expert_ids:     [T, K]  LongTensor of selected expert indices
        expert_weights: [T, K]  routing weights
        w1:             [E, H, D]
        b1:             [E, H]
        w2:             [E, D, H]
        b2:             [E, D]

    Returns:
        output: [T, D]
    """
    T, D = x.shape
    K = expert_ids.shape[1]
    E = w1.shape[0]

    # Make x contiguous to avoid view() errors on non-contiguous tensors
    x = x.contiguous()

    output = torch.zeros(T, D, dtype=x.dtype, device=x.device)

    # Flatten token-expert assignments: each (t, k) pair is one routing slot
    # token_indices[i] = which token,  flat_expert_ids[i] = which expert
    token_indices = (
        torch.arange(T, device=x.device)
        .unsqueeze(1)           # [T, 1]
        .expand(T, K)           # [T, K]
        .reshape(-1)            # [T*K]
    )
    flat_expert_ids = expert_ids.reshape(-1)    # [T*K]
    flat_weights = expert_weights.reshape(-1)  # [T*K]

    for e in range(E):
        mask = flat_expert_ids == e
        if not mask.any():
            continue

        tok_idx = token_indices[mask]   # [N_e] — token indices routed to expert e
        weights = flat_weights[mask]    # [N_e] — corresponding routing weights

        x_e = x[tok_idx]               # [N_e, D]

        # hidden = gelu(w1[e] @ x_e^T + b1[e])  -> [N_e, H]
        hidden = F.gelu(x_e @ w1[e].T + b1[e])

        # expert_out = w2[e] @ hidden^T + b2[e]  -> [N_e, D]
        expert_out = hidden @ w2[e].T + b2[e]

        # weighted scatter-add back to output
        weighted = weights.unsqueeze(1) * expert_out  # [N_e, D]
        output.scatter_add_(0, tok_idx.unsqueeze(1).expand_as(weighted), weighted)

    return output
