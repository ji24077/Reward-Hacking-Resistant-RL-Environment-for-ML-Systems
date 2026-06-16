"""
Slow but correct reference implementation of sparse MoE forward.
Do NOT modify this file.
"""

import torch
import torch.nn.functional as F


def moe_forward_reference(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    w1: torch.Tensor,
    b1: torch.Tensor,
    w2: torch.Tensor,
    b2: torch.Tensor,
) -> torch.Tensor:
    """
    Reference (oracle) implementation. Intentionally uses a simple loop for
    readability and correctness. This is the ground truth for judging.

    Args:
        x:              [T, D]
        expert_ids:     [T, K] LongTensor — selected expert indices per token
        expert_weights: [T, K] — routing weights, typically sum-to-1 per token
        w1:             [E, H, D]
        b1:             [E, H]
        w2:             [E, D, H]
        b2:             [E, D]

    Returns:
        output: [T, D]
    """
    T, D = x.shape
    K = expert_ids.shape[1]

    output = torch.zeros(T, D, dtype=x.dtype, device=x.device)

    for t in range(T):
        for k in range(K):
            e = expert_ids[t, k].item()
            w = expert_weights[t, k]

            # hidden = gelu(w1[e] @ x[t] + b1[e])
            hidden = F.gelu(w1[e] @ x[t] + b1[e])

            # expert_output = w2[e] @ hidden + b2[e]
            expert_out = w2[e] @ hidden + b2[e]

            output[t] = output[t] + w * expert_out

    return output
