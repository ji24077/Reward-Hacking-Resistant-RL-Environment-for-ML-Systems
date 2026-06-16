from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class MoEConfig:
    num_experts: int = 8
    token_dim: int = 64
    expert_hidden_dim: int = 128
    top_k: int = 2
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self):
        assert 1 <= self.top_k <= self.num_experts
        assert self.token_dim > 0
        assert self.expert_hidden_dim > 0


@dataclass
class MoEInputs:
    """All tensors for a single moe_forward call."""
    x: torch.Tensor             # [T, D]
    expert_ids: torch.Tensor    # [T, K]  LongTensor
    expert_weights: torch.Tensor  # [T, K]
    w1: torch.Tensor            # [E, H, D]
    b1: torch.Tensor            # [E, H]
    w2: torch.Tensor            # [E, D, H]
    b2: torch.Tensor            # [E, D]

    @property
    def T(self) -> int:
        return self.x.shape[0]

    @property
    def D(self) -> int:
        return self.x.shape[1]

    @property
    def K(self) -> int:
        return self.expert_ids.shape[1]

    @property
    def E(self) -> int:
        return self.w1.shape[0]

    @property
    def H(self) -> int:
        return self.w1.shape[1]

    def to(self, device: str) -> "MoEInputs":
        return MoEInputs(
            x=self.x.to(device),
            expert_ids=self.expert_ids.to(device),
            expert_weights=self.expert_weights.to(device),
            w1=self.w1.to(device),
            b1=self.b1.to(device),
            w2=self.w2.to(device),
            b2=self.b2.to(device),
        )

    def as_tuple(self):
        return (self.x, self.expert_ids, self.expert_weights,
                self.w1, self.b1, self.w2, self.b2)
