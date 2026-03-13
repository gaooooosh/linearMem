import torch
import torch.nn as nn


class FeatureSharpnessGate(nn.Module):
    """
    Parameter-free write gate based on feature sharpness:
        g(x) = ||x||_2^2 / (||x||_1^2 + eps)

    Input:
        x: sparse nonnegative feature, shape [..., d]

    Output:
        g: scalar gate, shape [..., 1]
    """
    def __init__(
        self,
        head_dim: int,
        eps: float = 1e-6,
        clamp_min: float | None = 0.0,
        clamp_max: float | None = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, x: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        l1 = x.sum(dim=-1, keepdim=True)
        l2_sq = (x * x).sum(dim=-1, keepdim=True)

        g = l2_sq / (l1 * l1 + self.eps)

        if self.clamp_min is not None or self.clamp_max is not None:
            g = torch.clamp(g, min=self.clamp_min, max=self.clamp_max)

        return g