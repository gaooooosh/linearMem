import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import FeatureSharpnessGate

class GatedTopkSoftplusKernel(nn.Module):
    """
    Sparse key feature map for memory writing:
        phi_k(x) = Gate(TopK(softplus(x)))

    Input:
        x: already projected key tensor, shape [..., d]

    Output:
        phi_k: same shape as x
    """
    def __init__(
        self,
        head_dim: int,
        topk: int = 8,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.topk = topk
        self.gate_fn = FeatureSharpnessGate(head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = F.softplus(x)

        k = self.topk
        if k is not None and k < phi.size(-1):
            vals, idx = torch.topk(phi, k, dim=-1)
            out = torch.zeros_like(phi)
            out.scatter_(-1, idx, vals)
            phi = out

        # g = self.gate_fn(phi)
        # phi = phi * g
        return phi


class PowTopkSoftplusKernel(nn.Module):
    """
    Sparse query feature map:
        phi_q(x) = Normalize(TopK(softplus(x)) ^ gamma)
    """
    def __init__(
        self,
        head_dim: int,
        topk: int = 8,
        gamma: float = 2.0,
        normalize: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.topk = topk
        self.gamma = gamma
        self.normalize = normalize
        self.eps = eps
        self.gate_fn = FeatureSharpnessGate(head_dim=head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = F.softplus(x)

        k = self.topk
        if k is not None and k < phi.size(-1):
            vals, idx = torch.topk(phi, k, dim=-1)
            out = torch.zeros_like(phi)
            out.scatter_(-1, idx, vals)
            phi = out

        if self.gamma != 1.0:
            phi = phi.pow(self.gamma)

        if self.normalize:
            phi = phi / (phi.sum(dim=-1, keepdim=True) + self.eps)
        
        g = self.gate_fn(phi)
        phi = phi * g
        
        return phi