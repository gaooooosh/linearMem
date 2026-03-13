import torch
import torch.nn as nn
import torch.nn.functional as F

class EluKernel(nn.Module):
    """
    Dense anchor feature map:
        phi_r(x) = exp(tau * (<normalize(x), normalize(c_r)> - 1))
    """
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        # x = F.normalize(x, dim=-1, eps=self.eps)
        # anchors = F.normalize(self.anchors, dim=-1, eps=self.eps)

        phi = F.elu(x) + 1
        k = 10
        if k is not None and k < phi.size(-1):
            vals, idx = torch.topk(phi, k, dim=-1)
            out = torch.zeros_like(phi)   
            out.scatter_(-1, idx, vals)
            phi = out
        return phi