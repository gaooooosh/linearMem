import torch
import torch.nn as nn
import torch.nn.functional as F

def make_fps_anchors(num_anchors, head_dim, candidate_factor=32, eps=1e-6, device=None, dtype=None):
    num_candidates = max(num_anchors * candidate_factor, num_anchors + 1)
    candidates = torch.randn(num_candidates, head_dim, device=device, dtype=dtype)
    candidates = F.normalize(candidates, dim=-1,eps=eps)

    selected = []
    first = 0
    selected.append(first)

    # 记录每个候选点到已选集合的最大相似度（越小越远）
    max_sim = candidates @ candidates[first:first+1].T
    max_sim = max_sim.squeeze(-1)

    for _ in range(1, num_anchors):
        idx = torch.argmin(max_sim)   # 选和当前集合最不相似的点
        selected.append(idx.item())
        sim = (candidates @ candidates[idx:idx+1].T).squeeze(-1)
        max_sim = torch.maximum(max_sim, sim)

    anchors = candidates[selected]
    anchors = F.normalize(anchors, dim=-1,eps=eps)
    return anchors

class AnchorKernel(nn.Module):
    """
    Dense anchor feature map:
        phi_r(x) = exp(tau * (<normalize(x), normalize(c_r)> - 1))

    输入:
        x: [B, H, T, D]
    输出:
        phi(x): [B, H, T, M]
    """
    def __init__(
        self,
        head_dim: int,
        num_anchors: int,
        tau: float = 8.0,
        learnable_anchors: bool = True,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_anchors = num_anchors
        self.tau = tau
        self.eps = eps

        anchors = make_fps_anchors(num_anchors, head_dim, candidate_factor=32, eps=self.eps, device=device, dtype=dtype)
        self.anchors = nn.Parameter(anchors, requires_grad=learnable_anchors)

    def forward(self, x: torch.Tensor, kind: str | None = None) -> torch.Tensor:
        # x: [B, H, T, D]
        x = F.normalize(x, dim=-1, eps=self.eps)

        # sim: [B, H, T, M]
        sim = torch.einsum("bhtd,md->bhtm", x, self.anchors)
        
        # phi: [B, H, T, M]
        phi = torch.exp(self.tau * (sim - 1.0))
        return phi