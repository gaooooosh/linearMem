import torch
import torch.nn as nn
import torch.nn.functional as F

def make_stable_fps_anchors(
    num_anchors: int,
    head_dim: int,
    candidate_factor: int = 32,
    eps: float = 1e-6,
    seed: int = 42,
    device=None,
    dtype=None,
):
    """
    更稳定的无训练 anchor 初始化：
    1. 固定 seed
    2. 生成球面候选点
    3. 加入 antipodal 对称点
    4. 在候选池上做 FPS
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    base_candidates = max((num_anchors * candidate_factor) // 2, num_anchors // 2 + 1)

    # 先生成一半候选
    c = torch.randn(base_candidates, head_dim, generator=g, device=device, dtype=dtype)
    c = F.normalize(c, dim=-1, eps=eps)

    # 加入正负对称候选
    candidates = torch.cat([c, -c], dim=0)   # [2 * base_candidates, D]

    # 用“最远二点”初始化，避免 first=0 的偏置
    sim_mat = candidates @ candidates.T
    # 找最不相似的一对点（近似最远）
    first = 0
    second = torch.argmin(sim_mat[first]).item()

    selected = [first, second]

    # 记录每个候选点到已选集合的最大相似度
    max_sim = torch.maximum(sim_mat[:, first], sim_mat[:, second])

    for _ in range(2, num_anchors):
        idx = torch.argmin(max_sim).item()
        selected.append(idx)
        max_sim = torch.maximum(max_sim, sim_mat[:, idx])

    anchors = candidates[selected]
    anchors = F.normalize(anchors, dim=-1, eps=eps)
    return anchors


class AnchorKernel(nn.Module):
    """
    Dense anchor feature map:
        phi_r(x) = exp(tau * (<normalize(x), normalize(c_r)> - 1))
    """
    def __init__(
        self,
        head_dim: int,
        num_anchors: int,
        tau: float = 8.0,
        learnable_anchors: bool = False,   # 推荐默认冻结
        eps: float = 1e-6,
        seed: int = 42,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_anchors = num_anchors
        self.tau = tau
        self.eps = eps

        anchors = make_stable_fps_anchors(
            num_anchors=num_anchors,
            head_dim=head_dim,
            candidate_factor=32,
            eps=eps,
            seed=seed,
            device=device,
            dtype=dtype,
        )
        self.anchors = nn.Parameter(anchors, requires_grad=learnable_anchors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = F.normalize(x, dim=-1, eps=self.eps)
        # anchors = F.normalize(self.anchors, dim=-1, eps=self.eps)

        sim = torch.einsum("bhtd,md->bhtm", F.elu(x), F.elu(self.anchors))

        logits = self.tau * sim
        phi = torch.softmax(logits, dim=-1)
        return phi