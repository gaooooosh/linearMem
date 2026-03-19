"""
针对NIAH任务优化的核函数（纯PyTorch实现）

核函数只负责特征映射 φ(x)，不改变线性注意力的基础算法。
输入: x: [..., d] （任意形状，最后一维是head_dim）
输出: phi: [..., d] （相同形状，特征映射后的结果）

基于探测实验结果的设计：
1. 稀有性增强：低频维度获得更高权重
2. 保留更多维度：40% vs 当前15.6%
3. 位置感知（可选）：早期token权重更高

注意：保持数值稳定性，避免极端值导致Triton编译问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RarityEnhancedKernel(nn.Module):
    """
    稀有性增强核（简化版）

    核心设计：
    1. 基于softplus的基础变换
    2. 自适应TopK稀疏化（保留更多维度）
    3. 简单归一化

    解决的问题：
    - Needle的独特信息往往在稀有维度上
    - 过度稀疏化会丢失关键信息
    """
    def __init__(
        self,
        head_dim: int,
        keep_ratio: float = 0.4,
        normalize: bool = True,
        eps: float = 1e-6,
        # 兼容现有接口的参数
        rarity_threshold: float = 0.4,  # 保留兼容性，暂不使用
        enhancement_factor: float = 2.5,  # 保留兼容性，暂不使用
        topk: int | None = None,
        gamma: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.keep_ratio = keep_ratio
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d] - 任意形状，最后一维是head_dim
            kind: 可选，用于区分Q或K

        Returns:
            phi: [..., d] - 相同形状，特征映射后的结果
        """
        # 1. 基础变换：softplus
        phi = F.softplus(x)

        dim = phi.shape[-1]

        # 2. 自适应稀疏化：保留keep_ratio的维度
        keep_k = max(1, int(dim * self.keep_ratio))
        if keep_k < dim:
            vals, idx = torch.topk(phi, keep_k, dim=-1)
            sparse_phi = torch.zeros_like(phi)
            sparse_phi.scatter_(-1, idx, vals)
            phi = sparse_phi

        # 3. 归一化
        if self.normalize:
            phi = phi / (phi.sum(dim=-1, keepdim=True) + self.eps)

        return phi


class PositionAwareKernel(nn.Module):
    """
    位置感知核（针对K向量）- 简化版

    核心设计：
    1. 基于softplus的基础变换
    2. 位置加权（早期token权重更高）
    3. 自适应稀疏化

    解决的问题：
    - 早期信息容易被后续噪音淹没
    - NIAH任务中Needle可能在序列早期
    """
    def __init__(
        self,
        head_dim: int,
        position_decay: float = 0.1,
        min_position_weight: float = 0.5,
        keep_ratio: float = 0.4,
        normalize: bool = True,
        eps: float = 1e-6,
        # 兼容现有接口的参数
        rarity_threshold: float = 0.4,  # 保留兼容性
        enhancement_factor: float = 2.5,  # 保留兼容性
        topk: int | None = None,
        gamma: float = 1.0,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.position_decay = position_decay
        self.min_position_weight = min_position_weight
        self.keep_ratio = keep_ratio
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d] - 对于K通常是 [batch, heads, seq, dim]
            kind: 用于区分Q或K，只有K会应用位置权重

        Returns:
            phi: [..., d]
        """
        # 1. 基础变换：softplus
        phi = F.softplus(x)

        original_shape = phi.shape
        dim = original_shape[-1]

        # 2. 位置加权（仅对K向量，且需要至少3D输入）
        if phi.dim() >= 3:
            seq_len = original_shape[-2]

            # 计算相对位置（0=开头，1=结尾）
            relative_pos = torch.linspace(0, 1, seq_len, device=phi.device, dtype=phi.dtype)

            # 早期token权重更高（线性衰减，更稳定）
            position_weight = 1.0 - (1.0 - self.min_position_weight) * relative_pos

            # 应用位置权重：[seq] -> [..., seq, 1]
            for _ in range(phi.dim() - 2):
                position_weight = position_weight.unsqueeze(0)
            position_weight = position_weight.unsqueeze(-1)

            phi = phi * position_weight

        # 3. 自适应稀疏化
        keep_k = max(1, int(dim * self.keep_ratio))
        if keep_k < dim:
            vals, idx = torch.topk(phi, keep_k, dim=-1)
            sparse_phi = torch.zeros_like(phi)
            sparse_phi.scatter_(-1, idx, vals)
            phi = sparse_phi

        # 4. 归一化
        if self.normalize:
            phi = phi / (phi.sum(dim=-1, keepdim=True) + self.eps)

        return phi


class DenseQueryKernel(nn.Module):
    """
    密集查询核（针对Q向量）- 简化版

    核心设计：
    1. 基于softplus的基础变换
    2. 不进行稀疏化，保留所有维度
    3. 简单归一化

    解决的问题：
    - Q需要精确匹配Needle，不能丢失维度
    """
    def __init__(
        self,
        head_dim: int,
        normalize: bool = True,
        eps: float = 1e-6,
        # 兼容现有接口的参数
        contrastive_factor: float = 1.5,  # 保留兼容性，暂不使用
        topk: int | None = None,
        gamma: float = 1.0,
        temperature: float = 1.0,  # 保留兼容性
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.normalize = normalize
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., d] - Q向量

        Returns:
            phi: [..., d] - 密集特征，不稀疏化
        """
        # 1. 基础变换：softplus
        phi = F.softplus(x)

        # 2. 简单归一化（密集，不稀疏化）
        if self.normalize:
            phi = phi / (phi.sum(dim=-1, keepdim=True) + self.eps)

        return phi


