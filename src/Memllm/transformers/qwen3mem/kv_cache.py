import torch
from typing import Optional, Tuple, List, Any, Dict
from transformers.cache_utils import Cache

class StreamingSinkCache(Cache):
    """
    适配 transformers 库风格。
    返回的是拼接了新 token 但尚未驱逐的 KV cache。
    内部存储则会保持在 sink + window 的长度内。
    """
    def __init__(
        self,
        window_length: int = 512,
        num_sink_tokens: int = 4,
    ) -> None:
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        
        # 存储 KV 对：List[torch.Tensor]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        
        self._seen_tokens = 0 

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        更新并返回驱逐前的完整缓存。
        """
        # 1. 拼接逻辑
        if len(self.key_cache) <= layer_idx:
            # 如果是该层的第一组 KV，直接作为初始缓存
            full_k = key_states
            full_v = value_states
            self.key_cache.append(full_k)
            self.value_cache.append(full_v)
        else:
            # 拼接新到来的 token: [batch, heads, seq_len, head_dim]
            full_k = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            full_v = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
            self.key_cache[layer_idx] = full_k
            self.value_cache[layer_idx] = full_v

        # 2. 准备返回结果（此时尚未驱逐）
        # 注意：这里返回的是引用，如果后续操作需要修改它，建议返回 full_k.detach() 或 clone()
        # 但在 transformers 流程中，通常直接返回拼接后的全量张量给 Attention 计算
        return_k, return_v = full_k, full_v

        # 3. 执行内部驱逐逻辑 (Eviction)
        # 保持内部存储的 size 不超过限定值
        curr_seq_len = self.key_cache[layer_idx].shape[2]
        max_cache_size = self.num_sink_tokens + self.window_length
        
        if curr_seq_len > max_cache_size:
            # 这里的逻辑等同于你提供的 evict_for_space
            k_sink = self.key_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            v_sink = self.value_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            
            k_recent = self.key_cache[layer_idx][:, :, -self.window_length:, :]
            v_recent = self.value_cache[layer_idx][:, :, -self.window_length:, :]
            
            self.key_cache[layer_idx] = torch.cat([k_sink, k_recent], dim=2)
            self.value_cache[layer_idx] = torch.cat([v_sink, v_recent], dim=2)

        # 更新计数
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        return return_k, return_v

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for i in range(len(self.key_cache)):
            self.key_cache[i] = self.key_cache[i].index_select(0, beam_idx)
            self.value_cache[i] = self.value_cache[i].index_select(0, beam_idx)