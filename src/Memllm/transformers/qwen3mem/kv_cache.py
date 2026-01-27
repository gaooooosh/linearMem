import torch
from typing import Optional, Tuple, List, Any, Dict
from transformers.cache_utils import Cache

class StreamingSinkCache(Cache):
    def __init__(self, window_length: int = 512, num_sink_tokens: int = 4, num_hidden_layers: int = 32) -> None:
        super().__init__()
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        
        # 预先根据模型层数创建好缓存空间，避免 len(self) == 0 的问题
        self.key_cache: List[Optional[torch.Tensor]] = [None] * num_hidden_layers
        self.value_cache: List[Optional[torch.Tensor]] = [None] * num_hidden_layers
        self._seen_tokens = 0

    @property
    def is_compileable(self) -> bool:
        # 明确告知 transformers 这个自定义 Cache 不支持 torch.compile
        return False

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        # 返回当前层在 dim=2 (sequence length) 上的长度
        return self.key_cache[layer_idx].shape[2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 1. 拼接逻辑
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # 2. 驱逐逻辑 (StreamingLLM 核心)
        curr_len = self.key_cache[layer_idx].shape[2]
        if curr_len > self.window_length + self.num_sink_tokens:
            k_sink = self.key_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            v_sink = self.value_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            
            k_recent = self.key_cache[layer_idx][:, :, -self.window_length:, :]
            v_recent = self.value_cache[layer_idx][:, :, -self.window_length:, :]
            
            self.key_cache[layer_idx] = torch.cat([k_sink, k_recent], dim=2)
            self.value_cache[layer_idx] = torch.cat([v_sink, v_recent], dim=2)

        # 3. 更新全局看到的 token 计数 (仅在第一层更新一次)
        if layer_idx == 0:
            if cache_kwargs is not None and "cache_position" in cache_kwargs:
                self._seen_tokens = cache_kwargs["cache_position"][-1].item() + 1
            else:
                self._seen_tokens += key_states.shape[2]

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def __len__(self):
        # 返回总层数，这样 len(self.layers) 检查就能通过
        return len(self.key_cache)

    @property
    def layers(self):
        # transformers 内部有时会访问 .layers 属性
        return self.key_cache
    def __init__(self, window_length: int = 512, num_sink_tokens: int = 4) -> None:
        # super().__init__() # 建议调用父类初始化
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self.num_sink_tokens = num_sink_tokens
        self._seen_tokens = 0 

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. 初始化或拼接
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # 2. 执行驱逐逻辑
        # 只有当长度超过窗口时才裁剪
        if self.key_cache[layer_idx].shape[2] > self.window_length + self.num_sink_tokens:
            k_sink = self.key_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            v_sink = self.value_cache[layer_idx][:, :, :self.num_sink_tokens, :]
            
            k_recent = self.key_cache[layer_idx][:, :, -self.window_length:, :]
            v_recent = self.value_cache[layer_idx][:, :, -self.window_length:, :]
            
            self.key_cache[layer_idx] = torch.cat([k_sink, k_recent], dim=2)
            self.value_cache[layer_idx] = torch.cat([v_sink, v_recent], dim=2)

        # 3. 更新全局看到的 token 数
        if layer_idx == 0:
            if cache_kwargs is not None and "cache_position" in cache_kwargs:
                self._seen_tokens = cache_kwargs["cache_position"][-1].item() + 1
            else:
                self._seen_tokens += key_states.shape[2]

        # 4. 重要：返回裁剪后的 cache
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[2]

    # 为了兼容 Transformers 的某些 mask 检查，确保返回的是当前实际缓存的长度
    def get_usable_length(self, deferred_move: int, layer_idx: Optional[int] = 0) -> int:
        return self.get_seq_length(layer_idx)