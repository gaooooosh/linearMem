# KV Cache Recurrent State Patch

为 Transformers KV Cache 添加线性注意力递归状态（Recurrent State）支持的 Monkey Patch。

## 概述

本模块为 `transformers` 库的 KV Cache 层添加了 `recurrent_state` 属性和相关方法，使其能够存储和管理线性注意力（Linear Attention）的递归状态矩阵。这对于实现 Flash Linear Attention (FLA) 等线性复杂度注意力机制非常有用。

## 安装与导入

```python
from swaa_patch import hack_kv_cache_recurrent_state, unhack_kv_cache_recurrent_state
```

## 快速开始

```python
import torch
from transformers import AutoModelForCausalLM, DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state

# 1. 在加载模型前应用 patch
hack_kv_cache_recurrent_state()

# 2. 加载模型和创建 cache
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
cache = DynamicCache()

# 3. 在推理中使用
# ... 见下方详细用法 ...
```

## API 参考

### 核心函数

#### `hack_kv_cache_recurrent_state()`

应用 monkey patch，为以下类添加 recurrent state 支持：
- `DynamicLayer`
- `DynamicSlidingWindowLayer`
- `QuantizedLayer`
- `Cache` (及其子类如 `DynamicCache`)

#### `unhack_kv_cache_recurrent_state()`

移除 monkey patch，恢复原始行为。用于清理或测试。

---

### Layer 级别方法

以下方法会被添加到每个 Cache Layer 对象上：

#### `state_update(recurrent_state, cache_kwargs=None)`

更新并返回当前层的递归状态。风格与 `DynamicLayer.update()` 一致。

**参数：**
- `recurrent_state` (`torch.Tensor | tuple[torch.Tensor, ...] | None`):
  - 新的递归状态。如果为 `None`，则只返回当前状态而不更新。
- `cache_kwargs` (`dict[str, Any] | None`, 可选):
  - 额外参数。支持 `init_state=True` 来初始化 `recurrent_dtype` 和 `recurrent_device`。

**返回：**
- `torch.Tensor | tuple[torch.Tensor, ...] | None`: 当前的递归状态。

**示例：**
```python
layer = DynamicLayer()

# 更新 recurrent state
new_state = torch.randn(batch_size, num_heads, head_dim, state_dim)
current_state = layer.state_update(new_state)

# 仅获取状态（不更新）
state = layer.state_update(None)

# 带 cache_kwargs 初始化 dtype/device
state = layer.state_update(new_state, cache_kwargs={"init_state": True})
print(layer.recurrent_dtype)  # torch.float32
print(layer.recurrent_device)  # cuda:0
```

#### `get_recurrent_state()`

获取当前层的递归状态。

**返回：**
- `torch.Tensor | tuple[torch.Tensor, ...] | None`: 当前的递归状态。

**示例：**
```python
state = layer.get_recurrent_state()
```

#### `set_recurrent_state(state)`

直接设置当前层的递归状态。

**参数：**
- `state` (`torch.Tensor | tuple[torch.Tensor, ...] | None`): 新的递归状态。

**示例：**
```python
layer.set_recurrent_state(new_state)
```

#### `is_recurrent_state_initialized()`

检查递归状态是否已被初始化（即是否曾被设置过）。

**返回：**
- `bool`: 如果递归状态已被设置过，返回 `True`；否则返回 `False`。

**示例：**
```python
layer = DynamicLayer()

# 初始状态：未初始化
print(layer.is_recurrent_state_initialized())  # False

# 设置状态后
layer.state_update(torch.randn(1, 8, 64, 128))
print(layer.is_recurrent_state_initialized())  # True

# reset 后
layer.reset()
print(layer.is_recurrent_state_initialized())  # False
```

#### `recurrent_state_initialized` 属性

直接访问初始化状态标志。

```python
print(layer.recurrent_state_initialized)  # True or False
```

---

### Cache 级别方法

以下方法会被添加到 `Cache` 及其子类（如 `DynamicCache`）上：

#### `state_update(recurrent_state, layer_idx, cache_kwargs=None)`

更新指定层的递归状态。风格与 `Cache.update()` 一致。

**参数：**
- `recurrent_state` (`torch.Tensor | tuple[torch.Tensor, ...] | None`):
  - 新的递归状态。如果为 `None`，则只返回当前状态而不更新。
- `layer_idx` (`int`):
  - 要更新的层索引。如果该层不存在，会自动创建（懒初始化）。
- `cache_kwargs` (`dict[str, Any] | None`, 可选):
  - 传递给 layer 的 `state_update` 方法的额外参数。

**返回：**
- `torch.Tensor | tuple[torch.Tensor, ...] | None`: 指定层的递归状态。

**示例：**
```python
cache = DynamicCache()

# 更新 layer 0 的 recurrent state
state = cache.state_update(recurrent_state_tensor, layer_idx=0)

# 仅获取 layer 0 的状态（不更新）
state = cache.state_update(None, layer_idx=0)

# 更新 layer 1
state = cache.state_update(another_state, layer_idx=1)
```

#### `is_recurrent_state_initialized(layer_idx)`

检查指定层的递归状态是否已被初始化。

**参数：**
- `layer_idx` (`int`): 要检查的层索引。

**返回：**
- `bool`: 如果指定层的递归状态已被初始化，返回 `True`；否则返回 `False`。
  - 如果层不存在，也返回 `False`。

**示例：**
```python
cache = DynamicCache()

# 检查 layer 0 是否已初始化
if not cache.is_recurrent_state_initialized(layer_idx=0):
    # 首次调用，需要初始化状态
    initial_state = create_initial_state()
    cache.state_update(initial_state, layer_idx=0)

# 获取已存在的状态
state = cache.state_update(None, layer_idx=0)

# 检查不存在的层（返回 False）
cache.is_recurrent_state_initialized(layer_idx=99)  # False
```

#### `get_recurrent_state(layer_idx)`

获取指定层的递归状态。

**参数：**
- `layer_idx` (`int`): 要获取状态的层索引。

**返回：**
- `torch.Tensor | tuple[torch.Tensor, ...] | None`: 指定层的递归状态。
  - 如果层不存在或状态未初始化，返回 `None`。

**示例：**
```python
cache = DynamicCache()

# 获取未初始化的状态（返回 None）
state = cache.get_recurrent_state(layer_idx=0)  # None

# 设置状态后获取
cache.state_update(recurrent_state_tensor, layer_idx=0)
state = cache.get_recurrent_state(layer_idx=0)  # 返回 tensor

# 获取不存在的层（返回 None）
state = cache.get_recurrent_state(layer_idx=99)  # None
```

---

## 完整使用示例

### 示例 1：在自定义 Linear Attention 中使用

```python
import torch
import torch.nn as nn
from transformers import DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state

hack_kv_cache_recurrent_state()


class LinearAttentionLayer(nn.Module):
    """简化的 Linear Attention 层示例"""

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: DynamicCache,
        layer_idx: int,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # 计算 Q, K, V
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置为 [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 获取初始 recurrent state
        initial_state = past_key_values.state_update(None, layer_idx)

        # 运行 linear attention kernel（伪代码）
        # 实际使用时替换为 fla.ops.linear_attn 等真实实现
        output, new_state = self._linear_attention_kernel(
            q, k, v, initial_state=initial_state
        )

        # 更新 recurrent state
        past_key_values.state_update(new_state, layer_idx)

        # 输出投影
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(output)

    def _linear_attention_kernel(self, q, k, v, initial_state):
        """Linear attention kernel（简化示例）"""
        # 这里应该使用真实的 linear attention 实现
        # 例如: from fla.ops.linear_attn import fused_recurrent_linear_attn

        if initial_state is None:
            # 初始化状态矩阵
            batch, heads, _, dim = q.shape
            state = torch.zeros(batch, heads, dim, dim, device=q.device, dtype=q.dtype)
        else:
            state = initial_state

        # 简化的 linear attention 计算
        # 状态更新: state = state + K^T @ V
        # 输出: O = Q @ state
        for t in range(q.shape[2]):
            qt = q[:, :, t:t+1, :]  # [batch, heads, 1, dim]
            kt = k[:, :, t:t+1, :]  # [batch, heads, 1, dim]
            vt = v[:, :, t:t+1, :]  # [batch, heads, 1, dim]

            # 更新状态
            state = state + kt.transpose(-1, -2) @ vt

            # 计算输出
            if t == 0:
                outputs = qt @ state
            else:
                outputs = torch.cat([outputs, qt @ state], dim=2)

        return outputs, state


# 使用示例
def demo():
    model = LinearAttentionLayer(hidden_size=512, num_heads=8)
    cache = DynamicCache()

    # 第一次前向传播
    hidden_states = torch.randn(2, 10, 512)  # [batch, seq, hidden]
    output = model(hidden_states, cache, layer_idx=0)

    print(f"Output shape: {output.shape}")
    print(f"Recurrent state shape: {cache.state_update(None, 0).shape}")

    # 继续生成（增量）
    hidden_states_2 = torch.randn(2, 5, 512)
    output_2 = model(hidden_states_2, cache, layer_idx=0)

    print(f"Output 2 shape: {output_2.shape}")


if __name__ == "__main__":
    demo()
```

### 示例 2：与标准 KV Cache 并行使用

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state

hack_kv_cache_recurrent_state()


def generate_with_recurrent_state():
    model_name = "Qwen/Qwen3-1.7B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    cache = DynamicCache()

    # 标准的 KV Cache 使用方式
    inputs = tokenizer("Hello, world!", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            past_key_values=cache,
            use_cache=True,
        )

    # KV Cache 已更新
    print(f"Cache layers: {len(cache)}")
    print(f"Layer 0 keys shape: {cache.layers[0].keys.shape}")

    # Recurrent state 初始为 None
    print(f"Layer 0 recurrent state: {cache.state_update(None, 0)}")

    # 可以同时存储 recurrent state
    # 假设我们有来自 linear attention 的状态
    batch_size = 1
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    state_dim = 128

    dummy_state = torch.randn(batch_size, num_heads, head_dim, state_dim, device=model.device)
    cache.state_update(dummy_state, layer_idx=0)

    print(f"Layer 0 recurrent state shape: {cache.state_update(None, 0).shape}")

    # 继续生成
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    next_inputs = {"input_ids": next_token}

    with torch.no_grad():
        next_outputs = model(
            **next_inputs,
            past_key_values=cache,
            use_cache=True,
        )

    print(f"Generated next token: {tokenizer.decode(next_token[0])}")


if __name__ == "__main__":
    generate_with_recurrent_state()
```

### 示例 3：重置和状态管理

```python
import torch
from transformers import DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state

hack_kv_cache_recurrent_state()


def state_management_demo():
    cache = DynamicCache()

    # 初始化一些 KV 和 recurrent state
    k = torch.randn(1, 8, 10, 64)
    v = torch.randn(1, 8, 10, 64)
    cache.update(k, v, layer_idx=0)

    state = torch.randn(1, 8, 64, 128)
    cache.state_update(state, layer_idx=0)

    print("Before reset:")
    print(f"  Keys shape: {cache.layers[0].keys.shape}")
    print(f"  Recurrent state shape: {cache.state_update(None, 0).shape}")

    # 重置特定层
    cache.layers[0].reset()

    print("\nAfter reset:")
    print(f"  Keys: {cache.layers[0].keys}")  # 应该被 zero_()
    print(f"  Recurrent state: {cache.state_update(None, 0)}")  # 应该是 None

    # 重新使用
    new_state = torch.randn(1, 8, 64, 128)
    cache.state_update(new_state, layer_idx=0)
    print(f"\nNew recurrent state shape: {cache.state_update(None, 0).shape}")


if __name__ == "__main__":
    state_management_demo()
```

---

## Recurrent State 的形状说明

线性注意力的 recurrent state 通常具有以下形状：

```
[batch_size, num_heads, head_dim, state_dim]
```

或对于某些实现（如 GLA）：

```
[batch_size, num_heads, head_dim, head_dim]  # KV 状态矩阵
```

具体形状取决于所使用的线性注意力 kernel 的实现。

---

## 注意事项

1. **调用顺序**：确保在加载模型或创建 Cache 对象之前调用 `hack_kv_cache_recurrent_state()`。

2. **状态覆盖**：`state_update()` 会直接覆盖之前的 recurrent state，不会进行累加或合并。

3. **内存管理**：recurrent state 存储在 GPU 上（如果 tensor 在 GPU），注意内存使用。

4. **与 KV Cache 独立**：recurrent state 的管理完全独立于标准的 key-value cache，两者可以同时使用。

5. **序列长度**：recurrent state 的大小与序列长度无关，这是线性注意力的优势之一。

---

## 相关资源

- [Flash Linear Attention (FLA)](https://github.com/sustcsonglin/flash-linear-attention)
- [Transformers Cache Utils](https://huggingface.co/docs/transformers/internal/cache_utils)
- [Gated Linear Attention Paper](https://arxiv.org/abs/2312.06635)
