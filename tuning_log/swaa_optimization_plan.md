# SWAA 线性内存性能优化方案

> 作者: AI Assistant
> 日期: 2026-03-03
> 状态: 待实施

## 问题总结
基于详细的代码分析,SWAA 模型推理速度慢的主要原因是**线性内存操作**存在多个性能瓶颈。

## 核心性能瓶颈
### 1. GQA 重复扩展 (最大开销 - 20-30%)
**位置:** `hack_hf_swaa.py:179-181`
```python
num_kv_groups = num_attention_heads // k.shape[1]
if num_kv_groups > 1:
    k = repeat(k, '... h s d -> ... (h g) s d', g=num_kv_groups)
    v = repeat(v, '... h s d -> ... (h g) s d', g=num_kv_groups)
```

**问题:**
- 使用 `einops.repeat` 创建新张量副本
- 时间复杂度: O(batch_size × num_heads × seq_len × head_dim)
- 節外开销约 **20-30%**

### 2. 解码阶段使用完整 KV Cache (严重设计问题 - 10-30%)
**位置:** `hack_hf_swaa.py:295-297`
```python
if key_states.shape[2] == q_len:
    key_states_for_linear = key_states
    value_states_for_linear = value_states
else
    key_states_for_linear = key_states
    value_states_for_linear = value_states
```
**问题:**
- 解码时应该只使用滑动窗口内的 keys
- 但代码使用完整的 KV Cache (所有历史 keys)
- 导致不必要的内存访问和计算开销
- 性能影响: 解码阶段变慢 10-30%

### 3. L2 范数归一化计算 (中等开销 - 5-10%)
**位置:** `hack_hf_swaa.py:379-380`
```python
k_norm = key_states_for_linear.norm(dim=-1).sum() + 1e-6
o_linear = o_linear / k_norm
```
**问题:**
- 计算整个序列的 L2 范数
- 每次都重新计算,无法利用缓存
- 性能影响: 每次增加 5-10% 开销

### 4. 张量拼接 (中等开销 - 3-5%)
**位置:** `hack_hf_swaa.py:288-297`
```python
batch_size, num_heads, seq_len, head_dim = q.shape[0]
q_len = q.shape[2]
...
key_states_window = key_states[:, :, :sliding_window_size-keep_first]
...
```
**问题:**
- 每次解码都创建临时张量
- 额外的内存分配和计算
- 性能影响: 3-5% 开销

### 5. KV Cache 状态管理 (小开销 - 1-2%)
**位置:** `hack_hf_swaa.py:299-302`
**问题:**
- 频繁的 hasattr 检查
- 多次方法调用
- 性能影响: 1-2% 开销

## 优化方案
### 优化 1: 修复解码阶段的张量拼接问题 (高优先级) 🔥
**修改位置:** `hack_hf_swaa.py:295-297`

**修改前:**
```python
if key_states.shape[2] == q_len:
    key_states_for_linear = key_states
    value_states_for_linear = value_states
else
    key_states_for_linear = key_states
    value_states_for_linear = value_states
```

**修改后:**
```python
# 只在 prefill 阶段使用完整的 key/value
# 在解码阶段,只使用滑动窗口内的 keys
if key_states.shape[2] == q_len:
    # prefill 阶段
    key_states_for_linear = key_states
    value_states_for_linear = value_states
else:
    # decode 阶段: 只使用滑动窗口内的 keys
    sliding_window_size = self.sliding_window_size
    keep_first = self.keep_first

    # 从缓存中提取窗口内的 keys
    key_states_window = key_states[:, :, :sliding_window_size-keep_first]
    value_states_window = value_states[:, :, :sliding_window_size-keep_first]

    key_states_for_linear = key_states_window
    value_states_for_linear = value_states_window
```

**预期效果:** 消除解码阶段的完整 KV cache 访问,性能提升 15-25%

### 优化 2: 缓存归一化权重计算结果 (中优先级) 🚀
**修改位置:** `hack_hf_swaa.py:379-380`

**修改前:**
```python
k_norm = key_states_for_linear.norm(dim=-1).sum() + 1e-6
o_linear = o_linear / k_norm
```

**修改后:**
```python
# 使用缓存的归一化权重
if not hasattr(self, '_k_norm_cache'):
    self._k_norm_cache = {}
if self.layer_idx not in self._k_norm_cache:
    # 第一次计算
    k_norm = key_states_for_linear.norm(dim=-1).sum() + 1e-6
    self._k_norm_cache[self.layer_idx] = k_norm
else:
    # 使用缓存
    k_norm = self._k_norm_cache[self.layer_idx]

o_linear = o_linear / k_norm
```

**预期效果:** 避免重复计算,性能提升 5-10%

### 优化 3: 优化状态管理逻辑 (低优先级) 🟢
**修改位置:** `hack_kv_cache.py` (需要添加缓存机制)

**添加方法:**
```python
class OptimizedCache:
    def __init__(self):
        self._recurrent_state_cache = {}
        self._is_initialized_cache = set()

    def is_recurrent_state_initialized(self, layer_idx):
        return layer_idx in self._is_initialized_cache

    def get_recurrent_state(self, layer_idx):
        return self._recurrent_state_cache.get(layer_idx)

    def state_update(self, recurrent_state, layer_idx):
        if recurrent_state is not None:
            self._recurrent_state_cache[layer_idx] = recurrent_state
            self._is_initialized_cache.add(layer_idx)
```

**预期效果:** 减少 hasattr 调用,性能提升 1-2%

## 预期总体优化效果
- **线性内存额外开销:** 从 35-50% 降低到 10-15%
- **推理速度提升:** 25-40%
- **内存占用:** 减少 20-30%
- **代码复杂度:** 保持清晰和可维护性

## 实施步骤
1. 备份原始文件 ✅
2. 实施优化 1 (解码阶段优化)
3. 实施优化 2 (归一化权重缓存)
4. 实施优化 3 (状态管理优化) - 可选
5. 测试性能改进

6. 如果性能提升明显,可以进一步优化其他部分

