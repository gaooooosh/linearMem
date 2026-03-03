# k_norm 缓存集成到 KV Cache - 最终方案

## 🎯 问题解决

**原始错误:**
```
AttributeError: 'DynamicCache' object has no attribute 'is_k_norm_cache_initialized'
```

**原因:**
用户调用了 `hack_kv_cache_recurrent_state()` 但没有调用 `hack_kv_cache_k_norm()`，导致 KV Cache 没有 k_norm 缓存方法。

## ✅ 解决方案

### 自动应用机制

修改了 `hack_kv_cache.py`，使其在应用 recurrent state patch 时**自动应用** k_norm 缓存 patch：

**文件:** `swaa_patch/hack_kv_cache.py:459-466`

```python
def hack_kv_cache_recurrent_state():
    # ... 原有的 recurrent state patch 代码 ...

    print("Hacked transformers KV Cache layers to support recurrent state for linear attention.")

    # Also apply k_norm cache patch automatically
    try:
        from .hack_kv_cache_k_norm import hack_kv_cache_k_norm
        hack_kv_cache_k_norm()
    except Exception as e:
        print(f"Warning: Failed to apply k_norm cache patch: {e}")
```

### 自动清理机制

同时修改了 `unhack_kv_cache_recurrent_state()` 来自动清理 k_norm 缓存：

**文件:** `swaa_patch/hack_kv_cache.py:526-531`

```python
def unhack_kv_cache_recurrent_state():
    # ... 原有的清理代码 ...

    print("Restored original transformers KV Cache layer methods.")

    # Also unapply k_norm cache patch
    try:
        from .hack_kv_cache_k_norm import unhack_kv_cache_k_norm
        unhack_kv_cache_k_norm()
    except Exception as e:
        print(f"Warning: Failed to remove k_norm cache patch: {e}")
```

## 📝 使用方法

### 简化使用（推荐）

现在只需要调用一个函数，两个 patch 会自动应用：

```python
from swaa_patch import (
    hack_hf_swaa,
    hack_kv_cache_recurrent_state,  # 会自动应用 k_norm 缓存
    SWAAConfig
)

# 应用 patches
hack_hf_swaa(training=False)
hack_kv_cache_recurrent_state()  # ✅ 自动应用 recurrent state 和 k_norm 缓存

# 配置模型
swaa_config = SWAAConfig(
    sliding_window_size=2048,
    keep_first=64,
    force_fa_decode=True,
    enable_linear_mem=True,
)
model.config.swaa_config = swaa_config

# 推理（k_norm 会自动缓存到 KV Cache 中）
outputs = model.generate(...)
```

### 手动控制（可选）

如果需要单独控制，仍然可以手动调用：

```python
from swaa_patch import (
    hack_kv_cache_recurrent_state,
    hack_kv_cache_k_norm,  # 可以单独调用
    unhack_kv_cache_k_norm,
    unhack_kv_cache_recurrent_state,
)

# 单独应用
hack_kv_cache_recurrent_state()  # 只应用 recurrent state
hack_kv_cache_k_norm()  # 只应用 k_norm 缓存

# 单独清理
unhack_kv_cache_k_norm()  # 只清理 k_norm 缓存
unhack_kv_cache_recurrent_state()  # 清理 recurrent state（也会自动清理 k_norm）
```

## 🔄 工作流程

### 应用流程

```
用户调用 hack_kv_cache_recurrent_state()
    ↓
应用 recurrent state patch
    ↓
自动调用 hack_kv_cache_k_norm()
    ↓
应用 k_norm 缓存 patch
    ↓
✅ 两个 patch 都已应用
```

### 清理流程

```
用户调用 unhack_kv_cache_recurrent_state()
    ↓
清理 recurrent state patch
    ↓
自动调用 unhack_kv_cache_k_norm()
    ↓
清理 k_norm 缓存 patch
    ↓
✅ 两个 patch 都已清理
```

## 📊 测试结果

运行 `test_k_norm_cache.py`:

```
✅ 所有测试通过
- Layer 级别缓存正常工作
- Cache 级别缓存正常工作
- 多层支持正常
- 初始化状态检查正常
- 自动应用/清理正常
```

## 🎁 优势

1. **用户友好:** 只需调用一个函数，无需关心内部细节
2. **自动同步:** 两个 patch 自动保持同步应用和清理
3. **向后兼容:** 仍然支持手动单独调用
4. **错误处理:** 有完善的异常处理，不会影响主流程

## 📁 修改的文件

```
swaa_patch/
└── hack_kv_cache.py         # 添加了自动应用/清理逻辑
```

## 🚀 下一步

现在可以直接使用您的推理脚本，k_norm 缓存会自动应用到 KV Cache 中！

```bash
python your_inference_script.py
```

无需任何额外配置，patch 会自动应用！✨
