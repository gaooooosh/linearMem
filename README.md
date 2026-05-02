# linearMem

`linearMem` 是一个研究型代码库，用于探索在冻结大语言模型参数的前提下，为滑动窗口注意力模型补充一条低成本全局记忆通道。项目当前实现了 SWAA（Sliding Window Attention Adaptation）补丁、Linear Memory 分支、KV cache recurrent state 扩展、定制 FlashAttention kernel，以及 RULER / lm-eval 评测入口。

当前研究定位不是“训练自由恢复精确长程检索”，而是：在 sliding-window attention 截断窗口外显式访问路径时，通过并联的递归线性记忆分支，为窗口外 token 的分布建模提供全局统计先验。已有实验显示该方法能稳定改善 exterior PPL，但在 RULER / LongBench / QA 检索类任务上尚未观察到同等稳定收益。

## 核心思路

模型推理时同时维护两条路径：

1. **局部精确分支**：使用带 sink token 的 sliding-window FlashAttention，保留窗口内 softmax attention 的建模质量。
2. **全局线性记忆分支**：将历史 `K,V` 通过训练自由 kernel 压缩到 recurrent state，并用当前 `Q` 读取该状态。
3. **输出融合**：在 attention `o_proj` 之前融合两条分支，支持 raw、centered、orth、orth_match 等融合方式。
4. **双状态缓存**：标准 KV cache 继续服务局部分支，新增 recurrent state 服务 Linear Memory 分支。

代码主要支持 Hugging Face Transformers 路径，并提供 vLLM 0.11.0 相关实验补丁。

## 目录结构

```text
linearMem/
├── swaa_patch/                 # SWAA / Linear Memory / KV cache monkey patch
│   ├── hack_hf_swaa.py         # Hugging Face attention forward 替换与融合逻辑
│   ├── hack_kv_cache.py        # recurrent state cache 支持
│   ├── swaa_config.py          # SWAAConfig
│   ├── kernel/                 # anchor / softplus / NIAH 等 kernel
│   └── serve_swaa.py           # vLLM OpenAI server 实验入口
├── flash-attention-SWAA/       # 修改版 FlashAttention / flash-attention-vllm
├── eval/
│   ├── scripts/run_ruler.py    # RULER 长文本评测入口
│   ├── scripts/eval_swaa_model.py
│   └── docs/RULER_GUIDE.md
├── TEST_CASE/                  # 手工长文本测试样例
├── qkv_analysis/               # Q/K/V 分析脚本与图表
├── experiments/                # 实验检查点与结果记录
├── tuning_log/                 # 优化记录与性能分析
├── METHOD.md                   # 方法草稿
├── paper_draft_sequence_mixed_attention.md
├── test.py                     # Qwen3-1.7B 推理 smoke test
├── test_*.py                   # 单元/接口测试
├── pixi.toml                   # Pixi 环境定义
└── setup_env.sh                # CUDA / 编译环境激活脚本
```

## 环境要求

推荐在 Linux + NVIDIA GPU 环境中运行。本仓库的 Pixi 配置面向 A100 / CUDA 13.1 系统环境，并安装 PyTorch 2.5.1 CUDA 12.4 wheel 以获得更好的兼容性。

主要依赖：

- Python 3.12
- PyTorch 2.5.1
- Transformers >= 4.48
- flash-linear-attention >= 0.4.1
- lm-eval
- 可选：vLLM、定制 FlashAttention-SWAA

安装环境：

```bash
pixi install
pixi run verify
```

如果只想检查基础包导入：

```bash
pixi run test-imports
pixi run test-swaa
```

## 编译定制 FlashAttention

如果使用 `attn_implementation="flash_attention_2"` 并需要 `keep_first`、`force_fa_decode` 等 SWAA 扩展参数，需要编译仓库内的定制 FlashAttention：

```bash
pixi run build-flash-attn
```

如需同时编译 vLLM 版本：

```bash
pixi run build-all
```

也可以进入子目录手动安装：

```bash
cd flash-attention-SWAA/flash-attention
PATCH_TORCH_CUDA_CHECK=1 python setup.py build_ext --inplace install
```

## 快速运行

运行 Hugging Face 推理 smoke test：

```bash
pixi run python test.py
```

`test.py` 默认加载 `Qwen/Qwen3-1.7B`，应用：

- `hack_kv_cache_recurrent_state()`
- `hack_hf_swaa(training=False)`
- `SWAAConfig(sliding_window_size=2048, keep_first=64, enable_linear_mem=True)`
- NIAH 风格的 `PositionAwareKernel` / `DenseQueryKernel`

最小使用示例：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from swaa_patch import SWAAConfig, hack_hf_swaa, hack_kv_cache_recurrent_state

hack_kv_cache_recurrent_state()
hack_hf_swaa(training=False)

model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": "cuda:0"},
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
).eval()

model.config.swaa_config = SWAAConfig(
    sliding_window_size=2048,
    keep_first=64,
    enable_linear_mem=True,
    flash_attn_weight=0.8,
    linear_mem_weight=0.2,
    linear_mem_mode="fused_chunk",
    linear_mem_blend_mode="orth_match",
)

prompt = "解释一下线性注意力为什么可以降低长序列复杂度。"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
outputs = model.generate(
    **inputs,
    max_new_tokens=128,
    past_key_values=DynamicCache(),
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

如需启用自定义 query/key kernel，可参考 `test.py` 中的 `model.config.kernel_q` 与 `model.config.kernel_k` 设置。

## RULER 长文本评测

RULER 入口位于 `eval/scripts/run_ruler.py`：

```bash
cd eval/scripts

# 32K 上下文，启用 Linear Memory
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768

# 禁用 Linear Memory 做对照
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768 --no-linear-mem

# 快速测试
python run_ruler.py --model Qwen/Qwen3-1.7B --limit 10
```

常用参数：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--context-length` | `32768` | 评测上下文长度 |
| `--sliding-window` | `2048` | 滑动窗口大小 |
| `--keep-first` | `4` | 始终保留的 sink token 数量 |
| `--enable-linear-mem` | `True` | 是否启用 Linear Memory |
| `--no-linear-mem` | `False` | 禁用 Linear Memory |
| `--linear-mem-mode` | `fused_recurrent` | `fused_recurrent` / `fused_chunk` / `chunk` |
| `--linear-mem-blend-mode` | `raw` | `raw` / `centered` / `orth` / `orth_match` |
| `--linear-kernel-family` | `softplus` | `softplus` / `niah` / `anchor` / `none` |
| `--active-layers` | 空 | 只在指定层启用 Linear Memory |
| `--beta-by-layer` | 空 | 逐层 beta，如 `0:0.08,18:0.16` |

结果会写入 `eval/results/ruler_{长度K}_{lm状态}_{timestamp}/`，包括配置、完整结果、表格和报告。

更详细说明见 [eval/docs/RULER_GUIDE.md](eval/docs/RULER_GUIDE.md)。

## lm-eval 评测

通用 lm-evaluation-harness 入口：

```bash
python eval/scripts/eval_swaa_model.py \
    --model_path Qwen/Qwen3-1.7B \
    --tasks hellaswag arc_easy \
    --output_dir ./eval_results \
    --batch_size 1 \
    --sliding_window_size 2048 \
    --keep_first 4 \
    --linear_mem_mode fused_chunk \
    --linear_mem_blend_mode orth_match \
    --linear_kernel_family softplus
```

禁用 Linear Memory：

```bash
python eval/scripts/eval_swaa_model.py \
    --model_path Qwen/Qwen3-1.7B \
    --tasks hellaswag \
    --disable_linear_mem
```

输出包括完整 JSON、摘要 JSON、按任务拆分的样本 JSONL 和执行日志。详见 [eval/scripts/README.md](eval/scripts/README.md)。

## 单元测试

无需加载大模型的接口测试：

```bash
pixi run python -m unittest test_linear_mem_interface.py
pixi run python test_kv_cache_patch.py
pixi run python test_k_norm_cache.py
```

其中 `test_linear_mem_interface.py` 主要检查 `linear_mem_ops` 与 FLA recurrent state 接口兼容性。

## vLLM 实验入口

`swaa_patch/serve_swaa.py` 提供 vLLM OpenAI-compatible server 的实验入口，会在启动时 patch vLLM 模型与 attention 类：

```bash
python swaa_patch/serve_swaa.py \
    --model Qwen/Qwen3-1.7B \
    --sliding-window-size 2048 \
    --keep-first 4 \
    --force-fa-decode True \
    --non-sliding-layers "[0, 1, 2, 3]"
```

注意：vLLM 路径依赖具体 vLLM 版本与本仓库的 `flash-attention-vllm` 修改版，建议优先使用 Hugging Face 路径复现实验。

## 当前研究状态

截至当前仓库记录，比较稳定的研究结论是：

- Linear Memory 更像对滑窗模型的全局统计先验，而不是精确检索替代品。
- 对 Qwen3-1.7B，`L0(0.08)+L18(0.16)` 在 PG19 exterior PPL 上表现较稳定，且收益随上下文长度增长。
- 收益主要集中在高惊讶度 token、中频内容词、段落起始位置和远窗外 token。
- RULER / LongBench / QA 类下游任务尚未稳定受益，因此 README 中的运行示例默认把它们作为边界验证和对照评测。

更多方法与实验叙述见：

- [METHOD.md](METHOD.md)
- [paper_draft_sequence_mixed_attention.md](paper_draft_sequence_mixed_attention.md)
- [tuning_log/SWAA线性内存性能优化完整报告.md](tuning_log/SWAA线性内存性能优化完整报告.md)

## 已知限制

- 本项目大量使用 monkey patch，需在加载模型前调用 patch 函数。
- `flash_attention_2` 路径依赖定制 FlashAttention 编译是否成功。
- 不同 Transformers / vLLM 版本的内部类路径可能变化，升级依赖后需要重新验证。
- 当前代码以研究迭代为主，部分目录包含实验草稿、历史报告和备份实现。
- 根目录暂未声明统一开源许可证；引用或发布前请先确认授权边界。

## 引用

如果基于本仓库继续研究或撰写论文，建议引用项目中的方法草稿与实验报告，并明确说明使用的是 training-free SWAA / Linear Memory 研究实现。
