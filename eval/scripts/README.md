# SWAA 模型评估脚本使用指南

## 概述

`eval_swaa_model.py` 是一个基于 lm-evaluation-harness 的评估脚本，支持对 SWAA (Sliding Window Attention Adaptation) 模型进行全面评估，并**自动保存详细的样本级推理结果**。

## 核心特性

✅ **详细结果保存**：保存每个样本的输入、输出和预测结果
✅ **多层次输出**：完整结果、摘要、按任务分类的样本文件
✅ **灵活配置**：支持所有 SWAA 参数和 lm_eval 参数
✅ **时间戳管理**：每次评估结果独立保存，便于对比分析

## 快速开始

### 基础用法

```bash
# 基本评估（使用默认参数）
python eval/scripts/eval_swaa_model.py \
    --model_path ./your_model \
    --tasks hellaswag arc_easy

# 指定输出目录
python eval/scripts/eval_swaa_model.py \
    --model_path ./your_model \
    --tasks hellaswag arc_easy \
    --output_dir ./my_results
```

### 高级用法

```bash
# 完整参数示例
python eval/scripts/eval_swaa_model.py \
    --model_path ./your_model \
    --tasks mmlu hellaswag arc_easy \
    --output_dir ./eval_results \
    --batch_size 4 \
    --device cuda:0 \
    --torch_dtype bfloat16 \
    --attn_implementation flash_attention_2 \
    --num_fewshot 5 \
    --sliding_window_size 2048 \
    --keep_first 4 \
    --linear_mem_mode fused_chunk \
    --linear_mem_blend_mode orth_match \
    --linear_kernel_family softplus \
    --flash_attn_weight 0.9 \
    --linear_mem_weight 0.1
```

```bash
# 禁用 linear memory
python eval/scripts/eval_swaa_model.py \
    --model_path ./your_model \
    --tasks hellaswag \
    --disable_linear_mem
```

### 测试模式（限制样本数）

```bash
# 快速测试（每个任务只评估 10 个样本）
python eval/scripts/eval_swaa_model.py \
    --model_path ./your_model \
    --tasks hellaswag \
    --limit 10
```

## 输出文件说明

每次评估会在 `output_dir` 中生成以下文件：

### 1. 完整结果文件
- **文件名**：`results_<timestamp>.json`
- **内容**：包含所有评估数据，包括样本级结果
- **大小**：较大（包含完整的输入输出数据）
- **用途**：完整记录，用于深度分析

```json
{
  "results": {...},      // 评估指标
  "configs": {...},      // 任务配置
  "samples": {           // 样本级详细结果
    "hellaswag": [
      {
        "doc_id": 0,
        "doc": {...},          // 原始文档
        "target": "...",       // 正确答案
        "arguments": [...],    // 模型输入
        "resps": [...],        // 模型输出
        "filtered_resps": [...] // 处理后的输出
      },
      ...
    ]
  }
}
```

### 2. 摘要文件
- **文件名**：`summary_<timestamp>.json`
- **内容**：仅包含评估指标，不包含样本数据
- **大小**：较小
- **用途**：快速查看评估结果

```json
{
  "results": {
    "hellaswag": {
      "acc": 0.4523,
      "acc_norm": 0.5532
    }
  },
  "n-shot": {...},
  "higher_is_better": {...}
}
```

### 3. 样本文件（按任务分类）
- **目录**：`samples_<timestamp>/`
- **文件**：`<task_name>.jsonl`
- **格式**：JSONL（每行一个样本的 JSON 对象）
- **用途**：便于程序化分析和处理

**示例** (`samples_20250101_120000/hellaswag.jsonl`)：
```json
{"doc_id": 0, "target": "A", "resps": [...], ...}
{"doc_id": 1, "target": "B", "resps": [...], ...}
{"doc_id": 2, "target": "C", "resps": [...], ...}
```

### 4. 执行日志
- **文件名**：`evaluation.log`
- **位置**：`eval/evaluation.log`
- **内容**：每个样本的推理日志，包括输入长度、输出内容等

## 参数详解

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--model_path` | str | 模型路径（本地路径或 HuggingFace 模型名） |
| `--tasks` | list[str] | 评估任务列表（如 `hellaswag arc_easy mmlu`） |

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `./eval_results` | 结果保存目录 |
| `--batch_size` | 1 | 批处理大小 |
| `--device` | `cuda:0` | 运行设备 |
| `--torch_dtype` | `bfloat16` | 数据类型（float32/float16/bfloat16） |
| `--attn_implementation` | `flash_attention_2` | attention 后端（flash_attention_2/eager/sdpa） |
| `--num_fewshot` | None | Few-shot 示例数量 |
| `--limit` | None | 限制每个任务的样本数（用于测试） |

### SWAA 专用参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sliding_window_size` | 2048 | 滑动窗口大小 |
| `--keep_first` | 4 | 保留的前 N 个 token |
| `--force_fa_decode` | False | 强制使用 Flash Attention 解码 |
| `--force_fa_decode_layers` | None | 指定哪些层强制 full-attention decode |
| `--non_sliding_layers` | [] | 不使用滑动窗口的层 |
| `--enable_linear_mem` | True | 启用线性记忆机制 |
| `--disable_linear_mem` | False | 禁用线性记忆机制 |
| `--flash_attn_weight` | 0.9 | Flash Attention 权重 |
| `--linear_mem_weight` | 0.1 | 线性记忆权重 |
| `--linear_mem_mode` | `fused_recurrent` | Linear Memory 执行模式 |
| `--linear_mem_blend_mode` | `raw` | flash attention 与 linear memory 的融合方式 |
| `--linear_kernel_family` | `softplus` | linear memory 使用的 kernel 族（softplus/niah/anchor/none） |
| `--num_anchors` | 64 | Anchor kernel 的 anchor 数，仅 `anchor` 模式生效 |
| `--tau` | 20.0 | Anchor kernel 温度，仅 `anchor` 模式生效 |

## 常见任务列表

### 常识推理
- `hellaswag` - HellaSwag
- `arc_easy`, `arc_challenge` - AI2 Reasoning Challenge
- `winogrande` - WinoGrande

### 知识问答
- `mmlu` - Massive Multitask Language Understanding
- `triviaqa` - TriviaQA

### 阅读理解
- `boolq` - Boolean Questions
- `piqa` - Physical Interaction QA

### 数学推理
- `gsm8k` - Grade School Math 8K

查看所有可用任务：
```bash
python -m lm_eval.tasks --list
```

## Python API 使用

除了命令行，也可以在 Python 代码中直接调用：

```python
from eval.scripts.eval_swaa_model import run_evaluation

results = run_evaluation(
    model_path="./your_model",
    tasks=["hellaswag", "arc_easy"],
    output_dir="./results",
    batch_size=4,
    num_fewshot=5,
    sliding_window_size=2048,
    enable_linear_mem=True,
)

# 访问结果
print(results["results"]["hellaswag"]["acc"])

# 访问样本数据
for sample in results["samples"]["hellaswag"]:
    print(f"Sample {sample['doc_id']}: {sample['filtered_resps']}")
```

## 结果分析示例

### 分析样本级结果

```python
import json

# 读取样本文件
with open("eval_results/samples_20250101_120000/hellaswag.jsonl") as f:
    samples = [json.loads(line) for line in f]

# 统计正确率
correct = sum(1 for s in samples if s["filtered_resps"][0] == s["target"])
print(f"准确率: {correct / len(samples):.2%}")

# 查看错误样本
errors = [s for s in samples if s["filtered_resps"][0] != s["target"]]
print(f"错误样本数: {len(errors)}")
```

### 对比多次评估结果

```python
import json
from pathlib import Path

results_dir = Path("eval_results")
all_summaries = {}

for summary_file in results_dir.glob("summary_*.json"):
    timestamp = summary_file.stem.replace("summary_", "")
    with open(summary_file) as f:
        all_summaries[timestamp] = json.load(f)

# 打印不同时间的准确率对比
for timestamp, summary in sorted(all_summaries.items()):
    acc = summary["results"]["hellaswag"]["acc"]
    print(f"{timestamp}: {acc:.4f}")
```

## 故障排除

### 1. 内存不足
```bash
# 减小 batch_size
--batch_size 1

# 限制样本数
--limit 100
```

### 2. 模型加载失败
```bash
# 确保路径正确
ls -la ./your_model

# 使用绝对路径
--model_path /absolute/path/to/your_model
```

### 3. 任务未找到
```bash
# 列出所有可用任务
python -m lm_eval.tasks --list

# 搜索特定任务
python -m lm_eval.tasks --list | grep mmlu
```

## 技术细节

### 关键机制

1. **log_samples 参数**：通过设置 `log_samples=True`，lm_eval 会在返回结果中包含每个样本的详细数据

2. **多层次保存**：
   - 完整结果：用于存档和深度分析
   - 摘要：用于快速查看
   - JSONL 样本文件：用于程序化处理

3. **时间戳管理**：每次评估使用独立时间戳，避免结果覆盖

### 性能优化

- 使用 `bfloat16` 减少显存占用
- 根据硬件调整 `batch_size`
- 使用 `--limit` 进行快速测试

## 参考资料

- [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [Python API 指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/python-api.md)
- [任务列表](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md)

## 更新日志

- **2025-01**：添加样本级详细结果保存功能
- **2025-01**：添加命令行接口和 Python API
- **2025-01**：支持多层次结果输出（完整/摘要/样本文件）
