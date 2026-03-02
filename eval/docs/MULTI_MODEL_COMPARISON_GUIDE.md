# RULER 32K 多模型对比评测指南

## 📊 在 lm-eval 中切换模型的方法

### 方法 1：命令行直接切换（最简单）

```bash
# 评测模型 1
lm-eval --model hf \
    --model_args pretrained=Qwen/Qwen3-1.7B \
    --tasks niah_single_1,niah_single_2,passkey \
    --device cuda:0 \
    --limit 100

# 评测模型 2（切换模型）
lm-eval --model hf \
    --model_args pretrained=path/to/model2 \
    --tasks niah_single_1,niah_single_2,passkey \
    --device cuda:0 \
    --limit 100
```

**核心参数**: 使用 `--model_args pretrained=模型路径` 来切换模型

### 方法 2：使用自定义 SWAA 模型

```bash
# 使用 SWAA 模型（滑动窗口 2048）
lm-eval --model swaa_hf \
    --model_args pretrained=Qwen/Qwen3-1.7B,sliding_window_size=2048,keep_first=4 \
    --tasks niah_single_1,passkey \
    --device cuda:0

# 切换到不同的 SWAA 配置（滑动窗口 4096）
lm-eval --model swaa_hf \
    --model_args pretrained=Qwen/Qwen3-1.7B,sliding_window_size=4096,keep_first=8 \
    --tasks niah_single_1,passkey \
    --device cuda:0
```

### 方法 3：使用多模型对比脚本（推荐）

#### 3.1 单个模型评测

```bash
python run_ruler_32k_comparison.py \
    --model Qwen/Qwen3-1.7B \
    --name "Qwen3-Base" \
    --device cuda:0 \
    --limit 100
```

#### 3.2 多个模型对比（命令行）

```bash
python run_ruler_32k_comparison.py \
    --models Qwen/Qwen3-1.7B path/to/model2 path/to/model3 \
    --device cuda:0 \
    --limit 100
```

#### 3.3 使用配置文件（推荐用于复杂对比）

```bash
# 1. 编辑配置文件
vim models_comparison_config.yaml

# 2. 运行对比评测
python run_ruler_32k_comparison.py \
    --config models_comparison_config.yaml \
    --device cuda:0 \
    --limit 100
```

## 🎯 lm-eval 核心参数说明

### 模型相关参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型类型 | `hf`, `swaa_hf` |
| `--model_args` | 模型参数 | `pretrained=模型路径,dtype=bfloat16` |

### 模型参数详解

```bash
--model_args <参数1>=<值1>,<参数2>=<值2>,...
```

**常用参数:**
- `pretrained`: 模型路径或 HuggingFace ID
- `dtype`: 数据类型 (`bfloat16`, `float16`, `float32`)
- `attn_implementation`: 注意力实现 (`flash_attention_2`, `eager`, `sdpa`)
- `device`: 设备 (`cuda:0`, `cuda:1`, `cpu`)
- `batch_size`: 批次大小

**SWAA 专用参数:**
- `sliding_window_size`: 滑动窗口大小
- `keep_first`: 保留的前 N 个 token
- `force_fa_decode`: 是否强制全注意力解码
- `non_sliding_layers`: 非滑动层列表

### 评测相关参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--tasks` | 任务列表 | `niah_single_1,passkey` |
| `--limit` | 样本数量限制 | `100` |
| `--num_fewshot` | Few-shot 数量 | `0` |
| `--batch_size` | 批次大小 | `1` |
| `--device` | 设备 | `cuda:0` |
| `--output_path` | 输出路径 | `results.json` |

## 📝 实战示例

### 示例 1：对比两个 HuggingFace 模型

```bash
# 模型 1
lm-eval --model hf \
    --model_args pretrained=Qwen/Qwen2-1.5B \
    --tasks niah_single_1,passkey \
    --limit 100 \
    --device cuda:0 \
    --output_path results_qwen2.json

# 模型 2
lm-eval --model hf \
    --model_args pretrained=meta-llama/Llama-2-7b-hf \
    --tasks niah_single_1,passkey \
    --limit 100 \
    --device cuda:0 \
    --output_path results_llama2.json
```

### 示例 2：对比不同 SWAA 配置

```bash
# 配置 1: SWAA 2048
python run_ruler_32k_comparison.py \
    --model Qwen/Qwen3-1.7B \
    --name "SWAA-2048" \
    --swaa-window 2048 \
    --keep-first 4 \
    --device cuda:0 \
    --limit 100

# 配置 2: SWAA 4096
python run_ruler_32k_comparison.py \
    --model Qwen/Qwen3-1.7B \
    --name "SWAA-4096" \
    --swaa-window 4096 \
    --keep-first 8 \
    --device cuda:0 \
    --limit 100
```

### 示例 3：使用配置文件对比多个模型

**配置文件 (`models.yaml`):**
```yaml
models:
  - name: "Baseline"
    path: "Qwen/Qwen3-1.7B"

  - name: "SWAA-2048"
    path: "Qwen/Qwen3-1.7B"
    swaa_window: 2048
    keep_first: 4

  - name: "SWAA-4096"
    path: "Qwen/Qwen3-1.7B"
    swaa_window: 4096
    keep_first: 8

  - name: "Custom-Model"
    path: "/path/to/your/model"
    swaa_window: 2048
    keep_first: 4
```

**运行评测:**
```bash
python run_ruler_32k_comparison.py \
    --config models.yaml \
    --device cuda:0 \
    --limit 100
```

## 🔧 高级技巧

### 1. 在不同 GPU 上并行评测

```bash
# GPU 0 评测模型 1
CUDA_VISIBLE_DEVICES=0 python run_ruler_32k_comparison.py \
    --model model1 --device cuda:0 &

# GPU 1 评测模型 2
CUDA_VISIBLE_DEVICES=1 python run_ruler_32k_comparison.py \
    --model model2 --device cuda:0 &

wait
```

### 2. 分批评测（节省内存）

```bash
# 分批评测任务
for task in niah_single_1 niah_single_2 passkey; do
    lm-eval --model hf \
        --model_args pretrained=model_path \
        --tasks $task \
        --limit 100 \
        --device cuda:0
done
```

### 3. 使用缓存加速

```bash
# 启用缓存
lm-eval --model hf \
    --model_args pretrained=model_path \
    --tasks niah_single_1 \
    --limit 100 \
    --cache_dir ./cache \
    --device cuda:0
```

## 📊 结果对比

### 自动对比（使用脚本）

```bash
python run_ruler_32k_comparison.py \
    --models model1 model2 model3 \
    --device cuda:0 \
    --limit 100
```

脚本会自动生成：
- 对比表格 (Markdown)
- 对比报告 (`comparison_report.md`)
- 对比数据 (`model_comparison.json`)

### 手动对比

```bash
# 查看结果
cat results_model1.json | jq '.results'
cat results_model2.json | jq '.results'
```

## 🎓 最佳实践

1. **统一评测环境**: 确保所有模型在相同的设备、批次大小等条件下评测
2. **记录配置**: 保存每个模型的配置信息（包括 SWAA 参数）
3. **多次运行**: 对重要实验进行多次运行以减少方差
4. **分批评测**: 大规模评测时分批进行，避免内存溢出
5. **结果归档**: 将评测结果保存到带时间戳的目录中

## 🐛 常见问题

### Q1: 如何切换不同的注意力实现？

```bash
--model_args pretrained=model_path,attn_implementation=flash_attention_2
# 或
--model_args pretrained=model_path,attn_implementation=eager
```

### Q2: 如何评测本地模型？

```bash
# 使用绝对路径
--model_args pretrained=/home/user/models/my_model

# 或相对路径
--model_args pretrained=./models/my_model
```

### Q3: 如何使用不同的数据类型？

```bash
# bfloat16 (推荐)
--model_args pretrained=model_path,dtype=bfloat16

# float16
--model_args pretrained=model_path,dtype=float16

# float32 (精度最高但最慢)
--model_args pretrained=model_path,dtype=float32
```

---

**更多帮助**: 查看 `EVALUATION_GUIDE.md` 或运行 `python run_ruler_32k_comparison.py --help`
