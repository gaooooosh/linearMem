# Linear Memory 对比评测使用指南

## 概述

本评测系统用于对比 **Linear Memory** 开启和关闭时的 RULER 长上下文评测性能差异。

## 核心特性

- ✅ **自动对比评测**：自动运行开启/关闭 Linear Mem 的两次评测
- ✅ **详细对比报告**：生成包含性能差异、改进百分比的综合报告
- ✅ **灵活配置**：支持配置文件和命令行参数配置
- ✅ **可调参数**：
  - 上下文长度（默认 32K）
  - 滑动窗口大小（默认 4K）
  - 测试样本数量（默认 100）

## 快速开始

### 方式 1：使用默认配置（推荐）

```bash
# 使用默认配置运行（32K 上下文，4K 滑动窗口，100 样本）
python eval/scripts/compare_linear_mem.py

# 或者使用快捷脚本
./eval/scripts/run_linear_mem_comparison.sh
```

### 方式 2：自定义参数

```bash
# 自定义模型和设备
python eval/scripts/compare_linear_mem.py \
  --model Qwen/Qwen3-1.7B \
  --device cuda:7

# 自定义上下文长度和滑动窗口
python eval/scripts/compare_linear_mem.py \
  --context-length 32768 \
  --sliding-window 4096

# 自定义样本数量（快速测试）
python eval/scripts/compare_linear_mem.py \
  --num-samples 10

# 完整自定义
python eval/scripts/compare_linear_mem.py \
  --model Qwen/Qwen3-1.7B \
  --device cuda:7 \
  --context-length 32768 \
  --sliding-window 4096 \
  --num-samples 100 \
  --output-dir eval/results/my_comparison
```

### 方式 3：使用环境变量（Shell 脚本）

```bash
# 设置环境变量
export MODEL=Qwen/Qwen3-1.7B
export DEVICE=cuda:7
export CONTEXT_LENGTH=32768
export SLIDING_WINDOW=4096
export NUM_SAMPLES=100

# 运行脚本
./eval/scripts/run_linear_mem_comparison.sh
```

## 配置文件说明

配置文件位于：`eval/configs/linear_mem_comparison.yaml`

### 主要配置项

```yaml
# 模型配置
model:
  name: "Qwen/Qwen3-1.7B"
  device: "cuda:7"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  batch_size: 1

# 评测配置
evaluation:
  context_length: 32768  # 上下文长度（32K）
  sliding_window: 4096   # 滑动窗口大小（4K）
  keep_first: 4
  num_samples: 100       # 每个任务的样本数量

  # RULER 任务列表
  tasks:
    - "niah_single_1"
    - "niah_single_2"
    - "niah_single_3"
    - "niah_multikey_1"
    - "niah_multivalue"
    - "niah_multiquery"
    - "passkey"
    - "ruler_vt"
    - "ruler_cwe"
    - "ruler_fwe"
```

### 自定义配置

1. **复制配置文件**：
   ```bash
   cp eval/configs/linear_mem_comparison.yaml eval/configs/my_comparison.yaml
   ```

2. **编辑配置**：
   ```yaml
   evaluation:
     context_length: 16384  # 改为 16K
     sliding_window: 2048   # 改为 2K
     num_samples: 50        # 改为 50 个样本
   ```

3. **使用自定义配置运行**：
   ```bash
   python eval/scripts/compare_linear_mem.py --config eval/configs/my_comparison.yaml
   ```

## 评测流程

评测脚本会依次执行以下三个阶段：

### Phase 1: Linear Mem ENABLED
- 使用 `enable_linear_mem=True` 运行 RULER 评测
- 保存结果到 `results_linear_mem_enabled.json`

### Phase 2: Linear Mem DISABLED
- 使用 `enable_linear_mem=False` 运行 RULER 评测
- 保存结果到 `results_linear_mem_disabled.json`

### Phase 3: 对比分析
- 对比两次评测的结果
- 计算每个任务的性能差异
- 生成对比报告

## 输出结果

评测完成后，结果保存在 `eval/results/linear_mem_comparison_<timestamp>/` 目录：

```
linear_mem_comparison_20240302_143025/
├── config.yaml                           # 使用的配置
├── results_linear_mem_enabled.json       # Linear Mem 开启的完整结果
├── results_linear_mem_disabled.json      # Linear Mem 关闭的完整结果
├── results_table_linear_mem_enabled.txt  # 开启结果的表格
├── results_table_linear_mem_disabled.txt # 关闭结果的表格
├── comparison_results.json               # 对比数据（JSON 格式）
└── linear_mem_comparison_report.md       # 📊 详细对比报告（推荐查看）
```

### 查看报告

```bash
# 查看详细对比报告
cat eval/results/linear_mem_comparison_*/linear_mem_comparison_report.md

# 或者使用 Markdown 查看器
code eval/results/linear_mem_comparison_*/linear_mem_comparison_report.md
```

## 报告内容

对比报告包含：

1. **概览**：
   - 平均准确率改进百分比
   - 改进的任务数量 vs 退化的任务数量
   - 总体结论（推荐开启/关闭）

2. **详细结果表格**：
   - 每个任务的性能对比
   - 启用/禁用的准确率
   - 差异和改进百分比

3. **任务分类分析**：
   - NIAH 任务的平均改进
   - 其他任务的平均改进

4. **建议**：
   - 基于结果的配置建议

## 参数调优建议

### 快速测试（开发调试）
```bash
--num-samples 10
```

### 标准评测
```bash
--num-samples 100
```

### 不同上下文长度测试
```bash
# 16K 上下文
--context-length 16384 --sliding-window 2048

# 32K 上下文（默认）
--context-length 32768 --sliding-window 4096

# 64K 上下文
--context-length 65536 --sliding-window 8192
```

### 不同滑动窗口测试
```bash
# 小窗口
--sliding-window 2048

# 中等窗口（默认）
--sliding-window 4096

# 大窗口
--sliding-window 8192
```

## 常见问题

### Q: 如何只运行部分任务？

A: 编辑配置文件中的 `evaluation.tasks` 列表，只保留需要的任务。

### Q: 评测过程中内存不足怎么办？

A: 尝试以下方法：
1. 减小 `batch_size`（在配置文件中）
2. 减小 `num_samples`
3. 使用更小的 `context_length`

### Q: 如何使用不同的模型？

A: 使用 `--model` 参数：
```bash
python eval/scripts/compare_linear_mem.py --model meta-llama/Llama-2-7b-hf
```

### Q: 如何解释对比结果？

A: 查看生成的 `linear_mem_comparison_report.md` 文件，其中包含：
- 平均改进百分比（正值表示 Linear Mem 有帮助）
- 每个任务的详细对比
- 基于数据的配置建议

## 技术细节

### Linear Mem 实现位置
- **核心实现**：`swaa_patch/hack_hf_swaa.py`
- **配置传递**：通过 `SWAAConfig` 的 `enable_linear_mem` 参数
- **模型包装器**：`eval/scripts/eval_swaa_model.py`

### 评测框架
- **基础框架**：lm-evaluation-harness
- **任务类型**：RULER 长上下文评测
- **指标**：准确率（accuracy）

## 示例输出

```
==========================================
Comparing Results: Linear Mem Enabled vs Disabled
==========================================

| Task | Metric | Enabled | Disabled | Difference | Improvement |
|------|--------|---------|----------|------------|-------------|
| niah_single_1 | acc | 0.9200 | 0.8800 | +0.0400 | +4.55% |
| niah_single_2 | acc | 0.9100 | 0.8700 | +0.0400 | +4.60% |
| passkey | acc | 0.9500 | 0.9300 | +0.0200 | +2.15% |
...

==========================================
Summary Statistics
==========================================

📊 Average Accuracy Improvement: +3.45%
✅ Linear Mem shows SIGNIFICANT IMPROVEMENT
```

## 联系与支持

如有问题或建议，请查看项目文档或提交 Issue。
