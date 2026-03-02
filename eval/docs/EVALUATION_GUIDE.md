# SWAA Model Evaluation Guide

本项目提供了一套完整的多维度评测系统，用于评估带有 SWAA (Sliding Window Attention Adaptation) 的语言模型。

## 📋 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [评测维度](#评测维度)
- [使用方法](#使用方法)
- [配置说明](#配置说明)
- [结果分析](#结果分析)
- [常见问题](#常见问题)

## ✨ 功能特性

- ✅ **多维度评测**: 覆盖长文本、常识推理、知识问答、真实性、数学推理、代码能力等
- ✅ **SWAA 支持**: 原生支持滑动窗口注意力配置
- ✅ **灵活配置**: 支持预设方案和自定义任务组合
- ✅ **自动化报告**: 生成详细的评测报告和可视化图表
- ✅ **结果对比**: 支持多次实验结果的横向对比

## 🚀 快速开始

### 1. 基础评测（推荐首次使用）

```bash
# 使用快速评测方案（约30分钟）
./eval.sh quick

# 或直接使用Python
python run_evaluation.py --preset quick --device cuda:0
```

### 2. 标准评测

```bash
# 标准多维度评测（约2-3小时）
./eval.sh standard
```

### 3. 全面评测

```bash
# 全面能力评测（约6-8小时）
./eval.sh comprehensive
```

### 4. 长文本专项评测

```bash
# 专注长文本能力（约1-2小时）
./eval.sh long_context_focus
```

## 📊 评测维度

### 1. 长文本理解能力 (Long Context)
- **passkey**: Passkey 检索任务
- **kv_retrieval**: 键值检索
- **longdoc_qa**: 长文档问答
- **needlehaystack**: 大海捞针任务

### 2. 常识推理 (Commonsense Reasoning)
- **hellaswag**: 常识推理
- **piqa**: 物理常识
- **siqa**: 社会常识
- **winogrande**: 代词消歧
- **openbookqa**: 开放书籍问答

### 3. 知识问答 (Knowledge QA)
- **arc_easy/challenge**: AI2 推理挑战
- **triviaqa**: 问答知识
- **nq_open**: 自然问题
- **webqs**: Web 问题

### 4. 真实性 (Truthfulness)
- **truthfulqa_mc1/mc2/gen**: 真实性问答

### 5. 数学推理 (Mathematical Reasoning)
- **gsm8k**: 小学数学
- **math**: 竞赛数学
- **asdiv**: 算术技能

### 6. 代码能力 (Code Generation)
- **humaneval**: 代码补全
- **mbpp**: Python 编程
- **multiple**: 多语言代码生成

### 7. 阅读理解 (Reading Comprehension)
- **lambada**: 词预测
- **wikitext**: 维基文本
- **pile_std**: Pile 数据集

### 8. 语言建模 (Language Modeling)
- **wikitext**: 困惑度评估
- **pile_***: 多领域困惑度

## 🎯 使用方法

### 命令行参数

```bash
python run_evaluation.py [OPTIONS]

必需参数 (二选一):
  --preset PRESET          使用预设评测方案
                           [quick|standard|comprehensive|long_context_focus]
  --tasks TASKS            自定义任务列表 (逗号分隔)

模型配置:
  --model MODEL            模型名称或路径 (默认: Qwen/Qwen3-1.7B)
  --device DEVICE          设备 (默认: cuda:0)
  --dtype DTYPE            数据类型 [bfloat16|float16|float32]
  --batch-size N           批次大小 (默认: 1)

SWAA 配置:
  --sliding-window N       滑动窗口大小
  --keep-first N           保留的前N个token

其他:
  --output-dir DIR         输出目录 (默认: eval_results)
  --num-fewshot N          Few-shot 示例数量
```

### 示例命令

```bash
# 1. 快速评测
python run_evaluation.py --preset quick --device cuda:0

# 2. 自定义任务评测
python run_evaluation.py --tasks hellaswag,arc_easy,gsm8k --device cuda:0

# 3. 自定义 SWAA 配置
python run_evaluation.py --preset standard \
    --sliding-window 4096 \
    --keep-first 8 \
    --device cuda:0

# 4. 指定输出目录
python run_evaluation.py --preset comprehensive \
    --output-dir my_eval_results \
    --device cuda:0

# 5. 使用不同模型
python run_evaluation.py --preset quick \
    --model path/to/your/model \
    --device cuda:0
```

## ⚙️ 配置说明

### 预设方案配置

配置文件位于: `eval_configs/comprehensive_eval.yaml`

```yaml
presets:
  quick:
    description: "快速评测核心能力"
    tasks:
      - hellaswag
      - arc_easy
      - truthfulqa_mc1
      - gsm8k
      - humaneval

  standard:
    description: "标准多维度评测"
    groups:
      - commonsense
      - knowledge
      - truthfulness
      - math
```

### 自定义任务组

可以在配置文件中定义新的任务组：

```yaml
task_groups:
  my_custom_group:
    description: "我的自定义评测组"
    tasks:
      - task1
      - task2
      - task3
    num_fewshot: 0
```

### SWAA 模型配置

```yaml
model_defaults:
  swaa:
    sliding_window_size: 2048  # 滑动窗口大小
    keep_first: 4              # 保留的前N个token
    force_fa_decode: false     # 是否强制全注意力解码
    non_sliding_layers: []     # 非滑动层列表
```

## 📈 结果分析

### 自动分析

评测完成后，运行分析脚本：

```bash
# 分析单个评测结果
python analyze_results.py --result-dir eval_results/eval_20240101_120000

# 对比多次评测结果
python analyze_results.py --compare eval_results/eval_* --output comparison
```

### 输出文件

评测结果目录结构：

```
eval_results/eval_YYYYMMDD_HHMMSS/
├── results.json              # 完整评测结果 (JSON)
├── results_table.txt         # 结果表格 (文本)
├── evaluation_report.md      # 评测报告 (Markdown)
└── analysis/                 # 分析结果
    ├── analysis_report.md    # 详细分析报告
    ├── metrics_comparison.png # 指标对比图
    └── radar_chart.png       # 雷达图
```

### 查看结果

```bash
# 查看摘要表格
cat eval_results/eval_*/results_table.txt

# 查看详细报告
cat eval_results/eval_*/evaluation_report.md

# 查看分析报告
cat eval_results/eval_*/analysis/analysis_report.md
```

### 可视化图表

- **metrics_comparison.png**: 各任务准确率柱状图
- **radar_chart.png**: 多维度能力雷达图
- **comparison.png**: 多次实验对比图（使用对比功能时）

## 🔧 高级用法

### 1. 使用自定义模型包装器

```python
from eval_swaa_model import SWAAHFLM

# 创建模型实例
model = SWAAHFLM(
    pretrained="Qwen/Qwen3-1.7B",
    device="cuda:0",
    sliding_window_size=2048,
    keep_first=4,
)

# 使用 lm-eval 进行评测
from lm_eval import evaluator
results = evaluator.simple_evaluate(
    model=model,
    tasks=["hellaswag", "arc_easy"],
)
```

### 2. 批量评测

```bash
# 创建批量评测脚本
for preset in quick standard comprehensive; do
    python run_evaluation.py --preset $preset --device cuda:0
done
```

### 3. 分布式评测

```bash
# 在不同GPU上运行不同任务
CUDA_VISIBLE_DEVICES=0 python run_evaluation.py --tasks hellaswag,arc_easy &
CUDA_VISIBLE_DEVICES=1 python run_evaluation.py --tasks gsm8k,truthfulqa &
wait
```

## 🐛 常见问题

### Q1: 评测中断后如何继续？

A: lm-eval 支持缓存机制，可以使用 `--use-cache` 参数：

```bash
python run_evaluation.py --preset quick --use-cache --cache-dir ./cache
```

### Q2: 内存不足怎么办？

A: 尝试以下方法：
1. 减小 batch size: `--batch-size 1`
2. 使用更小的模型精度: `--dtype float16`
3. 使用梯度检查点（需修改模型代码）

### Q3: 如何添加新的评测任务？

A: 参考 lm-evaluation-harness 文档添加自定义任务：
https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/task_guide.md

### Q4: 评测速度太慢？

A: 可以：
1. 使用 `quick` 预设
2. 减少评测任务数量
3. 使用 vLLM 或其他加速推理引擎（需修改模型包装器）

### Q5: 如何解释评测结果？

A: 关键指标：
- **Accuracy**: 准确率，越高越好
- **Perplexity**: 困惑度，越低越好
- **F1 Score**: F1 分数，越高越好

## 📚 参考资料

- [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [SWAA 论文](https://arxiv.org/abs/xxxx.xxxxx)
- [Qwen 模型文档](https://github.com/QwenLM/Qwen)

## 📝 更新日志

- **2024-01-01**: 初始版本发布
  - 支持多维度评测
  - SWAA 配置集成
  - 自动化报告生成

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License
