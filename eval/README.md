# SWAA 模型评测系统

本目录包含所有与 SWAA 模型评测相关的代码、配置和文档。

## 📁 目录结构

```
eval/
├── scripts/          # 评测脚本
│   ├── eval_swaa_model.py           # SWAA 自定义模型封装
│   ├── run_evaluation.py            # 通用评测脚本
│   ├── run_ruler_test.py            # RULER 小规模测试
│   ├── run_ruler_32k.py             # RULER 32K 评测
│   ├── run_ruler_32k_comparison.py  # 多模型对比评测
│   ├── analyze_results.py           # 结果分析工具
│   └── test_evaluation.py           # 评测系统测试
│
├── configs/          # 配置文件
│   ├── comprehensive_eval.yaml      # 综合评测配置
│   └── models_comparison_config.yaml # 多模型对比配置
│
├── docs/             # 文档
│   ├── EVAL_README.md               # 快速开始指南
│   ├── EVALUATION_GUIDE.md          # 详细使用文档
│   ├── MULTI_MODEL_COMPARISON_GUIDE.md # 多模型对比指南
│   ├── RULER_EVAL_STATUS.md         # RULER 评测状态
│   └── RULER_32K_TEST_STATUS.md     # 32K 测试状态
│
└── results/          # 评测结果输出目录
```

## 🚀 快速开始

### 1. 运行综合评测

```bash
# 快速测试（5个任务，每个10个样本）
python eval/scripts/run_evaluation.py --preset quick

# 标准评测（8个维度，每个100个样本）
python eval/scripts/run_evaluation.py --preset standard

# 完整评测（所有任务）
python eval/scripts/run_evaluation.py --preset comprehensive
```

### 2. RULER 长文本评测

```bash
# 小规模测试
python eval/scripts/run_ruler_test.py --test-size small

# 32K 长文本评测（100个样本/任务）
python eval/scripts/run_ruler_32k.py --device cuda:0

# 多模型对比评测
python eval/scripts/run_ruler_32k_comparison.py \
    --config eval/configs/models_comparison_config.yaml \
    --device cuda:0
```

### 3. 多模型对比

#### 方法 1：使用配置文件（推荐）

```bash
# 1. 编辑配置文件
vim eval/configs/models_comparison_config.yaml

# 2. 运行对比评测
python eval/scripts/run_ruler_32k_comparison.py \
    --config eval/configs/models_comparison_config.yaml \
    --device cuda:0 \
    --limit 100
```

#### 方法 2：命令行直接指定

```bash
# 单个模型
python eval/scripts/run_ruler_32k_comparison.py \
    --model Qwen/Qwen3-1.7B \
    --name "Qwen3-Base" \
    --device cuda:0

# 多个模型
python eval/scripts/run_ruler_32k_comparison.py \
    --models model1 model2 model3 \
    --device cuda:0
```

## 📊 评测维度

### 1. 长文本能力 (RULER)
- **NIAH (Needle In A Haystack)**: 海底捞针检索
- **Passkey**: 密钥检索
- **Variable Tracking**: 变量跟踪
- **Word Extraction**: 词汇提取

### 2. 常识推理
- **PIQA**: 物理常识推理
- **SIQA**: 社会常识推理
- **BoolQ**: 布尔问题回答

### 3. 知识问答
- **SciQ**: 科学问题
- **OpenbookQA**: 开放书籍问答
- **MMLU**: 多任务语言理解

### 4. 阅读理解
- **LAMBADA**: 词义预测
- **HellaSwag**: 常识推理补全
- **WinoGrande**: 代词消歧

## 🔧 SWAA 配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--sliding-window` | 滑动窗口大小 | 2048 |
| `--keep-first` | 保留前 N 个 token | 4 |
| `--force-fa-decode` | 强制全注意力解码 | False |
| `--attn` | 注意力实现 | flash_attention_2 |
| `--dtype` | 数据类型 | bfloat16 |

## 📝 示例配置

### SWAA 模型配置 (YAML)

```yaml
models:
  - name: "Qwen3-SWAA-2048"
    path: "Qwen/Qwen3-1.7B"
    swaa_window: 2048
    keep_first: 4

  - name: "Qwen3-SWAA-4096"
    path: "Qwen/Qwen3-1.7B"
    swaa_window: 4096
    keep_first: 8
```

### lm-eval 命令行参数

```bash
# 基础模型
--model hf \
--model_args pretrained=Qwen/Qwen3-1.7B

# SWAA 模型
--model swaa_hf \
--model_args pretrained=Qwen/Qwen3-1.7B,sliding_window_size=2048,keep_first=4
```

## 📈 结果分析

```bash
# 查看最新评测结果
python eval/scripts/analyze_results.py --result-dir eval/results/latest

# 生成可视化报告
python eval/scripts/analyze_results.py --visualize
```

## 📖 详细文档

- **快速开始**: [docs/EVAL_README.md](docs/EVAL_README.md)
- **完整指南**: [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md)
- **多模型对比**: [docs/MULTI_MODEL_COMPARISON_GUIDE.md](docs/MULTI_MODEL_COMPARISON_GUIDE.md)

## ⚙️ 依赖安装

```bash
# 使用 pixi 安装依赖
pixi add --pypi "lm_eval[hf]"
pixi add --pypi wonderwords nltk
pixi install
```

## 🐛 常见问题

### Q1: 如何切换模型？

使用 `--model_args pretrained=模型路径` 参数：

```bash
--model_args pretrained=path/to/your/model
```

### Q2: 如何修改滑动窗口大小？

```bash
--model swaa_hf \
--model_args pretrained=Qwen/Qwen3-1.7B,sliding_window_size=4096,keep_first=8
```

### Q3: 如何在不同 GPU 上运行？

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python eval/scripts/run_ruler_32k.py --device cuda:0

# GPU 1
CUDA_VISIBLE_DEVICES=1 python eval/scripts/run_ruler_32k.py --device cuda:0
```

## 📊 评测输出

评测结果保存在 `eval/results/` 目录下：

```
eval/results/
├── ruler_32k_20260302_132117/
│   ├── results.json          # 完整结果
│   ├── results_table.txt     # 结果表格
│   └── ruler_32k_report.md   # 详细报告
│
└── comparison_20260302_140000/
    ├── model_comparison.json  # 对比数据
    └── comparison_report.md   # 对比报告
```

## 🔗 相关资源

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [RULER Benchmark](https://github.com/hsiehjackson/RULER)
- [SWAA Paper](https://arxiv.org/abs/xxxx)

---

**最后更新**: 2026-03-02
