# 评测系统文件说明

## 📁 文件结构

```
linearMem/
├── eval_swaa_model.py          # 自定义 lm-eval 模型包装器（支持 SWAA）
├── run_evaluation.py           # 评测启动脚本
├── analyze_results.py          # 结果分析脚本
├── test_evaluation.py          # 快速测试脚本
├── eval.sh                     # Bash 快速启动脚本
├── EVALUATION_GUIDE.md         # 详细使用指南
│
├── eval_configs/
│   └── comprehensive_eval.yaml # 评测任务配置文件
│
└── eval_results/               # 评测结果输出目录（自动创建）
    └── eval_YYYYMMDD_HHMMSS/
        ├── results.json              # 完整评测结果
        ├── results_table.txt         # 结果表格
        ├── evaluation_report.md      # 评测报告
        └── analysis/                 # 分析结果
            ├── analysis_report.md    # 详细分析报告
            ├── metrics_comparison.png # 指标对比图
            └── radar_chart.png       # 雷达图
```

## 🚀 快速开始（三步走）

### 第一步：测试系统
```bash
# 确保评测系统正常工作（约5分钟）
python test_evaluation.py
```

### 第二步：运行评测
```bash
# 选择一个预设方案
./eval.sh quick              # 快速评测（30分钟）
./eval.sh standard           # 标准评测（2-3小时）
./eval.sh comprehensive      # 全面评测（6-8小时）
```

### 第三步：分析结果
```bash
# 自动分析最新评测结果
python analyze_results.py --result-dir eval_results/eval_*
```

## 📊 评测预设说明

| 预设 | 时长 | 评测维度 | 适用场景 |
|------|------|----------|----------|
| `quick` | ~30分钟 | 常识推理、知识问答、真实性、数学、代码 | 快速验证模型能力 |
| `standard` | ~2-3小时 | 常识、知识、真实性、数学 | 标准多维度评测 |
| `comprehensive` | ~6-8小时 | 长文本+标准+阅读理解 | 全面能力评估 |
| `long_context_focus` | ~1-2小时 | 长文本、阅读理解 | 专注长文本能力 |

## 🎯 常用命令

### 基础评测
```bash
# 1. 快速评测
python run_evaluation.py --preset quick --device cuda:0

# 2. 自定义任务
python run_evaluation.py --tasks hellaswag,arc_easy,gsm8k --device cuda:0

# 3. 修改 SWAA 配置
python run_evaluation.py --preset standard \
    --sliding-window 4096 \
    --keep-first 8 \
    --device cuda:0
```

### 结果分析
```bash
# 分析单个结果
python analyze_results.py --result-dir eval_results/eval_20240101_120000

# 对比多个结果
python analyze_results.py --compare eval_results/eval_* --output comparison
```

### 查看报告
```bash
# 查看评测摘要
cat eval_results/eval_*/results_table.txt

# 查看详细报告
cat eval_results/eval_*/evaluation_report.md

# 查看分析报告
cat eval_results/eval_*/analysis/analysis_report.md
```

## 🔧 核心组件说明

### 1. eval_swaa_model.py
- **功能**: lm-eval 的自定义模型包装器
- **特性**: 支持 SWAA 配置、滑动窗口注意力、动态缓存
- **关键类**: `SWAAHFLM`

### 2. run_evaluation.py
- **功能**: 评测启动入口
- **支持**: 预设方案、自定义任务、灵活配置
- **输出**: JSON 结果 + Markdown 报告

### 3. analyze_results.py
- **功能**: 结果分析与可视化
- **输出**: 对比图表、雷达图、详细报告
- **支持**: 单次分析、多次对比

### 4. eval_configs/comprehensive_eval.yaml
- **功能**: 任务配置文件
- **包含**: 8大评测维度、4个预设方案
- **可扩展**: 支持自定义任务组

## 📈 评测维度详情

1. **长文本理解** (Long Context)
   - passkey, kv_retrieval, longdoc_qa, needlehaystack

2. **常识推理** (Commonsense)
   - hellaswag, piqa, siqa, winogrande, openbookqa

3. **知识问答** (Knowledge)
   - arc_easy, arc_challenge, triviaqa, nq_open, webqs

4. **真实性** (Truthfulness)
   - truthfulqa_mc1, truthfulqa_mc2, truthfulqa_gen

5. **数学推理** (Math)
   - gsm8k, math, asdiv

6. **代码能力** (Code)
   - humaneval, mbpp, multiple

7. **阅读理解** (Reading)
   - lambada, wikitext, pile_std

8. **语言建模** (Language Modeling)
   - wikitext, pile_arxiv, pile_github, pile_stackexchange

## 💡 使用建议

### 首次使用
1. 先运行 `python test_evaluation.py` 验证系统
2. 使用 `quick` 预设快速了解模型能力
3. 根据结果选择合适的预设进行深度评测

### 日常评测
- **开发阶段**: 使用 `quick` 预设快速迭代
- **模型对比**: 使用 `standard` 预设
- **论文实验**: 使用 `comprehensive` 预设

### 结果解读
- **Accuracy**: 准确率，0-1之间，越高越好
- **Perplexity**: 困惑度，越低越好
- **F1 Score**: F1分数，0-1之间，越高越好

## 🐛 故障排除

### 问题1: CUDA 内存不足
```bash
# 解决方案：减小 batch size
python run_evaluation.py --preset quick --batch-size 1
```

### 问题2: 评测中断
```bash
# 解决方案：使用缓存继续
python run_evaluation.py --preset quick --use-cache --cache-dir ./cache
```

### 问题3: 任务不存在
```bash
# 解决方案：列出所有可用任务
lm-eval ls tasks
```

## 📚 更多信息

详细使用指南请查看: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

## 🎓 学习路径

1. **入门** (30分钟)
   - 运行 `test_evaluation.py`
   - 运行 `./eval.sh quick`
   - 查看评测报告

2. **进阶** (2小时)
   - 尝试不同预设
   - 自定义任务组合
   - 分析结果对比

3. **高级** (半天)
   - 修改配置文件
   - 添加自定义任务
   - 集成到 CI/CD

## ✅ 检查清单

使用前确保：
- [x] 已安装 lm-eval: `pixi add "lm-eval[hf]"`
- [x] 已安装依赖: `pixi install`
- [x] CUDA 可用（或使用 CPU）
- [x] 磁盘空间充足（至少10GB）

开始评测：
- [ ] 运行测试脚本验证
- [ ] 选择合适的预设
- [ ] 检查评测结果
- [ ] 分析报告和图表

---

**Happy Evaluating! 🎉**

有问题请查看 [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) 或提交 Issue。
