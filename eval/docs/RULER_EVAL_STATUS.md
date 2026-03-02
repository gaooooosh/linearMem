# RULER 测试运行总结

## ✅ 已完成的工作

### 1. 系统准备
- ✅ 安装了 `lm-eval[hf]` 依赖
- ✅ 安装了 RULER 所需的额外依赖：`wonderwords` 和 `nltk`
- ✅ 创建了自定义的 SWAA 模型包装器 (`eval_swaa_model.py`)
- ✅ 修复了模型参数传递的问题（device 和 batch_size 参数冲突）

### 2. 脚本创建
- ✅ 创建了 `run_ruler_test.py` - 专门用于 RULER 评测的脚本
- ✅ 支持 4 种测试规模：mini (2 tasks), small (4 tasks), medium (7 tasks), full (12 tasks)
- ✅ 集成了 SWAA 配置支持
- ✅ 自动生成评测报告

### 3. 测试配置
- **测试规模**: mini (niah_single_1 + passkey)
- **样本数量**: 3 个样本（小规模测试）
- **设备**: cuda:7
- **SWAA 配置**:
  - sliding_window_size: 2048
  - keep_first: 4
  - force_fa_decode: False

## 📊 RULER 任务说明

RULER (Real-World Long-Context Evaluation Benchmark) 包含以下任务：

### Needle In A Haystack (NIAH)
- **niah_single_1/2/3**: 在长文本中找到单个"针"
- **niah_multikey**: 检索多个键
- **niah_multivalue**: 检索多个值
- **niah_multiquery**: 多个查询

### 其他任务
- **passkey**: 密钥检索任务
- **ruler_vt**: 变量跟踪
- **ruler_cwe**: 代码词提取
- **ruler_fwe**: 频繁词提取
- **ruler_qa_hotpot**: HotpotQA 长文本问答
- **ruler_qa_squad**: SQuAD 长文本问答

## 🔧 使用方法

### 基础命令
```bash
# Mini 测试 (2 tasks, 最快)
pixi run python run_ruler_test.py --size mini --device cuda:7 --limit 5

# Small 测试 (4 tasks)
pixi run python run_ruler_test.py --size small --device cuda:7 --limit 5

# Medium 测试 (7 tasks)
pixi run python run_ruler_test.py --size medium --device cuda:7 --limit 10

# Full 测试 (12 tasks, 完整评测)
pixi run python run_ruler_test.py --size full --device cuda:7
```

### 自定义 SWAA 配置
```bash
# 使用更大的滑动窗口
pixi run python run_ruler_test.py --size mini --device cuda:7 \\
    --sliding-window 4096 --keep-first 8
```

### 查看结果
```bash
# 查看最新的评测结果
cat eval_results/ruler_*/ruler_report.md

# 查看结果表格
cat eval_results/ruler_*/results_table.txt

# 查看完整 JSON 结果
cat eval_results/ruler_*/results.json
```

## ⚠️ 已知问题

### 1. 模型参数传递
- **问题**: lm-eval 的 `simple_evaluate` 函数会自动传递 `device` 和 `batch_size` 参数
- **解决**: 在 `model_args` 字符串中不包含这两个参数

### 2. RULER 依赖
- **问题**: RULER 任务需要额外的包
- **解决**: 已安装 `wonderwords` 和 `nltk`

### 3. Triton 版本警告
- **警告**: `Current Triton version 3.1.0 is below the recommended 3.2.0`
- **影响**: 不影响功能，仅为警告信息

## 📁 输出文件

评测完成后会生成以下文件：

```
eval_results/ruler_{size}_{timestamp}/
├── results.json          # 完整的评测结果（JSON格式）
├── results_table.txt     # 结果表格（文本格式）
└── ruler_report.md       # 详细的评测报告
```

## 🎯 下一步建议

### 1. 完整运行测试
```bash
# 运行 mini 测试（约5-10分钟）
pixi run python run_ruler_test.py --size mini --device cuda:7 --limit 10
```

### 2. 逐步扩大规模
```bash
# 如果 mini 测试成功，运行 small 测试
pixi run python run_ruler_test.py --size small --device cuda:7 --limit 10

# 然后 medium 测试
pixi run python run_ruler_test.py --size medium --device cuda:7 --limit 10
```

### 3. 完整评测
```bash
# 最后运行完整的 RULER 评测（可能需要数小时）
pixi run python run_ruler_test.py --size full --device cuda:7
```

### 4. 结果分析
```bash
# 使用分析脚本生成可视化报告
python analyze_results.py --result-dir eval_results/ruler_mini_*
```

## 💡 优化建议

### 提高评测速度
1. **增加 batch size**（如果内存允许）:
   ```bash
   pixi run python run_ruler_test.py --size mini --batch-size 4 --device cuda:7
   ```

2. **并行评测**（在多个 GPU 上）:
   ```bash
   # GPU 0 运行一部分任务
   CUDA_VISIBLE_DEVICES=0 python run_ruler_test.py --tasks niah_single_1 &

   # GPU 1 运行另一部分任务
   CUDA_VISIBLE_DEVICES=1 python run_ruler_test.py --tasks passkey &

   wait
   ```

### 调试模式
如果遇到问题，可以添加 `--limit 1` 只测试1个样本：
```bash
pixi run python run_ruler_test.py --size mini --device cuda:7 --limit 1
```

## 📊 预期结果

RULER 任务的主要指标是 **准确率 (Accuracy)**，范围 0-1，越高越好。

典型的长文本模型在 RULER 上的表现：
- **NIAH 任务**: 0.7-0.9（好的模型）
- **Passkey 任务**: 0.8-0.95（好的模型）
- **QA 任务**: 0.5-0.7（取决于数据集）

## 🔗 相关资源

- [RULER 论文](https://arxiv.org/abs/2409.17791)
- [lm-evaluation-harness 文档](https://github.com/EleutherAI/lm-evaluation-harness)
- [项目评测指南](EVALUATION_GUIDE.md)
- [快速开始指南](EVAL_README.md)

---

**状态**: ✅ 系统已就绪，可以开始 RULER 评测

**最后更新**: 2026-03-02
