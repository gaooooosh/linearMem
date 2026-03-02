# 可用评测任务列表

本文档列出了 lm-evaluation-harness 中实际可用的评测任务。

## 📋 任务可用性检查结果

### ✅ 可用任务

#### 长文本理解 (Long Context)
- ✓ `passkey` - 密钥检索任务

#### 常识推理 (Commonsense Reasoning)
- ✓ `hellaswag` - 常识推理补全
- ✓ `piqa` - 物理常识推理
- ✓ `winogrande` - 代词消歧
- ✓ `openbookqa` - 开放书籍问答

#### 知识问答 (Knowledge QA)
- ✓ `arc_easy` - ARC 简单题
- ✓ `arc_challenge` - ARC 挑战题
- ✓ `triviaqa` - 琐碎问答
- ✓ `nq_open` - 自然问题
- ✓ `webqs` - Web问题

#### 真实性与幻觉 (Truthfulness)
- ✓ `truthfulqa_mc1` - 真实性问答 (多选题1)
- ✓ `truthfulqa_mc2` - 真实性问答 (多选题2)
- ✓ `truthfulqa_gen` - 真实性问答 (生成式)

#### 数学推理 (Mathematical Reasoning)
- ✓ `gsm8k` - 小学数学应用题
- ✓ `asdiv` - 算术技能

#### 代码能力 (Code Generation)
- ✓ `humaneval` - HumanEval 代码生成
- ✓ `mbpp` - Mostly Basic Python Problems

#### 阅读理解 (Reading Comprehension)
- ✓ `lambada` - LAMBADA 词义预测
- ✓ `lambada_openai` - LAMBADA OpenAI版本
- ✓ `wikitext` - WikiText 困惑度

#### 语言建模 (Language Modeling)
- ✓ `pile_arxiv` - arXiv 论文
- ✓ `pile_github` - GitHub 代码
- ✓ `pile_stackexchange` - StackExchange 问答

### ❌ 不可用任务 (已从配置中移除)

以下任务在 lm-eval 标准库中不存在:

- ✗ `siqa` - 社会常识推理 (Social IQA)
- ✗ `math` - MATH 数据集
- ✗ `multiple` - 多语言代码生成
- ✗ `pile_std` - Pile标准数据集
- ✗ `kv_retrieval` - 键值检索
- ✗ `longdoc_qa` - 长文档问答
- ✗ `needlehaystack` - 海底捞针

## 🎯 推荐的评测方式

### 1. 使用预设方案 (推荐)

```bash
# 快速评测 (5个核心任务,约30分钟)
python eval/scripts/run_evaluation.py --preset quick --device cuda:0

# 标准评测 (4个维度,约2-3小时)
python eval/scripts/run_evaluation.py --preset standard --device cuda:0

# 全面评测 (6个维度,约6-8小时)
python eval/scripts/run_evaluation.py --preset comprehensive --device cuda:0

# 长文本专项评测
python eval/scripts/run_evaluation.py --preset long_context_focus --device cuda:0
```

### 2. 使用具体任务列表

```bash
# 指定具体任务
python eval/scripts/run_evaluation.py \
    --tasks hellaswag,piqa,winogrande,arc_easy \
    --device cuda:0
```

### 3. 长文本评测 (RULER)

对于需要长文本能力的评测,推荐使用专门的 RULER 脚本:

```bash
# RULER 小规模测试
python eval/scripts/run_ruler_test.py --size mini --device cuda:0

# RULER 32K 长文本评测
python eval/scripts/run_ruler_32k.py --device cuda:0

# 多模型对比
python eval/scripts/run_ruler_32k_comparison.py \
    --config eval/configs/models_comparison_config.yaml \
    --device cuda:0
```

## 📊 可用的 RULER 任务

RULER 是专门的长文本评测基准,包含以下任务:

- `niah_single_1/2/3` - 单针大海捞针
- `niah_multikey` - 多键检索
- `niah_multivalue` - 多值检索
- `niah_multiquery` - 多查询
- `passkey` - 密钥检索
- `ruler_vt` - 变量跟踪
- `ruler_cwe` - 代码词提取
- `ruler_fwe` - 频繁词提取

## 🔍 如何查看所有可用任务

```bash
# 列出所有可用任务
python -c "from lm_eval.tasks import TaskManager; tm = TaskManager(); print('\n'.join(sorted(tm.all_tasks)))"
```

## 📝 注意事项

1. **任务组 vs 任务**: `--tasks` 参数接受具体任务名,不接受任务组名
2. **预设方案**: 使用 `--preset` 会自动展开任务组
3. **RULER 任务**: 需要使用 `run_ruler_*.py` 脚本,不在标准评测中
4. **自定义任务**: 可以通过 `--tasks` 指定任何 lm-eval 支持的任务
