# linearMem

[English version](README_EN.md)

本仓库是一个关于 decoder-only 语言模型长上下文适配的研究项目，重点关注训练自由的推理期改造。

项目研究的问题是：当冻结语言模型在 sliding-window attention 设置下推理时，窗口外的大部分 token 不再具有显式注意力路径。是否可以在不继续训练模型参数的前提下，为模型增加一条低成本的全局记忆通道，使窗口外信息仍然能影响下一 token 的预测？

## 研究动机

Sliding-window attention 可以显著降低长上下文推理成本，同时保留局部窗口内原始 softmax attention 的建模方式。但它的限制也很直接：除少量保留的 sink tokens 外，窗口外 token 基本失去了被当前 token 精确访问的路径。

本项目的工作假设是：可以利用模型原有的 query、key、value 投影构造一条 recurrent linear-memory 分支，将历史上下文压缩为一个全局统计状态。该分支不替代原始 softmax attention，而是作为 sliding-window attention 的辅助信号。

## 方法概述

当前实现将每个 attention layer 视为两条并行路径：

1. 局部分支：使用 sliding-window FlashAttention。
2. 全局分支：由变换后的 keys 和 values 累积 recurrent state。

全局分支使用当前 query 从 recurrent state 中读取信息，并在输出投影之前将该信号融合回 attention output。实现中支持多种训练自由 feature map、多种融合方式，以及只在部分层注入 linear memory 的设置。

当前代码主要面向 Hugging Face Transformers，同时包含 vLLM 和定制 FlashAttention kernel 的实验性支持。

## 当前观察

项目仍处于探索阶段。目前最稳定的观察是：在 sliding-window 评测下，该方法能够改善窗口外 token 的语言建模表现，尤其是在距离当前窗口较远的位置。

但这些收益不应被理解为已经解决长程精确检索。当前实验中，perplexity 的改善并不能稳定转化到 RULER 或长文档 QA 等检索密集型任务上。因此，这条额外分支更适合被理解为一种全局统计先验，而不是精确记忆机制。

## 仓库结构

```text
swaa_patch/             Transformers、KV cache state 与 linear memory 的核心 patch
flash-attention-SWAA/   SWAA 实验使用的定制 FlashAttention kernel
eval/                   RULER 与 lm-eval 相关评测脚本
qkv_analysis/           Query / key / value 统计分析脚本
tuning_log/             实现、调参和性能优化记录
METHOD.md               方法笔记
paper_draft_sequence_mixed_attention.md
test.py                 最小推理测试脚本
pixi.toml               可复现开发环境
```

## 快速开始

本项目使用 Pixi 管理环境：

```bash
pixi install
pixi run verify
```

最小推理测试：

```bash
pixi run python test.py
```

评测脚本位于 `eval/`。例如：

```bash
cd eval/scripts
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768
```

部分实验需要使用 `flash-attention-SWAA/` 中的定制 FlashAttention kernel。

## 项目状态

这是一个研究代码库，而不是稳定的软件库。部分接口仍可能变化，且若干组件依赖对模型内部实现的 monkey patch。本仓库主要用于方法开发、受控实验，以及分析训练自由 linear memory 机制在长上下文建模中的作用。
