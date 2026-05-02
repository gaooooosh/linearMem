# linearMem

[中文版](README.md)

This repository contains an ongoing research project on training-free long-context adaptation for decoder-only language models.

The project studies whether a frozen language model evaluated with sliding-window attention can benefit from an additional low-cost global memory path. The goal is not to extend the model by further training, but to modify the inference-time attention computation so that information outside the local window can still contribute to next-token prediction.

## Motivation

Sliding-window attention reduces the cost of long-context inference while preserving the original softmax attention mechanism inside a local neighborhood. However, it also removes the explicit attention path to most tokens outside the active window, except for any preserved sink tokens.

The working hypothesis of this project is that a recurrent linear-memory branch, constructed from the model's existing query, key, and value projections, can provide a compressed global statistical state. This branch is used as an auxiliary signal for the original sliding-window attention rather than as a replacement for softmax attention.

## Method Overview

The implemented method treats each attention layer as two parallel paths:

1. A local branch using sliding-window FlashAttention.
2. A global branch that accumulates a recurrent state from transformed keys and values.

The global branch reads from the recurrent state using the current query, and the resulting signal is fused into the attention output before the output projection. The implementation supports several training-free feature maps, multiple fusion rules, and layer-selective injection.

The current implementation focuses on Hugging Face Transformers, with experimental support for vLLM and customized FlashAttention kernels.

## Current Findings

The project remains exploratory. The most stable observation so far is an improvement in exterior-token language modeling under sliding-window evaluation, especially for positions far outside the active window.

These gains should not be interpreted as solved long-range retrieval. In current experiments, perplexity improvements do not reliably transfer to retrieval-heavy benchmarks such as RULER or long-document QA tasks. The added branch is therefore better understood as a global statistical prior than as an exact memory mechanism.

## Repository Layout

```text
swaa_patch/             Core monkey patches for Transformers, KV cache state, and linear memory
flash-attention-SWAA/   Customized FlashAttention kernels used by the SWAA experiments
eval/                   Evaluation scripts, including RULER and lm-eval based workflows
qkv_analysis/           Analysis scripts for query/key/value statistics
tuning_log/             Notes and reports from implementation and optimization experiments
METHOD.md               Method notes
paper_draft_sequence_mixed_attention.md
test.py                 Minimal inference smoke test
pixi.toml               Reproducible development environment
```

## Getting Started

The project uses Pixi for environment management:

```bash
pixi install
pixi run verify
```

A small inference example is available in:

```bash
pixi run python test.py
```

Evaluation scripts are under `eval/`. For example:

```bash
cd eval/scripts
python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768
```

Some experiments require the customized FlashAttention kernels in `flash-attention-SWAA/`.

## Status

This is a research codebase rather than a polished library. Interfaces may change, and several components rely on monkey patching internal model implementations. The repository is mainly intended to support method development, controlled experiments, and analysis of training-free linear memory mechanisms for long-context modeling.
