#!/bin/bash
# 确保 Pixi 内部的 CUDA 路径优先级最高
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 算子编译优化：针对 A100 (SXM4 架构)
export TORCH_CUDA_ARCH_LIST="8.0"
# Flash Attention CUDA 架构设置 (用于 setup.py)
export FLASH_ATTN_CUDA_ARCHS="80"

# 强制使用 Pixi 安装的 GCC 13
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++

# 增加编译时的并行度，A100 所在节点通常 CPU 核心很多
export MAX_JOBS=$(nproc)

# 跳过 CUDA 版本严格检查
export TORCH_DONT_CHECK_COMPILER_ABI=ON

# 让 PyTorch 使用正确的 CUDA 路径
export CUDA_PATH=$CUDA_HOME
