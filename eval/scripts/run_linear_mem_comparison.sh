#!/bin/bash
# Linear Memory Comparison Evaluation - Quick Start Script
# 快速运行 Linear Mem 对比评测

set -e  # Exit on error

# Default configuration
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
DEVICE="${DEVICE:-cuda:7}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
SLIDING_WINDOW="${SLIDING_WINDOW:-4096}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

# Build command
CMD="python eval/scripts/compare_linear_mem.py"

# Add arguments
CMD+=" --model $MODEL"
CMD+=" --device $DEVICE"
CMD+=" --context-length $CONTEXT_LENGTH"
CMD+=" --sliding-window $SLIDING_WINDOW"
CMD+=" --num-samples $NUM_SAMPLES"

if [ -n "$OUTPUT_DIR" ]; then
    CMD+=" --output-dir $OUTPUT_DIR"
fi

echo "=========================================="
echo "Linear Memory Comparison Evaluation"
echo "=========================================="
echo "Model: $MODEL"
echo "Device: $DEVICE"
echo "Context Length: $CONTEXT_LENGTH"
echo "Sliding Window: $SLIDING_WINDOW"
echo "Num Samples: $NUM_SAMPLES"
echo "=========================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "Starting evaluation..."
echo ""

# Run the command
$CMD

echo ""
echo "=========================================="
echo "Evaluation completed!"
echo "=========================================="
