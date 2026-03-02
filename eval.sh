#!/bin/bash
# =============================================================================
# SWAA Model Evaluation Quick Start Script
# =============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        SWAA Model Multi-Dimensional Evaluation             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Default parameters
DEVICE=${DEVICE:-"cuda:0"}
MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
PRESET=${1:-"quick"}

# Display configuration
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Model:  ${GREEN}${MODEL}${NC}"
echo -e "  Device: ${GREEN}${DEVICE}${NC}"
echo -e "  Preset: ${GREEN}${PRESET}${NC}"
echo ""

# Available presets
echo -e "${YELLOW}Available Presets:${NC}"
echo -e "  ${GREEN}quick${NC}              - Quick evaluation (~30 min)"
echo -e "  ${GREEN}standard${NC}           - Standard evaluation (~2-3 hours)"
echo -e "  ${GREEN}comprehensive${NC}      - Comprehensive evaluation (~6-8 hours)"
echo -e "  ${GREEN}long_context_focus${NC} - Long context evaluation (~1-2 hours)"
echo ""

# Run evaluation
echo -e "${BLUE}Starting evaluation...${NC}"
echo ""

python run_evaluation.py \
    --preset "${PRESET}" \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --batch-size 1 \
    --output-dir eval_results

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Evaluation completed successfully!${NC}"
    echo ""

    # Find the latest result directory
    LATEST_RESULT=$(ls -td eval_results/eval_* | head -1)

    echo -e "${YELLOW}Results saved to: ${GREEN}${LATEST_RESULT}${NC}"
    echo ""
    echo -e "${YELLOW}To analyze results:${NC}"
    echo -e "  python analyze_results.py --result-dir ${LATEST_RESULT}"
    echo ""
    echo -e "${YELLOW}To view the report:${NC}"
    echo -e "  cat ${LATEST_RESULT}/evaluation_report.md"
    echo ""
else
    echo ""
    echo -e "${RED}❌ Evaluation failed!${NC}"
    exit 1
fi
