#!/usr/bin/env python3
"""
Quick test script to verify the evaluation system is working correctly.

This script runs a minimal evaluation to ensure:
1. Custom SWAA model wrapper loads correctly
2. lm-eval integration works
3. Results can be saved and analyzed
"""

import sys
import torch
from pathlib import Path

print("=" * 80)
print("SWAA Evaluation System - Quick Test")
print("=" * 80)
print()

# Test 1: Import dependencies
print("Test 1: Checking dependencies...")
try:
    from lm_eval import evaluator
    from swaa_patch import SWAAConfig, hack_hf_swaa, hack_kv_cache_recurrent_state
    from eval_swaa_model import SWAAHFLM
    print("✓ All dependencies imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check CUDA availability
print("\nTest 2: Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")
    device = "cuda:0"
else:
    print("⚠ CUDA not available, using CPU (evaluation will be slow)")
    device = "cpu"

# Test 3: Load custom model wrapper
print("\nTest 3: Loading SWAA model wrapper...")
try:
    model = SWAAHFLM(
        pretrained="Qwen/Qwen3-1.7B",
        device=device,
        torch_dtype="bfloat16",
        sliding_window_size=2048,
        keep_first=4,
        batch_size=1,
    )
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Run minimal evaluation
print("\nTest 4: Running minimal evaluation (hellaswag, 10 samples)...")
try:
    # Use a small subset for quick testing
    results = evaluator.simple_evaluate(
        model=model,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=1,
        limit=10,  # Only evaluate 10 samples for quick test
    )

    print("✓ Evaluation completed successfully")

    # Display results
    if "results" in results:
        print("\nSample Results:")
        for task, metrics in results["results"].items():
            print(f"  {task}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")

except Exception as e:
    print(f"✗ Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Save and analyze results
print("\nTest 5: Testing result saving...")
try:
    import json
    from datetime import datetime

    output_dir = Path("eval_results") / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "test_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✓ Results saved to: {results_file}")

except Exception as e:
    print(f"✗ Result saving failed: {e}")
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ All tests passed! The evaluation system is ready to use.")
print("=" * 80)
print()
print("Next steps:")
print("  1. Run quick evaluation:     ./eval.sh quick")
print("  2. Run standard evaluation:  ./eval.sh standard")
print("  3. Custom evaluation:        python run_evaluation.py --tasks hellaswag,arc_easy")
print()
