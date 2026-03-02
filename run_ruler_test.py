#!/usr/bin/env python3
"""
RULER Long Context Evaluation Script (Small Scale)

RULER is a comprehensive long-context benchmark. This script runs a small-scale
test to evaluate your SWAA model's long-context capabilities.

Available RULER tasks:
- niah_single_1/2/3: Needle In A Haystack (single needle)
- niah_multikey: Multiple keys to retrieve
- niah_multivalue: Multiple values to retrieve
- niah_multiquery: Multiple queries
- ruler_cwe: Code Word Extraction
- ruler_fwe: Frequent Word Extraction
- ruler_vt: Variable Tracking
- ruler_qa_hotpot: HotpotQA long-context
- ruler_qa_squad: SQuAD long-context
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path for custom model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and register custom model BEFORE importing lm_eval
from eval_swaa_model import SWAAHFLM  # This registers the model

from lm_eval import evaluator
from lm_eval.utils import make_table


def run_ruler_evaluation(args):
    """Run RULER evaluation with SWAA model."""
    print("\n" + "=" * 80)
    print("RULER Long Context Evaluation (Small Scale)")
    print("=" * 80)

    # Define task sets for different test sizes
    task_sets = {
        "mini": [
            "niah_single_1",  # Basic needle in haystack
            "passkey",  # Passkey retrieval
        ],
        "small": [
            "niah_single_1",
            "niah_single_2",
            "passkey",
            "ruler_vt",  # Variable tracking
        ],
        "medium": [
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey",
            "passkey",
            "ruler_vt",
            "ruler_cwe",  # Code word extraction
        ],
        "full": [
            "niah_single_1",
            "niah_single_2",
            "niah_single_3",
            "niah_multikey",
            "niah_multivalue",
            "niah_multiquery",
            "passkey",
            "ruler_vt",
            "ruler_cwe",
            "ruler_fwe",
            "ruler_qa_hotpot",
            "ruler_qa_squad",
        ],
    }

    # Select tasks based on size
    tasks = task_sets.get(args.size, task_sets["mini"])

    print(f"\n📋 Test Size: {args.size}")
    print(f"📊 Tasks: {tasks}")
    print(f"🎯 Model: {args.model}")
    print(f"💻 Device: {args.device}")
    print(f"🔧 SWAA Window: {args.sliding_window}")

    # Build model arguments
    # NOTE: device and batch_size are passed separately to simple_evaluate,
    # not in model_args string
    model_args = {
        "pretrained": args.model,
        # "device": args.device,  # Passed separately
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn,
        # "batch_size": args.batch_size,  # Passed separately
        "sliding_window_size": args.sliding_window,
        "keep_first": args.keep_first,
        "force_fa_decode": False,
        "non_sliding_layers": [],
    }

    model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])

    print(f"\n🤖 Model Args: {model_args_str}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ruler_{args.size}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output: {output_dir}")
    print("\n" + "=" * 80)
    print("Starting RULER Evaluation...")
    print("=" * 80 + "\n")

    # Run evaluation
    try:
        results = evaluator.simple_evaluate(
            model="swaa_hf",
            model_args=model_args_str,
            tasks=tasks,
            num_fewshot=0,  # RULER tasks don't use few-shot
            batch_size=args.batch_size,
            max_batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,  # Limit number of samples for small-scale test
        )

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print results table
        print("\n" + "=" * 80)
        print("RULER Evaluation Results")
        print("=" * 80 + "\n")
        print(make_table(results))

        # Save formatted table
        table_file = output_dir / "results_table.txt"
        with open(table_file, "w") as f:
            f.write(make_table(results))

        # Generate report
        generate_ruler_report(results, output_dir, args, tasks)

        print(f"\n✅ Results saved to: {output_dir}")
        print(f"   - Full results: {results_file}")
        print(f"   - Summary table: {table_file}")

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def generate_ruler_report(results: dict, output_dir: Path, args: argparse.Namespace, tasks: list):
    """Generate RULER-specific evaluation report."""
    report_file = output_dir / "ruler_report.md"

    with open(report_file, "w") as f:
        f.write("# RULER Long Context Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Model:** {args.model}\n")
        f.write(f"- **Device:** {args.device}\n")
        f.write(f"- **Test Size:** {args.size}\n")
        f.write(f"- **Sample Limit:** {args.limit or 'Full'}\n")
        f.write(f"- **SWAA Window:** {args.sliding_window}\n")
        f.write(f"- **Keep First:** {args.keep_first}\n\n")

        f.write("## Tasks Evaluated\n\n")
        for task in tasks:
            f.write(f"- {task}\n")
        f.write("\n")

        if "results" in results:
            f.write("## Performance Summary\n\n")
            f.write("| Task | Metric | Score |\n")
            f.write("|------|--------|-------|\n")

            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"| {task_name} | {metric} | {value:.4f} |\n")

            # Calculate average if we have accuracy scores
            accuracies = []
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if "acc" in metric.lower() and isinstance(value, (int, float)):
                        accuracies.append(value)

            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                f.write(f"\n**Average Accuracy:** {avg_acc:.4f}\n\n")

        f.write("## Detailed Results\n\n")
        f.write("```json\n")
        f.write(json.dumps(results.get("results", {}), indent=2))
        f.write("\n```\n")

        # Add interpretation
        f.write("\n## Interpretation\n\n")
        f.write("### RULER Tasks Explanation\n\n")
        f.write("**NIAH (Needle In A Haystack):**\n")
        f.write("- Tests the model's ability to find specific information in long contexts\n")
        f.write("- Single: One needle to find\n")
        f.write("- Multi-key: Multiple keys to retrieve\n")
        f.write("- Multi-value: Multiple values per key\n")
        f.write("- Multi-query: Multiple queries per context\n\n")

        f.write("**Passkey:**\n")
        f.write("- Tests ability to recall a passkey buried in long text\n\n")

        f.write("**Variable Tracking (VT):**\n")
        f.write("- Tests ability to track variable assignments through long code\n\n")

        f.write("**Code Word Extraction (CWE):**\n")
        f.write("- Tests ability to extract specific words meeting criteria\n\n")

        f.write("**Frequent Word Extraction (FWE):**\n")
        f.write("- Tests ability to identify most frequent words\n\n")

    print(f"   - RULER report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RULER long context evaluation with SWAA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mini test (2 tasks, fastest)
  python run_ruler_test.py --size mini --device cuda:0

  # Small test (4 tasks)
  python run_ruler_test.py --size small --device cuda:0

  # Medium test (7 tasks)
  python run_ruler_test.py --size medium --limit 10 --device cuda:0

  # Full test (all RULER tasks)
  python run_ruler_test.py --size full --device cuda:0
        """,
    )

    # Test size
    parser.add_argument(
        "--size",
        choices=["mini", "small", "medium", "full"],
        default="mini",
        help="Test size: mini(2), small(4), medium(7), full(12) tasks",
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Data type",
    )
    parser.add_argument(
        "--attn",
        default="flash_attention_2",
        help="Attention implementation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size",
    )

    # SWAA configuration
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=2048,
        help="SWAA sliding window size",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=4,
        help="SWAA keep_first tokens",
    )

    # Evaluation settings
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples per task (for small-scale test)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Output directory",
    )

    args = parser.parse_args()

    # Suggest limit for small-scale test
    if args.size in ["mini", "small"] and args.limit is None:
        args.limit = 5
        print(f"💡 Auto-setting sample limit to {args.limit} for {args.size} test")

    # Run evaluation
    run_ruler_evaluation(args)


if __name__ == "__main__":
    main()
