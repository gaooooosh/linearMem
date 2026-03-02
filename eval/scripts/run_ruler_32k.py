#!/usr/bin/env python3
"""
RULER 32K Long Context Evaluation Script

Run RULER evaluation with 32K context length and 100 samples per task.
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
from eval_swaa_model import SWAAHFLM

from lm_eval import evaluator
from lm_eval.utils import make_table


def run_ruler_32k_evaluation(args):
    """Run RULER evaluation with 32K context length."""
    print("\n" + "=" * 80)
    print("RULER 32K Long Context Evaluation")
    print("=" * 80)

    # Define RULER tasks for comprehensive evaluation
    tasks = [
        "niah_single_1",
        "niah_single_2",
        "niah_single_3",
        "niah_multikey_1",
        "niah_multivalue",
        "niah_multiquery",
        "passkey",
        "ruler_vt",
        "ruler_cwe",
        "ruler_fwe",
    ]

    print(f"\n📋 Test Configuration:")
    print(f"  Context Length: 32K tokens")
    print(f"  Samples per task: {args.limit}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Total samples: {len(tasks) * args.limit}")
    print(f"\n📊 Tasks: {tasks}")
    print(f"\n🎯 Model: {args.model}")
    print(f"💻 Device: {args.device}")
    print(f"🔧 SWAA Window: {args.sliding_window}")

    # Build model arguments
    # NOTE: device and batch_size are passed separately to simple_evaluate
    model_args = {
        "pretrained": args.model,
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn,
        "sliding_window_size": args.sliding_window,
        "keep_first": args.keep_first,
        "force_fa_decode": False,
        "non_sliding_layers": [],
    }

    model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])

    # Generation kwargs for 32K context
    gen_kwargs = {
        "max_gen_toks": 128,  # Maximum generation tokens
        "temperature": 0.0,  # Greedy decoding for deterministic results
        "top_p": 1.0,
        "until": ["\n", "</s>", "<|endoftext|>"],  # Stop sequences
    }

    gen_kwargs_str = ",".join([f"{k}={v}" for k, v in gen_kwargs.items()])

    print(f"\n🤖 Model Args: {model_args_str}")
    print(f"\n🔧 Generation Args: {gen_kwargs_str}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        # Use default eval/results directory
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "results"
    else:
        output_base = Path(args.output_dir)

    output_dir = output_base / f"ruler_32k_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output: {output_dir}")
    print("\n" + "=" * 80)
    print("Starting RULER 32K Evaluation...")
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
            limit=args.limit,
            gen_kwargs=gen_kwargs_str,
            log_samples=False,  # Don't log all samples to save space
        )

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print results table
        print("\n" + "=" * 80)
        print("RULER 32K Evaluation Results")
        print("=" * 80 + "\n")
        print(make_table(results))

        # Save formatted table
        table_file = output_dir / "results_table.txt"
        with open(table_file, "w") as f:
            f.write(make_table(results))

        # Generate report
        generate_32k_report(results, output_dir, args, tasks)

        print(f"\n✅ Results saved to: {output_dir}")
        print(f"   - Full results: {results_file}")
        print(f"   - Summary table: {table_file}")

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_32k_report(results: dict, output_dir: Path, args: argparse.Namespace, tasks: list):
    """Generate RULER 32K evaluation report."""
    report_file = output_dir / "ruler_32k_report.md"

    with open(report_file, "w") as f:
        f.write("# RULER 32K Long Context Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- **Context Length:** 32K tokens\n")
        f.write(f"- **Samples per Task:** {args.limit}\n")
        f.write(f"- **Total Tasks:** {len(tasks)}\n")
        f.write(f"- **Total Samples:** {len(tasks) * args.limit}\n")
        f.write(f"- **Model:** {args.model}\n")
        f.write(f"- **Device:** {args.device}\n")
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

            # Track accuracy scores
            accuracies = []
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"| {task_name} | {metric} | {value:.4f} |\n")
                        if "acc" in metric.lower():
                            accuracies.append(value)

            if accuracies:
                avg_acc = sum(accuracies) / len(accuracies)
                f.write(f"\n**Average Accuracy:** {avg_acc:.4f} ({avg_acc*100:.2f}%)\n\n")

                # Performance categorization
                f.write("## Performance Analysis\n\n")
                if avg_acc >= 0.9:
                    f.write("✅ **Excellent**: Model performs exceptionally well on 32K context tasks.\n")
                elif avg_acc >= 0.7:
                    f.write("✅ **Good**: Model shows strong performance on 32K context tasks.\n")
                elif avg_acc >= 0.5:
                    f.write("⚠️ **Moderate**: Model has reasonable performance but room for improvement.\n")
                else:
                    f.write("❌ **Needs Improvement**: Model struggles with 32K context tasks.\n")

        f.write("\n## Detailed Results\n\n")
        f.write("```json\n")
        f.write(json.dumps(results.get("results", {}), indent=2))
        f.write("\n```\n")

        # Add interpretation guide
        f.write("\n## Interpretation Guide\n\n")
        f.write("### RULER Tasks at 32K Context\n\n")
        f.write("**NIAH (Needle In A Haystack):**\n")
        f.write("- Tests ability to find specific information in 32K token contexts\n")
        f.write("- Single: One needle to find\n")
        f.write("- Multi-key: Multiple keys to retrieve\n")
        f.write("- Multi-value: Multiple values per key\n")
        f.write("- Multi-query: Multiple queries per context\n\n")

        f.write("**Passkey:**\n")
        f.write("- Tests ability to recall a passkey buried in 32K tokens\n\n")

        f.write("**Variable Tracking (VT):**\n")
        f.write("- Tests ability to track variable assignments through long code\n\n")

        f.write("**Word Extraction (CWE/FWE):**\n")
        f.write("- Tests ability to extract specific/frequent words from long text\n\n")

        f.write("### Expected Performance\n\n")
        f.write("For a well-performing long-context model at 32K:\n")
        f.write("- **NIAH tasks**: > 0.8 accuracy\n")
        f.write("- **Passkey**: > 0.9 accuracy\n")
        f.write("- **VT/CWE/FWE**: > 0.7 accuracy\n")

    print(f"   - 32K report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run RULER 32K evaluation with SWAA model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--device",
        default="cuda:7",
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
        default=100,
        help="Number of samples per task (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (default: eval/results)",
    )

    args = parser.parse_args()

    # Run evaluation
    run_ruler_32k_evaluation(args)


if __name__ == "__main__":
    main()
