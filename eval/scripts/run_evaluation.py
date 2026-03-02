#!/usr/bin/env python3
"""
SWAA Model Evaluation Script

This script provides a unified interface for evaluating SWAA models
using lm-evaluation-harness with multiple dimensions.

Usage:
    # Quick evaluation (30 min)
    python run_evaluation.py --preset quick --device cuda:0

    # Standard evaluation (2-3 hours)
    python run_evaluation.py --preset standard --device cuda:0

    # Comprehensive evaluation (6-8 hours)
    python run_evaluation.py --preset comprehensive --device cuda:0

    # Custom tasks
    python run_evaluation.py --tasks hellaswag,arc_easy --device cuda:0

    # Long context focus
    python run_evaluation.py --preset long_context_focus --device cuda:0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Add current directory to path for custom model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and register custom model BEFORE importing lm_eval
from eval_swaa_model import SWAAHFLM  # This registers the model

from lm_eval import evaluator
from lm_eval.utils import make_table


def load_config(config_path: str = None) -> dict:
    """Load evaluation configuration."""
    if config_path is None:
        # Use default config path relative to this script
        script_dir = Path(__file__).parent
        config_path = script_dir.parent / "configs" / "comprehensive_eval.yaml"

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_tasks_from_preset(preset: str, config: dict) -> list:
    """Get task list from preset name."""
    if preset not in config["presets"]:
        raise ValueError(
            f"Unknown preset: {preset}. "
            f"Available: {list(config['presets'].keys())}"
        )

    preset_config = config["presets"][preset]
    tasks = []

    # Add tasks from groups
    if "groups" in preset_config:
        for group in preset_config["groups"]:
            if group in config["task_groups"]:
                tasks.extend(config["task_groups"][group]["tasks"])

    # Add individual tasks
    if "tasks" in preset_config:
        tasks.extend(preset_config["tasks"])

    return list(set(tasks))  # Remove duplicates


def build_model_args(args, config: dict) -> str:
    """Build model arguments string for lm-eval."""
    model_defaults = config.get("model_defaults", {})

    # Base arguments
    # NOTE: device and batch_size are passed separately to simple_evaluate,
    # not in model_args string
    model_args = {
        "pretrained": args.model,
        # "device": args.device,  # Passed separately
        "torch_dtype": args.dtype or model_defaults.get("torch_dtype", "bfloat16"),
        "attn_implementation": args.attn or model_defaults.get(
            "attn_implementation", "flash_attention_2"
        ),
        # "batch_size": args.batch_size or model_defaults.get("batch_size", 1),  # Passed separately
        "max_chunk_size": args.max_chunk_size,
    }

    # SWAA configuration
    swaa_config = model_defaults.get("swaa", {})
    if args.sliding_window:
        swaa_config["sliding_window_size"] = args.sliding_window
    if args.keep_first:
        swaa_config["keep_first"] = args.keep_first
    if args.enable_linear_mem is not None:
        swaa_config["enable_linear_mem"] = args.enable_linear_mem

    model_args.update(swaa_config)

    # Convert to string format
    args_list = [f"{k}={v}" for k, v in model_args.items()]
    return ",".join(args_list)


def run_evaluation(args):
    """Run lm-evaluation-harness with SWAA model."""
    print("\n" + "=" * 80)
    print("SWAA Model Evaluation")
    print("=" * 80)

    # Load configuration
    config = load_config(args.config)

    # Determine tasks
    if args.tasks:
        tasks = args.tasks.split(",")
    elif args.preset:
        tasks = get_tasks_from_preset(args.preset, config)
        print(f"\n📋 Using preset: {args.preset}")
    else:
        raise ValueError("Either --tasks or --preset must be specified")

    print(f"📊 Evaluation tasks: {tasks}")

    # Build model arguments
    model_args = build_model_args(args, config)
    print(f"\n🤖 Model arguments: {model_args}")

    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        # Use default eval/results directory
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "results"
    else:
        output_base = Path(args.output_dir)

    output_dir = output_base / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Results will be saved to: {output_dir}")
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80 + "\n")

    # Run evaluation
    try:
        # Build evaluation kwargs
        eval_kwargs = {
            "model": "swaa_hf",
            "model_args": model_args,
            "tasks": tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
            "max_batch_size": args.max_batch_size,
            "device": args.device,
        }

        # Add cache settings if specified
        if args.use_cache:
            eval_kwargs["use_cache"] = args.cache_dir if args.cache_dir else "lm_cache"

        # Add metadata for task configuration (e.g., max_seq_lengths for passkey)
        if args.metadata:
            try:
                eval_kwargs["metadata"] = json.loads(args.metadata)
                print(f"\n📋 Metadata: {eval_kwargs['metadata']}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --metadata: {e}")

        # Add generation kwargs
        if args.gen_kwargs:
            try:
                eval_kwargs["gen_kwargs"] = json.loads(args.gen_kwargs)
                print(f"\n🔧 Gen kwargs: {eval_kwargs['gen_kwargs']}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in --gen_kwargs: {e}")

        results = evaluator.simple_evaluate(**eval_kwargs)

        # Save results
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print results table
        print("\n" + "=" * 80)
        print("Evaluation Results")
        print("=" * 80 + "\n")
        print(make_table(results))

        # Save formatted table
        table_file = output_dir / "results_table.txt"
        with open(table_file, "w") as f:
            f.write(make_table(results))

        print(f"\n✅ Results saved to: {output_dir}")
        print(f"   - Full results: {results_file}")
        print(f"   - Summary table: {table_file}")

        # Generate report
        generate_report(results, output_dir, args, tasks)

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def generate_report(results: dict, output_dir: Path, args: argparse.Namespace, tasks: list):
    """Generate a detailed evaluation report."""
    report_file = output_dir / "evaluation_report.md"

    with open(report_file, "w") as f:
        f.write("# SWAA Model Evaluation Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Model Configuration\n\n")
        f.write(f"- **Model:** {args.model}\n")
        f.write(f"- **Device:** {args.device}\n")
        f.write(f"- **Preset:** {args.preset or 'Custom'}\n")
        f.write(f"- **Tasks:** {', '.join(tasks)}\n\n")

        if "results" in results:
            f.write("## Performance Summary\n\n")
            f.write("| Task | Metric | Score |\n")
            f.write("|------|--------|-------|\n")

            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        if "acc" in metric.lower() or "ppl" in metric.lower():
                            f.write(f"| {task_name} | {metric} | {value:.4f} |\n")

        f.write("\n## Detailed Results\n\n")
        f.write("```json\n")
        f.write(json.dumps(results.get("results", {}), indent=2))
        f.write("\n```\n")

    print(f"   - Evaluation report: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SWAA models using lm-evaluation-harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick evaluation
  python run_evaluation.py --preset quick --device cuda:0

  # Comprehensive evaluation
  python run_evaluation.py --preset comprehensive --device cuda:0

  # Custom tasks
  python run_evaluation.py --tasks hellaswag,arc_easy,gsm8k --device cuda:0

  # With custom SWAA config
  python run_evaluation.py --preset standard --sliding_window 4096 --keep_first 8

  # Disable linear memory operations
  python run_evaluation.py --preset standard --enable-linear-mem false

  # Enable linear memory operations explicitly
  python run_evaluation.py --preset standard --enable-linear-mem true
        """,
    )

    # Evaluation mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--preset",
        choices=["quick", "standard", "comprehensive", "long_context_focus"],
        help="Use a predefined evaluation preset",
    )
    mode_group.add_argument("--tasks", help="Comma-separated list of tasks")

    # Model configuration
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="Model name or path (default: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--device", default="cuda:0", help="Device to use (default: cuda:0)"
    )
    parser.add_argument("--dtype", help="Data type (bfloat16, float16, float32)")
    parser.add_argument(
        "--attn", help="Attention implementation (flash_attention_2, eager, sdpa)"
    )
    parser.add_argument("--batch-size", default="auto", help="Batch size (default: auto)")
    parser.add_argument("--max-batch-size", type=int, help="Max batch size")

    # SWAA configuration
    parser.add_argument(
        "--sliding-window", type=int, help="SWAA sliding window size"
    )
    parser.add_argument("--keep-first", type=int, help="SWAA keep_first tokens")
    parser.add_argument(
        "--enable-linear-mem",
        type=lambda x: x.lower() == 'true',
        default=None,
        help="Enable/disable linear memory operations (true/false). Default: from config file"
    )

    # Memory optimization
    parser.add_argument(
        "--max-chunk-size", type=int, default=2048,
        help="Max chunk size for processing long sequences (default: 2048)"
    )

    # Evaluation settings
    parser.add_argument(
        "--num-fewshot", type=int, help="Number of few-shot examples"
    )
    parser.add_argument("--output-dir", default=None, help="Output directory (default: eval/results)")
    parser.add_argument("--config", default=None, help="Config file path (default: eval/configs/comprehensive_eval.yaml)")

    # Cache settings
    parser.add_argument("--use-cache", action="store_true", help="Use cache")
    parser.add_argument("--cache-dir", help="Cache directory")

    # Advanced settings for long context tasks
    parser.add_argument(
        "--metadata",
        type=str,
        help='Metadata as JSON string for task configuration (e.g., \'{"max_seq_lengths":[4096,8192]}\')',
    )
    parser.add_argument(
        "--gen-kwargs",
        type=str,
        help='Generation kwargs as JSON string (e.g., \'{"until":["\\n\\n"],"max_gen_toks":128}\')',
    )

    args = parser.parse_args()

    # Run evaluation
    run_evaluation(args)


if __name__ == "__main__":
    main()
