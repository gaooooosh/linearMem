#!/usr/bin/env python3
"""
Linear Memory Comparison Evaluation Script

对比 Linear Mem 开启和关闭时的 RULER 评测性能
支持自定义配置：上下文长度、滑动窗口大小、样本数量
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import yaml

# Add current directory to path for custom model import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and register custom model BEFORE importing lm_eval
from eval_swaa_model import SWAAHFLM

from lm_eval import evaluator
from lm_eval.utils import make_table


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_single_evaluation(
    model_name: str,
    device: str,
    dtype: str,
    attn: str,
    batch_size: int,
    sliding_window: int,
    keep_first: int,
    enable_linear_mem: bool,
    tasks: List[str],
    limit: int,
    output_dir: Path,
    gen_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run a single evaluation with specific Linear Mem configuration.

    Args:
        model_name: Model name or path
        device: Device to use
        dtype: Data type
        attn: Attention implementation
        batch_size: Batch size
        sliding_window: Sliding window size
        keep_first: Keep first tokens
        enable_linear_mem: Whether to enable Linear Mem
        tasks: List of tasks to evaluate
        limit: Number of samples per task
        output_dir: Output directory
        gen_kwargs: Generation kwargs

    Returns:
        Evaluation results dictionary
    """
    linear_mem_status = "ENABLED" if enable_linear_mem else "DISABLED"

    print("\n" + "=" * 80)
    print(f"Running Evaluation: Linear Mem {linear_mem_status}")
    print("=" * 80)
    print(f"🎯 Model: {model_name}")
    print(f"💻 Device: {device}")
    print(f"🔧 Sliding Window: {sliding_window}")
    print(f"🔄 Linear Mem: {enable_linear_mem}")
    print(f"📊 Tasks: {len(tasks)}")
    print(f"📏 Samples per task: {limit}")
    print("=" * 80 + "\n")

    # Build model arguments
    model_args = {
        "pretrained": model_name,
        "device": device,
        "torch_dtype": dtype,
        "attn_implementation": attn,
        "sliding_window_size": sliding_window,
        "keep_first": keep_first,
        "force_fa_decode": False,
        "non_sliding_layers": [],
        "enable_linear_mem": enable_linear_mem,
    }

    model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])

    # Build generation kwargs string
    gen_kwargs_str = ",".join([f"{k}={v}" for k, v in gen_kwargs.items()])

    print(f"🤖 Model Args: {model_args_str}")
    print(f"🔧 Generation Args: {gen_kwargs_str}\n")

    # Run evaluation
    try:
        results = evaluator.simple_evaluate(
            model="swaa_hf",
            model_args=model_args_str,
            tasks=tasks,
            num_fewshot=0,
            batch_size=batch_size,
            max_batch_size=batch_size,
            device=device,
            limit=limit,
            gen_kwargs=gen_kwargs_str,
            log_samples=False,
        )

        # Save results
        suffix = "enabled" if enable_linear_mem else "disabled"
        results_file = output_dir / f"results_linear_mem_{suffix}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Print results table
        print(f"\n{'='*80}")
        print(f"Results: Linear Mem {linear_mem_status}")
        print(f"{'='*80}\n")
        print(make_table(results))

        # Save formatted table
        table_file = output_dir / f"results_table_linear_mem_{suffix}.txt"
        with open(table_file, "w") as f:
            f.write(make_table(results))

        print(f"\n✅ Results saved to:")
        print(f"   - {results_file}")
        print(f"   - {table_file}\n")

        return results

    except Exception as e:
        print(f"\n❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def compare_results(
    results_enabled: Dict[str, Any],
    results_disabled: Dict[str, Any],
    output_dir: Path,
):
    """
    Compare results between Linear Mem enabled and disabled.

    Args:
        results_enabled: Results with Linear Mem enabled
        results_disabled: Results with Linear Mem disabled
        output_dir: Output directory
    """
    print("\n" + "=" * 80)
    print("Comparing Results: Linear Mem Enabled vs Disabled")
    print("=" * 80 + "\n")

    comparison = {}

    if "results" in results_enabled and "results" in results_disabled:
        # Compare each task
        all_tasks = set(results_enabled["results"].keys()) | set(
            results_disabled["results"].keys()
        )

        print("| Task | Metric | Enabled | Disabled | Difference | Improvement |")
        print("|------|--------|---------|----------|------------|-------------|")

        for task in sorted(all_tasks):
            enabled_results = results_enabled["results"].get(task, {})
            disabled_results = results_disabled["results"].get(task, {})

            # Find common metrics
            all_metrics = set(enabled_results.keys()) | set(disabled_results.keys())

            for metric in sorted(all_metrics):
                enabled_val = enabled_results.get(metric)
                disabled_val = disabled_results.get(metric)

                if isinstance(enabled_val, (int, float)) and isinstance(
                    disabled_val, (int, float)
                ):
                    diff = enabled_val - disabled_val
                    improvement = (
                        ((enabled_val - disabled_val) / disabled_val * 100)
                        if disabled_val != 0
                        else 0
                    )

                    print(
                        f"| {task} | {metric} | {enabled_val:.4f} | {disabled_val:.4f} | "
                        f"{diff:+.4f} | {improvement:+.2f}% |"
                    )

                    # Store in comparison dict
                    if task not in comparison:
                        comparison[task] = {}
                    comparison[task][metric] = {
                        "enabled": enabled_val,
                        "disabled": disabled_val,
                        "difference": diff,
                        "improvement_percent": improvement,
                    }

    # Calculate average improvements
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80 + "\n")

    # Track accuracy improvements
    acc_improvements = []
    for task, metrics in comparison.items():
        for metric, values in metrics.items():
            if "acc" in metric.lower():
                acc_improvements.append(values["improvement_percent"])

    if acc_improvements:
        avg_improvement = sum(acc_improvements) / len(acc_improvements)
        print(f"📊 Average Accuracy Improvement: {avg_improvement:+.2f}%")

        if avg_improvement > 5:
            print("✅ Linear Mem shows SIGNIFICANT IMPROVEMENT")
        elif avg_improvement > 0:
            print("✅ Linear Mem shows MODEST IMPROVEMENT")
        elif avg_improvement > -5:
            print("⚠️ Linear Mem shows MODEST DEGRADATION")
        else:
            print("❌ Linear Mem shows SIGNIFICANT DEGRADATION")

    # Save comparison results
    comparison_file = output_dir / "comparison_results.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n✅ Comparison saved to: {comparison_file}")

    # Generate comparison report
    generate_comparison_report(comparison, output_dir, acc_improvements)


def generate_comparison_report(
    comparison: Dict[str, Any], output_dir: Path, acc_improvements: List[float]
):
    """Generate a detailed comparison report in Markdown."""

    report_file = output_dir / "linear_mem_comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# Linear Memory Comparison Report\n\n")
        f.write(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        f.write("## Overview\n\n")
        f.write(
            "This report compares the performance of RULER evaluation with "
            "Linear Memory enabled vs disabled.\n\n"
        )

        if acc_improvements:
            avg_improvement = sum(acc_improvements) / len(acc_improvements)
            f.write("### Summary\n\n")
            f.write(f"- **Average Accuracy Improvement:** {avg_improvement:+.2f}%\n")

            positive_improvements = [x for x in acc_improvements if x > 0]
            negative_improvements = [x for x in acc_improvements if x < 0]

            f.write(
                f"- **Tasks Improved:** {len(positive_improvements)}/{len(acc_improvements)}\n"
            )
            f.write(
                f"- **Tasks Degraded:** {len(negative_improvements)}/{len(acc_improvements)}\n\n"
            )

            if avg_improvement > 5:
                f.write("**Conclusion:** ✅ Linear Mem shows SIGNIFICANT IMPROVEMENT\n\n")
            elif avg_improvement > 0:
                f.write("**Conclusion:** ✅ Linear Mem shows MODEST IMPROVEMENT\n\n")
            elif avg_improvement > -5:
                f.write("**Conclusion:** ⚠️ Linear Mem shows MODEST DEGRADATION\n\n")
            else:
                f.write("**Conclusion:** ❌ Linear Mem shows SIGNIFICANT DEGRADATION\n\n")

        f.write("## Detailed Results\n\n")
        f.write("| Task | Metric | Enabled | Disabled | Difference | Improvement |\n")
        f.write("|------|--------|---------|----------|------------|-------------|\n")

        for task, metrics in sorted(comparison.items()):
            for metric, values in sorted(metrics.items()):
                f.write(
                    f"| {task} | {metric} | {values['enabled']:.4f} | "
                    f"{values['disabled']:.4f} | {values['difference']:+.4f} | "
                    f"{values['improvement_percent']:+.2f}% |\n"
                )

        f.write("\n## Task Categories\n\n")

        # Group by task type
        niah_tasks = {k: v for k, v in comparison.items() if "niah" in k.lower()}
        other_tasks = {k: v for k, v in comparison.items() if "niah" not in k.lower()}

        if niah_tasks:
            f.write("### NIAH Tasks (Needle In A Haystack)\n\n")
            niah_improvements = []
            for task, metrics in niah_tasks.items():
                for metric, values in metrics.items():
                    if "acc" in metric.lower():
                        niah_improvements.append(values["improvement_percent"])

            if niah_improvements:
                avg_niah = sum(niah_improvements) / len(niah_improvements)
                f.write(f"- **Average Improvement:** {avg_niah:+.2f}%\n\n")

        if other_tasks:
            f.write("### Other Tasks\n\n")
            other_improvements = []
            for task, metrics in other_tasks.items():
                for metric, values in metrics.items():
                    if "acc" in metric.lower():
                        other_improvements.append(values["improvement_percent"])

            if other_improvements:
                avg_other = sum(other_improvements) / len(other_improvements)
                f.write(f"- **Average Improvement:** {avg_other:+.2f}%\n\n")

        f.write("## Recommendations\n\n")

        if acc_improvements and sum(acc_improvements) / len(acc_improvements) > 0:
            f.write(
                "Based on these results, **enabling Linear Mem is recommended** "
                "for this model and task configuration.\n\n"
            )
        else:
            f.write(
                "Based on these results, **disabling Linear Mem may be preferable** "
                "for this model and task configuration.\n\n"
            )

    print(f"✅ Comparison report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare RULER evaluation with Linear Mem enabled vs disabled",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="eval/configs/linear_mem_comparison.yaml",
        help="Path to configuration file",
    )

    # Override configuration with command line arguments
    parser.add_argument(
        "--model",
        type=str,
        help="Model name or path (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use (overrides config)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        help="Context length in tokens (overrides config)",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        help="Sliding window size (overrides config)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples per task (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Make relative path absolute from project root
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / args.config

    print(f"\n📋 Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Apply command line overrides
    model_name = args.model or config["model"]["name"]
    device = args.device or config["model"]["device"]
    dtype = config["model"]["dtype"]
    attn = config["model"]["attn_implementation"]
    batch_size = config["model"]["batch_size"]

    context_length = args.context_length or config["evaluation"]["context_length"]
    sliding_window = args.sliding_window or config["evaluation"]["sliding_window"]
    keep_first = config["evaluation"]["keep_first"]
    num_samples = args.num_samples or config["evaluation"]["num_samples"]
    tasks = config["evaluation"]["tasks"]

    gen_kwargs = config["evaluation"]["generation"]

    # Setup output directory
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = Path(config["output"]["base_dir"])
        if not output_base.is_absolute():
            project_root = Path(__file__).parent.parent.parent
            output_base = project_root / output_base

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / f"linear_mem_comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("Linear Memory Comparison Evaluation")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Device: {device}")
    print(f"  Context Length: {context_length}")
    print(f"  Sliding Window: {sliding_window}")
    print(f"  Samples per task: {num_samples}")
    print(f"  Total tasks: {len(tasks)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 80 + "\n")

    # Save configuration
    config_save_file = output_dir / "config.yaml"
    with open(config_save_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"✅ Configuration saved to: {config_save_file}\n")

    # Run evaluation with Linear Mem ENABLED
    print("\n" + "▶" * 40)
    print("▶ Phase 1: Evaluating with Linear Mem ENABLED")
    print("▶" * 40 + "\n")

    results_enabled = run_single_evaluation(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn=attn,
        batch_size=batch_size,
        sliding_window=sliding_window,
        keep_first=keep_first,
        enable_linear_mem=True,
        tasks=tasks,
        limit=num_samples,
        output_dir=output_dir,
        gen_kwargs=gen_kwargs,
    )

    # Run evaluation with Linear Mem DISABLED
    print("\n" + "▶" * 40)
    print("▶ Phase 2: Evaluating with Linear Mem DISABLED")
    print("▶" * 40 + "\n")

    results_disabled = run_single_evaluation(
        model_name=model_name,
        device=device,
        dtype=dtype,
        attn=attn,
        batch_size=batch_size,
        sliding_window=sliding_window,
        keep_first=keep_first,
        enable_linear_mem=False,
        tasks=tasks,
        limit=num_samples,
        output_dir=output_dir,
        gen_kwargs=gen_kwargs,
    )

    # Compare results
    print("\n" + "▶" * 40)
    print("▶ Phase 3: Comparing Results")
    print("▶" * 40 + "\n")

    compare_results(results_enabled, results_disabled, output_dir)

    print("\n" + "=" * 80)
    print("✅ Evaluation Complete!")
    print("=" * 80)
    print(f"\n📁 All results saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
