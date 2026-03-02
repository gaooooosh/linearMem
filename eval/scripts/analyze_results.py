#!/usr/bin/env python3
"""
Evaluation Results Analysis Script

Analyze and visualize evaluation results from lm-evaluation-harness.

Usage:
    python analyze_results.py --result-dir eval_results/eval_20240101_120000
    python analyze_results.py --compare eval_results/eval_* --output comparison_report.md
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_results(result_dir: Path) -> dict:
    """Load evaluation results from directory."""
    results_file = result_dir / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")

    with open(results_file, "r") as f:
        return json.load(f)


def extract_metrics(results: dict) -> Dict[str, Dict[str, float]]:
    """Extract key metrics from results."""
    metrics = {}

    for task_name, task_results in results.get("results", {}).items():
        task_metrics = {}

        for metric_name, value in task_results.items():
            if isinstance(value, (int, float)):
                # Normalize metric names
                if "acc" in metric_name.lower():
                    task_metrics["accuracy"] = value
                elif "ppl" in metric_name.lower():
                    task_metrics["perplexity"] = value
                elif "f1" in metric_name.lower():
                    task_metrics["f1_score"] = value
                elif "bleu" in metric_name.lower():
                    task_metrics["bleu"] = value
                elif "rouge" in metric_name.lower():
                    task_metrics["rouge"] = value

        if task_metrics:
            metrics[task_name] = task_metrics

    return metrics


def create_metrics_dataframe(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Create pandas DataFrame from metrics."""
    rows = []
    for task_name, task_metrics in metrics.items():
        for metric_name, value in task_metrics.items():
            rows.append(
                {"task": task_name, "metric": metric_name, "value": value}
            )

    return pd.DataFrame(rows)


def plot_metrics_comparison(df: pd.DataFrame, output_path: Path):
    """Create bar plot comparing metrics across tasks."""
    plt.figure(figsize=(14, 8))

    # Pivot for plotting
    plot_df = df[df["metric"] == "accuracy"].copy()

    if len(plot_df) > 0:
        sns.barplot(data=plot_df, x="task", y="value", palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Task")
        plt.ylabel("Accuracy")
        plt.title("Model Performance Across Tasks")
        plt.tight_layout()
        plt.savefig(output_path / "metrics_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Saved metrics comparison plot to: {output_path / 'metrics_comparison.png'}")


def plot_radar_chart(metrics: Dict[str, Dict[str, float]], output_path: Path):
    """Create radar chart for multi-dimensional analysis."""
    # Prepare data
    categories = list(metrics.keys())
    accuracy_values = [
        metrics[cat].get("accuracy", 0) for cat in categories
    ]

    # Number of categories
    N = len(categories)

    # Create angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    accuracy_values += accuracy_values[:1]
    angles += angles[:1]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    ax.plot(angles, accuracy_values, "o-", linewidth=2, label="Accuracy")
    ax.fill(angles, accuracy_values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Dimensional Performance", size=15, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    plt.tight_layout()
    plt.savefig(output_path / "radar_chart.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Saved radar chart to: {output_path / 'radar_chart.png'}")


def generate_summary_report(
    metrics: Dict[str, Dict[str, float]],
    result_dir: Path,
    output_path: Path,
):
    """Generate markdown summary report."""
    report_file = output_path / "analysis_report.md"

    with open(report_file, "w") as f:
        f.write("# Evaluation Results Analysis\n\n")
        f.write(f"**Result Directory:** `{result_dir}`\n\n")

        # Overall statistics
        f.write("## Overall Statistics\n\n")
        all_accuracies = [
            m.get("accuracy", 0) for m in metrics.values() if "accuracy" in m
        ]
        if all_accuracies:
            f.write(f"- **Average Accuracy:** {np.mean(all_accuracies):.4f}\n")
            f.write(f"- **Median Accuracy:** {np.median(all_accuracies):.4f}\n")
            f.write(f"- **Max Accuracy:** {np.max(all_accuracies):.4f}\n")
            f.write(f"- **Min Accuracy:** {np.min(all_accuracies):.4f}\n\n")

        # Task breakdown
        f.write("## Task Breakdown\n\n")
        f.write("| Task | Accuracy | Perplexity |\n")
        f.write("|------|----------|------------|\n")

        for task_name, task_metrics in sorted(metrics.items()):
            acc = task_metrics.get("accuracy", "N/A")
            ppl = task_metrics.get("perplexity", "N/A")

            acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
            ppl_str = f"{ppl:.4f}" if isinstance(ppl, float) else ppl

            f.write(f"| {task_name} | {acc_str} | {ppl_str} |\n")

        # Visualizations
        f.write("\n## Visualizations\n\n")
        f.write("### Metrics Comparison\n\n")
        f.write("![Metrics Comparison](metrics_comparison.png)\n\n")
        f.write("### Radar Chart\n\n")
        f.write("![Radar Chart](radar_chart.png)\n\n")

        # Recommendations
        f.write("## Analysis & Recommendations\n\n")

        if all_accuracies:
            weak_tasks = [
                task
                for task, m in metrics.items()
                if m.get("accuracy", 1.0) < np.median(all_accuracies)
            ]
            strong_tasks = [
                task
                for task, m in metrics.items()
                if m.get("accuracy", 0.0) >= np.percentile(all_accuracies, 75)
            ]

            if weak_tasks:
                f.write("### Areas for Improvement\n\n")
                for task in weak_tasks:
                    f.write(f"- **{task}**: {metrics[task].get('accuracy', 0):.4f}\n")
                f.write("\n")

            if strong_tasks:
                f.write("### Strengths\n\n")
                for task in strong_tasks:
                    f.write(f"- **{task}**: {metrics[task].get('accuracy', 0):.4f}\n")
                f.write("\n")

    print(f"📄 Saved analysis report to: {report_file}")


def analyze_single_result(result_dir: Path):
    """Analyze a single evaluation result."""
    print("\n" + "=" * 80)
    print(f"Analyzing results from: {result_dir}")
    print("=" * 80 + "\n")

    # Load results
    results = load_results(result_dir)

    # Extract metrics
    metrics = extract_metrics(results)

    # Create output directory
    output_path = result_dir / "analysis"
    output_path.mkdir(exist_ok=True)

    # Create visualizations
    df = create_metrics_dataframe(metrics)

    if len(df) > 0:
        plot_metrics_comparison(df, output_path)

    if len(metrics) > 0:
        plot_radar_chart(metrics, output_path)

    # Generate report
    generate_summary_report(metrics, result_dir, output_path)

    print("\n✅ Analysis complete!\n")


def compare_results(result_dirs: List[Path], output_dir: Path):
    """Compare multiple evaluation results."""
    print("\n" + "=" * 80)
    print("Comparing evaluation results")
    print("=" * 80 + "\n")

    all_metrics = {}

    for result_dir in result_dirs:
        exp_name = result_dir.name
        results = load_results(result_dir)
        all_metrics[exp_name] = extract_metrics(results)

    # Create comparison DataFrame
    comparison_data = []

    for exp_name, metrics in all_metrics.items():
        for task_name, task_metrics in metrics.items():
            if "accuracy" in task_metrics:
                comparison_data.append(
                    {
                        "experiment": exp_name,
                        "task": task_name,
                        "accuracy": task_metrics["accuracy"],
                    }
                )

    df = pd.DataFrame(comparison_data)

    # Plot comparison
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 8))
    sns.barplot(data=df, x="task", y="accuracy", hue="experiment", palette="Set2")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Comparison")
    plt.legend(title="Experiment", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Saved comparison plot to: {output_dir / 'comparison.png'}")

    # Generate comparison report
    report_file = output_dir / "comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# Evaluation Results Comparison\n\n")

        for exp_name in all_metrics.keys():
            f.write(f"## {exp_name}\n\n")
            metrics = all_metrics[exp_name]

            f.write("| Task | Accuracy |\n")
            f.write("|------|----------|\n")

            for task_name, task_metrics in sorted(metrics.items()):
                acc = task_metrics.get("accuracy", "N/A")
                acc_str = f"{acc:.4f}" if isinstance(acc, float) else acc
                f.write(f"| {task_name} | {acc_str} |\n")

            f.write("\n")

        f.write("## Visualization\n\n")
        f.write("![Comparison](comparison.png)\n")

    print(f"📄 Saved comparison report to: {report_file}")
    print("\n✅ Comparison complete!\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze evaluation results")

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--result-dir", type=Path, help="Analyze single result directory"
    )
    mode_group.add_argument(
        "--compare", nargs="+", type=Path, help="Compare multiple result directories"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_comparison"),
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    if args.result_dir:
        analyze_single_result(args.result_dir)
    elif args.compare:
        compare_results(args.compare, args.output)


if __name__ == "__main__":
    main()
