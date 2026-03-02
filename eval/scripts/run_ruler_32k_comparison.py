#!/usr/bin/env python3
"""
RULER 32K 多模型对比评测脚本

用于对比多个模型在 RULER 32K 任务上的性能表现。

使用方法:
    # 单个模型评测
    python run_ruler_32k_comparison.py --model model1 --device cuda:0

    # 多个模型对比
    python run_ruler_32k_comparison.py --models model1 model2 model3 --device cuda:0

    # 使用配置文件
    python run_ruler_32k_comparison.py --config models_config.yaml
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lm_eval import evaluator
from lm_eval.utils import make_table


# RULER 32K 任务列表
RULER_32K_TASKS = [
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


def evaluate_single_model(model_config: dict, args):
    """
    评测单个模型

    Args:
        model_config: 模型配置字典
            {
                "name": "模型名称",
                "path": "模型路径",
                "swaa_window": 2048,  # 可选
                "keep_first": 4,      # 可选
            }
        args: 命令行参数

    Returns:
        评测结果字典
    """
    model_name = model_config["name"]
    model_path = model_config["path"]

    print("\n" + "=" * 80)
    print(f"评测模型: {model_name}")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"设备: {args.device}")
    print(f"SWAA 窗口: {model_config.get('swaa_window', 'N/A')}")
    print(f"Keep First: {model_config.get('keep_first', 'N/A')}")

    # 构建模型参数
    model_args = {
        "pretrained": model_path,
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn,
    }

    # 添加 SWAA 配置（如果有）
    if "swaa_window" in model_config:
        model_args["sliding_window_size"] = model_config["swaa_window"]
    if "keep_first" in model_config:
        model_args["keep_first"] = model_config["keep_first"]

    # 转换为字符串
    model_args_str = ",".join([f"{k}={v}" for k, v in model_args.items()])

    # 选择模型类型
    model_type = "swaa_hf" if "swaa_window" in model_config else "hf"

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        # Use default eval/results directory
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "results"
    else:
        output_base = Path(args.output_dir)

    output_dir = output_base / f"{model_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n开始评测 {len(RULER_32K_TASKS)} 个任务...")
    print(f"任务列表: {RULER_32K_TASKS}")

    try:
        # 运行评测
        results = evaluator.simple_evaluate(
            model=model_type,
            model_args=model_args_str,
            tasks=RULER_32K_TASKS,
            num_fewshot=0,
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
        )

        # 保存结果
        results_file = output_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # 保存表格
        table_file = output_dir / "results_table.txt"
        with open(table_file, "w") as f:
            f.write(make_table(results))

        # 打印结果
        print("\n" + "=" * 80)
        print(f"{model_name} 评测结果")
        print("=" * 80)
        print(make_table(results))

        # 提取关键指标
        metrics = extract_metrics(results)

        print(f"\n✅ {model_name} 评测完成!")
        print(f"结果保存至: {output_dir}")

        return {
            "model_name": model_name,
            "model_path": model_path,
            "results": results,
            "metrics": metrics,
            "output_dir": str(output_dir),
        }

    except Exception as e:
        print(f"\n❌ {model_name} 评测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def extract_metrics(results: dict) -> dict:
    """提取关键指标"""
    metrics = {}

    if "results" in results:
        for task_name, task_results in results["results"].items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    metrics[f"{task_name}_{metric}"] = value

    return metrics


def compare_models(all_results: list, output_dir: Path):
    """
    对比多个模型的评测结果

    Args:
        all_results: 所有模型的评测结果列表
        output_dir: 输出目录
    """
    if not all_results:
        print("没有有效的评测结果可供对比")
        return

    print("\n" + "=" * 80)
    print("多模型对比结果")
    print("=" * 80)

    # 创建对比表格
    comparison = {}

    for result in all_results:
        if result is None:
            continue

        model_name = result["model_name"]
        metrics = result["metrics"]

        for metric_name, value in metrics.items():
            if metric_name not in comparison:
                comparison[metric_name] = {}
            comparison[metric_name][model_name] = value

    # 打印对比表格
    print("\n| 指标 |", end="")
    for result in all_results:
        if result:
            print(f" {result['model_name']} |", end="")
    print()

    print("|------|", end="")
    for _ in all_results:
        if _:
            print("------|", end="")
    print()

    for metric_name in sorted(comparison.keys()):
        print(f"| {metric_name} |", end="")
        for result in all_results:
            if result and result["model_name"] in comparison[metric_name]:
                value = comparison[metric_name][result["model_name"]]
                print(f" {value:.4f} |", end="")
            else:
                print(" N/A |", end="")
        print()

    # 保存对比结果
    comparison_file = output_dir / "model_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    # 生成 Markdown 报告
    generate_comparison_report(all_results, comparison, output_dir)

    print(f"\n✅ 对比结果保存至: {output_dir}")


def generate_comparison_report(all_results: list, comparison: dict, output_dir: Path):
    """生成 Markdown 对比报告"""
    report_file = output_dir / "comparison_report.md"

    with open(report_file, "w") as f:
        f.write("# RULER 32K 多模型对比评测报告\n\n")
        f.write(f"**评测时间:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 评测配置\n\n")
        f.write(f"- **任务数量:** {len(RULER_32K_TASKS)}\n")
        f.write(f"- **任务列表:** {', '.join(RULER_32K_TASKS)}\n")
        f.write(f"- **样本数量/任务:** {all_results[0]['results'].get('config', {}).get('limit', 'N/A') if all_results else 'N/A'}\n\n")

        f.write("## 模型列表\n\n")
        for result in all_results:
            if result:
                f.write(f"### {result['model_name']}\n")
                f.write(f"- **路径:** `{result['model_path']}`\n")
                f.write(f"- **结果目录:** `{result['output_dir']}`\n\n")

        f.write("## 性能对比\n\n")
        f.write("| 指标 |")
        for result in all_results:
            if result:
                f.write(f" {result['model_name']} |")
        f.write("\n")

        f.write("|------|")
        for _ in all_results:
            if _:
                f.write("------|")
        f.write("\n")

        for metric_name in sorted(comparison.keys()):
            f.write(f"| {metric_name} |")
            for result in all_results:
                if result and result["model_name"] in comparison[metric_name]:
                    value = comparison[metric_name][result["model_name"]]
                    f.write(f" {value:.4f} |")
                else:
                    f.write(" N/A |")
            f.write("\n")

        f.write("\n## 分析\n\n")
        f.write("### 最佳表现\n\n")
        for metric_name in sorted(comparison.keys()):
            values = comparison[metric_name]
            if values:
                best_model = max(values, key=values.get)
                best_value = values[best_model]
                f.write(f"- **{metric_name}**: {best_model} ({best_value:.4f})\n")

    print(f"📄 对比报告: {report_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RULER 32K 多模型对比评测",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 模型配置
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--model",
        help="单个模型路径",
    )
    model_group.add_argument(
        "--models",
        nargs="+",
        help="多个模型路径（空格分隔）",
    )
    model_group.add_argument(
        "--config",
        help="模型配置文件 (YAML格式)",
    )

    # 评测配置
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="设备 (default: cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="数据类型 (default: bfloat16)",
    )
    parser.add_argument(
        "--attn",
        default="flash_attention_2",
        help="注意力实现 (default: flash_attention_2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="批次大小 (default: 1)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="每任务样本数 (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录 (default: eval/results)",
    )

    # 模型名称
    parser.add_argument(
        "--name",
        help="模型名称（单模型模式）",
    )

    # SWAA 配置
    parser.add_argument(
        "--swaa-window",
        type=int,
        help="SWAA 滑动窗口大小",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        help="SWAA keep_first tokens",
    )

    args = parser.parse_args()

    # 准备模型配置列表
    models_to_evaluate = []

    if args.model:
        # 单模型模式
        model_config = {
            "name": args.name or Path(args.model).name,
            "path": args.model,
        }
        if args.swaa_window:
            model_config["swaa_window"] = args.swaa_window
        if args.keep_first:
            model_config["keep_first"] = args.keep_first
        models_to_evaluate.append(model_config)

    elif args.models:
        # 多模型模式
        for model_path in args.models:
            model_config = {
                "name": Path(model_path).name,
                "path": model_path,
            }
            models_to_evaluate.append(model_config)

    elif args.config:
        # 配置文件模式
        import yaml
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        models_to_evaluate = config.get("models", [])

    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        # Use default eval/results directory
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "results"
    else:
        output_base = Path(args.output_dir)

    output_dir = output_base / f"comparison_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("RULER 32K 多模型对比评测")
    print("=" * 80)
    print(f"\n将评测 {len(models_to_evaluate)} 个模型:")
    for i, model_config in enumerate(models_to_evaluate, 1):
        print(f"  {i}. {model_config['name']} ({model_config['path']})")
    print()

    # 评测所有模型
    all_results = []
    for model_config in models_to_evaluate:
        result = evaluate_single_model(model_config, args)
        all_results.append(result)

    # 对比结果
    compare_models(all_results, output_dir)

    print("\n" + "=" * 80)
    print("✅ 所有评测完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
