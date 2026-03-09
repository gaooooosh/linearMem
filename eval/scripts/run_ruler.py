#!/usr/bin/env python3
"""
简化的 RULER 评测脚本

专注于 RULER 长文本评测，支持：
- 自定义上下文长度
- Linear Memory 开关
- Sliding Window 配置
- Keep First 配置
- 完整的配置记录

Usage:
    # 基础用法
    python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768

    # 禁用 linear mem
    python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 16384 --no-linear-mem

    # 自定义滑动窗口
    python run_ruler.py --model Qwen/Qwen3-1.7B --sliding-window 4096 --keep-first 8

    # 指定输出目录
    python run_ruler.py --model Qwen/Qwen3-1.7B --output-dir ./my_results
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


# RULER 任务列表
RULER_TASKS = [
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
    "ruler_qa_hotpot"
]


def run_ruler_evaluation(args):
    """Run RULER evaluation with full configuration tracking."""
    print("\n" + "=" * 80)
    print("RULER 长文本评测")
    print("=" * 80)

    # 构建完整配置
    config = {
        "model": args.model,
        "context_length": args.context_length,
        "sliding_window": args.sliding_window,
        "keep_first": args.keep_first,
        "enable_linear_mem": args.enable_linear_mem,
        "non_sliding_layers": args.non_sliding_layers,
        "flash_attn_weight": args.flash_attn_weight,
        "linear_mem_weight": args.linear_mem_weight,
        "linear_mem_mode": args.linear_mem_mode,
        "device": args.device,
        "dtype": args.dtype,
        "attn_implementation": args.attn,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "timestamp": datetime.now().isoformat(),
    }

    # 打印配置
    print(f"\n📋 评测配置:")
    print(f"  模型: {config['model']}")
    print(f"  上下文长度: {config['context_length']} tokens")
    print(f"  滑动窗口: {config['sliding_window']}")
    print(f"  保留前N个token: {config['keep_first']}")
    print(f"  启用Linear Memory: {config['enable_linear_mem']}")
    print(f"  非滑动层: {config['non_sliding_layers']}")
    print(f"  Flash Attention权重: {config['flash_attn_weight']}")
    print(f"  Linear Memory权重: {config['linear_mem_weight']}")
    print(f"  Linear Memory模式: {config['linear_mem_mode']}")
    print(f"  设备: {config['device']}")
    print(f"  数据类型: {config['dtype']}")
    print(f"  注意力实现: {config['attn_implementation']}")
    print(f"  批次大小: {config['batch_size']}")
    print(f"  每任务样本数: {config['limit']}")

    print(f"\n📊 评测任务 ({len(RULER_TASKS)} 个):")
    for task in RULER_TASKS:
        print(f"  - {task}")

    # 构建模型参数
    # 注意：所有参数必须是可哈希的类型，因为 lm-eval 会将 model_args 传递给 custom_dataset
    # 而 custom_dataset 中的 get_tokenizer 使用了 @cache 装饰器
    model_args = {
        "pretrained": args.model,
        "torch_dtype": args.dtype,
        "attn_implementation": args.attn,
        "sliding_window_size": args.sliding_window,
        "keep_first": args.keep_first,
        "enable_linear_mem": args.enable_linear_mem,
        # non_sliding_layers 已经是元组（在第464行转换）
        "non_sliding_layers": args.non_sliding_layers,
        "force_fa_decode": False,
        "max_chunk_size": args.max_chunk_size,
        "flash_attn_weight": args.flash_attn_weight,
        "linear_mem_weight": args.linear_mem_weight,
        "linear_mem_mode": args.linear_mem_mode,
    }

    # 生成参数
    gen_kwargs = {
        "max_gen_toks": 128,
        "temperature": 0.0,
        "top_p": 1.0,
        "until": ["\n", "</s>", "###"],
    }

    print(f"\n🤖 模型参数:")
    for k, v in model_args.items():
        print(f"   {k}: {v}")

    print(f"\n📊 RULER 测试配置:")
    print(f"   测试长度: {args.context_length} tokens ({args.context_length//1024}K)")

    print(f"\n🔧 生成参数:")
    for k, v in gen_kwargs.items():
        print(f"   {k}: {v}")

    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir is None:
        script_dir = Path(__file__).parent
        output_base = script_dir.parent / "results"
    else:
        output_base = Path(args.output_dir)

    # 生成描述性的输出目录名
    linear_mem_str = "lm" if args.enable_linear_mem else "no_lm"
    output_dir_name = f"ruler_{args.context_length//1024}k_{linear_mem_str}_{timestamp}"
    output_dir = output_base / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 输出目录: {output_dir}")

    # 保存配置
    config_file = output_dir / "config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"   - 配置文件: {config_file}")

    print("\n" + "=" * 80)
    print("开始评测...")
    print("=" * 80 + "\n")

    # 处理 batch_size
    batch_size = args.batch_size
    if batch_size != "auto":
        try:
            batch_size = int(batch_size)
        except ValueError:
            print(f"⚠️ 无效的 batch_size 值: {batch_size}，使用 'auto'")
            batch_size = "auto"

    # 运行评测
    try:
        # 构建传递给 RULER 任务的 metadata
        # metadata 会被传递给 custom_dataset 函数（如 niah_single_1）
        # 注意：
        # 1. max_seq_lengths 使用元组（可哈希），因为 @cache 装饰器要求参数可哈希
        # 2. 只传递 tokenizer 需要的参数，避免传递不可哈希的列表参数
        metadata = {
            "max_seq_lengths": (args.context_length,),  # 使用元组而不是列表
            "pretrained": args.model,  # tokenizer 只需要这个参数
        }

        results = evaluator.simple_evaluate(
            model="swaa_hf",
            model_args=model_args,
            tasks=RULER_TASKS,
            num_fewshot=0,
            batch_size=batch_size,
            device=args.device,
            limit=args.limit,
            gen_kwargs=gen_kwargs,
            log_samples=False,
            metadata=metadata,  # 通过 metadata 传递测试长度
        )

        # 添加配置信息到结果
        results["eval_config"] = config

        # 保存完整结果
        results_file = output_dir / "results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str, ensure_ascii=False)

        # 打印结果表格
        print("\n" + "=" * 80)
        print("评测结果")
        print("=" * 80 + "\n")
        print(make_table(results))

        # 保存结果表格
        table_file = output_dir / "results_table.txt"
        with open(table_file, "w", encoding="utf-8") as f:
            f.write(make_table(results))

        # 生成报告
        generate_report(results, output_dir, config)

        print(f"\n✅ 结果已保存到: {output_dir}")
        print(f"   - 完整结果: {results_file}")
        print(f"   - 结果表格: {table_file}")
        print(f"   - 评测报告: {output_dir / 'report.md'}")

        # 打印摘要
        print_summary(results)

        return results

    except Exception as e:
        print(f"\n❌ 评测失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_report(results: dict, output_dir: Path, config: dict):
    """生成详细的评测报告，包含所有配置信息。"""
    report_file = output_dir / "report.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# RULER 长文本评测报告\n\n")
        f.write(f"**评测时间:** {config['timestamp']}\n\n")

        # 模型配置
        f.write("## 模型配置\n\n")
        f.write(f"| 配置项 | 值 |\n")
        f.write(f"|--------|----|\n")
        f.write(f"| 模型 | {config['model']} |\n")
        f.write(f"| 上下文长度 | {config['context_length']} tokens |\n")
        f.write(f"| 滑动窗口大小 | {config['sliding_window']} |\n")
        f.write(f"| 保留前N个token | {config['keep_first']} |\n")
        f.write(f"| 启用Linear Memory | {config['enable_linear_mem']} |\n")
        f.write(f"| 非滑动层索引 | {config['non_sliding_layers']} |\n")
        f.write(f"| Flash Attention权重 | {config['flash_attn_weight']} |\n")
        f.write(f"| Linear Memory权重 | {config['linear_mem_weight']} |\n")
        f.write(f"| Linear Memory模式 | {config['linear_mem_mode']} |\n")
        f.write(f"| 设备 | {config['device']} |\n")
        f.write(f"| 数据类型 | {config['dtype']} |\n")
        f.write(f"| 注意力实现 | {config['attn_implementation']} |\n")
        f.write(f"| 批次大小 | {config['batch_size']} |\n")
        f.write(f"| 每任务样本数 | {config['limit']} |\n\n")

        # 评测结果
        if "results" in results:
            f.write("## 评测结果\n\n")
            f.write("| 任务 | 指标 | 分数 |\n")
            f.write("|------|------|------|\n")

            accuracies = []
            for task_name, task_results in results["results"].items():
                for metric, value in task_results.items():
                    if isinstance(value, (int, float)):
                        f.write(f"| {task_name} | {metric} | {value:.4f} |\n")
                        if "acc" in metric.lower():
                            accuracies.append((task_name, value))

            # 性能摘要
            if accuracies:
                avg_acc = sum(v for _, v in accuracies) / len(accuracies)
                f.write(f"\n### 性能摘要\n\n")
                f.write(f"**平均准确率:** {avg_acc:.4f} ({avg_acc*100:.2f}%)\n\n")

                # 性能分级
                if avg_acc >= 0.9:
                    f.write("✅ **优秀**: 模型在该上下文长度下表现卓越。\n\n")
                elif avg_acc >= 0.7:
                    f.write("✅ **良好**: 模型在该上下文长度下表现良好。\n\n")
                elif avg_acc >= 0.5:
                    f.write("⚠️ **中等**: 模型表现一般，有改进空间。\n\n")
                else:
                    f.write("❌ **需改进**: 模型在该上下文长度下表现较差。\n\n")

                # 各任务详情
                f.write("### 各任务表现\n\n")
                for task_name, acc in sorted(accuracies, key=lambda x: x[1], reverse=True):
                    status = "✅" if acc >= 0.8 else "⚠️" if acc >= 0.5 else "❌"
                    f.write(f"- {status} **{task_name}**: {acc:.4f} ({acc*100:.2f}%)\n")

        # 详细结果 JSON
        f.write("\n## 详细结果 (JSON)\n\n")
        f.write("```json\n")
        f.write(json.dumps(results.get("results", {}), indent=2, ensure_ascii=False))
        f.write("\n```\n")

        # 任务说明
        f.write("\n## RULER 任务说明\n\n")
        f.write("### NIAH (Needle In A Haystack)\n")
        f.write("测试在长文本中查找特定信息的能力\n")
        f.write("- **niah_single_1/2/3**: 单针检索，不同难度\n")
        f.write("- **niah_multikey_1**: 多键检索\n")
        f.write("- **niah_multivalue**: 多值检索\n")
        f.write("- **niah_multiquery**: 多查询检索\n\n")

        f.write("### Passkey\n")
        f.write("测试从长文本中回忆密码的能力\n\n")

        f.write("### Variable Tracking (VT)\n")
        f.write("测试在长代码中追踪变量赋值的能力\n\n")

        f.write("### Word Extraction\n")
        f.write("- **ruler_cwe**: 常见词提取\n")
        f.write("- **ruler_fwe**: 频繁词提取\n")

    print(f"   - 评测报告: {report_file}")


def print_summary(results: dict):
    """打印结果摘要。"""
    if "results" not in results:
        return

    print("\n" + "=" * 80)
    print("📊 结果摘要")
    print("=" * 80)

    accuracies = []
    for task_name, task_results in results["results"].items():
        for metric, value in task_results.items():
            if isinstance(value, (int, float)) and "acc" in metric.lower():
                accuracies.append((task_name, value))

    if accuracies:
        avg_acc = sum(v for _, v in accuracies) / len(accuracies)
        print(f"\n平均准确率: {avg_acc:.4f} ({avg_acc*100:.2f}%)")

        print("\n各任务表现:")
        for task_name, acc in sorted(accuracies, key=lambda x: x[1], reverse=True):
            bar_len = int(acc * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(f"  {task_name:20s} [{bar}] {acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="简化的 RULER 长文本评测脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础评测 (32K上下文)
  python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 32768

  # 禁用 linear memory
  python run_ruler.py --model Qwen/Qwen3-1.7B --context-length 16384 --no-linear-mem

  # 自定义滑动窗口和保留token
  python run_ruler.py --model Qwen/Qwen3-1.7B --sliding-window 4096 --keep-first 8

  # 指定非滑动层
  python run_ruler.py --model Qwen/Qwen3-1.7B --non-sliding-layers 0,1,2,3

  # 快速测试 (少量样本)
  python run_ruler.py --model Qwen/Qwen3-1.7B --limit 10
        """,
    )

    # 模型配置
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-1.7B",
        help="模型名称或路径 (默认: Qwen/Qwen3-1.7B)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="设备 (默认: cuda:0)",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="数据类型 (默认: bfloat16)",
    )
    parser.add_argument(
        "--attn",
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="注意力实现 (默认: flash_attention_2)",
    )
    parser.add_argument(
        "--batch-size",
        type=str,
        default="auto",
        help="批次大小，支持 'auto' 自动检测或具体数字 (默认: auto)",
    )

    # RULER 配置
    parser.add_argument(
        "--context-length",
        type=int,
        default=32768,
        help="上下文长度 (默认: 32768)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="每任务样本数 (默认: 100)",
    )

    # SWAA / Linear Memory 配置
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=2048,
        help="滑动窗口大小 (默认: 2048)",
    )
    parser.add_argument(
        "--keep-first",
        type=int,
        default=4,
        help="保留前N个token (默认: 4)",
    )
    parser.add_argument(
        "--enable-linear-mem",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=True,
        help="启用 Linear Memory 操作 (默认: True)",
    )
    parser.add_argument(
        "--no-linear-mem",
        action="store_true",
        help="禁用 Linear Memory 操作",
    )
    parser.add_argument(
        "--non-sliding-layers",
        type=str,
        default="",
        help="不使用滑动注意力的层索引，逗号分隔 (如: 0,1,2,3)",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2048,
        help="处理长序列的最大块大小 (默认: 2048)",
    )
    parser.add_argument(
        "--flash-attn-weight",
        type=float,
        default=0.9,
        help="混合注意力中 flash attention 输出的权重 (默认: 0.9)",
    )
    parser.add_argument(
        "--linear-mem-weight",
        type=float,
        default=0.1,
        help="混合注意力中 linear memory 输出的权重 (默认: 0.1)",
    )
    parser.add_argument(
        "--linear-mem-mode",
        type=str,
        default="fused_recurrent",
        choices=["fused_recurrent", "fused_chunk", "chunk"],
        help="Linear Memory 操作模式 (默认: fused_recurrent)",
    )

    # 输出配置
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录 (默认: eval/results)",
    )

    args = parser.parse_args()

    # 处理 no-linear-mem 标志
    if args.no_linear_mem:
        args.enable_linear_mem = False

    # 解析 non-sliding-layers（转换为元组以保证可哈希性）
    if args.non_sliding_layers:
        args.non_sliding_layers = tuple(int(x.strip()) for x in args.non_sliding_layers.split(",") if x.strip())
    else:
        args.non_sliding_layers = tuple()  # 使用空元组而不是空列表

    # 运行评测
    run_ruler_evaluation(args)


if __name__ == "__main__":
    main()
