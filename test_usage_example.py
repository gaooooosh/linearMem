#!/usr/bin/env python3
"""
使用示例：展示如何在 test.py 中使用 case_text() 函数

这个文件演示了三种使用方式：
1. 直接使用字符串
2. 使用 case_text() 读取文件内容
3. 使用字典格式提供自定义显示名称（适合长文本）
"""

from test import case_text

# 方式1: 直接字符串
test_prompts_simple = [
    "你好，你是谁?",
    "法国的首都是哪里？",
]

# 方式2: 使用 case_text() 读取文件
test_prompts_with_file = [
    "你好，你是谁?",
    case_text("TEST_CASE/niah_single_1.txt"),  # 读取文件内容
]

# 方式3: 使用字典格式，为长文本提供简短的显示名称
test_prompts_advanced = [
    "你好，你是谁?",
    {
        "display": "[大海捞针测试] niah_single_1.txt",
        "content": case_text("TEST_CASE/niah_single_1.txt")
    },
]

if __name__ == "__main__":
    print("=" * 60)
    print("示例1: 简单字符串")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts_simple, 1):
        print(f"{i}. {prompt}")

    print("\n" + "=" * 60)
    print("示例2: 混合使用字符串和文件")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts_with_file, 1):
        if len(str(prompt)) > 100:
            print(f"{i}. {str(prompt)[:100]}... [已截断]")
        else:
            print(f"{i}. {prompt}")

    print("\n" + "=" * 60)
    print("示例3: 使用字典格式提供显示名称")
    print("=" * 60)
    for i, prompt in enumerate(test_prompts_advanced, 1):
        if isinstance(prompt, dict):
            print(f"{i}. {prompt['display']}")
        else:
            print(f"{i}. {prompt}")

    print("\n✅ 所有格式都可以直接用在 test.py 的 test_prompts 列表中！")
