#!/usr/bin/env python3
"""
Test script for Qwen3-1.7B model with SWAA (Sliding Window Attention Adaptation).

This script demonstrates how to:
1. Patch transformers with SWAA support
2. Load Qwen3-1.7B model with flash_attention_2
3. Configure sliding window attention
4. Run inference with the model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from swaa_patch import SWAAConfig, hack_hf_swaa,hack_kv_cache_recurrent_state
from swaa_patch.kernel.AnchorKernel import AnchorKernel
from swaa_patch.kernel.EluKernel import EluKernel
from swaa_patch.kernel.SoftplusKernel import GatedTopkSoftplusKernel, PowTopkSoftplusKernel
from swaa_patch.kernel.NIAHKernel import PositionAwareKernel, DenseQueryKernel, RarityEnhancedKernel
import math
from pathlib import Path
from typing import Union


def case_text(file_path: str) -> str:
    """
    Read text content from a test case file.

    Args:
        file_path: Path to the test case file (relative or absolute)

    Returns:
        The content of the file as a string

    Example:
        >>> test_prompts = [
        ...     "你好，你是谁?",
        ...     case_text("TEST_CASE/niah_single_1.txt"),
        ... ]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test case file not found: {file_path}")

    with open(path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    return content


def parse_test_prompt(prompt: Union[str, dict]) -> tuple[str, str]:
    """
    Parse a test prompt and return (display_prompt, actual_prompt).

    Args:
        prompt: Can be a string or a dict with 'display' and 'content' keys

    Returns:
        Tuple of (display_text, actual_content)
    """
    if isinstance(prompt, dict):
        return prompt.get('display', prompt.get('content', '')), prompt.get('content', '')
    return prompt, prompt

def main():
    # =========================================================================
    # 1. SWAA Patch Setup
    # =========================================================================
    # Apply SWAA patch to transformers before loading the model
    # training=False means we're using the model for inference
    hack_kv_cache_recurrent_state()
    hack_hf_swaa(training=False)

    # =========================================================================
    # 2. Model Configuration
    # =========================================================================
    model_name = "Qwen/Qwen3-1.7B"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print("-" * 50)

    # =========================================================================
    # 3. Load Tokenizer and Model
    # =========================================================================
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    ).eval()

    # =========================================================================
    # 4. Configure SWAA
    # =========================================================================

    num_layers = model.config.num_hidden_layers
    non_sliding_layers = []  # Every other layer uses full attention

    swaa_config = SWAAConfig(
        sliding_window_size=2048,
        keep_first=64,
        force_fa_decode=False,
        non_sliding_layers=[],
        enable_linear_mem=True,
        flash_attn_weight=0.8,
        linear_mem_weight=0.2,
        linear_mem_mode="fused_chunk",
        linear_mem_blend_mode="orth_match",

    )

    linear_kernel_k = PositionAwareKernel(
    head_dim=model.config.head_dim, topk=None,
    device=device,
    dtype = torch.bfloat16
    )
    linear_kernel_q = DenseQueryKernel(
    head_dim=model.config.head_dim, topk=None, gamma = 2.0,
    device=device,
    dtype = torch.bfloat16,
    normalize=True,
    )

    ######
    # linear_kernel_k = GatedTopkSoftplusKernel(
    # head_dim=model.config.head_dim, topk=None,
    # device=device,
    # dtype = torch.bfloat16
    # )
    # linear_kernel_q = PowTopkSoftplusKernel(
    # head_dim=model.config.head_dim, topk=20, gamma = 5.0,
    # device=device,
    # dtype = torch.bfloat16,
    # normalize=True,
    # )


    # Attach SWAA config to model config
    model.config.swaa_config = swaa_config
    model.config.kernel_k = linear_kernel_k
    model.config.kernel_q = linear_kernel_q
    print(f"SWAA Config:")
    print(f"  - sliding_window_size: {swaa_config.sliding_window_size}")
    print(f"  - keep_first: {swaa_config.keep_first}")
    print(f"  - force_fa_decode: {swaa_config.force_fa_decode}")
    print(f"  - non_sliding_layers: {swaa_config.non_sliding_layers}")
    print(f"  - enable_linear_mem: {swaa_config.enable_linear_mem}")
    print(f"  - flash_attn_weight: {swaa_config.flash_attn_weight}")
    print(f"  - linear_mem_weight: {swaa_config.linear_mem_weight}")
    print(f"  - linear_mem_blend_mode: {swaa_config.linear_mem_blend_mode}")
    print(f"  - mark: {swaa_config.mark}")
    print("-" * 50)

    # =========================================================================
    # 5. Test Inference
    # =========================================================================
    # Example: Mix string prompts and file-based prompts
    # For long content from files, you can use dict format to provide a short display name
    test_prompts = [
        "你好，你是谁?",
        "法国的首都是哪里？",
        # "解释线性注意力机制。",
        case_text("TEST_CASE/niah_single_1.txt")
        # Example: Using file content with display name
        # {
        #     "display": "[文件] TEST_CASE/niah_single_1.txt",
        #     "content": case_text("TEST_CASE/niah_single_1.txt")
        # },
        # Or simply: case_text("TEST_CASE/niah_single_1.txt"),
    ]

    for i, prompt_item in enumerate(test_prompts):
        # Parse the prompt (handle both string and dict format)
        display_prompt, actual_prompt = parse_test_prompt(prompt_item)

        # Apply Qwen3 chat template with no thinking mode
        messages = [{"role": "user", "content": actual_prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        print(f"\n[Test {i+1}]")
        # Display short description (truncate if too long)
        if len(display_prompt) > 100:
            print(f"Prompt: {display_prompt[:100]}...")
        else:
            print(f"Prompt: {display_prompt}")
        print("-" * 30)

        inputs = tokenizer(text, return_tensors="pt").to(device)
        input_length = inputs['input_ids'].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                num_beams=1,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=DynamicCache(),
            )

        # Decode only the generated part (skip input tokens)
        generated_tokens = outputs[0][input_length:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        print(f"Response: {response}")
        print("-" * 30)

    # =========================================================================
    # 6. Memory Usage Summary
    # =========================================================================
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        print(f"\n[Memory Usage]")
        print(f"  - Allocated: {allocated:.2f} GB")
        print(f"  - Reserved: {reserved:.2f} GB")

    print("\n[Done] All tests completed successfully!")


if __name__ == "__main__":
    main()
