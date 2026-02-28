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
    device = "cuda:7" if torch.cuda.is_available() else "cpu"

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
    # SWAA Configuration for Qwen3-1.7B (28 layers)
    # - sliding_window_size: 2048 tokens for sliding window
    # - keep_first: 64 sink tokens at the beginning
    # - force_fa_decode: Use full attention during decoding
    # - non_sliding_layers: Layers that use full attention (every other layer)
    num_layers = model.config.num_hidden_layers
    non_sliding_layers = list(range(0, num_layers, 2))  # Every other layer uses full attention

    swaa_config = SWAAConfig(
        sliding_window_size=2048,
        keep_first=64,
        force_fa_decode=True,
        non_sliding_layers=non_sliding_layers,
    )

    # Attach SWAA config to model config
    model.config.swaa_config = swaa_config

    print(f"SWAA Config:")
    print(f"  - sliding_window_size: {swaa_config.sliding_window_size}")
    print(f"  - keep_first: {swaa_config.keep_first}")
    print(f"  - force_fa_decode: {swaa_config.force_fa_decode}")
    print(f"  - non_sliding_layers: {swaa_config.non_sliding_layers}")
    print(f"  - mark: {swaa_config.mark}")
    print("-" * 50)

    # =========================================================================
    # 5. Test Inference
    # =========================================================================
    test_prompts = [
        "Hello, who are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
    ]

    for i, prompt in enumerate(test_prompts):
        print(f"\n[Test {i+1}]")
        print(f"Prompt: {prompt}")
        print("-" * 30)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                past_key_values=DynamicCache(),
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
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
