"""
SWAA Patch - Sliding Window Attention Adaptation Patches

This package provides monkey patches for:
1. HuggingFace Transformers models to support SWAA (Sliding Window Attention Adaptation)
2. vLLM inference engine for SWAA support (optional)
3. KV Cache layers to support linear attention recurrent state

Quick Start:
    from swaa_patch import hack_hf_swaa, hack_kv_cache_recurrent_state, SWAAConfig

    # Apply patches
    hack_hf_swaa(training=False)
    hack_kv_cache_recurrent_state()

    # Configure SWAA
    swaa_config = SWAAConfig(
        sliding_window_size=2048,
        keep_first=64,
        force_fa_decode=True,
        non_sliding_layers=[0, 2, 4, 6],
    )
    model.config.swaa_config = swaa_config

For more details, see:
- README_KV_CACHE.md for KV Cache recurrent state usage
"""

# 1. Import Configuration

from .swaa_config import SWAAConfig


# 2. Import Hugging Face Hack
# This module depends on transformers/torch.
from .hack_hf_swaa import hack_hf_swaa


# 3. Import vLLM Hack (Optional Dependency)
# This module strictly depends on vllm. We wrap it in a try-except block
# to prevent the package from crashing if the user does not have vllm installed.
try:
    from .hack_vllm_0110_swaa import hack_vllm_swaa, LLMSWAA
except ImportError:
    # If vllm is not installed (or compatible), define dummy objects.
    # These will only raise errors if the user explicitly tries to use them.

    def hack_vllm_swaa(*args, **kwargs):
        raise ImportError(
            "Cannot call 'hack_vllm_swaa'. 'vllm' does not appear to be installed "
            "in your environment, or the version is incompatible.\n"
            "Please ensure vllm is installed to use this feature."
        )


# 4. Import KV Cache Hack for Linear Attention Recurrent State
# This module patches transformers KV Cache to support recurrent state.
from .hack_kv_cache import (
    hack_kv_cache_recurrent_state,
    unhack_kv_cache_recurrent_state,
)


__all__ = [
    "hack_hf_swaa",
    "hack_vllm_swaa",
    "hack_kv_cache_recurrent_state",
    "unhack_kv_cache_recurrent_state",
    "SWAAConfig"
]