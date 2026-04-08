#!/usr/bin/env python3
"""
Custom lm-eval model wrapper for SWAA (Sliding Window Attention Adaptation) models.

This wrapper allows lm-evaluation-harness to evaluate models with SWAA configuration,
supporting various attention mechanisms and cache strategies.
"""

import os
import sys
import json
import ast
import argparse
import importlib
import inspect
from pathlib import Path

# Add project root to sys.path for importing swaa_patch
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from datetime import datetime
from typing import Optional, List, Tuple
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DynamicCache
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import simple_evaluate
from lm_eval.utils import handle_non_serializable
from swaa_patch import SWAAConfig, hack_hf_swaa, hack_kv_cache_recurrent_state
from swaa_patch.kernel.AnchorKernel import AnchorKernel
from swaa_patch.kernel.NIAHKernel import PositionAwareKernel, DenseQueryKernel
from swaa_patch.kernel.SoftplusKernel import GatedTopkSoftplusKernel, PowTopkSoftplusKernel
# Global log file path
EVAL_LOG_FILE = Path(__file__).parent.parent / "evaluation.log"
SWAA_DECODE_NORM_STATE_KEY = "linear_k_sum_state"
SWAA_DECODE_NORM_SCOPE = "cached_decode"
_REQUIRED_SWAA_CACHE_METHODS = (
    "state_update",
    "is_recurrent_state_initialized",
    "get_recurrent_state",
    "linear_k_sum_state_update",
    "is_linear_k_sum_state_initialized",
    "get_linear_k_sum_state",
)
_ORIG_SWAA_BLEND_LINEAR_OUTPUT = None


def _init_log_file():
    """Initialize a fresh log file at the start of evaluation."""
    with open(EVAL_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"{'='*80}\n")
        f.write(f"Evaluation Log - Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*80}\n\n")


def _log_sample(method: str, idx: int, input_len: int, output: str, extra_info: str = ""):
    """Log a single sample's information."""
    with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{method}] Sample #{idx}\n")
        f.write(f"  Input Length: {input_len} tokens\n")
        if extra_info:
            f.write(f"  {extra_info}\n")
        f.write(f"  Output: {output[:500]}{'...' if len(output) > 500 else ''}\n")
        f.write(f"  {'-'*60}\n")


def _parse_bool(value) -> bool:
    """Parse common bool-like values passed through CLI or lm-eval model_args."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return bool(value)


def _parse_int_list(value) -> List[int]:
    """Parse ints from Python-list strings, tuples, or comma/space separated strings."""
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, int):
            return [parsed]
        if isinstance(parsed, (list, tuple)):
            return [int(item) for item in parsed]

        cleaned = stripped.replace(",", " ")
        return [int(item) for item in cleaned.split() if item]

    raise TypeError(f"Unsupported list value type: {type(value)!r}")


def _parse_float_dict(value) -> dict[int, float]:
    """Parse {layer: beta} maps from dicts, Python literals, or k:v CSV strings."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return {int(key): float(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        out = {}
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError(f"Unsupported beta_by_layer entry: {item!r}")
            out[int(item[0])] = float(item[1])
        return out
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return {}
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None

        if isinstance(parsed, dict):
            return {int(key): float(val) for key, val in parsed.items()}
        if isinstance(parsed, (list, tuple)):
            return _parse_float_dict(parsed)

        out = {}
        for piece in stripped.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if ":" not in piece:
                raise ValueError(
                    "beta_by_layer must look like '0:0.08,18:0.16' or '{0: 0.08, 18: 0.16}'"
                )
            key, val = piece.split(":", 1)
            out[int(key.strip())] = float(val.strip())
        return out
    raise TypeError(f"Unsupported beta_by_layer type: {type(value)!r}")


def _parse_force_fa_decode(value) -> bool | List[int]:
    """
    force_fa_decode supports either a global bool or a per-layer list in SWAAConfig.
    """
    if isinstance(value, (list, tuple)):
        parsed = _parse_int_list(value)
        return parsed if parsed else False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        parsed = _parse_int_list(value)
        return parsed if parsed else False
    return bool(value)


def _current_self_attn():
    """Best-effort lookup of the current attention module from the Python stack."""
    frame = inspect.currentframe()
    try:
        caller = frame.f_back if frame is not None else None
        while caller is not None:
            self_obj = caller.f_locals.get("self")
            if self_obj is not None and hasattr(self_obj, "layer_idx"):
                return self_obj
            caller = caller.f_back
        return None
    finally:
        del frame


def _norm_match_mem(o_base: torch.Tensor, o_mem: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    base_f = o_base.float()
    mem_f = o_mem.float()
    base_norm = base_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    mem_norm = mem_f.norm(dim=-1, keepdim=True).clamp_min(eps)
    return mem_f * (base_norm / mem_norm)


def make_layer_beta_blend(
    *,
    beta: float | None,
    beta_by_layer: dict[int, float] | None,
    boundary: int,
):
    default_beta = float(beta or 0.0)
    beta_map = {int(k): float(v) for k, v in (beta_by_layer or {}).items()}

    def resolve_beta(layer_idx: int) -> float:
        if beta_map:
            return float(beta_map.get(layer_idx, 0.0))
        return default_beta

    def blend(o_base, o_mem, *, layer_idx: int):
        layer_beta = resolve_beta(layer_idx)
        if layer_beta <= 0.0:
            return o_base

        base_f = o_base.float()
        mem_f = o_mem.float()
        seq_len = base_f.shape[-2] if base_f.ndim == 3 else 1
        if seq_len > 1:
            positions = torch.arange(seq_len, device=base_f.device).float()
            gate = (positions >= boundary).float()
            gate = gate.view(1, seq_len, 1) if base_f.ndim == 3 else gate.view(seq_len, 1)
        else:
            gate = 1.0

        mem_scaled = _norm_match_mem(base_f, mem_f)
        return (base_f + layer_beta * gate * mem_scaled).to(o_base.dtype)

    return blend


def _normalize_stop_sequences(value) -> List[str]:
    """Normalize lm-eval/HF stop specifications into a flat string list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _coerce_gen_kwargs(value) -> dict:
    """
    lm-eval passes generate_until requests as `(context, gen_kwargs)`.
    Older local callers may still pass `(context, until)`, so keep that path working.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    return {"until": value}


def _get_head_dim(config) -> int:
    """Infer head_dim for configs that do not expose it directly."""
    head_dim = getattr(config, "head_dim", None)
    if head_dim is not None:
        return head_dim

    hidden_size = getattr(config, "hidden_size", None)
    num_attention_heads = getattr(config, "num_attention_heads", None)
    if hidden_size is None or num_attention_heads is None:
        raise AttributeError("Unable to infer head_dim from model config.")
    return hidden_size // num_attention_heads


def try_get_model_head_dim(pretrained: str) -> Optional[int]:
    """Best-effort head_dim inference for config logging without loading model weights."""
    try:
        config = AutoConfig.from_pretrained(pretrained, trust_remote_code=True)
    except Exception:
        return None

    try:
        return _get_head_dim(config)
    except AttributeError:
        return None


def _serialize_runtime_value(value):
    """Convert runtime-only objects into JSON-friendly values."""
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    if isinstance(value, torch.device):
        return str(value)
    return value


def _validate_swaa_decode_norm_patch() -> dict:
    """
    Ensure eval uses the patched DynamicCache that carries decode norm-correction state.
    """
    cache = DynamicCache()
    missing_cache_methods = [name for name in _REQUIRED_SWAA_CACHE_METHODS if not hasattr(cache, name)]
    if missing_cache_methods:
        raise RuntimeError(
            "eval_swaa_model.py requires the current SWAA decode norm correction patch, "
            f"but DynamicCache is missing methods: {missing_cache_methods}. "
            "Expected `hack_kv_cache_recurrent_state()` to patch "
            f"`{SWAA_DECODE_NORM_STATE_KEY}` support before evaluation."
        )

    return {
        "decode_norm_correction": True,
        "decode_norm_state_key": SWAA_DECODE_NORM_STATE_KEY,
        "decode_norm_scope": SWAA_DECODE_NORM_SCOPE,
        "required_cache_methods": list(_REQUIRED_SWAA_CACHE_METHODS),
    }


def _resolve_linear_kernel_defs(
    kernel_family: str,
    num_anchors: int,
    tau: float,
) -> tuple[Optional[dict], Optional[dict], str]:
    """Central source of truth for q/k kernel classes and fixed hyperparameters."""
    family = kernel_family.lower()

    if family == "softplus":
        kernel_q_def = {
            "class": PowTopkSoftplusKernel,
            "class_name": "PowTopkSoftplusKernel",
            "constructor_kwargs": {
                "topk": 20,
                "gamma": 5.0,
                "normalize": True,
            },
        }
        kernel_k_def = {
            "class": GatedTopkSoftplusKernel,
            "class_name": "GatedTopkSoftplusKernel",
            "constructor_kwargs": {
                "topk": None,
            },
        }
        return kernel_q_def, kernel_k_def, family

    if family == "niah":
        kernel_q_def = {
            "class": DenseQueryKernel,
            "class_name": "DenseQueryKernel",
            "constructor_kwargs": {
                "topk": None,
                "gamma": 2.0,
                "normalize": True,
            },
        }
        kernel_k_def = {
            "class": PositionAwareKernel,
            "class_name": "PositionAwareKernel",
            "constructor_kwargs": {
                "topk": None,
            },
        }
        return kernel_q_def, kernel_k_def, family

    if family == "anchor":
        anchor_kwargs = {
            "num_anchors": num_anchors,
            "tau": tau,
        }
        kernel_q_def = {
            "class": AnchorKernel,
            "class_name": "AnchorKernel",
            "constructor_kwargs": dict(anchor_kwargs),
        }
        kernel_k_def = {
            "class": AnchorKernel,
            "class_name": "AnchorKernel",
            "constructor_kwargs": dict(anchor_kwargs),
        }
        return kernel_q_def, kernel_k_def, family

    if family in {"none", "identity", "raw"}:
        return None, None, family

    raise ValueError(
        f"Unsupported linear_kernel_family: {kernel_family}. "
        "Expected one of: softplus, niah, anchor, none."
    )


def resolve_linear_kernel_specs(
    kernel_family: str,
    num_anchors: int,
    tau: float,
    *,
    head_dim: Optional[int] = None,
    device=None,
    dtype=None,
) -> tuple[dict, dict]:
    """Return JSON-friendly specs for the effective q/k kernels."""
    kernel_q_def, kernel_k_def, family = _resolve_linear_kernel_defs(
        kernel_family=kernel_family,
        num_anchors=num_anchors,
        tau=tau,
    )

    shared_kwargs = {
        "head_dim": head_dim,
        "device": _serialize_runtime_value(device),
        "dtype": _serialize_runtime_value(dtype),
    }

    def build_spec(role: str, kernel_def: Optional[dict]) -> dict:
        if kernel_def is None:
            return {
                "role": role,
                "family": family,
                "class_name": None,
                "constructor_kwargs": dict(shared_kwargs),
            }

        return {
            "role": role,
            "family": family,
            "class_name": kernel_def["class_name"],
            "constructor_kwargs": {
                **shared_kwargs,
                **{
                    key: _serialize_runtime_value(value)
                    for key, value in kernel_def["constructor_kwargs"].items()
                },
            },
        }

    return build_spec("q", kernel_q_def), build_spec("k", kernel_k_def)


def _instantiate_linear_kernel(kernel_def: Optional[dict], head_dim: int, device, dtype):
    """Instantiate a single linear-memory kernel from its resolved definition."""
    if kernel_def is None:
        return None

    constructor_kwargs = {
        "head_dim": head_dim,
        **kernel_def["constructor_kwargs"],
        "device": device,
        "dtype": dtype,
    }
    return kernel_def["class"](**constructor_kwargs)


def _build_linear_kernels(
    kernel_family: str,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    num_anchors: int,
    tau: float,
):
    """Build the q/k feature-map modules expected by hack_hf_swaa.py."""
    kernel_q_def, kernel_k_def, _ = _resolve_linear_kernel_defs(
        kernel_family=kernel_family,
        num_anchors=num_anchors,
        tau=tau,
    )
    kernel_q = _instantiate_linear_kernel(kernel_q_def, head_dim=head_dim, device=device, dtype=dtype)
    kernel_k = _instantiate_linear_kernel(kernel_k_def, head_dim=head_dim, device=device, dtype=dtype)
    return kernel_q, kernel_k


@register_model("swaa_hf")
class SWAAHFLM(LM):
    """
    HuggingFace model with SWAA support for lm-evaluation-harness.

    This class wraps a HuggingFace CausalLM with SWAA configuration,
    allowing evaluation of sliding window attention models.
    """

    def __init__(
        self,
        pretrained: str,
        device: str = "cuda:0",
        torch_dtype: str = "bfloat16",
        attn_implementation: str = "flash_attention_2",
        # SWAA Configuration
        sliding_window_size: int = 2048,
        keep_first: int = 4,
        force_fa_decode: bool = False,
        non_sliding_layers: Optional[List[int]] = None,
        enable_linear_mem: bool = True,
        flash_attn_weight: float = 0.9,
        linear_mem_weight: float = 0.1,
        linear_mem_mode: str = "fused_recurrent",
        linear_mem_blend_mode: str = "raw",
        linear_kernel_family: str = "softplus",
        active_layers: Optional[List[int] | str] = None,
        beta_by_layer: Optional[dict | str] = None,
        # Anchor kernel configuration
        num_anchors: int = 64,
        tau: float = 20.0,
        force_fa_decode_layers: Optional[List[int]] = None,
        # Generation config
        batch_size: Optional[int] = None,
        max_length: int = 4096,
        # Memory optimization for long sequences
        max_chunk_size: int = 2048,
        **kwargs,
    ):
        # Handle batch_size from kwargs (lm-eval may pass it)
        batch_size = batch_size or kwargs.pop("batch_size", 1)

        if isinstance(batch_size, str) and batch_size.isdigit():
            batch_size = int(batch_size)

        non_sliding_layers = _parse_int_list(non_sliding_layers)
        force_fa_decode_layers = _parse_int_list(force_fa_decode_layers)
        force_fa_decode = _parse_force_fa_decode(force_fa_decode)
        if force_fa_decode_layers:
            force_fa_decode = force_fa_decode_layers
        active_layers = _parse_int_list(active_layers)
        beta_by_layer = _parse_float_dict(beta_by_layer)
        if not active_layers and beta_by_layer:
            active_layers = sorted(beta_by_layer)

        enable_linear_mem = _parse_bool(enable_linear_mem)
        num_anchors = int(num_anchors)
        tau = float(tau)

        # Call parent init with batch_size
        super().__init__()

        # Apply SWAA patches
        hack_kv_cache_recurrent_state()
        hack_hf_swaa(training=False)
        self.swaa_runtime = _validate_swaa_decode_norm_patch()

        # Device setup
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.torch_dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self.batch_size = batch_size
        self.max_length = max_length
        self.max_chunk_size = max_chunk_size

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            trust_remote_code=True,
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            device_map={"": self.device},
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).eval()

        # Configure SWAA
        self.swaa_config = SWAAConfig(
            sliding_window_size=sliding_window_size,
            keep_first=keep_first,
            force_fa_decode=force_fa_decode,
            non_sliding_layers=non_sliding_layers,
            enable_linear_mem=enable_linear_mem,
            flash_attn_weight=flash_attn_weight,
            linear_mem_weight=linear_mem_weight,
            linear_mem_mode=linear_mem_mode,
            linear_mem_blend_mode=linear_mem_blend_mode,
        )
        self.model.config.swaa_config = self.swaa_config
        self.active_layers = list(active_layers)
        self.beta_by_layer = dict(beta_by_layer)
        self._install_layer_selective_linear_mem(
            enable_linear_mem=enable_linear_mem,
            active_layers=self.active_layers,
            beta=linear_mem_weight,
            beta_by_layer=self.beta_by_layer,
            sliding_window_size=sliding_window_size,
            keep_first=keep_first,
        )

        head_dim = _get_head_dim(self.model.config)
        self.kernel_q_spec, self.kernel_k_spec = resolve_linear_kernel_specs(
            kernel_family=linear_kernel_family,
            num_anchors=num_anchors,
            tau=tau,
            head_dim=head_dim,
            device=self.device,
            dtype=self.torch_dtype,
        )

        if enable_linear_mem:
            linear_kernel_q, linear_kernel_k = _build_linear_kernels(
                kernel_family=linear_kernel_family,
                head_dim=head_dim,
                device=self.device,
                dtype=self.torch_dtype,
                num_anchors=num_anchors,
                tau=tau,
            )
        else:
            linear_kernel_q, linear_kernel_k = None, None
        self.model.config.kernel_q = linear_kernel_q
        self.model.config.kernel_k = linear_kernel_k


        print(f"\n{'='*60}")
        print(f"SWAA Model Loaded: {pretrained}")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Dtype: {self.torch_dtype}")
        print(f"Attention: {attn_implementation}")
        print(f"\nSWAA Configuration:")
        print(f"  - sliding_window_size: {sliding_window_size}")
        print(f"  - keep_first: {keep_first}")
        print(f"  - force_fa_decode: {force_fa_decode}")
        print(f"  - non_sliding_layers: {non_sliding_layers}")
        print(f"  - enable_linear_mem: {enable_linear_mem}")
        print(f"  - flash_attn_weight: {flash_attn_weight}")
        print(f"  - linear_mem_weight: {linear_mem_weight}")
        print(f"  - linear_mem_mode: {linear_mem_mode}")
        print(f"  - linear_mem_blend_mode: {linear_mem_blend_mode}")
        print(f"  - active_layers: {self.active_layers}")
        print(f"  - beta_by_layer: {self.beta_by_layer}")
        print(f"  - decode_norm_correction: {self.swaa_runtime['decode_norm_correction']}")
        print(f"  - decode_norm_state_key: {self.swaa_runtime['decode_norm_state_key']}")
        print(f"  - decode_norm_scope: {self.swaa_runtime['decode_norm_scope']}")
        print(f"  - mark: {self.swaa_config.mark}")
        print(f"\nLinear Kernel Configuration:")
        print(f"  - linear_kernel_family: {linear_kernel_family}")
        print(f"  - num_anchors: {num_anchors}")
        print(f"  - tau: {tau}")
        print(f"  - kernel_q: {self.kernel_q_spec}")
        print(f"  - kernel_k: {self.kernel_k_spec}")
        print(f"{'='*60}\n")

        # Initialize fresh evaluation log
        _init_log_file()

    def _install_layer_selective_linear_mem(
        self,
        *,
        enable_linear_mem: bool,
        active_layers: List[int],
        beta: float,
        beta_by_layer: dict[int, float],
        sliding_window_size: int,
        keep_first: int,
    ) -> None:
        global _ORIG_SWAA_BLEND_LINEAR_OUTPUT

        swaa_mod = importlib.import_module("swaa_patch.hack_hf_swaa")
        if _ORIG_SWAA_BLEND_LINEAR_OUTPUT is None:
            _ORIG_SWAA_BLEND_LINEAR_OUTPUT = swaa_mod.blend_linear_output
        else:
            swaa_mod.blend_linear_output = _ORIG_SWAA_BLEND_LINEAR_OUTPUT

        if not enable_linear_mem:
            return

        active_set = set(int(layer_idx) for layer_idx in active_layers) if active_layers else None
        beta_map = {int(k): float(v) for k, v in beta_by_layer.items()}
        if active_set is None and not beta_map:
            return

        boundary = sliding_window_size + keep_first
        custom_blend = make_layer_beta_blend(
            beta=beta,
            beta_by_layer=beta_map,
            boundary=boundary,
        )

        def patched_blend(
            o_base,
            o_mem,
            flash_attn_weight=1.0,
            linear_mem_weight=0.1,
            blend_mode="raw",
            eps=1e-6,
        ):
            del flash_attn_weight, linear_mem_weight, blend_mode, eps
            self_attn = _current_self_attn()
            layer_idx = int(self_attn.layer_idx) if self_attn is not None and hasattr(self_attn, "layer_idx") else -1
            return custom_blend(o_base, o_mem, layer_idx=layer_idx)

        swaa_mod.blend_linear_output = patched_blend

        if active_set is None:
            return

        for layer_idx, decoder_layer in enumerate(self.model.model.layers):
            attn = decoder_layer.self_attn
            orig_forward = attn.forward

            def layer_forward(
                *args,
                __orig=orig_forward,
                __attn=attn,
                __layer_idx=layer_idx,
                **kwargs,
            ):
                if __layer_idx not in active_set:
                    prev = __attn.config.swaa_config.enable_linear_mem
                    __attn.config.swaa_config.enable_linear_mem = False
                    try:
                        return __orig(*args, **kwargs)
                    finally:
                        __attn.config.swaa_config.enable_linear_mem = prev
                return __orig(*args, **kwargs)

            attn.forward = layer_forward

    def tok_encode(self, string: str) -> List[int]:
        """Encode string to token ids."""
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: List[int]) -> str:
        """Decode token ids to string."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _create_cache(self):
        """Create a new cache instance for each generation."""
        cache = DynamicCache()
        missing_cache_methods = [name for name in _REQUIRED_SWAA_CACHE_METHODS if not hasattr(cache, name)]
        if missing_cache_methods:
            raise RuntimeError(
                "DynamicCache lost required SWAA decode norm correction methods during evaluation: "
                f"{missing_cache_methods}"
            )
        return cache

    def _model_call(
        self,
        inps: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            inps: Input token ids [batch_size, seq_len]
            attention_mask: Attention mask
            past_key_values: KV cache for incremental decoding

        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        with torch.no_grad():
            outputs = self.model(
                input_ids=inps,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return outputs.logits

    def _model_generate(
        self,
        context: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
    ) -> str:
        """
        Generate text given a context.

        Args:
            context: Input text
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter

        Returns:
            Generated text
        """
        inputs = self.tokenizer(context, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
                pad_token_id=self.tokenizer.eos_token_id,
                past_key_values=self._create_cache(),
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        """
        Compute log-likelihood of continuations given contexts.

        Args:
            requests: List of Instance objects with arguments = (context, continuation)

        Returns:
            List of (log_likelihood, is_greedy) tuples
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="loglikelihood", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            context, continuation = request.arguments

            # Tokenize
            context_tokens = self.tok_encode(context)
            continuation_tokens = self.tok_encode(continuation)

            # Full sequence
            full_tokens = context_tokens + continuation_tokens
            input_ids = torch.tensor([full_tokens], device=self.device)

            # Get logits
            with torch.no_grad():
                logits = self._model_call(input_ids)

            # Compute log likelihood for continuation tokens
            cont_start = len(context_tokens)
            cont_logits = logits[0, cont_start - 1 : -1, :]  # [cont_len, vocab_size]
            cont_tokens_tensor = torch.tensor(continuation_tokens, device=self.device)

            # Calculate log probabilities
            log_probs = torch.nn.functional.log_softmax(cont_logits, dim=-1)
            token_log_probs = log_probs[range(len(continuation_tokens)), cont_tokens_tensor]

            # Sum log probabilities
            total_log_prob = token_log_probs.sum().item()

            # Check if greedy decoding would produce the continuation
            greedy_tokens = cont_logits.argmax(dim=-1)
            is_greedy = (greedy_tokens == cont_tokens_tensor).all().item()

            results.append((total_log_prob, is_greedy))

            # Log sample
            _log_sample(
                method="loglikelihood",
                idx=idx,
                input_len=len(full_tokens),
                output=f"log_prob={total_log_prob:.4f}, is_greedy={is_greedy}",
                extra_info=f"Context: {len(context_tokens)} tokens | Continuation: {len(continuation_tokens)} tokens"
            )

        return results

    def loglikelihood_rolling(self, requests) -> List[float]:
        """
        Compute log-likelihood of sequences (for perplexity evaluation).

        Args:
            requests: List of Instance objects with arguments = (sequence,)

        Returns:
            List of log-likelihoods
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="loglikelihood_rolling", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            (sequence,) = request.arguments

            tokens = self.tok_encode(sequence)

            # Chunked processing to avoid OOM on long sequences
            total_log_prob = 0.0
            chunk_size = self.max_chunk_size

            for chunk_start in range(0, len(tokens), chunk_size):
                chunk_end = min(chunk_start + chunk_size + 1, len(tokens))  # +1 for next token prediction
                chunk_tokens = tokens[chunk_start:chunk_end]

                if len(chunk_tokens) < 2:
                    continue

                input_ids = torch.tensor([chunk_tokens], device=self.device)

                with torch.no_grad():
                    logits = self._model_call(input_ids)

                # Compute log probabilities for this chunk
                log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
                chunk_token_log_probs = log_probs[range(len(chunk_tokens) - 1), chunk_tokens[1:]]

                total_log_prob += chunk_token_log_probs.sum().item()

                # Clear cache to free memory
                torch.cuda.empty_cache()

            results.append(total_log_prob)

            # Log sample
            _log_sample(
                method="loglikelihood_rolling",
                idx=idx,
                input_len=len(tokens),
                output=f"log_prob={total_log_prob:.4f}",
                extra_info=f"Chunks: {(len(tokens) + chunk_size - 1) // chunk_size}"
            )

        return results

    def generate_until(self, requests) -> List[str]:
        """
        Generate text until stopping criteria.

        Args:
            requests: List of Instance objects with arguments = (context, gen_kwargs)

        Returns:
            List of generated texts
        """
        results = []

        pbar = tqdm(enumerate(requests), total=len(requests), desc="generate_until", leave=True)
        for idx, request in pbar:
            # Extract arguments from Instance object
            context, generation_spec = request.arguments
            gen_kwargs = _coerce_gen_kwargs(generation_spec)

            # Tokenize context
            inputs = self.tokenizer(context, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            until = _normalize_stop_sequences(gen_kwargs.pop("until", None))
            max_new_tokens = int(gen_kwargs.pop("max_gen_toks", gen_kwargs.pop("max_new_tokens", 256)))
            do_sample = _parse_bool(gen_kwargs.pop("do_sample", False))
            temperature = float(gen_kwargs.pop("temperature", 1.0))
            top_p = float(gen_kwargs.pop("top_p", 1.0))
            num_beams = int(gen_kwargs.pop("num_beams", 1))
            gen_kwargs.pop("stop_strings", None)
            gen_kwargs.pop("tokenizer", None)
            gen_kwargs.pop("past_key_values", None)
            gen_kwargs.pop("pad_token_id", None)

            generate_kwargs = dict(gen_kwargs)
            generate_kwargs.update(
                {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": do_sample,
                    "num_beams": num_beams,
                    "pad_token_id": self.tokenizer.eos_token_id,
                    "past_key_values": self._create_cache(),
                    "stop_strings": until if until else None,
                    "tokenizer": self.tokenizer,
                }
            )
            if do_sample:
                generate_kwargs["temperature"] = temperature
                generate_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generate_kwargs,
                )

            generated_tokens = outputs[0, input_len:]
            generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Apply stopping criteria
            for stop in until:
                if stop and stop in generated:
                    generated = generated.split(stop)[0].strip()

            results.append(generated)

            # Log sample
            output_len = generated_tokens.shape[0]
            _log_sample(
                method="generate_until",
                idx=idx,
                input_len=input_len,
                output=generated,
                extra_info=(
                    f"Output Length: {output_len} tokens | Stop: {until} | "
                    f"Gen kwargs: max_new_tokens={max_new_tokens}, do_sample={do_sample}, "
                    f"temperature={temperature if do_sample else 'n/a'}, top_p={top_p if do_sample else 'n/a'}"
                )
            )

        return results


def run_evaluation(
    model_path: str,
    tasks: List[str],
    output_dir: str = "./eval_results",
    batch_size: int = 1,
    device: str = "cuda:0",
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    num_fewshot: Optional[int] = None,
    limit: Optional[int] = None,
    # SWAA parameters
    sliding_window_size: int = 2048,
    keep_first: int = 4,
    force_fa_decode: bool = False,
    force_fa_decode_layers: Optional[List[int]] = None,
    non_sliding_layers: Optional[List[int]] = None,
    enable_linear_mem: bool = True,
    flash_attn_weight: float = 0.9,
    linear_mem_weight: float = 0.1,
    linear_mem_mode: str = "fused_recurrent",
    linear_mem_blend_mode: str = "raw",
    linear_kernel_family: str = "softplus",
    active_layers: Optional[List[int] | str] = None,
    beta_by_layer: Optional[dict | str] = None,
    num_anchors: int = 64,
    tau: float = 20.0,
    **kwargs,
):
    """
    Run evaluation with detailed sample-level result saving.

    Args:
        model_path: Path to the pretrained model
        tasks: List of task names to evaluate
        output_dir: Directory to save evaluation results
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        torch_dtype: Torch dtype for model
        num_fewshot: Number of few-shot examples
        limit: Limit number of samples per task
        sliding_window_size: SWAA sliding window size
        keep_first: SWAA keep_first parameter
        force_fa_decode: SWAA force_fa_decode parameter
        non_sliding_layers: SWAA non_sliding_layers parameter
        enable_linear_mem: SWAA enable_linear_mem parameter
        flash_attn_weight: SWAA flash_attn_weight parameter
        linear_mem_weight: SWAA linear_mem_weight parameter
        **kwargs: Additional arguments passed to simple_evaluate

    Returns:
        Dictionary containing evaluation results
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*80}")
    print(f"Starting Evaluation")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Output Directory: {output_dir}")
    print(f"Timestamp: {timestamp}")
    print(
        "Decode Norm Correction: "
        f"enabled (state={SWAA_DECODE_NORM_STATE_KEY}, scope={SWAA_DECODE_NORM_SCOPE})"
    )
    print(f"{'='*80}\n")

    # Build model arguments
    model_args = {
        "pretrained": model_path,
        "device": device,
        "torch_dtype": torch_dtype,
        "attn_implementation": attn_implementation,
        "batch_size": batch_size,
        "sliding_window_size": sliding_window_size,
        "keep_first": keep_first,
        "force_fa_decode": str(force_fa_decode_layers) if force_fa_decode_layers else force_fa_decode,
        "non_sliding_layers": str(non_sliding_layers or []),
        "enable_linear_mem": enable_linear_mem,
        "flash_attn_weight": flash_attn_weight,
        "linear_mem_weight": linear_mem_weight,
        "linear_mem_mode": linear_mem_mode,
        "linear_mem_blend_mode": linear_mem_blend_mode,
        "linear_kernel_family": linear_kernel_family,
        "active_layers": str(active_layers) if active_layers is not None else "",
        "beta_by_layer": json.dumps(beta_by_layer, ensure_ascii=False, sort_keys=True) if beta_by_layer else "",
        "num_anchors": num_anchors,
        "tau": tau,
    }

    # Run evaluation with log_samples=True to save detailed results
    results = simple_evaluate(
        model="swaa_hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        device=device,
        limit=limit,
        log_samples=True,  # Enable sample-level logging
        **kwargs,
    )
    results["swaa_runtime"] = {
        "decode_norm_correction": True,
        "decode_norm_state_key": SWAA_DECODE_NORM_STATE_KEY,
        "decode_norm_scope": SWAA_DECODE_NORM_SCOPE,
        "required_cache_methods": list(_REQUIRED_SWAA_CACHE_METHODS),
    }

    # Save complete results with samples
    results_file = output_path / f"results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, default=handle_non_serializable, indent=2, ensure_ascii=False)
    print(f"\n✓ Complete results saved to: {results_file}")

    # Save summary (without samples for quick viewing)
    summary = {
        "results": results.get("results", {}),
        "configs": results.get("configs", {}),
        "versions": results.get("versions", {}),
        "n-shot": results.get("n-shot", {}),
        "higher_is_better": results.get("higher_is_better", {}),
        "n-samples": results.get("n-samples", {}),
        "swaa_runtime": results.get("swaa_runtime", {}),
    }
    summary_file = output_path / f"summary_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, default=handle_non_serializable, indent=2, ensure_ascii=False)
    print(f"✓ Summary saved to: {summary_file}")

    # Save detailed sample results in a more readable format
    if "samples" in results:
        samples_dir = output_path / f"samples_{timestamp}"
        samples_dir.mkdir(exist_ok=True)

        for task_name, task_samples in results["samples"].items():
            task_file = samples_dir / f"{task_name}.jsonl"
            with open(task_file, "w", encoding="utf-8") as f:
                for sample in task_samples:
                    f.write(json.dumps(sample, default=handle_non_serializable, ensure_ascii=False) + "\n")
            print(f"✓ Task '{task_name}' samples saved to: {task_file}")

    # Print summary to console
    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")
    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
    print(f"{'='*80}\n")

    return results


def main():
    """Command-line interface for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate SWAA models with detailed result logging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python eval_swaa_model.py --model_path ./model --tasks hellaswag arc_easy

  # With custom output directory and few-shot
  python eval_swaa_model.py --model_path ./model --tasks mmlu --num_fewshot 5 --output_dir ./results

  # Limit samples for testing
  python eval_swaa_model.py --model_path ./model --tasks hellaswag --limit 10

Output Files:
  - results_<timestamp>.json: Complete evaluation results including all samples
  - summary_<timestamp>.json: Summary metrics without individual samples
  - samples_<timestamp>/<task>.jsonl: Per-task sample-level results in JSONL format
  - evaluation.log: Detailed execution log
        """
    )

    # Required arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained model"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        required=True,
        help="List of task names to evaluate (e.g., hellaswag arc_easy mmlu)"
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results (default: ./eval_results)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run evaluation on (default: cuda:0)"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for model (default: bfloat16)"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager", "sdpa"],
        help="Attention implementation (default: flash_attention_2)"
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of few-shot examples (default: task-specific)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (for testing)"
    )

    # SWAA-specific arguments
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=2048,
        help="SWAA sliding window size (default: 2048)"
    )
    parser.add_argument(
        "--keep_first",
        type=int,
        default=4,
        help="SWAA keep_first parameter (default: 4)"
    )
    parser.add_argument(
        "--force_fa_decode",
        action="store_true",
        help="Force flash attention during decoding"
    )
    parser.add_argument(
        "--force_fa_decode_layers",
        type=int,
        nargs="*",
        default=None,
        help="Specific layers that should force full-attention decode"
    )
    parser.add_argument(
        "--non_sliding_layers",
        type=int,
        nargs="*",
        default=[],
        help="Layers that should not use sliding window (default: [])"
    )
    parser.add_argument(
        "--enable_linear_mem",
        dest="enable_linear_mem",
        action="store_true",
        help="Enable linear memory mechanism (default: True)"
    )
    parser.add_argument(
        "--disable_linear_mem",
        "--no_linear_mem",
        dest="enable_linear_mem",
        action="store_false",
        help="Disable linear memory mechanism"
    )
    parser.add_argument(
        "--flash_attn_weight",
        type=float,
        default=0.9,
        help="Flash attention weight (default: 0.9)"
    )
    parser.add_argument(
        "--linear_mem_weight",
        type=float,
        default=0.1,
        help="Linear memory weight (default: 0.1)"
    )
    parser.add_argument(
        "--linear_mem_mode",
        type=str,
        default="fused_recurrent",
        choices=["fused_recurrent", "fused_chunk", "chunk"],
        help="Linear memory execution mode (default: fused_recurrent)"
    )
    parser.add_argument(
        "--linear_mem_blend_mode",
        type=str,
        default="raw",
        choices=["raw", "centered", "orth", "orth_match"],
        help="How to blend linear memory with flash attention (default: raw)"
    )
    parser.add_argument(
        "--active_layers",
        type=str,
        default="",
        help="Comma-separated active layer indices for layer-selective linear memory (default: all layers)"
    )
    parser.add_argument(
        "--beta_by_layer",
        type=str,
        default="",
        help="Per-layer beta map, e.g. '0:0.08,18:0.16' or '{0:0.08,18:0.16}'"
    )
    parser.add_argument(
        "--linear_kernel_family",
        type=str,
        default="softplus",
        choices=["softplus", "niah", "anchor", "none"],
        help="Kernel family used by hack_hf_swaa linear memory (default: softplus)"
    )
    parser.add_argument(
        "--num_anchors",
        type=int,
        default=64,
        help="Number of anchors when linear_kernel_family=anchor (default: 64)"
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=20.0,
        help="Anchor kernel temperature when linear_kernel_family=anchor (default: 20.0)"
    )

    parser.set_defaults(enable_linear_mem=True)

    args = parser.parse_args()

    # Run evaluation
    run_evaluation(
        model_path=args.model_path,
        tasks=args.tasks,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        sliding_window_size=args.sliding_window_size,
        keep_first=args.keep_first,
        force_fa_decode=args.force_fa_decode,
        force_fa_decode_layers=args.force_fa_decode_layers,
        non_sliding_layers=args.non_sliding_layers,
        enable_linear_mem=args.enable_linear_mem,
        flash_attn_weight=args.flash_attn_weight,
        linear_mem_weight=args.linear_mem_weight,
        linear_mem_mode=args.linear_mem_mode,
        linear_mem_blend_mode=args.linear_mem_blend_mode,
        linear_kernel_family=args.linear_kernel_family,
        active_layers=args.active_layers,
        beta_by_layer=args.beta_by_layer,
        num_anchors=args.num_anchors,
        tau=args.tau,
    )


if __name__ == "__main__":
    main()
