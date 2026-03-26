"""
Monkey patch for transformers KV Cache to support linear attention recurrent state.

This module patches DynamicLayer and its subclasses to add support for storing
linear attention recurrent state alongside the standard key-value cache.
"""

from __future__ import annotations

from typing import Any
import torch
from transformers.cache_utils import (
    DynamicLayer,
    DynamicSlidingWindowLayer,
    QuantizedLayer,
    CacheLayerMixin,
)


# Store original methods for chaining
_original_dynamic_layer_lazy_init = DynamicLayer.lazy_initialization
_original_dynamic_layer_update = DynamicLayer.update
_original_dynamic_layer_reset = DynamicLayer.reset

_original_sliding_layer_lazy_init = DynamicSlidingWindowLayer.lazy_initialization
_original_sliding_layer_update = DynamicSlidingWindowLayer.update
_original_sliding_layer_reset = DynamicSlidingWindowLayer.reset

_original_quantized_layer_lazy_init = QuantizedLayer.lazy_initialization
_original_quantized_layer_update = QuantizedLayer.update
_original_quantized_layer_reset = QuantizedLayer.reset


def _init_recurrent_state(self: CacheLayerMixin) -> None:
    """Initialize the recurrent state container if not exists."""
    if not hasattr(self, 'recurrent_state') or self.recurrent_state is None:
        self.recurrent_state = None
        self.recurrent_state_initialized = False


def _init_k_norm_cache(self: CacheLayerMixin) -> None:
    """Initialize the k_norm cache container if not exists."""
    if not hasattr(self, 'k_norm_cache') or self.k_norm_cache is None:
        self.k_norm_cache = None
        self.k_norm_cache_initialized = False


def _init_linear_k_sum_state(self: CacheLayerMixin) -> None:
    """Initialize the cumulative transformed-k state used for decode normalization."""
    if not hasattr(self, 'linear_k_sum_state') or self.linear_k_sum_state is None:
        self.linear_k_sum_state = None
        self.linear_k_sum_state_initialized = False


def _init_hash_mem(self: CacheLayerMixin) -> None:
    """Initialize the hash_mem container if not exists.

    hash_mem structure: dict[int -> dict] where each dict contains:
        - "S": Tensor with shape [B, H, R, M, D]
        - "Z": Tensor with shape [B, H, R, M]
    where H=num_heads, D=head_dim, R=hash tables, M=2^b buckets
    """
    if not hasattr(self, 'hash_mem') or self.hash_mem is None:
        self.hash_mem = {}
        self.hash_mem_initialized = False


def dynamic_layer_lazy_init_swaa(
    self: DynamicLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization that also initializes recurrent state, k_norm cache, and hash_mem."""
    # Call original initialization
    _original_dynamic_layer_lazy_init(self, key_states, value_states)
    # Initialize recurrent state container
    _init_recurrent_state(self)
    # Initialize k_norm cache container
    _init_k_norm_cache(self)
    # Initialize cumulative transformed-k state
    _init_linear_k_sum_state(self)
    # Initialize hash_mem container
    _init_hash_mem(self)


def dynamic_layer_update_swaa(
    self: DynamicLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Patched update method that also handles recurrent state.

    This method:
    1. Ensures recurrent_state is initialized (lazy init)
    2. Passes through any recurrent_state from cache_kwargs
    3. Returns the standard key-value states

    Usage in attention layer:
        cache_kwargs = {
            "recurrent_state": new_recurrent_state,  # From linear attention kernel
        }
        key_states, value_states = past_key_values.update(
            key_states, value_states, layer_idx, cache_kwargs
        )
    """
    # Lazy initialization for recurrent state
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

    # Lazy initialization for k_norm cache
    if not hasattr(self, 'k_norm_cache'):
        _init_k_norm_cache(self)

    # Lazy initialization for decode denominator state
    if not hasattr(self, 'linear_k_sum_state'):
        _init_linear_k_sum_state(self)

    # Lazy initialization for hash_mem
    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    # Handle recurrent state update from cache_kwargs
    if cache_kwargs is not None:
        new_recurrent_state = cache_kwargs.get('recurrent_state', None)
        if new_recurrent_state is not None:
            self.recurrent_state = new_recurrent_state

    # Call original update for key-value states
    return _original_dynamic_layer_update(self, key_states, value_states, cache_kwargs)


def dynamic_layer_reset_swaa(self: DynamicLayer) -> None:
    """Patched reset that also clears recurrent state, k_norm cache, and hash_mem."""
    _original_dynamic_layer_reset(self)
    if hasattr(self, 'recurrent_state'):
        self.recurrent_state = None
        self.recurrent_state_initialized = False
    if hasattr(self, 'k_norm_cache'):
        self.k_norm_cache = None
        self.k_norm_cache_initialized = False
    if hasattr(self, 'linear_k_sum_state'):
        self.linear_k_sum_state = None
        self.linear_k_sum_state_initialized = False
    if hasattr(self, 'hash_mem'):
        self.hash_mem = {}
        self.hash_mem_initialized = False


def sliding_layer_lazy_init_swaa(
    self: DynamicSlidingWindowLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization for DynamicSlidingWindowLayer."""
    _original_sliding_layer_lazy_init(self, key_states, value_states)
    _init_recurrent_state(self)
    _init_k_norm_cache(self)
    _init_linear_k_sum_state(self)
    _init_hash_mem(self)


def sliding_layer_update_swaa(
    self: DynamicSlidingWindowLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched update for DynamicSlidingWindowLayer with recurrent state, k_norm cache, and hash_mem support."""
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

    if not hasattr(self, 'k_norm_cache'):
        _init_k_norm_cache(self)

    if not hasattr(self, 'linear_k_sum_state'):
        _init_linear_k_sum_state(self)

    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    if cache_kwargs is not None:
        new_recurrent_state = cache_kwargs.get('recurrent_state', None)
        if new_recurrent_state is not None:
            self.recurrent_state = new_recurrent_state

    return _original_sliding_layer_update(self, key_states, value_states, cache_kwargs)


def sliding_layer_reset_swaa(self: DynamicSlidingWindowLayer) -> None:
    """Patched reset for DynamicSlidingWindowLayer."""
    _original_sliding_layer_reset(self)
    if hasattr(self, 'recurrent_state'):
        self.recurrent_state = None
        self.recurrent_state_initialized = False
    if hasattr(self, 'k_norm_cache'):
        self.k_norm_cache = None
        self.k_norm_cache_initialized = False
    if hasattr(self, 'linear_k_sum_state'):
        self.linear_k_sum_state = None
        self.linear_k_sum_state_initialized = False
    if hasattr(self, 'hash_mem'):
        self.hash_mem = {}
        self.hash_mem_initialized = False


def quantized_layer_lazy_init_swaa(
    self: QuantizedLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization for QuantizedLayer."""
    _original_quantized_layer_lazy_init(self, key_states, value_states)
    _init_recurrent_state(self)
    _init_k_norm_cache(self)
    _init_linear_k_sum_state(self)
    _init_hash_mem(self)


def quantized_layer_update_swaa(
    self: QuantizedLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched update for QuantizedLayer with recurrent state, k_norm cache, and hash_mem support."""
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

    if not hasattr(self, 'k_norm_cache'):
        _init_k_norm_cache(self)

    if not hasattr(self, 'linear_k_sum_state'):
        _init_linear_k_sum_state(self)

    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    if cache_kwargs is not None:
        new_recurrent_state = cache_kwargs.get('recurrent_state', None)
        if new_recurrent_state is not None:
            self.recurrent_state = new_recurrent_state

    return _original_quantized_layer_update(self, key_states, value_states, cache_kwargs)


def quantized_layer_reset_swaa(self: QuantizedLayer) -> None:
    """Patched reset for QuantizedLayer."""
    _original_quantized_layer_reset(self)
    if hasattr(self, 'recurrent_state'):
        self.recurrent_state = None
        self.recurrent_state_initialized = False
    if hasattr(self, 'k_norm_cache'):
        self.k_norm_cache = None
        self.k_norm_cache_initialized = False
    if hasattr(self, 'linear_k_sum_state'):
        self.linear_k_sum_state = None
        self.linear_k_sum_state_initialized = False
    if hasattr(self, 'hash_mem'):
        self.hash_mem = {}
        self.hash_mem_initialized = False


def get_recurrent_state(self: CacheLayerMixin) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """
    Get the recurrent state for this cache layer.

    Returns:
        The recurrent state tensor(s) or None if not initialized.
    """
    return getattr(self, 'recurrent_state', None)


def set_recurrent_state(
    self: CacheLayerMixin,
    state: torch.Tensor | tuple[torch.Tensor, ...] | None
) -> None:
    """
    Set the recurrent state for this cache layer.

    Args:
        state: The new recurrent state tensor(s) or None to clear.
    """
    self.recurrent_state = state
    self.recurrent_state_initialized = state is not None


def is_recurrent_state_initialized(self: CacheLayerMixin) -> bool:
    """
    Check if the recurrent state has been initialized.

    Returns:
        `bool`: True if the recurrent state has been set, False otherwise.
    """
    return getattr(self, 'recurrent_state_initialized', False)


# ============== k_norm Cache Methods ==============

def get_k_norm_cache(self: CacheLayerMixin) -> torch.Tensor | None:
    """
    Get the k_norm cache for this cache layer.

    Returns:
        The k_norm cache tensor or None if not initialized.
    """
    return getattr(self, 'k_norm_cache', None)


def set_k_norm_cache(
    self: CacheLayerMixin,
    k_norm: torch.Tensor | None
) -> None:
    """
    Set the k_norm cache for this cache layer.

    Args:
        k_norm: The new k_norm tensor or None to clear.
    """
    self.k_norm_cache = k_norm
    self.k_norm_cache_initialized = k_norm is not None


def is_k_norm_cache_initialized(self: CacheLayerMixin) -> bool:
    """
    Check if the k_norm cache has been initialized.

    Returns:
        `bool`: True if the k_norm cache has been set, False otherwise.
    """
    return getattr(self, 'k_norm_cache_initialized', False)


# ============== Decode Denominator State Methods ==============

def get_linear_k_sum_state(self: CacheLayerMixin) -> torch.Tensor | None:
    """
    Get the cumulative transformed-k state for this cache layer.

    Returns:
        The cumulative transformed-k sum tensor with shape `[B, H, D]`, or
        None if not initialized.
    """
    return getattr(self, 'linear_k_sum_state', None)


def set_linear_k_sum_state(
    self: CacheLayerMixin,
    state: torch.Tensor | None,
) -> None:
    """
    Set the cumulative transformed-k state for this cache layer.

    Args:
        state: The new cumulative transformed-k sum tensor or None to clear.
    """
    self.linear_k_sum_state = state
    self.linear_k_sum_state_initialized = state is not None


def is_linear_k_sum_state_initialized(self: CacheLayerMixin) -> bool:
    """
    Check if the cumulative transformed-k state has been initialized.

    Returns:
        `bool`: True if the state has been set, False otherwise.
    """
    return getattr(self, 'linear_k_sum_state_initialized', False)


def linear_k_sum_state_update(
    self: CacheLayerMixin,
    state: torch.Tensor | None,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """
    Update the cumulative transformed-k state in-place and return the current value.

    Args:
        state (`torch.Tensor`, *optional*):
            The new cumulative transformed-k state. If None, no update is
            performed and the current value is returned.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Reserved for future use.

    Returns:
        `torch.Tensor` or `None`:
            The current cumulative transformed-k state after update.
    """
    if not hasattr(self, 'linear_k_sum_state'):
        _init_linear_k_sum_state(self)

    if state is not None:
        self.linear_k_sum_state = state
        self.linear_k_sum_state_initialized = True

    return self.linear_k_sum_state


# ============== Hash State Methods ==============

def get_hash_state(
    self: CacheLayerMixin,
    layer_idx: int | None = None
) -> dict[str, torch.Tensor] | None:
    """
    Get the hash state for this cache layer.

    Args:
        layer_idx: The layer index to get hash state for.
                   If None, uses the layer's own hash_mem.

    Returns:
        dict with keys "S" and "Z", or None if not initialized.
        - S: Tensor with shape [B, H, R, M, D]
        - Z: Tensor with shape [B, H, R, M]
    """
    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    if layer_idx is None:
        # For single-layer access (deprecated, use layer_idx in Cache)
        return self.hash_mem if self.hash_mem else None

    return self.hash_mem.get(layer_idx, None)


def set_hash_state(
    self: CacheLayerMixin,
    layer_idx: int,
    state_dict: dict[str, torch.Tensor] | None
) -> None:
    """
    Set the hash state for a specific layer index.

    Args:
        layer_idx: The layer index to set hash state for.
        state_dict: dict with keys "S" and "Z", or None to clear.
            - S: Tensor with shape [B, H, R, M, D]
            - Z: Tensor with shape [B, H, R, M]
    """
    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    if state_dict is None:
        if layer_idx in self.hash_mem:
            del self.hash_mem[layer_idx]
    else:
        self.hash_mem[layer_idx] = state_dict
        self.hash_mem_initialized = True


def is_hash_state_initialized(
    self: CacheLayerMixin,
    layer_idx: int | None = None
) -> bool:
    """
    Check if the hash state has been initialized for a specific layer.

    Args:
        layer_idx: The layer index to check. If None, checks the overall flag.

    Returns:
        `bool`: True if the hash state has been set, False otherwise.
    """
    if not hasattr(self, 'hash_mem'):
        return False

    if layer_idx is None:
        return getattr(self, 'hash_mem_initialized', False)

    return layer_idx in self.hash_mem


def clear_hash_state(
    self: CacheLayerMixin,
    layer_idx: int | None = None
) -> None:
    """
    Clear the hash state for a specific layer or all layers.

    Args:
        layer_idx: The layer index to clear. If None, clears all hash states.
    """
    if not hasattr(self, 'hash_mem'):
        return

    if layer_idx is None:
        self.hash_mem = {}
        self.hash_mem_initialized = False
    elif layer_idx in self.hash_mem:
        del self.hash_mem[layer_idx]


def hash_state_update(
    self: CacheLayerMixin,
    layer_idx: int,
    new_state: dict[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Update the hash state for a specific layer and return the current state.

    This method mirrors the style of `k_norm_update()` for consistent API.

    Args:
        layer_idx (`int`):
            The layer index to update.
        new_state (`dict[str, torch.Tensor]`, *optional*):
            The new hash state dict with keys "S" and "Z".
            If None, no update is performed, just retrieval.
        device (`torch.device`, *optional*):
            Target device to align tensors. If state exists and device differs,
            tensors will be moved to this device.
        dtype (`torch.dtype`, *optional*):
            Target dtype to align tensors. If state exists and dtype differs,
            tensors will be cast to this dtype.

    Returns:
        `dict[str, torch.Tensor]` or `None`:
            The current hash state after update (or before if no update).
            Tensors are aligned to device/dtype if specified.

    Example:
        ```python
        # First call: set initial state
        state = cache.hash_state_update(0, {"S": S_tensor, "Z": Z_tensor})

        # Subsequent calls: update and get
        new_state = cache.hash_state_update(0, {"S": new_S, "Z": new_Z})

        # Just retrieve without update
        state = cache.hash_state_update(0)

        # With device/dtype alignment
        state = cache.hash_state_update(0, device=query.device, dtype=query.dtype)
        ```
    """
    if not hasattr(self, 'hash_mem'):
        _init_hash_mem(self)

    # Update the hash state if provided
    if new_state is not None:
        self.hash_mem[layer_idx] = new_state
        self.hash_mem_initialized = True

    # Retrieve current state
    state = self.hash_mem.get(layer_idx, None)

    # Align device/dtype if requested and state exists
    if state is not None and (device is not None or dtype is not None):
        aligned_state = {}
        for key, tensor in state.items():
            if device is not None and tensor.device != device:
                tensor = tensor.to(device)
            if dtype is not None and tensor.dtype != dtype:
                tensor = tensor.to(dtype)
            aligned_state[key] = tensor
        return aligned_state

    return state


def k_norm_update(
    self: CacheLayerMixin,
    k_norm: torch.Tensor | None,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """
    Update the k_norm cache in-place, and return the current k_norm value.

    Args:
        k_norm (`torch.Tensor`, *optional*):
            The new k_norm value to cache. If None, no update is performed.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Additional arguments for the cache (reserved for future use).

    Returns:
        `torch.Tensor` or `None`:
            The current k_norm value after update (or before if no update).
    """
    # Lazy initialization: ensure k_norm_cache attribute exists
    if not hasattr(self, 'k_norm_cache'):
        _init_k_norm_cache(self)

    # Update the k_norm cache if provided
    if k_norm is not None:
        self.k_norm_cache = k_norm
        self.k_norm_cache_initialized = True

    return self.k_norm_cache


def state_update(
    self: CacheLayerMixin,
    recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """
    Update the recurrent state cache in-place, and return the current recurrent state.

    This method mirrors the style of `DynamicLayer.update()` for key-value states,
    but operates on the recurrent state for linear attention.

    Args:
        recurrent_state (`torch.Tensor` or `tuple[torch.Tensor, ...]`, *optional*):
            The new recurrent state to cache. If None, no update is performed.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Additional arguments for the cache. Currently supports:
            - `init_state`: If True and recurrent_state is not None, initialize dtype/device from it

    Returns:
        `torch.Tensor` or `tuple[torch.Tensor, ...]` or `None`:
            The current recurrent state after update (or before if no update).

    Example:
        ```python
        # In linear attention layer forward:
        # After computing new recurrent state from kernel
        new_recurrent_state = linear_attention_kernel(q, k, v, initial_state=cache.get_recurrent_state())

        # Update and get the state
        current_state = cache.state_update(new_recurrent_state)

        # Or use in DynamicCache:
        state = cache.state_update(recurrent_state, layer_idx=layer_idx)
        ```
    """
    # Lazy initialization: ensure recurrent_state attribute exists
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

    # Update the recurrent state if provided
    if recurrent_state is not None:
        self.recurrent_state = recurrent_state
        self.recurrent_state_initialized = True

        # Optionally extract dtype/device for future use
        if cache_kwargs is not None and cache_kwargs.get('init_state', False):
            if isinstance(recurrent_state, torch.Tensor):
                self.recurrent_dtype = recurrent_state.dtype
                self.recurrent_device = recurrent_state.device
            elif isinstance(recurrent_state, (tuple, list)) and len(recurrent_state) > 0:
                self.recurrent_dtype = recurrent_state[0].dtype
                self.recurrent_device = recurrent_state[0].device

    return self.recurrent_state


def cache_state_update(
    self,
    recurrent_state: torch.Tensor | tuple[torch.Tensor, ...] | None,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """
    Update the recurrent state for a specific layer in the cache.

    This method is attached to Cache classes (DynamicCache, etc.) and provides
    a layer-indexed interface similar to `Cache.update()` for key-value states.

    Args:
        recurrent_state (`torch.Tensor` or `tuple[torch.Tensor, ...]`, *optional*):
            The new recurrent state to cache.
        layer_idx (`int`):
            The index of the layer to update.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Additional arguments passed to the layer's state_update method.

    Returns:
        `torch.Tensor` or `tuple[torch.Tensor, ...]` or `None`:
            The recurrent state for the specified layer.

    Example:
        ```python
        from transformers import DynamicCache
        from swaa_patch import hack_kv_cache_recurrent_state

        hack_kv_cache_recurrent_state()

        cache = DynamicCache()

        # Update recurrent state for layer 0
        state = cache.state_update(recurrent_state_tensor, layer_idx=0)

        # Get recurrent state for layer 0
        state = cache.state_update(None, layer_idx=0)  # No update, just retrieve
        ```
    """
    # Ensure layer exists (for lazy layer creation)
    while len(self.layers) <= layer_idx:
        if hasattr(self, 'layer_class_to_replicate'):
            self.layers.append(self.layer_class_to_replicate())
        else:
            raise IndexError(f"Layer {layer_idx} does not exist and cannot be created lazily")

    return self.layers[layer_idx].state_update(recurrent_state, cache_kwargs)


def cache_is_recurrent_state_initialized(
    self,
    layer_idx: int,
) -> bool:
    """
    Check if the recurrent state for a specific layer has been initialized.

    This method is attached to Cache classes (DynamicCache, etc.) and provides
    a layer-indexed interface to check initialization status.

    Args:
        layer_idx (`int`):
            The index of the layer to check.

    Returns:
        `bool`: True if the recurrent state for the specified layer has been
                initialized, False otherwise.

    Example:
        ```python
        from transformers import DynamicCache
        from swaa_patch import hack_kv_cache_recurrent_state

        hack_kv_cache_recurrent_state()

        cache = DynamicCache()

        # Check if layer 0 has initialized recurrent state
        if not cache.is_recurrent_state_initialized(layer_idx=0):
            # First time, need to initialize
            initial_state = create_initial_state()
            cache.state_update(initial_state, layer_idx=0)

        # Get existing state
        state = cache.state_update(None, layer_idx=0)
        ```
    """
    # If layer doesn't exist yet, it's not initialized
    if layer_idx >= len(self.layers):
        return False

    return self.layers[layer_idx].is_recurrent_state_initialized()


# ============== Cache-level k_norm Methods ==============

def cache_k_norm_update(
    self,
    k_norm: torch.Tensor | None,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """
    Update the k_norm cache for a specific layer in the cache.

    Args:
        k_norm (`torch.Tensor`, *optional*):
            The new k_norm value to cache.
        layer_idx (`int`):
            The index of the layer to update.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Additional arguments passed to the layer's k_norm_update method.

    Returns:
        `torch.Tensor` or `None`:
            The k_norm value for the specified layer.
    """
    # Ensure layer exists (for lazy layer creation)
    while len(self.layers) <= layer_idx:
        if hasattr(self, 'layer_class_to_replicate'):
            self.layers.append(self.layer_class_to_replicate())
        else:
            raise IndexError(f"Layer {layer_idx} does not exist and cannot be created lazily")

    return self.layers[layer_idx].k_norm_update(k_norm, cache_kwargs)


def cache_is_k_norm_cache_initialized(
    self,
    layer_idx: int,
) -> bool:
    """
    Check if the k_norm cache for a specific layer has been initialized.

    Args:
        layer_idx (`int`):
            The index of the layer to check.

    Returns:
        `bool`: True if the k_norm cache for the specified layer has been
                initialized, False otherwise.
    """
    # If layer doesn't exist yet, it's not initialized
    if layer_idx >= len(self.layers):
        return False

    return self.layers[layer_idx].is_k_norm_cache_initialized()


def cache_get_k_norm_cache(
    self,
    layer_idx: int,
) -> torch.Tensor | None:
    """
    Get the k_norm cache for a specific layer in the cache.

    Args:
        layer_idx (`int`):
            The index of the layer to get the k_norm from.

    Returns:
        `torch.Tensor` or `None`:
            The k_norm value for the specified layer, or None if not initialized
            or the layer doesn't exist.
    """
    # If layer doesn't exist, return None
    if layer_idx >= len(self.layers):
        return None

    return self.layers[layer_idx].get_k_norm_cache()


def cache_linear_k_sum_state_update(
    self,
    state: torch.Tensor | None,
    layer_idx: int,
    cache_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor | None:
    """
    Update the cumulative transformed-k state for a specific layer in the cache.

    Args:
        state (`torch.Tensor`, *optional*):
            The new cumulative transformed-k state.
        layer_idx (`int`):
            The index of the layer to update.
        cache_kwargs (`dict[str, Any]`, *optional*):
            Additional arguments passed to the layer method.

    Returns:
        `torch.Tensor` or `None`:
            The cumulative transformed-k state for the specified layer.
    """
    while len(self.layers) <= layer_idx:
        if hasattr(self, 'layer_class_to_replicate'):
            self.layers.append(self.layer_class_to_replicate())
        else:
            raise IndexError(f"Layer {layer_idx} does not exist and cannot be created lazily")

    return self.layers[layer_idx].linear_k_sum_state_update(state, cache_kwargs)


def cache_is_linear_k_sum_state_initialized(
    self,
    layer_idx: int,
) -> bool:
    """
    Check if the cumulative transformed-k state for a specific layer has been initialized.
    """
    if layer_idx >= len(self.layers):
        return False
    return self.layers[layer_idx].is_linear_k_sum_state_initialized()


def cache_get_linear_k_sum_state(
    self,
    layer_idx: int,
) -> torch.Tensor | None:
    """
    Get the cumulative transformed-k state for a specific layer in the cache.
    """
    if layer_idx >= len(self.layers):
        return None
    return self.layers[layer_idx].get_linear_k_sum_state()


def cache_get_recurrent_state(
    self,
    layer_idx: int,
) -> torch.Tensor | tuple[torch.Tensor, ...] | None:
    """
    Get the recurrent state for a specific layer in the cache.

    This method is attached to Cache classes (DynamicCache, etc.) and provides
    a layer-indexed interface to retrieve the recurrent state.

    Args:
        layer_idx (`int`):
            The index of the layer to get the state from.

    Returns:
        `torch.Tensor` or `tuple[torch.Tensor, ...]` or `None`:
            The recurrent state for the specified layer, or None if not initialized
            or the layer doesn't exist.

    Example:
        ```python
        from transformers import DynamicCache
        from swaa_patch import hack_kv_cache_recurrent_state

        hack_kv_cache_recurrent_state()

        cache = DynamicCache()

        # Get recurrent state for layer 0 (returns None if not initialized)
        state = cache.get_recurrent_state(layer_idx=0)

        # After setting state
        cache.state_update(recurrent_state_tensor, layer_idx=0)
        state = cache.get_recurrent_state(layer_idx=0)  # Returns the tensor

        # Non-existent layer returns None
        state = cache.get_recurrent_state(layer_idx=99)  # None
        ```
    """
    # If layer doesn't exist, return None
    if layer_idx >= len(self.layers):
        return None

    return self.layers[layer_idx].get_recurrent_state()


# ============== Cache-level Hash State Methods ==============

def cache_get_hash_state(
    self,
    layer_idx: int,
) -> dict[str, torch.Tensor] | None:
    """
    Get the hash state for a specific layer in the cache.

    Args:
        layer_idx (`int`):
            The index of the layer to get the hash state from.

    Returns:
        `dict[str, torch.Tensor]` or `None`:
            The hash state dict with keys "S" and "Z" for the specified layer,
            or None if not initialized or the layer doesn't exist.
    """
    # If layer doesn't exist, return None
    if layer_idx >= len(self.layers):
        return None

    return self.layers[layer_idx].get_hash_state(layer_idx)


def cache_set_hash_state(
    self,
    layer_idx: int,
    state_dict: dict[str, torch.Tensor] | None
) -> None:
    """
    Set the hash state for a specific layer in the cache.

    Args:
        layer_idx (`int`):
            The index of the layer to set the hash state for.
        state_dict (`dict[str, torch.Tensor]` or `None`):
            The hash state dict with keys "S" and "Z", or None to clear.
    """
    # Ensure layer exists (for lazy layer creation)
    while len(self.layers) <= layer_idx:
        if hasattr(self, 'layer_class_to_replicate'):
            self.layers.append(self.layer_class_to_replicate())
        else:
            raise IndexError(f"Layer {layer_idx} does not exist and cannot be created lazily")

    self.layers[layer_idx].set_hash_state(layer_idx, state_dict)


def cache_is_hash_state_initialized(
    self,
    layer_idx: int,
) -> bool:
    """
    Check if the hash state for a specific layer has been initialized.

    Args:
        layer_idx (`int`):
            The index of the layer to check.

    Returns:
        `bool`: True if the hash state for the specified layer has been
                initialized, False otherwise.
    """
    # If layer doesn't exist yet, it's not initialized
    if layer_idx >= len(self.layers):
        return False

    return self.layers[layer_idx].is_hash_state_initialized(layer_idx)


def cache_clear_hash_state(
    self,
    layer_idx: int | None = None
) -> None:
    """
    Clear the hash state for a specific layer or all layers.

    Args:
        layer_idx (`int`, *optional*):
            The index of the layer to clear. If None, clears all layers.
    """
    if layer_idx is None:
        # Clear all layers
        for layer in self.layers:
            layer.clear_hash_state()
    elif layer_idx < len(self.layers):
        # Clear specific layer
        self.layers[layer_idx].clear_hash_state(layer_idx)


def cache_hash_state_update(
    self,
    layer_idx: int,
    new_state: dict[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> dict[str, torch.Tensor] | None:
    """
    Update the hash state for a specific layer and return the current state.

    This method provides a layer-indexed interface similar to `k_norm_update()`.

    Args:
        layer_idx (`int`):
            The index of the layer to update.
        new_state (`dict[str, torch.Tensor]`, *optional*):
            The new hash state dict with keys "S" and "Z".
            If None, no update is performed, just retrieval.
        device (`torch.device`, *optional*):
            Target device to align tensors.
        dtype (`torch.dtype`, *optional*):
            Target dtype to align tensors.

    Returns:
        `dict[str, torch.Tensor]` or `None`:
            The hash state for the specified layer after update,
            with tensors aligned to device/dtype if specified.

    Example:
        ```python
        # First call: initialize
        state = cache.hash_state_update(0, {"S": S_tensor, "Z": Z_tensor})

        # Subsequent calls: update and get
        state = cache.hash_state_update(0, {"S": new_S, "Z": new_Z})

        # Just retrieve without update, with device alignment
        state = cache.hash_state_update(0, device=query.device, dtype=query.dtype)
        ```
    """
    # Ensure layer exists (for lazy layer creation)
    while len(self.layers) <= layer_idx:
        if hasattr(self, 'layer_class_to_replicate'):
            self.layers.append(self.layer_class_to_replicate())
        else:
            raise IndexError(f"Layer {layer_idx} does not exist and cannot be created lazily")

    return self.layers[layer_idx].hash_state_update(layer_idx, new_state, device, dtype)


def hack_kv_cache_recurrent_state():
    """
    Apply monkey patch to transformers KV Cache layers to support recurrent state,
    k_norm cache, decode denominator state, and hash state.

    This patches the following classes:
    - DynamicLayer
    - DynamicSlidingWindowLayer
    - QuantizedLayer
    - Cache (and its subclasses like DynamicCache)

    After patching, each layer will have:
    - `recurrent_state` attribute for storing linear attention state
    - `recurrent_state_initialized` attribute (bool) indicating if state has been set
    - `get_recurrent_state()` method to retrieve the state
    - `set_recurrent_state(state)` method to set the state
    - `state_update(state, cache_kwargs)` method to update and return state
    - `is_recurrent_state_initialized()` method to check initialization status

    - `k_norm_cache` attribute for storing key norm values
    - `k_norm_cache_initialized` attribute (bool) indicating if cache has been set
    - `get_k_norm_cache()` method to retrieve the cache
    - `set_k_norm_cache(k_norm)` method to set the cache
    - `k_norm_update(k_norm, cache_kwargs)` method to update and return cache
    - `is_k_norm_cache_initialized()` method to check initialization status

    - `linear_k_sum_state` attribute for storing cumulative transformed-k sums
    - `linear_k_sum_state_initialized` attribute (bool)
    - `get_linear_k_sum_state()` method to retrieve the state
    - `set_linear_k_sum_state(state)` method to set the state
    - `linear_k_sum_state_update(state, cache_kwargs)` method to update and return state
    - `is_linear_k_sum_state_initialized()` method to check initialization status

    - `hash_mem` attribute for storing hash state (dict[int -> dict])
    - `hash_mem_initialized` attribute (bool) indicating if state has been set
    - `get_hash_state(layer_idx)` method to retrieve hash state
    - `set_hash_state(layer_idx, state_dict)` method to set hash state
    - `is_hash_state_initialized(layer_idx)` method to check initialization status
    - `clear_hash_state(layer_idx)` method to clear hash state
    - `hash_state_update(layer_idx, new_state, device, dtype)` method to update and return state

    The Cache class will have:
    - `state_update(state, layer_idx, cache_kwargs)` method for layer-indexed updates
    - `k_norm_update(k_norm, layer_idx, cache_kwargs)` method for layer-indexed k_norm updates
    - `linear_k_sum_state_update(state, layer_idx, cache_kwargs)` method for
      layer-indexed decode denominator-state updates
    - `is_linear_k_sum_state_initialized(layer_idx)` method for layer-indexed checks
    - `get_linear_k_sum_state(layer_idx)` method for layer-indexed retrieval
    - `get_hash_state(layer_idx)` method for layer-indexed hash state retrieval
    - `set_hash_state(layer_idx, state_dict)` method for layer-indexed hash state setting
    - `is_hash_state_initialized(layer_idx)` method for layer-indexed hash state check
    - `clear_hash_state(layer_idx)` method for layer-indexed hash state clearing
    - `hash_state_update(layer_idx, new_state, device, dtype)` method for layer-indexed hash state update

    The recurrent state, k_norm cache, and hash state are automatically managed during:
    - lazy_initialization: Creates empty containers with initialized=False
    - update: Accepts recurrent_state via cache_kwargs, sets initialized=True
    - reset: Clears the state and sets initialized=False
    """
    from transformers.cache_utils import Cache

    # Patch DynamicLayer
    DynamicLayer.lazy_initialization = dynamic_layer_lazy_init_swaa
    DynamicLayer.update = dynamic_layer_update_swaa
    DynamicLayer.reset = dynamic_layer_reset_swaa
    DynamicLayer.get_recurrent_state = get_recurrent_state
    DynamicLayer.set_recurrent_state = set_recurrent_state
    DynamicLayer.state_update = state_update
    DynamicLayer.is_recurrent_state_initialized = is_recurrent_state_initialized

    # Patch DynamicSlidingWindowLayer
    DynamicSlidingWindowLayer.lazy_initialization = sliding_layer_lazy_init_swaa
    DynamicSlidingWindowLayer.update = sliding_layer_update_swaa
    DynamicSlidingWindowLayer.reset = sliding_layer_reset_swaa
    DynamicSlidingWindowLayer.get_recurrent_state = get_recurrent_state
    DynamicSlidingWindowLayer.set_recurrent_state = set_recurrent_state
    DynamicSlidingWindowLayer.state_update = state_update
    DynamicSlidingWindowLayer.is_recurrent_state_initialized = is_recurrent_state_initialized

    # Patch QuantizedLayer
    QuantizedLayer.lazy_initialization = quantized_layer_lazy_init_swaa
    QuantizedLayer.update = quantized_layer_update_swaa
    QuantizedLayer.reset = quantized_layer_reset_swaa
    QuantizedLayer.get_recurrent_state = get_recurrent_state
    QuantizedLayer.set_recurrent_state = set_recurrent_state
    QuantizedLayer.state_update = state_update
    QuantizedLayer.is_recurrent_state_initialized = is_recurrent_state_initialized

    # Patch Cache class with layer-indexed state_update
    Cache.state_update = cache_state_update
    Cache.is_recurrent_state_initialized = cache_is_recurrent_state_initialized
    Cache.get_recurrent_state = cache_get_recurrent_state

    # Patch k_norm cache methods for Layer classes
    DynamicLayer.get_k_norm_cache = get_k_norm_cache
    DynamicLayer.set_k_norm_cache = set_k_norm_cache
    DynamicLayer.k_norm_update = k_norm_update
    DynamicLayer.is_k_norm_cache_initialized = is_k_norm_cache_initialized
    DynamicLayer.get_linear_k_sum_state = get_linear_k_sum_state
    DynamicLayer.set_linear_k_sum_state = set_linear_k_sum_state
    DynamicLayer.linear_k_sum_state_update = linear_k_sum_state_update
    DynamicLayer.is_linear_k_sum_state_initialized = is_linear_k_sum_state_initialized

    DynamicSlidingWindowLayer.get_k_norm_cache = get_k_norm_cache
    DynamicSlidingWindowLayer.set_k_norm_cache = set_k_norm_cache
    DynamicSlidingWindowLayer.k_norm_update = k_norm_update
    DynamicSlidingWindowLayer.is_k_norm_cache_initialized = is_k_norm_cache_initialized
    DynamicSlidingWindowLayer.get_linear_k_sum_state = get_linear_k_sum_state
    DynamicSlidingWindowLayer.set_linear_k_sum_state = set_linear_k_sum_state
    DynamicSlidingWindowLayer.linear_k_sum_state_update = linear_k_sum_state_update
    DynamicSlidingWindowLayer.is_linear_k_sum_state_initialized = is_linear_k_sum_state_initialized

    QuantizedLayer.get_k_norm_cache = get_k_norm_cache
    QuantizedLayer.set_k_norm_cache = set_k_norm_cache
    QuantizedLayer.k_norm_update = k_norm_update
    QuantizedLayer.is_k_norm_cache_initialized = is_k_norm_cache_initialized
    QuantizedLayer.get_linear_k_sum_state = get_linear_k_sum_state
    QuantizedLayer.set_linear_k_sum_state = set_linear_k_sum_state
    QuantizedLayer.linear_k_sum_state_update = linear_k_sum_state_update
    QuantizedLayer.is_linear_k_sum_state_initialized = is_linear_k_sum_state_initialized

    # Patch k_norm cache methods for Cache class
    Cache.k_norm_update = cache_k_norm_update
    Cache.is_k_norm_cache_initialized = cache_is_k_norm_cache_initialized
    Cache.get_k_norm_cache = cache_get_k_norm_cache
    Cache.linear_k_sum_state_update = cache_linear_k_sum_state_update
    Cache.is_linear_k_sum_state_initialized = cache_is_linear_k_sum_state_initialized
    Cache.get_linear_k_sum_state = cache_get_linear_k_sum_state

    # Patch hash state methods for Layer classes
    DynamicLayer.get_hash_state = get_hash_state
    DynamicLayer.set_hash_state = set_hash_state
    DynamicLayer.is_hash_state_initialized = is_hash_state_initialized
    DynamicLayer.clear_hash_state = clear_hash_state
    DynamicLayer.hash_state_update = hash_state_update

    DynamicSlidingWindowLayer.get_hash_state = get_hash_state
    DynamicSlidingWindowLayer.set_hash_state = set_hash_state
    DynamicSlidingWindowLayer.is_hash_state_initialized = is_hash_state_initialized
    DynamicSlidingWindowLayer.clear_hash_state = clear_hash_state
    DynamicSlidingWindowLayer.hash_state_update = hash_state_update

    QuantizedLayer.get_hash_state = get_hash_state
    QuantizedLayer.set_hash_state = set_hash_state
    QuantizedLayer.is_hash_state_initialized = is_hash_state_initialized
    QuantizedLayer.clear_hash_state = clear_hash_state
    QuantizedLayer.hash_state_update = hash_state_update

    # Patch hash state methods for Cache class
    Cache.get_hash_state = cache_get_hash_state
    Cache.set_hash_state = cache_set_hash_state
    Cache.is_hash_state_initialized = cache_is_hash_state_initialized
    Cache.clear_hash_state = cache_clear_hash_state
    Cache.hash_state_update = cache_hash_state_update

    print("Hacked transformers KV Cache layers to support recurrent state, k_norm cache, linear_k_sum_state, and hash state for linear attention.")


def unhack_kv_cache_recurrent_state():
    """
    Remove the monkey patch and restore original methods.

    This is useful for cleanup or testing purposes.
    """
    from transformers.cache_utils import Cache

    # Restore DynamicLayer
    DynamicLayer.lazy_initialization = _original_dynamic_layer_lazy_init
    DynamicLayer.update = _original_dynamic_layer_update
    DynamicLayer.reset = _original_dynamic_layer_reset
    if hasattr(DynamicLayer, 'get_recurrent_state'):
        delattr(DynamicLayer, 'get_recurrent_state')
    if hasattr(DynamicLayer, 'set_recurrent_state'):
        delattr(DynamicLayer, 'set_recurrent_state')
    if hasattr(DynamicLayer, 'state_update'):
        delattr(DynamicLayer, 'state_update')
    if hasattr(DynamicLayer, 'is_recurrent_state_initialized'):
        delattr(DynamicLayer, 'is_recurrent_state_initialized')

    # Restore DynamicSlidingWindowLayer
    DynamicSlidingWindowLayer.lazy_initialization = _original_sliding_layer_lazy_init
    DynamicSlidingWindowLayer.update = _original_sliding_layer_update
    DynamicSlidingWindowLayer.reset = _original_sliding_layer_reset
    if hasattr(DynamicSlidingWindowLayer, 'get_recurrent_state'):
        delattr(DynamicSlidingWindowLayer, 'get_recurrent_state')
    if hasattr(DynamicSlidingWindowLayer, 'set_recurrent_state'):
        delattr(DynamicSlidingWindowLayer, 'set_recurrent_state')
    if hasattr(DynamicSlidingWindowLayer, 'state_update'):
        delattr(DynamicSlidingWindowLayer, 'state_update')
    if hasattr(DynamicSlidingWindowLayer, 'is_recurrent_state_initialized'):
        delattr(DynamicSlidingWindowLayer, 'is_recurrent_state_initialized')

    # Restore QuantizedLayer
    QuantizedLayer.lazy_initialization = _original_quantized_layer_lazy_init
    QuantizedLayer.update = _original_quantized_layer_update
    QuantizedLayer.reset = _original_quantized_layer_reset
    if hasattr(QuantizedLayer, 'get_recurrent_state'):
        delattr(QuantizedLayer, 'get_recurrent_state')
    if hasattr(QuantizedLayer, 'set_recurrent_state'):
        delattr(QuantizedLayer, 'set_recurrent_state')
    if hasattr(QuantizedLayer, 'state_update'):
        delattr(QuantizedLayer, 'state_update')
    if hasattr(QuantizedLayer, 'is_recurrent_state_initialized'):
        delattr(QuantizedLayer, 'is_recurrent_state_initialized')

    # Restore Cache
    if hasattr(Cache, 'state_update'):
        delattr(Cache, 'state_update')
    if hasattr(Cache, 'is_recurrent_state_initialized'):
        delattr(Cache, 'is_recurrent_state_initialized')
    if hasattr(Cache, 'get_recurrent_state'):
        delattr(Cache, 'get_recurrent_state')
    if hasattr(Cache, 'k_norm_update'):
        delattr(Cache, 'k_norm_update')
    if hasattr(Cache, 'is_k_norm_cache_initialized'):
        delattr(Cache, 'is_k_norm_cache_initialized')
    if hasattr(Cache, 'get_k_norm_cache'):
        delattr(Cache, 'get_k_norm_cache')
    if hasattr(Cache, 'linear_k_sum_state_update'):
        delattr(Cache, 'linear_k_sum_state_update')
    if hasattr(Cache, 'is_linear_k_sum_state_initialized'):
        delattr(Cache, 'is_linear_k_sum_state_initialized')
    if hasattr(Cache, 'get_linear_k_sum_state'):
        delattr(Cache, 'get_linear_k_sum_state')
    if hasattr(Cache, 'get_hash_state'):
        delattr(Cache, 'get_hash_state')
    if hasattr(Cache, 'set_hash_state'):
        delattr(Cache, 'set_hash_state')
    if hasattr(Cache, 'is_hash_state_initialized'):
        delattr(Cache, 'is_hash_state_initialized')
    if hasattr(Cache, 'clear_hash_state'):
        delattr(Cache, 'clear_hash_state')
    if hasattr(Cache, 'hash_state_update'):
        delattr(Cache, 'hash_state_update')

    # Remove k_norm cache methods from layer classes
    for layer_class in [DynamicLayer, DynamicSlidingWindowLayer, QuantizedLayer]:
        if hasattr(layer_class, 'get_k_norm_cache'):
            delattr(layer_class, 'get_k_norm_cache')
        if hasattr(layer_class, 'set_k_norm_cache'):
            delattr(layer_class, 'set_k_norm_cache')
        if hasattr(layer_class, 'k_norm_update'):
            delattr(layer_class, 'k_norm_update')
        if hasattr(layer_class, 'is_k_norm_cache_initialized'):
            delattr(layer_class, 'is_k_norm_cache_initialized')
        if hasattr(layer_class, 'get_linear_k_sum_state'):
            delattr(layer_class, 'get_linear_k_sum_state')
        if hasattr(layer_class, 'set_linear_k_sum_state'):
            delattr(layer_class, 'set_linear_k_sum_state')
        if hasattr(layer_class, 'linear_k_sum_state_update'):
            delattr(layer_class, 'linear_k_sum_state_update')
        if hasattr(layer_class, 'is_linear_k_sum_state_initialized'):
            delattr(layer_class, 'is_linear_k_sum_state_initialized')

    # Remove hash state methods from layer classes
    for layer_class in [DynamicLayer, DynamicSlidingWindowLayer, QuantizedLayer]:
        if hasattr(layer_class, 'get_hash_state'):
            delattr(layer_class, 'get_hash_state')
        if hasattr(layer_class, 'set_hash_state'):
            delattr(layer_class, 'set_hash_state')
        if hasattr(layer_class, 'is_hash_state_initialized'):
            delattr(layer_class, 'is_hash_state_initialized')
        if hasattr(layer_class, 'clear_hash_state'):
            delattr(layer_class, 'clear_hash_state')
        if hasattr(layer_class, 'hash_state_update'):
            delattr(layer_class, 'hash_state_update')

    print("Restored original transformers KV Cache layer methods.")
