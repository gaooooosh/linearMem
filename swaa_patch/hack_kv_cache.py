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


def dynamic_layer_lazy_init_swaa(
    self: DynamicLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization that also initializes recurrent state."""
    # Call original initialization
    _original_dynamic_layer_lazy_init(self, key_states, value_states)
    # Initialize recurrent state container
    _init_recurrent_state(self)


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

    # Handle recurrent state update from cache_kwargs
    if cache_kwargs is not None:
        new_recurrent_state = cache_kwargs.get('recurrent_state', None)
        if new_recurrent_state is not None:
            self.recurrent_state = new_recurrent_state

    # Call original update for key-value states
    return _original_dynamic_layer_update(self, key_states, value_states, cache_kwargs)


def dynamic_layer_reset_swaa(self: DynamicLayer) -> None:
    """Patched reset that also clears recurrent state."""
    _original_dynamic_layer_reset(self)
    if hasattr(self, 'recurrent_state'):
        self.recurrent_state = None


def sliding_layer_lazy_init_swaa(
    self: DynamicSlidingWindowLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization for DynamicSlidingWindowLayer."""
    _original_sliding_layer_lazy_init(self, key_states, value_states)
    _init_recurrent_state(self)


def sliding_layer_update_swaa(
    self: DynamicSlidingWindowLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched update for DynamicSlidingWindowLayer with recurrent state support."""
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

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


def quantized_layer_lazy_init_swaa(
    self: QuantizedLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor
) -> None:
    """Patched lazy_initialization for QuantizedLayer."""
    _original_quantized_layer_lazy_init(self, key_states, value_states)
    _init_recurrent_state(self)


def quantized_layer_update_swaa(
    self: QuantizedLayer,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    cache_kwargs: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Patched update for QuantizedLayer with recurrent state support."""
    if not hasattr(self, 'recurrent_state'):
        _init_recurrent_state(self)

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


def hack_kv_cache_recurrent_state():
    """
    Apply monkey patch to transformers KV Cache layers to support recurrent state.

    This patches the following classes:
    - DynamicLayer
    - DynamicSlidingWindowLayer
    - QuantizedLayer
    - Cache (and its subclasses like DynamicCache)

    After patching, each layer will have:
    - `recurrent_state` attribute for storing linear attention state
    - `get_recurrent_state()` method to retrieve the state
    - `set_recurrent_state(state)` method to set the state
    - `state_update(state, cache_kwargs)` method to update and return state

    The Cache class will have:
    - `state_update(state, layer_idx, cache_kwargs)` method for layer-indexed updates

    The recurrent state is automatically managed during:
    - lazy_initialization: Creates empty state container
    - update: Accepts recurrent_state via cache_kwargs
    - reset: Clears the recurrent state
    """
    from transformers.cache_utils import Cache

    # Patch DynamicLayer
    DynamicLayer.lazy_initialization = dynamic_layer_lazy_init_swaa
    DynamicLayer.update = dynamic_layer_update_swaa
    DynamicLayer.reset = dynamic_layer_reset_swaa
    DynamicLayer.get_recurrent_state = get_recurrent_state
    DynamicLayer.set_recurrent_state = set_recurrent_state
    DynamicLayer.state_update = state_update

    # Patch DynamicSlidingWindowLayer
    DynamicSlidingWindowLayer.lazy_initialization = sliding_layer_lazy_init_swaa
    DynamicSlidingWindowLayer.update = sliding_layer_update_swaa
    DynamicSlidingWindowLayer.reset = sliding_layer_reset_swaa
    DynamicSlidingWindowLayer.get_recurrent_state = get_recurrent_state
    DynamicSlidingWindowLayer.set_recurrent_state = set_recurrent_state
    DynamicSlidingWindowLayer.state_update = state_update

    # Patch QuantizedLayer
    QuantizedLayer.lazy_initialization = quantized_layer_lazy_init_swaa
    QuantizedLayer.update = quantized_layer_update_swaa
    QuantizedLayer.reset = quantized_layer_reset_swaa
    QuantizedLayer.get_recurrent_state = get_recurrent_state
    QuantizedLayer.set_recurrent_state = set_recurrent_state
    QuantizedLayer.state_update = state_update

    # Patch Cache class with layer-indexed state_update
    Cache.state_update = cache_state_update

    print("Hacked transformers KV Cache layers to support recurrent state for linear attention.")


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

    # Restore Cache
    if hasattr(Cache, 'state_update'):
        delattr(Cache, 'state_update')

    print("Restored original transformers KV Cache layer methods.")
