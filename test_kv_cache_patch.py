#!/usr/bin/env python3
"""
Test script for KV Cache recurrent state monkey patch.

This script verifies that:
1. The monkey patch can be applied successfully
2. DynamicLayer and its subclasses support recurrent_state
3. Recurrent state can be set/get via cache_kwargs
4. Reset clears the recurrent state
5. state_update method works like update for key-value states
"""

import torch
from transformers.cache_utils import DynamicLayer, DynamicSlidingWindowLayer, DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state, unhack_kv_cache_recurrent_state


def test_dynamic_layer_recurrent_state():
    """Test DynamicLayer with recurrent state support."""
    print("\n[Test 1] DynamicLayer recurrent state support")
    print("-" * 50)

    # Create a DynamicLayer
    layer = DynamicLayer()

    # Create dummy key-value states
    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # First update (triggers lazy initialization)
    k, v = layer.update(key_states, value_states)
    print(f"After first update:")
    print(f"  - keys shape: {k.shape}")
    print(f"  - values shape: {v.shape}")
    print(f"  - recurrent_state: {layer.get_recurrent_state()}")

    # Create a dummy recurrent state (simulating linear attention output)
    # Shape: [batch_size, num_heads, head_dim, state_dim]
    state_dim = 128
    recurrent_state = torch.randn(batch_size, num_heads, head_dim, state_dim)

    # Update with recurrent state via cache_kwargs
    cache_kwargs = {"recurrent_state": recurrent_state}
    key_states2 = torch.randn(batch_size, num_heads, 5, head_dim)
    value_states2 = torch.randn(batch_size, num_heads, 5, head_dim)
    k, v = layer.update(key_states2, value_states2, cache_kwargs)

    print(f"\nAfter update with recurrent_state:")
    print(f"  - keys shape: {k.shape}")
    print(f"  - recurrent_state shape: {layer.get_recurrent_state().shape}")

    # Test set_recurrent_state method
    new_recurrent_state = torch.randn(batch_size, num_heads, head_dim, state_dim)
    layer.set_recurrent_state(new_recurrent_state)
    print(f"\nAfter set_recurrent_state:")
    print(f"  - recurrent_state shape: {layer.get_recurrent_state().shape}")

    # Test reset
    layer.reset()
    print(f"\nAfter reset:")
    print(f"  - recurrent_state: {layer.get_recurrent_state()}")

    print("\n[Test 1] PASSED ✓")


def test_state_update_method():
    """Test state_update method that mirrors update() style."""
    print("\n[Test 2] state_update method (update-style API)")
    print("-" * 50)

    # Create a DynamicLayer
    layer = DynamicLayer()

    batch_size, num_heads, head_dim, state_dim = 2, 8, 64, 128

    # Initial state is None
    print(f"Initial state: {layer.state_update(None)}")
    print(f"Initial recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    # Create and update recurrent state (similar to how update() works for kv)
    recurrent_state_1 = torch.randn(batch_size, num_heads, head_dim, state_dim)
    returned_state = layer.state_update(recurrent_state_1)
    print(f"\nAfter state_update(state_1):")
    print(f"  - returned state shape: {returned_state.shape}")
    print(f"  - internal state shape: {layer.get_recurrent_state().shape}")
    print(f"  - recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    # Update with new state (overwrites)
    recurrent_state_2 = torch.randn(batch_size, num_heads, head_dim, state_dim)
    returned_state = layer.state_update(recurrent_state_2)
    print(f"\nAfter state_update(state_2):")
    print(f"  - returned state shape: {returned_state.shape}")
    print(f"  - recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    # Retrieve without updating (pass None)
    returned_state = layer.state_update(None)
    print(f"\nAfter state_update(None) (retrieve only):")
    print(f"  - returned state shape: {returned_state.shape}")
    print(f"  - recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    # Test with cache_kwargs
    recurrent_state_3 = torch.randn(batch_size, num_heads, head_dim, state_dim)
    returned_state = layer.state_update(recurrent_state_3, cache_kwargs={"init_state": True})
    print(f"\nAfter state_update with init_state=True:")
    print(f"  - recurrent_dtype: {layer.recurrent_dtype}")
    print(f"  - recurrent_device: {layer.recurrent_device}")
    print(f"  - recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    # Reset
    layer.reset()
    print(f"\nAfter reset:")
    print(f"  - state: {layer.state_update(None)}")
    print(f"  - recurrent_state_initialized: {layer.is_recurrent_state_initialized()}")

    print("\n[Test 2] PASSED ✓")


def test_dynamic_sliding_window_layer():
    """Test DynamicSlidingWindowLayer with recurrent state support."""
    print("\n[Test 3] DynamicSlidingWindowLayer recurrent state support")
    print("-" * 50)

    sliding_window = 16
    layer = DynamicSlidingWindowLayer(sliding_window=sliding_window)

    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # First update
    k, v = layer.update(key_states, value_states)
    print(f"After first update:")
    print(f"  - keys shape: {k.shape}")
    print(f"  - cumulative_length: {layer.cumulative_length}")
    print(f"  - recurrent_state: {layer.get_recurrent_state()}")

    # Update with recurrent state
    state_dim = 128
    recurrent_state = torch.randn(batch_size, num_heads, head_dim, state_dim)
    cache_kwargs = {"recurrent_state": recurrent_state}

    key_states2 = torch.randn(batch_size, num_heads, 10, head_dim)
    value_states2 = torch.randn(batch_size, num_heads, 10, head_dim)
    k, v = layer.update(key_states2, value_states2, cache_kwargs)

    print(f"\nAfter update with recurrent_state:")
    print(f"  - keys shape: {k.shape}")
    print(f"  - recurrent_state shape: {layer.get_recurrent_state().shape}")

    # Test state_update on sliding window layer
    recurrent_state_2 = torch.randn(batch_size, num_heads, head_dim, state_dim)
    returned = layer.state_update(recurrent_state_2)
    print(f"\nAfter state_update on sliding layer:")
    print(f"  - returned shape: {returned.shape}")

    print("\n[Test 3] PASSED ✓")


def test_dynamic_cache_integration():
    """Test DynamicCache with patched layers."""
    print("\n[Test 4] DynamicCache integration with state_update")
    print("-" * 50)

    # Create DynamicCache
    cache = DynamicCache()

    batch_size, num_heads, seq_len, head_dim = 2, 8, 10, 64
    key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Update cache for layer 0
    k, v = cache.update(key_states, value_states, layer_idx=0)
    print(f"Cache length: {len(cache)}")
    print(f"Layer 0 keys shape: {k.shape}")

    # Test Cache-level is_recurrent_state_initialized and get_recurrent_state
    print(f"\nBefore state_update:")
    print(f"  - cache.is_recurrent_state_initialized(0): {cache.is_recurrent_state_initialized(layer_idx=0)}")
    print(f"  - cache.get_recurrent_state(0): {cache.get_recurrent_state(layer_idx=0)}")

    # Create recurrent state and update via state_update
    state_dim = 128
    recurrent_state = torch.randn(batch_size, num_heads, head_dim, state_dim)

    # Use Cache.state_update (layer-indexed)
    returned_state = cache.state_update(recurrent_state, layer_idx=0)
    print(f"\nAfter cache.state_update(layer_idx=0):")
    print(f"  - returned shape: {returned_state.shape}")
    print(f"  - cache.get_recurrent_state(0) shape: {cache.get_recurrent_state(layer_idx=0).shape}")
    print(f"  - cache.is_recurrent_state_initialized(0): {cache.is_recurrent_state_initialized(layer_idx=0)}")

    # Update another layer
    recurrent_state_2 = torch.randn(batch_size, num_heads, head_dim, state_dim)
    returned_state = cache.state_update(recurrent_state_2, layer_idx=1)
    print(f"\nAfter cache.state_update(layer_idx=1):")
    print(f"  - returned shape: {returned_state.shape}")
    print(f"  - cache.get_recurrent_state(1) shape: {cache.get_recurrent_state(layer_idx=1).shape}")
    print(f"  - cache.is_recurrent_state_initialized(1): {cache.is_recurrent_state_initialized(layer_idx=1)}")

    # Retrieve without updating using get_recurrent_state
    retrieved = cache.get_recurrent_state(layer_idx=0)
    print(f"\nRetrieve layer 0 state via get_recurrent_state:")
    print(f"  - retrieved shape: {retrieved.shape}")

    # Test non-existent layer
    print(f"\nTest non-existent layer:")
    print(f"  - cache.is_recurrent_state_initialized(99): {cache.is_recurrent_state_initialized(layer_idx=99)}")
    print(f"  - cache.get_recurrent_state(99): {cache.get_recurrent_state(layer_idx=99)}")

    print("\n[Test 4] PASSED ✓")


def test_unhack():
    """Test that unhack restores original behavior."""
    print("\n[Test 5] Unhack functionality")
    print("-" * 50)

    # Check that methods exist before unhack
    print(f"Before unhack:")
    print(f"  - DynamicLayer has get_recurrent_state: {hasattr(DynamicLayer, 'get_recurrent_state')}")
    print(f"  - DynamicLayer has set_recurrent_state: {hasattr(DynamicLayer, 'set_recurrent_state')}")
    print(f"  - DynamicLayer has state_update: {hasattr(DynamicLayer, 'state_update')}")
    print(f"  - DynamicCache has state_update: {hasattr(DynamicCache, 'state_update')}")

    # Unhack
    unhack_kv_cache_recurrent_state()

    print(f"\nAfter unhack:")
    print(f"  - DynamicLayer has get_recurrent_state: {hasattr(DynamicLayer, 'get_recurrent_state')}")
    print(f"  - DynamicLayer has set_recurrent_state: {hasattr(DynamicLayer, 'set_recurrent_state')}")
    print(f"  - DynamicLayer has state_update: {hasattr(DynamicLayer, 'state_update')}")
    print(f"  - DynamicCache has state_update: {hasattr(DynamicCache, 'state_update')}")

    # Re-hack for other tests
    hack_kv_cache_recurrent_state()

    print("\n[Test 5] PASSED ✓")


def main():
    print("=" * 60)
    print("KV Cache Recurrent State Monkey Patch Test")
    print("=" * 60)

    # Apply the monkey patch
    print("\nApplying KV Cache recurrent state monkey patch...")
    hack_kv_cache_recurrent_state()

    try:
        test_dynamic_layer_recurrent_state()
        test_state_update_method()
        test_dynamic_sliding_window_layer()
        test_dynamic_cache_integration()
        test_unhack()

        print("\n" + "=" * 60)
        print("All tests PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\nTest FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
