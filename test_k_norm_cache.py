"""
Test script to verify k_norm caching in KV Cache.
"""

import torch
from transformers import DynamicCache
from swaa_patch import hack_kv_cache_recurrent_state, unhack_kv_cache_recurrent_state


def test_k_norm_cache():
    """Test k_norm caching functionality."""
    print("=" * 60)
    print("Testing k_norm caching in KV Cache (integrated)")
    print("=" * 60)

    # Apply the patch (now includes both recurrent state and k_norm cache)
    hack_kv_cache_recurrent_state()

    # Create a cache
    cache = DynamicCache()

    # Test layer 0
    layer_idx = 0

    # Check initialization status
    print(f"\n1. Initial status for layer {layer_idx}:")
    is_initialized = cache.is_k_norm_cache_initialized(layer_idx)
    print(f"   k_norm cache initialized: {is_initialized}")
    assert not is_initialized, "Should not be initialized yet"

    # Create a test k_norm value
    k_norm_value = torch.tensor(123.456)
    print(f"\n2. Setting k_norm value: {k_norm_value.item():.3f}")

    # Update k_norm cache
    returned_value = cache.k_norm_update(k_norm_value, layer_idx)
    print(f"   Returned value: {returned_value.item():.3f}")
    assert returned_value.item() == k_norm_value.item(), "Returned value should match"

    # Check initialization status again
    print(f"\n3. After setting k_norm:")
    is_initialized = cache.is_k_norm_cache_initialized(layer_idx)
    print(f"   k_norm cache initialized: {is_initialized}")
    assert is_initialized, "Should be initialized now"

    # Retrieve cached value
    print(f"\n4. Retrieving cached value:")
    cached_value = cache.k_norm_update(None, layer_idx)
    print(f"   Cached value: {cached_value.item():.3f}")
    assert cached_value.item() == k_norm_value.item(), "Cached value should match"

    # Also test get_k_norm_cache method
    print(f"\n5. Testing get_k_norm_cache method:")
    retrieved_value = cache.get_k_norm_cache(layer_idx)
    print(f"   Retrieved value: {retrieved_value.item():.3f}")
    assert retrieved_value.item() == k_norm_value.item(), "Retrieved value should match"

    # Test multiple layers
    print(f"\n6. Testing multiple layers:")
    for layer in [0, 1, 2]:
        k_norm = torch.tensor(float(layer * 100 + 50))
        cache.k_norm_update(k_norm, layer)
        retrieved = cache.get_k_norm_cache(layer)
        print(f"   Layer {layer}: set={k_norm.item():.1f}, get={retrieved.item():.1f}")
        assert retrieved.item() == k_norm.item(), f"Layer {layer} values should match"

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)

    # Cleanup
    print("\n7. Cleaning up...")
    unhack_kv_cache_recurrent_state()
    print("   Removed k_norm cache patch")


if __name__ == "__main__":
    test_k_norm_cache()
