"""
Test Attention Weights Capture - Verify attention weights are captured correctly.

Tests the new AttentionWrapper functionality that captures attention patterns
(softmax weights) during tracing.
"""

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlxterp import InterpretableModel


def test_attention_weights_captured():
    """Test that attention weights are captured during tracing."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights captured")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is Paris"

    with model.trace(text) as trace:
        pass

    # Check that attention weights are captured
    attention_weight_keys = [
        k for k in trace.activations.keys()
        if ".attention_weights" in k
    ]

    print(f"  Found {len(attention_weight_keys)} attention weight activations")

    assert len(attention_weight_keys) > 0, "No attention weights captured!"

    # Check at least one layer
    for key in attention_weight_keys[:3]:
        print(f"  {key}")

    print("PASSED: Attention weights are captured!")
    return True


def test_attention_weights_shape():
    """Test that attention weights have correct shape."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights shape")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello world"
    tokens = tokenizer.encode(text)
    seq_len = len(tokens)

    with model.trace(text) as trace:
        pass

    # Find attention weights for layer 0
    key = None
    for k in trace.activations.keys():
        if "layers.0.self_attn.attention_weights" in k:
            key = k
            break

    assert key is not None, "Layer 0 attention weights not found"

    weights = trace.activations[key]
    mx.eval(weights)

    print(f"  Attention weights shape: {weights.shape}")
    print(f"  Expected: (1, num_heads, {seq_len}, {seq_len})")

    # Shape should be (batch, num_heads, seq_len, seq_len)
    assert len(weights.shape) == 4, f"Expected 4D tensor, got {len(weights.shape)}D"
    assert weights.shape[0] == 1, f"Batch should be 1, got {weights.shape[0]}"
    assert weights.shape[2] == seq_len, f"Query seq_len should be {seq_len}, got {weights.shape[2]}"
    assert weights.shape[3] == seq_len, f"Key seq_len should be {seq_len}, got {weights.shape[3]}"

    print("PASSED: Attention weights shape is correct!")
    return True


def test_attention_weights_sum_to_one():
    """Test that attention weights sum to 1 along the key dimension (proper softmax)."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights sum to 1")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox"

    with model.trace(text) as trace:
        pass

    # Find attention weights for layer 5
    key = None
    for k in trace.activations.keys():
        if "layers.5.self_attn.attention_weights" in k:
            key = k
            break

    assert key is not None, "Layer 5 attention weights not found"

    weights = trace.activations[key]
    mx.eval(weights)

    # Sum along key dimension (last axis)
    # For causal attention, each row should sum to 1 (or close to it)
    row_sums = mx.sum(weights, axis=-1)
    mx.eval(row_sums)

    # Convert to numpy for easier checking
    row_sums_np = np.array(row_sums)

    # All sums should be very close to 1
    print(f"  Row sum mean: {row_sums_np.mean():.6f}")
    print(f"  Row sum min: {row_sums_np.min():.6f}")
    print(f"  Row sum max: {row_sums_np.max():.6f}")

    assert np.allclose(row_sums_np, 1.0, atol=1e-5), "Attention weights don't sum to 1!"

    print("PASSED: Attention weights properly normalized!")
    return True


def test_attention_weights_causal_mask():
    """Test that attention weights respect causal masking (no attending to future)."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights causal mask")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "One two three four five"

    with model.trace(text) as trace:
        pass

    # Find attention weights
    key = None
    for k in trace.activations.keys():
        if "layers.3.self_attn.attention_weights" in k:
            key = k
            break

    assert key is not None, "Layer 3 attention weights not found"

    weights = trace.activations[key]
    mx.eval(weights)

    # Convert to numpy
    weights_np = np.array(weights)

    # Check upper triangle (future positions) is zero
    # For position i, positions j > i should have zero attention
    batch, num_heads, seq_q, seq_k = weights_np.shape

    # Check a few positions
    violations = 0
    for q_pos in range(seq_q):
        for k_pos in range(q_pos + 1, seq_k):
            # Future position - should be zero
            future_attention = weights_np[0, :, q_pos, k_pos]
            if np.any(future_attention > 1e-5):
                violations += 1

    print(f"  Checked causal mask, violations: {violations}")

    assert violations == 0, f"Found {violations} causal mask violations!"

    print("PASSED: Causal mask is respected!")
    return True


def test_attention_weights_multiple_layers():
    """Test that attention weights are captured for all layers."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights for multiple layers")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Test"

    with model.trace(text) as trace:
        pass

    num_layers = len(model.layers)

    # Count layers with attention weights
    layers_with_weights = 0
    for i in range(num_layers):
        found = False
        for k in trace.activations.keys():
            if f"layers.{i}.self_attn.attention_weights" in k:
                found = True
                break
        if found:
            layers_with_weights += 1

    print(f"  Total layers: {num_layers}")
    print(f"  Layers with attention weights: {layers_with_weights}")

    assert layers_with_weights == num_layers, \
        f"Expected {num_layers} layers with attention weights, got {layers_with_weights}"

    print("PASSED: All layers have attention weights!")
    return True


def test_attention_weights_not_nan_inf():
    """Test that attention weights don't contain NaN or Inf."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights not NaN/Inf")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox jumps over the lazy dog"

    with model.trace(text) as trace:
        pass

    # Check all attention weights
    issues = []
    checked = 0
    for key, value in trace.activations.items():
        if ".attention_weights" in key:
            mx.eval(value)

            has_nan = bool(mx.any(mx.isnan(value)))
            has_inf = bool(mx.any(mx.isinf(value)))

            if has_nan or has_inf:
                issues.append(f"{key}: NaN={has_nan}, Inf={has_inf}")
            checked += 1

    print(f"  Checked {checked} attention weight tensors")

    if issues:
        for issue in issues:
            print(f"  ISSUE: {issue}")

    assert len(issues) == 0, f"Found {len(issues)} attention weights with NaN/Inf"

    print("PASSED: No NaN/Inf in attention weights!")
    return True


def test_attention_weights_range():
    """Test that attention weights are in valid range [0, 1]."""
    print("\n" + "=" * 60)
    print("TEST: Attention weights in valid range")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Machine learning"

    with model.trace(text) as trace:
        pass

    # Check attention weight ranges
    for key, value in trace.activations.items():
        if ".attention_weights" in key:
            mx.eval(value)

            min_val = float(mx.min(value))
            max_val = float(mx.max(value))

            assert min_val >= -1e-5, f"{key}: min value {min_val} < 0"
            assert max_val <= 1.0 + 1e-5, f"{key}: max value {max_val} > 1"

    print("PASSED: Attention weights in valid range [0, 1]!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ATTENTION WEIGHTS CAPTURE TEST SUITE")
    print("Testing that attention patterns are correctly captured")
    print("=" * 70)

    tests = [
        test_attention_weights_captured,
        test_attention_weights_shape,
        test_attention_weights_sum_to_one,
        test_attention_weights_causal_mask,
        test_attention_weights_multiple_layers,
        test_attention_weights_not_nan_inf,
        test_attention_weights_range,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAILED: {test.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    if failed == 0:
        print("ALL ATTENTION WEIGHTS TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
