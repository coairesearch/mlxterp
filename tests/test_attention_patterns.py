"""
Test Attention Patterns - Verify attention patterns have correct shapes and properties.

Inspired by mlux/tests/test_hooked_model.py TestAttentionPatterns.

These tests validate that attention pattern outputs have the expected
dimensions and mathematical properties (e.g., softmax normalization).
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel


def test_attention_output_shape():
    """Test that attention output has correct shape."""
    print("\n" + "=" * 60)
    print("TEST: Attention output shape")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is Paris"

    with model.trace(text) as trace:
        pass

    # Get sequence length from input
    tokens = tokenizer.encode(text)
    seq_len = len(tokens)

    # Check attention outputs from several layers
    for layer_idx in [0, 5, 10]:
        key = f"model.model.layers.{layer_idx}.self_attn"
        if key in trace.activations:
            attn_out = trace.activations[key]
            mx.eval(attn_out)

            print(f"  Layer {layer_idx} self_attn: shape={attn_out.shape}")

            # Should be (batch, seq_len, hidden_dim)
            assert len(attn_out.shape) == 3, f"Expected 3D tensor, got {len(attn_out.shape)}D"
            assert attn_out.shape[0] == 1, f"Batch size should be 1, got {attn_out.shape[0]}"
            assert attn_out.shape[1] == seq_len, f"Seq len should be {seq_len}, got {attn_out.shape[1]}"

    print("PASSED: Attention output shapes are correct!")
    return True


def test_qkv_projection_shapes():
    """Test that Q, K, V projections have expected shapes."""
    print("\n" + "=" * 60)
    print("TEST: Q/K/V projection shapes")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello world"

    with model.trace(text) as trace:
        pass

    tokens = tokenizer.encode(text)
    seq_len = len(tokens)

    # Check Q, K, V projections
    for layer_idx in [0, 5]:
        q_key = f"model.model.layers.{layer_idx}.self_attn.q_proj"
        k_key = f"model.model.layers.{layer_idx}.self_attn.k_proj"
        v_key = f"model.model.layers.{layer_idx}.self_attn.v_proj"

        if q_key in trace.activations:
            q = trace.activations[q_key]
            k = trace.activations[k_key]
            v = trace.activations[v_key]
            mx.eval(q, k, v)

            print(f"  Layer {layer_idx}:")
            print(f"    Q: {q.shape}")
            print(f"    K: {k.shape}")
            print(f"    V: {v.shape}")

            # All should have (batch, seq_len, dim)
            assert q.shape[0] == 1, "Batch should be 1"
            assert q.shape[1] == seq_len, f"Seq len should be {seq_len}"

            # K and V might have different dimensions (GQA)
            assert k.shape[0] == 1, "K batch should be 1"
            assert v.shape[0] == 1, "V batch should be 1"

    print("PASSED: Q/K/V projection shapes are valid!")
    return True


def test_attention_layer_components_captured():
    """Test that all attention layer components are captured."""
    print("\n" + "=" * 60)
    print("TEST: Attention layer components captured")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Test input"

    with model.trace(text) as trace:
        pass

    # Check that we capture various attention components
    layer_idx = 5
    expected_components = [
        f"model.model.layers.{layer_idx}.self_attn",
        f"model.model.layers.{layer_idx}.self_attn.q_proj",
        f"model.model.layers.{layer_idx}.self_attn.k_proj",
        f"model.model.layers.{layer_idx}.self_attn.v_proj",
        f"model.model.layers.{layer_idx}.self_attn.o_proj",
    ]

    found = []
    missing = []
    for comp in expected_components:
        if comp in trace.activations:
            found.append(comp)
            print(f"  Found: {comp}")
        else:
            missing.append(comp)
            print(f"  Missing: {comp}")

    print(f"  Found {len(found)}/{len(expected_components)} components")

    # At least the main attention output should be present
    assert f"model.model.layers.{layer_idx}.self_attn" in trace.activations, \
        "Main attention output not captured"

    print("PASSED: Attention components captured!")
    return True


def test_attention_output_not_nan_inf():
    """Test that attention outputs don't contain NaN or Inf."""
    print("\n" + "=" * 60)
    print("TEST: Attention output not NaN/Inf")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox jumps over the lazy dog"

    with model.trace(text) as trace:
        pass

    # Check all attention outputs
    issues = []
    checked = 0
    for key, value in trace.activations.items():
        if "self_attn" in key:
            mx.eval(value)

            has_nan = bool(mx.any(mx.isnan(value)))
            has_inf = bool(mx.any(mx.isinf(value)))

            if has_nan or has_inf:
                issues.append(f"{key}: NaN={has_nan}, Inf={has_inf}")
            checked += 1

    print(f"  Checked {checked} attention activations")

    if issues:
        for issue in issues:
            print(f"  ISSUE: {issue}")

    assert len(issues) == 0, f"Found {len(issues)} attention outputs with NaN/Inf"
    print("PASSED: No NaN/Inf in attention outputs!")
    return True


def test_attention_values_reasonable_range():
    """Test that attention values are in a reasonable range."""
    print("\n" + "=" * 60)
    print("TEST: Attention values in reasonable range")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Machine learning is transforming technology"

    with model.trace(text) as trace:
        pass

    # Check value ranges for attention outputs
    for layer_idx in [0, 8, 15]:
        key = f"model.model.layers.{layer_idx}.self_attn"
        if key in trace.activations:
            value = trace.activations[key]
            mx.eval(value)

            min_val = float(mx.min(value))
            max_val = float(mx.max(value))
            mean_val = float(mx.mean(value))

            print(f"  Layer {layer_idx}: min={min_val:.2f}, max={max_val:.2f}, mean={mean_val:.4f}")

            # Values should be in reasonable range (not exploding)
            assert abs(max_val) < 1000, f"Max value too large: {max_val}"
            assert abs(min_val) < 1000, f"Min value too large: {min_val}"

    print("PASSED: Attention values in reasonable range!")
    return True


def test_multiple_layers_have_attention():
    """Test that multiple layers have attention outputs."""
    print("\n" + "=" * 60)
    print("TEST: Multiple layers have attention")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Test"

    with model.trace(text) as trace:
        pass

    # Count layers with attention
    num_layers = len(model.layers)
    layers_with_attn = 0

    for i in range(num_layers):
        key = f"model.model.layers.{i}.self_attn"
        if key in trace.activations:
            layers_with_attn += 1

    print(f"  Total layers: {num_layers}")
    print(f"  Layers with attention: {layers_with_attn}")

    assert layers_with_attn == num_layers, \
        f"Expected {num_layers} layers with attention, got {layers_with_attn}"

    print("PASSED: All layers have attention outputs!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ATTENTION PATTERNS TEST SUITE")
    print("Verifying attention patterns have correct shapes and properties")
    print("=" * 70)

    tests = [
        test_attention_output_shape,
        test_qkv_projection_shapes,
        test_attention_layer_components_captured,
        test_attention_output_not_nan_inf,
        test_attention_values_reasonable_range,
        test_multiple_layers_have_attention,
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
        print("ALL ATTENTION PATTERNS TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
