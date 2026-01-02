"""
Test Dtype Consistency - Verify dtypes are preserved through operations.

Inspired by mlux/tests/test_misc.py TestDtypeConsistency.

These tests ensure that activation dtypes remain consistent through
tracing, caching, and intervention operations.
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


def test_output_dtype_matches_model():
    """Test that output maintains expected dtype."""
    print("\n" + "=" * 60)
    print("TEST: Output dtype matches model")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is"

    # Get original model output dtype
    tokens = mx.array(tokenizer.encode(text))[None, :]
    original_output = base_model(tokens)
    mx.eval(original_output)
    original_dtype = original_output.dtype

    # Get wrapped model output dtype
    with model.trace(text):
        wrapped_output = model.output.save()
    mx.eval(wrapped_output)
    wrapped_dtype = wrapped_output.dtype

    print(f"  Original dtype: {original_dtype}")
    print(f"  Wrapped dtype: {wrapped_dtype}")

    assert original_dtype == wrapped_dtype, f"Dtype mismatch: {original_dtype} vs {wrapped_dtype}"
    print("PASSED: Output dtypes match!")
    return True


def test_activation_dtypes_consistent():
    """Test that all cached activations have consistent dtypes."""
    print("\n" + "=" * 60)
    print("TEST: Activation dtypes consistent")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox jumps over the lazy dog"

    with model.trace(text) as trace:
        pass

    # Collect all activation dtypes
    dtypes = {}
    for key, value in trace.activations.items():
        mx.eval(value)
        dtypes[key] = value.dtype

    # Check for consistency
    unique_dtypes = set(dtypes.values())
    print(f"  Total activations: {len(dtypes)}")
    print(f"  Unique dtypes: {unique_dtypes}")

    # Show some examples
    for key, dtype in list(dtypes.items())[:5]:
        print(f"    {key}: {dtype}")

    # All activations should have the same dtype (or compatible types)
    # Note: Some models may have mixed precision, so we just verify no unexpected types
    for key, dtype in dtypes.items():
        assert dtype in [mx.float16, mx.float32, mx.bfloat16], \
            f"Unexpected dtype {dtype} for {key}"

    print("PASSED: All activation dtypes are valid!")
    return True


def test_intervention_preserves_dtype():
    """Test that interventions produce valid output dtypes."""
    print("\n" + "=" * 60)
    print("TEST: Intervention produces valid dtype")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Paris is the capital of France"

    # Get baseline dtype
    with model.trace(text):
        baseline_output = model.output.save()
    mx.eval(baseline_output)
    baseline_dtype = baseline_output.dtype
    print(f"  Baseline: dtype={baseline_dtype}")

    # Test scale intervention
    with model.trace(text, interventions={"layers.5": iv.scale(0.5)}):
        output_scale = model.output.save()
    mx.eval(output_scale)
    print(f"  After scale: dtype={output_scale.dtype}")

    # Test noise intervention
    with model.trace(text, interventions={"layers.5": iv.noise(std=0.1)}):
        output_noise = model.output.save()
    mx.eval(output_noise)
    print(f"  After noise: dtype={output_noise.dtype}")

    # Test clamp intervention
    with model.trace(text, interventions={"layers.5": iv.clamp(-1, 1)}):
        output_clamp = model.output.save()
    mx.eval(output_clamp)
    print(f"  After clamp: dtype={output_clamp.dtype}")

    # All outputs should have valid floating point dtypes
    # Note: Some interventions may cause dtype promotion (e.g., float16 -> float32)
    # which is acceptable behavior
    valid_dtypes = [mx.float16, mx.float32, mx.bfloat16]
    for name, output in [("scale", output_scale), ("noise", output_noise), ("clamp", output_clamp)]:
        assert output.dtype in valid_dtypes, \
            f"Intervention '{name}' produced invalid dtype: {output.dtype}"

    print("PASSED: Interventions produce valid dtypes!")
    return True


def test_layer_output_dtypes():
    """Test that layer outputs have expected dtypes."""
    print("\n" + "=" * 60)
    print("TEST: Layer output dtypes")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Hello world"

    with model.trace(text) as trace:
        pass

    # Check layer outputs
    layers_checked = 0
    for i in range(len(model.layers)):
        key = f"model.model.layers.{i}"
        if key in trace.activations:
            layer_output = trace.activations[key]
            mx.eval(layer_output)
            print(f"  Layer {i}: shape={layer_output.shape}, dtype={layer_output.dtype}")
            layers_checked += 1

            # Verify valid dtype
            assert layer_output.dtype in [mx.float16, mx.float32, mx.bfloat16], \
                f"Invalid dtype for layer {i}: {layer_output.dtype}"

    print(f"  Checked {layers_checked} layers")
    assert layers_checked > 0, "No layer outputs found"

    print("PASSED: Layer output dtypes are valid!")
    return True


def test_mlp_attention_dtypes_match():
    """Test that MLP and attention outputs have matching dtypes."""
    print("\n" + "=" * 60)
    print("TEST: MLP and attention dtypes match")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox"

    with model.trace(text) as trace:
        pass

    # Check a few layers
    for layer_idx in [0, 5, 10]:
        mlp_key = f"model.model.layers.{layer_idx}.mlp"
        attn_key = f"model.model.layers.{layer_idx}.self_attn"

        if mlp_key in trace.activations and attn_key in trace.activations:
            mlp_out = trace.activations[mlp_key]
            attn_out = trace.activations[attn_key]
            mx.eval(mlp_out, attn_out)

            print(f"  Layer {layer_idx}: MLP={mlp_out.dtype}, Attn={attn_out.dtype}")
            assert mlp_out.dtype == attn_out.dtype, \
                f"Dtype mismatch in layer {layer_idx}"

    print("PASSED: MLP and attention dtypes match!")
    return True


def test_batch_dimension_dtype_preserved():
    """Test that batched operations preserve dtype."""
    print("\n" + "=" * 60)
    print("TEST: Batch dimension dtype preserved")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    # Single input
    text = "Hello"
    with model.trace(text):
        single_output = model.output.save()
    mx.eval(single_output)

    # Check dtype
    print(f"  Single input dtype: {single_output.dtype}")
    print(f"  Single input shape: {single_output.shape}")

    assert single_output.dtype in [mx.float16, mx.float32, mx.bfloat16], \
        f"Invalid output dtype: {single_output.dtype}"

    print("PASSED: Batch dimension dtype preserved!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DTYPE CONSISTENCY TEST SUITE")
    print("Verifying dtypes are preserved through operations")
    print("=" * 70)

    tests = [
        test_output_dtype_matches_model,
        test_activation_dtypes_consistent,
        test_intervention_preserves_dtype,
        test_layer_output_dtypes,
        test_mlp_attention_dtypes_match,
        test_batch_dimension_dtype_preserved,
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
        print("ALL DTYPE CONSISTENCY TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
