"""
Test Edge Cases - Verify mlxterp handles edge cases correctly.

Inspired by mlux/tests/test_misc.py TestEdgeCasesAndSpecialInputs.

These tests verify that mlxterp handles unusual inputs correctly:
- Single token input
- Very long input
- Special characters and unicode
- Empty interventions
- Various text types
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


def test_single_token_input():
    """Test handling of minimal single-token input."""
    print("\n" + "=" * 60)
    print("TEST: Single token input")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    # Very short inputs
    short_inputs = ["Hi", "A", "!", "1"]

    for text in short_inputs:
        tokens = tokenizer.encode(text)
        print(f"  '{text}' -> {len(tokens)} tokens")

        with model.trace(text):
            output = model.output.save()
        mx.eval(output)

        # Output should be valid
        assert not mx.any(mx.isnan(output)), f"NaN in output for '{text}'"
        assert not mx.any(mx.isinf(output)), f"Inf in output for '{text}'"

    print("PASSED: Single token inputs handled correctly!")
    return True


def test_very_long_input():
    """Test handling of very long input sequences."""
    print("\n" + "=" * 60)
    print("TEST: Very long input")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    # Create a long input (100+ words)
    long_text = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
    tokens = tokenizer.encode(long_text)
    print(f"  Input length: {len(long_text)} chars, {len(tokens)} tokens")

    with model.trace(long_text):
        output = model.output.save()
    mx.eval(output)

    print(f"  Output shape: {output.shape}")

    # Verify output is valid
    assert not mx.any(mx.isnan(output)), "NaN in output for long input"
    assert not mx.any(mx.isinf(output)), "Inf in output for long input"
    assert output.shape[1] == len(tokens), f"Output seq length mismatch"

    print("PASSED: Very long input handled correctly!")
    return True


def test_special_characters():
    """Test handling of special characters, unicode, and emoji."""
    print("\n" + "=" * 60)
    print("TEST: Special characters")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    special_inputs = [
        "Hello 世界!",  # Chinese
        "Bonjour le monde!",  # French
        "Привет мир!",  # Russian
        "こんにちは世界",  # Japanese
        "Test with 🎉 emoji 🚀",  # Emoji
        "Math: x² + y² = z²",  # Math symbols
        "Code: def f(x): return x * 2",  # Code
        "Punctuation: !@#$%^&*()",  # Special chars
    ]

    all_passed = True
    for text in special_inputs:
        try:
            with model.trace(text):
                output = model.output.save()
            mx.eval(output)

            has_nan = mx.any(mx.isnan(output))
            has_inf = mx.any(mx.isinf(output))

            if has_nan or has_inf:
                print(f"  '{text[:30]}...': FAIL (NaN/Inf)")
                all_passed = False
            else:
                print(f"  '{text[:30]}...': OK")
        except Exception as e:
            print(f"  '{text[:30]}...': ERROR - {e}")
            all_passed = False

    assert all_passed, "Some special character inputs failed"
    print("PASSED: Special characters handled correctly!")
    return True


def test_empty_interventions():
    """Test that empty interventions dict produces unmodified output."""
    print("\n" + "=" * 60)
    print("TEST: Empty interventions")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is"

    # Without interventions
    with model.trace(text):
        output_no_intervention = model.output.save()
    mx.eval(output_no_intervention)

    # With empty interventions dict
    with model.trace(text, interventions={}):
        output_empty_intervention = model.output.save()
    mx.eval(output_empty_intervention)

    # Outputs should be identical
    diff = mx.max(mx.abs(output_no_intervention - output_empty_intervention))
    mx.eval(diff)

    print(f"  Max difference: {float(diff):.10f}")
    assert float(diff) < 1e-10, "Empty interventions changed output"

    print("PASSED: Empty interventions produce identical output!")
    return True


def test_whitespace_only_input():
    """Test handling of whitespace-only input."""
    print("\n" + "=" * 60)
    print("TEST: Whitespace only input")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    whitespace_inputs = [
        " ",
        "   ",
        "\t",
        "\n",
    ]

    for text in whitespace_inputs:
        tokens = tokenizer.encode(text)
        if len(tokens) > 0:  # Some tokenizers might produce empty for whitespace
            with model.trace(text):
                output = model.output.save()
            mx.eval(output)
            print(f"  Whitespace ({len(text)} chars): {len(tokens)} tokens, output shape={output.shape}")

    print("PASSED: Whitespace input handled!")
    return True


def test_repeated_traces():
    """Test that repeated traces produce consistent results."""
    print("\n" + "=" * 60)
    print("TEST: Repeated traces consistency")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Test input"
    num_repeats = 10

    outputs = []
    for i in range(num_repeats):
        with model.trace(text):
            output = model.output.save()
        mx.eval(output)
        outputs.append(output)

    # All outputs should be identical
    all_same = True
    for i in range(1, num_repeats):
        diff = mx.max(mx.abs(outputs[0] - outputs[i]))
        mx.eval(diff)
        if float(diff) > 1e-10:
            all_same = False
            print(f"  Run 0 vs Run {i}: diff={float(diff):.2e}")

    if all_same:
        print(f"  All {num_repeats} runs produced identical outputs")

    assert all_same, "Repeated traces produced different outputs"
    print("PASSED: Repeated traces are consistent!")
    return True


def test_alternating_interventions():
    """Test alternating between intervention and no intervention."""
    print("\n" + "=" * 60)
    print("TEST: Alternating interventions")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Paris is the capital of France"

    # Get baseline
    with model.trace(text):
        baseline = model.output.save()
    mx.eval(baseline)

    # Alternate interventions
    for i in range(5):
        if i % 2 == 0:
            # With intervention
            with model.trace(text, interventions={"layers.5": iv.scale(0.5)}):
                output = model.output.save()
            mx.eval(output)
        else:
            # Without intervention
            with model.trace(text):
                output = model.output.save()
            mx.eval(output)

            # This should match baseline
            diff = mx.max(mx.abs(baseline - output))
            mx.eval(diff)
            if float(diff) > 1e-10:
                print(f"  Iteration {i}: baseline mismatch, diff={float(diff):.2e}")
                assert False, "Non-intervention output doesn't match baseline"

    print("  All non-intervention runs matched baseline")
    print("PASSED: Alternating interventions work correctly!")
    return True


def test_numeric_input_handling():
    """Test handling of numeric-only inputs."""
    print("\n" + "=" * 60)
    print("TEST: Numeric input handling")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    numeric_inputs = [
        "123",
        "3.14159",
        "1 + 2 = 3",
        "42",
        "1000000",
        "0.001",
    ]

    for text in numeric_inputs:
        with model.trace(text):
            output = model.output.save()
        mx.eval(output)

        assert not mx.any(mx.isnan(output)), f"NaN for '{text}'"
        assert not mx.any(mx.isinf(output)), f"Inf for '{text}'"
        print(f"  '{text}': OK")

    print("PASSED: Numeric inputs handled correctly!")
    return True


def test_mixed_case_input():
    """Test handling of mixed case inputs."""
    print("\n" + "=" * 60)
    print("TEST: Mixed case input")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    mixed_inputs = [
        "HELLO WORLD",
        "hello world",
        "HeLLo WoRLd",
        "hElLO wOrLD",
    ]

    outputs = []
    for text in mixed_inputs:
        with model.trace(text):
            output = model.output.save()
        mx.eval(output)
        outputs.append(output)
        print(f"  '{text}': shape={output.shape}")

    # Different cases should produce different outputs (model is case-sensitive)
    # Just verify all outputs are valid
    for i, output in enumerate(outputs):
        assert not mx.any(mx.isnan(output)), f"NaN for input {i}"
        assert not mx.any(mx.isinf(output)), f"Inf for input {i}"

    print("PASSED: Mixed case inputs handled correctly!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("EDGE CASES TEST SUITE")
    print("Verifying mlxterp handles unusual inputs correctly")
    print("=" * 70)

    tests = [
        test_single_token_input,
        test_very_long_input,
        test_special_characters,
        test_empty_interventions,
        test_whitespace_only_input,
        test_repeated_traces,
        test_alternating_interventions,
        test_numeric_input_handling,
        test_mixed_case_input,
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
        print("ALL EDGE CASE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
