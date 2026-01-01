"""
Test Output Equivalence - Verify InterpretableModel produces identical outputs to original model.

Inspired by mlux/tests/test_hooked_model.py TestLogitEquivalence.

These tests ensure that wrapping a model with InterpretableModel doesn't change
its behavior when no interventions are applied.
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel


def test_simple_prompt_equivalence():
    """Test that InterpretableModel produces identical outputs to original model."""
    print("\n" + "=" * 60)
    print("TEST: Simple prompt equivalence")
    print("=" * 60)

    # Load model
    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")

    # Get output from original model
    tokens = mx.array(tokenizer.encode("Hello, world!"))
    tokens = tokens[None, :]  # Add batch dimension

    original_output = base_model(tokens)
    mx.eval(original_output)

    # Wrap with InterpretableModel
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    # Get output from wrapped model (no interventions)
    with model.trace(tokens):
        wrapped_output = model.output.save()

    mx.eval(wrapped_output)

    # Compare outputs
    diff = mx.max(mx.abs(original_output - wrapped_output))
    mx.eval(diff)

    print(f"  Original output shape: {original_output.shape}")
    print(f"  Wrapped output shape: {wrapped_output.shape}")
    print(f"  Max difference: {float(diff):.10f}")

    # Allow small numerical differences due to floating-point
    assert float(diff) < 1e-5, f"Outputs differ by {float(diff)}"

    print("PASSED: Outputs are equivalent!")
    return True


def test_various_prompts_equivalence():
    """Test output equivalence across different prompt types."""
    print("\n" + "=" * 60)
    print("TEST: Various prompts equivalence")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    prompts = [
        "The capital of France is",
        "def fibonacci(n):",
        "In 1969, Neil Armstrong",
        "2 + 2 =",
        "Hello",  # Short prompt
    ]

    all_passed = True
    for prompt in prompts:
        tokens = mx.array(tokenizer.encode(prompt))
        tokens = tokens[None, :]

        # Original
        original_output = base_model(tokens)
        mx.eval(original_output)

        # Wrapped
        with model.trace(tokens):
            wrapped_output = model.output.save()
        mx.eval(wrapped_output)

        # Compare
        diff = mx.max(mx.abs(original_output - wrapped_output))
        mx.eval(diff)

        status = "PASS" if float(diff) < 1e-5 else "FAIL"
        print(f"  '{prompt[:30]}...': diff={float(diff):.2e} [{status}]")

        if float(diff) >= 1e-5:
            all_passed = False

    assert all_passed, "Some prompts have output differences"
    print("PASSED: All prompts produce equivalent outputs!")
    return True


def test_string_vs_token_input_equivalence():
    """Test that string input and token input produce same outputs."""
    print("\n" + "=" * 60)
    print("TEST: String vs token input equivalence")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox jumps over the lazy dog"

    # Input as string
    with model.trace(text):
        string_output = model.output.save()
    mx.eval(string_output)

    # Input as tokens
    tokens = mx.array(tokenizer.encode(text))
    tokens = tokens[None, :]
    with model.trace(tokens):
        token_output = model.output.save()
    mx.eval(token_output)

    # Compare
    diff = mx.max(mx.abs(string_output - token_output))
    mx.eval(diff)

    print(f"  String input shape: {string_output.shape}")
    print(f"  Token input shape: {token_output.shape}")
    print(f"  Max difference: {float(diff):.10f}")

    assert float(diff) < 1e-5, f"Outputs differ by {float(diff)}"
    print("PASSED: String and token inputs produce equivalent outputs!")
    return True


def test_argmax_equivalence():
    """Test that argmax (predicted token) is identical between original and wrapped."""
    print("\n" + "=" * 60)
    print("TEST: Argmax equivalence")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    prompts = [
        "The capital of France is",
        "One plus one equals",
        "The color of the sky is",
    ]

    all_passed = True
    for prompt in prompts:
        tokens = mx.array(tokenizer.encode(prompt))
        tokens = tokens[None, :]

        # Original
        original_output = base_model(tokens)
        mx.eval(original_output)
        original_pred = int(mx.argmax(original_output[0, -1]))

        # Wrapped
        with model.trace(tokens):
            wrapped_output = model.output.save()
        mx.eval(wrapped_output)
        wrapped_pred = int(mx.argmax(wrapped_output[0, -1]))

        original_token = tokenizer.decode([original_pred])
        wrapped_token = tokenizer.decode([wrapped_pred])

        status = "PASS" if original_pred == wrapped_pred else "FAIL"
        print(f"  '{prompt}': original='{original_token}', wrapped='{wrapped_token}' [{status}]")

        if original_pred != wrapped_pred:
            all_passed = False

    assert all_passed, "Predicted tokens differ"
    print("PASSED: Argmax predictions are identical!")
    return True


def test_multiple_traces_consistency():
    """Test that multiple traces produce consistent outputs."""
    print("\n" + "=" * 60)
    print("TEST: Multiple traces consistency")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Paris is the capital of France"

    outputs = []
    for i in range(3):
        with model.trace(text):
            output = model.output.save()
        mx.eval(output)
        outputs.append(output)

    # Compare all outputs
    all_same = True
    for i in range(1, len(outputs)):
        diff = mx.max(mx.abs(outputs[0] - outputs[i]))
        mx.eval(diff)
        if float(diff) >= 1e-10:
            all_same = False
            print(f"  Run 0 vs Run {i}: diff={float(diff):.2e}")

    if all_same:
        print("  All 3 runs produced identical outputs")

    assert all_same, "Multiple traces produced different outputs"
    print("PASSED: Multiple traces are consistent!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("OUTPUT EQUIVALENCE TEST SUITE")
    print("Verifying InterpretableModel produces identical outputs to base model")
    print("=" * 70)

    tests = [
        test_simple_prompt_equivalence,
        test_various_prompts_equivalence,
        test_string_vs_token_input_equivalence,
        test_argmax_equivalence,
        test_multiple_traces_consistency,
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
        print("ALL OUTPUT EQUIVALENCE TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
