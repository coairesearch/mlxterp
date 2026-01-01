"""
Test Model Restoration - Ensure model state is unchanged after hooks/interventions.

Inspired by mlux/tests/test_hooks_advanced.py TestModelRestoration.

These tests verify that after running with hooks, caches, or interventions,
the model returns to its original state and produces identical outputs.
"""

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


def test_model_unchanged_after_trace():
    """Test that model outputs are identical before and after tracing."""
    print("\n" + "=" * 60)
    print("TEST: Model unchanged after trace")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is"
    tokens = mx.array(tokenizer.encode(text))[None, :]

    # Output before any tracing
    output_before = base_model(tokens)
    mx.eval(output_before)

    # Run several traces
    for _ in range(3):
        with model.trace(text) as trace:
            _ = trace.activations  # Access activations

    # Output after tracing
    output_after = base_model(tokens)
    mx.eval(output_after)

    diff = mx.max(mx.abs(output_before - output_after))
    mx.eval(diff)

    print(f"  Max difference: {float(diff):.10f}")
    assert float(diff) < 1e-10, f"Model changed after tracing: diff={float(diff)}"

    print("PASSED: Model unchanged after tracing!")
    return True


def test_model_unchanged_after_interventions():
    """Test that model outputs are identical before and after interventions."""
    print("\n" + "=" * 60)
    print("TEST: Model unchanged after interventions")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is"
    tokens = mx.array(tokenizer.encode(text))[None, :]

    # Output before interventions
    output_before = base_model(tokens)
    mx.eval(output_before)

    # Run several interventions
    with model.trace(text, interventions={"layers.5": iv.scale(0.5)}):
        pass

    with model.trace(text, interventions={"layers.10.mlp": iv.zero_out}):
        pass

    with model.trace(text, interventions={"layers.3": iv.noise(std=0.1)}):
        pass

    # Output after interventions (should be same as before)
    output_after = base_model(tokens)
    mx.eval(output_after)

    diff = mx.max(mx.abs(output_before - output_after))
    mx.eval(diff)

    print(f"  Max difference: {float(diff):.10f}")
    assert float(diff) < 1e-10, f"Model changed after interventions: diff={float(diff)}"

    print("PASSED: Model unchanged after interventions!")
    return True


def test_model_unchanged_after_activation_saving():
    """Test that saving activations doesn't modify model state."""
    print("\n" + "=" * 60)
    print("TEST: Model unchanged after activation saving")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The quick brown fox"
    tokens = mx.array(tokenizer.encode(text))[None, :]

    # Output before
    output_before = base_model(tokens)
    mx.eval(output_before)

    # Save lots of activations
    with model.trace(text) as trace:
        for i in range(len(model.layers)):
            layer_act = trace.activations.get(f"model.model.layers.{i}")
            if layer_act is not None:
                mx.eval(layer_act)

    # Output after
    output_after = base_model(tokens)
    mx.eval(output_after)

    diff = mx.max(mx.abs(output_before - output_after))
    mx.eval(diff)

    print(f"  Max difference: {float(diff):.10f}")
    assert float(diff) < 1e-10, f"Model changed after activation saving: diff={float(diff)}"

    print("PASSED: Model unchanged after activation saving!")
    return True


def test_sequential_operations_dont_leak():
    """Test that sequential operations don't interfere with each other."""
    print("\n" + "=" * 60)
    print("TEST: Sequential operations don't leak")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "Paris is the capital of France"

    # Baseline output
    with model.trace(text):
        baseline = model.output.save()
    mx.eval(baseline)

    # Series of different operations
    operations = [
        ("Scale", {"layers.5": iv.scale(2.0)}),
        ("Zero out", {"layers.10": iv.zero_out}),
        ("Add noise", {"layers.3": iv.noise(std=1.0)}),
        ("Clamp", {"layers.8": iv.clamp(-1, 1)}),
    ]

    for name, interventions in operations:
        with model.trace(text, interventions=interventions):
            _ = model.output.save()

    # After all operations, baseline should be unchanged
    with model.trace(text):
        after_ops = model.output.save()
    mx.eval(after_ops)

    diff = mx.max(mx.abs(baseline - after_ops))
    mx.eval(diff)

    print(f"  Max difference from baseline: {float(diff):.10f}")
    assert float(diff) < 1e-10, f"Operations leaked state: diff={float(diff)}"

    print("PASSED: Sequential operations don't leak!")
    return True


def test_intervention_is_temporary():
    """Test that interventions only affect the current trace."""
    print("\n" + "=" * 60)
    print("TEST: Interventions are temporary")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    text = "The capital of France is"

    # Get baseline output
    with model.trace(text):
        baseline = model.output.save()
    mx.eval(baseline)

    # Apply intervention that should change output
    with model.trace(text, interventions={"layers.5": iv.zero_out}):
        with_intervention = model.output.save()
    mx.eval(with_intervention)

    # Verify intervention changed output
    diff_intervention = mx.max(mx.abs(baseline - with_intervention))
    mx.eval(diff_intervention)
    print(f"  Intervention effect: {float(diff_intervention):.4f}")
    assert float(diff_intervention) > 0.01, "Intervention should change output"

    # Now trace without intervention - should match baseline
    with model.trace(text):
        after_intervention = model.output.save()
    mx.eval(after_intervention)

    diff_after = mx.max(mx.abs(baseline - after_intervention))
    mx.eval(diff_after)
    print(f"  After intervention: {float(diff_after):.10f}")
    assert float(diff_after) < 1e-10, "Intervention leaked to subsequent trace"

    print("PASSED: Interventions are temporary!")
    return True


def test_different_prompts_dont_interfere():
    """Test that tracing different prompts doesn't cause interference."""
    print("\n" + "=" * 60)
    print("TEST: Different prompts don't interfere")
    print("=" * 60)

    base_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    prompts = [
        "Hello world",
        "The quick brown fox",
        "Machine learning is",
        "Paris is the capital of France",
    ]

    # Get baseline outputs for each prompt
    baselines = {}
    for prompt in prompts:
        with model.trace(prompt):
            baselines[prompt] = model.output.save()
        mx.eval(baselines[prompt])

    # Run traces in different order with interventions
    for prompt in reversed(prompts):
        with model.trace(prompt, interventions={"layers.5": iv.scale(0.5)}):
            pass

    # Verify baselines are still reproducible
    all_match = True
    for prompt in prompts:
        with model.trace(prompt):
            current = model.output.save()
        mx.eval(current)

        diff = mx.max(mx.abs(baselines[prompt] - current))
        mx.eval(diff)

        if float(diff) >= 1e-10:
            print(f"  '{prompt[:20]}...': diff={float(diff):.2e} [FAIL]")
            all_match = False
        else:
            print(f"  '{prompt[:20]}...': OK")

    assert all_match, "Some prompts don't match baseline"
    print("PASSED: Different prompts don't interfere!")
    return True


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("MODEL RESTORATION TEST SUITE")
    print("Verifying model state is preserved after operations")
    print("=" * 70)

    tests = [
        test_model_unchanged_after_trace,
        test_model_unchanged_after_interventions,
        test_model_unchanged_after_activation_saving,
        test_sequential_operations_dont_leak,
        test_intervention_is_temporary,
        test_different_prompts_dont_interfere,
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
        print("ALL MODEL RESTORATION TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - please check output above")
        exit(1)
