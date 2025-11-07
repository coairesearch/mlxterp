"""
Test that interventions work with real models, not just activation capture.
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv

def test_real_model_interventions():
    """Test interventions on real mlx-lm model"""
    try:
        from mlx_lm import load

        print("Loading real model...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        print(f"Model loaded: {len(model.layers)} layers\n")

        # Test 1: Baseline (no intervention)
        print("Test 1: Baseline output")
        with model.trace("Hello, how are you?") as trace:
            baseline_layer_5 = model.layers[5].output.save()
            baseline_output = model.output.save()

        print(f"Layer 5 norm: {mx.linalg.norm(baseline_layer_5):.4f}")
        print(f"Output norm: {mx.linalg.norm(baseline_output):.4f}\n")

        # Test 2: Scale intervention on layer 5
        print("Test 2: Scale layer 5 by 0.5")
        with model.trace("Hello, how are you?", interventions={'layers.5': iv.scale(0.5)}) as trace:
            modified_layer_5 = model.layers[5].output.save()
            modified_output = model.output.save()

        print(f"Layer 5 norm: {mx.linalg.norm(modified_layer_5):.4f}")
        print(f"Output norm: {mx.linalg.norm(modified_output):.4f}\n")

        # Verify intervention had an effect
        layer_5_diff = mx.linalg.norm(baseline_layer_5 - modified_layer_5)
        output_diff = mx.linalg.norm(baseline_output - modified_output)

        print(f"Layer 5 difference: {layer_5_diff:.4f}")
        print(f"Output difference: {output_diff:.4f}\n")

        if layer_5_diff > 0.1 and output_diff > 0.1:
            print("✅ SUCCESS! Interventions work on real models!")
            return True
        else:
            print("❌ FAILED: Intervention didn't affect output")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_real_model_interventions()
