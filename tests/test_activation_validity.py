"""
Test that captured activations are actually valid, not NaN or corrupted.
"""

import mlx.core as mx
from mlxterp import InterpretableModel

def test_activation_validity():
    """Test that activations are valid arrays"""
    try:
        from mlx_lm import load

        print("Loading model...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        print("Running forward pass...")
        with model.trace("Hello, how are you?") as trace:
            layer_0 = model.layers[0].output.save()
            layer_5 = model.layers[5].output.save()
            output = model.output.save()

        print(f"\nLayer 0:")
        print(f"  Shape: {layer_0.shape}")
        print(f"  dtype: {layer_0.dtype}")
        print(f"  Has NaN: {mx.any(mx.isnan(layer_0)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(layer_0)).item()}")
        print(f"  Min: {mx.min(layer_0).item():.4f}")
        print(f"  Max: {mx.max(layer_0).item():.4f}")
        print(f"  Mean: {mx.mean(layer_0).item():.4f}")

        print(f"\nLayer 5:")
        print(f"  Shape: {layer_5.shape}")
        print(f"  dtype: {layer_5.dtype}")
        print(f"  Has NaN: {mx.any(mx.isnan(layer_5)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(layer_5)).item()}")
        print(f"  Min: {mx.min(layer_5).item():.4f}")
        print(f"  Max: {mx.max(layer_5).item():.4f}")
        print(f"  Mean: {mx.mean(layer_5).item():.4f}")

        print(f"\nOutput:")
        print(f"  Shape: {output.shape}")
        print(f"  dtype: {output.dtype}")
        print(f"  Has NaN: {mx.any(mx.isnan(output)).item()}")
        print(f"  Has Inf: {mx.any(mx.isinf(output)).item()}")
        print(f"  First few logits: {output[0, -1, :10]}")

        # Check if any have NaN or Inf
        has_problems = (
            mx.any(mx.isnan(layer_0)).item() or mx.any(mx.isinf(layer_0)).item() or
            mx.any(mx.isnan(layer_5)).item() or mx.any(mx.isinf(layer_5)).item() or
            mx.any(mx.isnan(output)).item() or mx.any(mx.isinf(output)).item()
        )

        if not has_problems:
            print("\n✅ SUCCESS! All activations are valid (no NaN/Inf)")
            return True
        else:
            print("\n❌ FAILED: Activations contain NaN or Inf")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_activation_validity()
