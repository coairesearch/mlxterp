"""
Test script to verify layer activation capture works with real mlx-lm models.

This tests the method-wrapping approach for models with property-based layers.
"""

import mlx.core as mx
from mlxterp import InterpretableModel

def test_simple_model():
    """First verify simple models still work"""
    import mlx.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [nn.Linear(64, 64) for _ in range(4)]

        def __call__(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    print("Testing simple model...")
    model = InterpretableModel(SimpleModel())
    input_data = mx.random.normal((1, 64))

    with model.trace(input_data) as trace:
        print(f"Activations in context: {list(trace.activations.keys())}")
        layer_0 = model.layers[0].output.save()
        layer_2 = model.layers[2].output.save()
        output = model.output.save()

    print(f"✅ Layer 0: {layer_0.shape if layer_0 is not None else 'None'}")
    print(f"✅ Layer 2: {layer_2.shape if layer_2 is not None else 'None'}")
    print(f"✅ Output: {output.shape if output is not None else 'None'}")
    print()


def test_real_model():
    """Test with a real mlx-lm model"""
    try:
        from mlx_lm import load

        print("Loading real model (this may take a moment)...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        print(f"Model loaded: {len(model.layers)} layers")
        print()

        print("Testing layer activation capture...")
        with model.trace("Hello, how are you?") as trace:
            # Try to capture individual layer outputs
            layer_0 = model.layers[0].output.save()
            layer_5 = model.layers[5].output.save()
            layer_10 = model.layers[10].output.save()
            output = model.output.save()

        print(f"✅ Layer 0: {layer_0.shape if layer_0 is not None else 'None'}")
        print(f"✅ Layer 5: {layer_5.shape if layer_5 is not None else 'None'}")
        print(f"✅ Layer 10: {layer_10.shape if layer_10 is not None else 'None'}")
        print(f"✅ Output: {output.shape}")
        print()

        # Verify activations are actually captured
        if layer_0 is not None and layer_5 is not None and layer_10 is not None:
            print("🎉 SUCCESS! Real model layer activations are captured!")
        else:
            print("❌ FAILED: Layer activations are None")

    except ImportError:
        print("⚠️  mlx-lm not installed. Run: uv add mlx-lm")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_simple_model()
    test_real_model()
