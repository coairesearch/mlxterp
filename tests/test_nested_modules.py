"""
Test if we can access nested modules like model.layers[3].attn.output
"""

import mlx.core as mx
from mlxterp import InterpretableModel

def test_nested_access():
    """Test nested module access"""
    try:
        from mlx_lm import load

        print("Loading model...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        # Inspect layer structure
        print(f"\nLayer 0 attributes:")
        layer_0_attrs = [attr for attr in dir(model.layers._layers[0]) if not attr.startswith('_')]
        print(f"  {layer_0_attrs[:10]}...")

        print(f"\nTrying to access nested modules...")
        with model.trace("Hello world") as trace:
            # Try to access nested submodules
            try:
                layer_0_full = model.layers[0].output.save()
                print(f"✅ Layer 0 output: {layer_0_full.shape}")
            except Exception as e:
                print(f"❌ Layer 0 output failed: {e}")

            # Try to access a submodule (if it exists)
            try:
                # Common transformer components
                if hasattr(model.layers._layers[0], 'self_attn'):
                    attn_out = model.layers[0].self_attn.output.save()
                    print(f"✅ Layer 0 self_attn output: {attn_out.shape if attn_out else 'None'}")
                elif hasattr(model.layers._layers[0], 'attention'):
                    attn_out = model.layers[0].attention.output.save()
                    print(f"✅ Layer 0 attention output: {attn_out.shape if attn_out else 'None'}")
                else:
                    print("⚠️  No standard attention submodule found")
            except Exception as e:
                print(f"❌ Nested module access failed: {e}")

            output = model.output.save()
            print(f"✅ Model output: {output.shape}")

        # Check what was captured
        print(f"\nCaptured activations: {list(trace.activations.keys())[:5]}...")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_nested_access()
