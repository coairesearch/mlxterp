"""
Test that the proxy API (model.layers[i].output.save()) works with real models.

This verifies the fix for the naming mismatch between wrappers and proxies.
"""

import mlx.core as mx
from mlxterp import InterpretableModel

def test_proxy_api_real_model():
    """Test that model.layers[i].output.save() works"""
    try:
        from mlx_lm import load

        print("Loading model...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        print(f"Model loaded with {len(model.layers)} layers\n")

        # Test the proxy API
        print("Testing proxy API: model.layers[i].output.save()")
        with model.trace("Hello, how are you?") as trace:
            # This should now work!
            layer_0 = model.layers[0].output.save()
            layer_5 = model.layers[5].output.save()
            layer_10 = model.layers[10].output.save()
            output = model.output.save()

        # Verify captures
        print(f"\nResults:")
        print(f"  Layer 0: {layer_0.shape if layer_0 is not None else 'None'}")
        print(f"  Layer 5: {layer_5.shape if layer_5 is not None else 'None'}")
        print(f"  Layer 10: {layer_10.shape if layer_10 is not None else 'None'}")
        print(f"  Output: {output.shape if output is not None else 'None'}")

        if layer_0 is not None and layer_5 is not None and layer_10 is not None:
            print("\n✅ SUCCESS! Proxy API works with real models!")
            return True
        else:
            print("\n❌ FAILED: Some activations are None")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_nested_proxy_access():
    """Test nested module access: model.layers[i].self_attn.output.save()"""
    try:
        from mlx_lm import load

        print("\n\nTesting nested proxy access...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        with model.trace("Hello world") as trace:
            # Test nested access
            layer_5_attn = model.layers[5].self_attn.output.save()
            layer_3_mlp = model.layers[3].mlp.output.save()

        print(f"\nResults:")
        print(f"  Layer 5 attention: {layer_5_attn.shape if layer_5_attn is not None else 'None'}")
        print(f"  Layer 3 MLP: {layer_3_mlp.shape if layer_3_mlp is not None else 'None'}")

        if layer_5_attn is not None and layer_3_mlp is not None:
            print("\n✅ SUCCESS! Nested proxy access works!")
            return True
        else:
            print("\n❌ FAILED: Some nested activations are None")
            return False

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    result1 = test_proxy_api_real_model()
    result2 = test_nested_proxy_access()

    if result1 and result2:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! The proxy API is fully functional!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Some tests failed")
        print("="*60)
