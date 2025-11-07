"""
Comprehensive test showing what we can actually capture now.
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv

def test_comprehensive():
    """Test what activations we can capture"""
    try:
        from mlx_lm import load

        print("Loading model...")
        base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
        model = InterpretableModel(base_model, tokenizer=tokenizer)

        print(f"Model loaded with {len(model.layers)} layers\n")

        # Run a forward pass
        print("Running forward pass with tracing...")
        with model.trace("Hello, how are you?") as trace:
            pass  # Just run forward, capture everything

        # Show what was captured
        print(f"\n✅ Captured {len(trace.activations)} activations!")
        print("\nFirst 20 captured activations:")
        for i, name in enumerate(list(trace.activations.keys())[:20]):
            activation = trace.activations[name]
            if isinstance(activation, mx.array):
                print(f"  {i+1}. {name}: {activation.shape}")
            else:
                print(f"  {i+1}. {name}: {type(activation)}")

        # Test accessing specific activations
        print("\n\nAccessing specific activations:")

        # Try to get layer 0
        if 'model.model.layers.0' in trace.activations:
            layer_0 = trace.activations['model.model.layers.0']
            print(f"✅ Layer 0: {layer_0.shape}")

        # Try to get attention from layer 5
        if 'model.model.layers.5.self_attn' in trace.activations:
            attn_5 = trace.activations['model.model.layers.5.self_attn']
            print(f"✅ Layer 5 attention: {attn_5.shape}")

        # Try to get q_proj from layer 3
        if 'model.model.layers.3.self_attn.q_proj' in trace.activations:
            q_proj_3 = trace.activations['model.model.layers.3.self_attn.q_proj']
            print(f"✅ Layer 3 Q projection: {q_proj_3.shape}")

        # Output
        if '__model_output__' in trace.activations:
            output = trace.activations['__model_output__']
            print(f"✅ Model output: {output.shape}")

        # Test intervention
        print("\n\nTesting intervention on layer 5...")
        baseline_arr = None
        if 'model.model.layers.5' in trace.activations:
            baseline_arr = trace.activations['model.model.layers.5']
            print(f"Baseline layer 5 shape: {baseline_arr.shape}")

        with model.trace("Hello, how are you?", interventions={'model.model.layers.5': iv.scale(0.5)}) as trace2:
            pass

        if baseline_arr is not None and 'model.model.layers.5' in trace2.activations:
            modified_arr = trace2.activations['model.model.layers.5']
            print(f"Modified layer 5 shape: {modified_arr.shape}")

            # Compute norm of difference (not difference of norms)
            diff = baseline_arr - modified_arr
            diff_norm = mx.linalg.norm(diff)
            mx.eval(diff_norm)
            diff_val = float(diff_norm)
            print(f"Difference norm: {diff_val:.4f}")

            if diff_val > 0.1:
                print("✅ Intervention worked!")
            else:
                print("❌ Intervention didn't have effect")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_comprehensive()
