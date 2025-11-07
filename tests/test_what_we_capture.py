#!/usr/bin/env python3
"""
Debug: What are we actually capturing with model.layers[i].output?
"""

import mlx.core as mx
from mlxterp import InterpretableModel
from mlx_lm import load

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

text = "Paris is the capital of France"

print(f"\nTracing: '{text}'\n")

with model.trace(text) as trace:
    layer_0_out = model.layers[0].output.save()
    layer_5_out = model.layers[5].output.save()
    layer_15_out = model.layers[15].output.save()
    final_out = model.output.save()

print(f"Layer 0 output shape: {layer_0_out.shape}")
print(f"Layer 5 output shape: {layer_5_out.shape}")
print(f"Layer 15 output shape: {layer_15_out.shape}")
print(f"Final output shape: {final_out.shape}")

print(f"\n{'='*60}")
print("Checking activation keys captured:")
print(f"{'='*60}\n")

# Get all keys
all_keys = sorted(trace.activations.keys())
print(f"Total activations captured: {len(all_keys)}\n")

# Show layer-related keys
layer_keys = [k for k in all_keys if 'layers.' in k and not ('self_attn' in k or 'mlp' in k or 'input_layernorm' in k or 'post_attention_layernorm' in k)]
print(f"Layer output keys ({len(layer_keys)}):")
for key in layer_keys[:20]:
    shape = trace.activations[key].shape
    print(f"  {key}: {shape}")

if len(layer_keys) > 20:
    print(f"  ... and {len(layer_keys) - 20} more")

print(f"\n{'='*60}")
print("Understanding the architecture:")
print(f"{'='*60}\n")

# Check what model.layers[0] actually is
print(f"model.layers[0] type: {type(model.layers[0]._module)}")
print(f"model.layers[0] attributes: {dir(model.layers[0]._module)[:10]}...")

# Check if it's a residual block
if hasattr(model.model.model, 'layers'):
    layer_0 = model.model.model.layers[0]
    print(f"\nActual layer 0 module: {type(layer_0)}")
    print(f"Layer 0 __call__ signature: ", end="")
    import inspect
    sig = inspect.signature(layer_0.__call__)
    print(sig)

print(f"\n{'='*60}")
print("Key insight:")
print(f"{'='*60}")
print("""
Transformer layers typically have this structure:

def layer_forward(x):
    # Self-attention with residual
    x = x + self_attn(layernorm(x))

    # MLP with residual
    x = x + mlp(layernorm(x))

    return x  # This is the RESIDUAL STREAM output

When we capture model.layers[i].output, we get the residual stream
AFTER that layer. This means:
- Patching layer 0 replaces the residual stream at position 0
- This affects ALL subsequent layers
- That's why patching ANY layer gives perfect recovery!

For proper activation patching, we should patch specific components
like self_attn or mlp, not the entire layer output.
""")
