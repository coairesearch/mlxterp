"""
Understand how MLX modules expose their children.
"""

from mlx_lm import load
import mlx.nn as nn

# Load model
model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')

# Get first layer
layer_0 = model.model.layers[0]

print(f"Layer 0 type: {type(layer_0)}")
print(f"\nDirect attributes: {[a for a in dir(layer_0) if not a.startswith('_')][:15]}")
print()

# Check MLX's built-in methods for discovering children
if hasattr(layer_0, 'children'):
    print("layer_0.children():")
    for name, child in layer_0.children().items():
        print(f"  {name}: {type(child).__name__}")
print()

if hasattr(layer_0, 'leaf_modules'):
    print("layer_0.leaf_modules():")
    for name, module in list(layer_0.leaf_modules().items())[:10]:
        print(f"  {name}: {type(module).__name__}")
print()

# Try to access known submodules directly
print("Direct access attempts:")
attrs_to_try = ['self_attn', 'attention', 'attn', 'mlp', 'feed_forward', 'ffn']
for attr in attrs_to_try:
    if hasattr(layer_0, attr):
        val = getattr(layer_0, attr)
        print(f"  ✅ {attr}: {type(val).__name__}")
    else:
        print(f"  ❌ {attr}: not found")
