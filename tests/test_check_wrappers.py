"""
Check if wrappers are actually in place after patching
"""

from mlxterp import InterpretableModel
from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Check layer types before tracing
print("Before tracing:")
print(f"  model.model.layers type: {type(base_model.model.layers)}")
print(f"  model.model.layers[0] type: {type(base_model.model.layers[0])}")
print(f"  ID of layers list: {id(base_model.model.layers)}")

# Start trace
with model.trace("Hello") as trace:
    print("\nDuring tracing:")
    print(f"  model.model.layers type: {type(base_model.model.layers)}")
    print(f"  model.model.layers[0] type: {type(base_model.model.layers[0])}")
    print(f"  ID of layers list: {id(base_model.model.layers)}")

    # Check if it's a wrapper
    layer_0 = base_model.model.layers[0]
    print(f"  Is layer 0 a wrapper? {hasattr(layer_0, '_wrapped_layer')}")
    print(f"  Layer 0 class name: {type(layer_0).__name__}")

print("\nAfter tracing:")
print(f"  model.model.layers type: {type(base_model.model.layers)}")
print(f"  model.model.layers[0] type: {type(base_model.model.layers[0])}")
print(f"  ID of layers list: {id(base_model.model.layers)}")
