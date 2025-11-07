"""
Check if model.layers property returns the same list each time
"""

from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')

# Get layers multiple times
layers1 = base_model.model.layers
layers2 = base_model.model.layers
layers3 = base_model.model.layers

print(f"First access:  ID={id(layers1)}")
print(f"Second access: ID={id(layers2)}")
print(f"Third access:  ID={id(layers3)}")
print(f"\nSame object? {layers1 is layers2 is layers3}")

# Check if items are the same
print(f"\nFirst item same? {layers1[0] is layers2[0]}")

# Try modifying
class DummyWrapper:
    pass

layers1[0] = DummyWrapper()
print(f"\nAfter modifying layers1[0]:")
print(f"  layers1[0] type: {type(layers1[0])}")
print(f"  layers2[0] type: {type(layers2[0])}")  # If same list, should be DummyWrapper
print(f"  layers3[0] type: {type(layers3[0])}")
