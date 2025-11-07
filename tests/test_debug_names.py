"""
Debug script to see what names are actually in trace.activations
"""

from mlxterp import InterpretableModel
from mlx_lm import load

# Load model
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Run trace
with model.trace("Hello") as trace:
    pass

# Show first 20 activation names
print("First 20 activation names:")
for i, name in enumerate(list(trace.activations.keys())[:20]):
    print(f"  {i+1}. {name}")

# Check specifically for layer names
print("\nLooking for layer-related names:")
for name in trace.activations.keys():
    if 'layers' in name and 'layers.0' in name:
        print(f"  Found: {name}")
        break

# Try different searches
searches = [
    "layers.0",
    "model.layers.0",
    "model.model.layers.0",
    "layers.5",
    "model.model.layers.5.self_attn",
]

print("\nSearching for specific names:")
for search in searches:
    found = search in trace.activations
    print(f"  '{search}': {'✅ Found' if found else '❌ Not found'}")
