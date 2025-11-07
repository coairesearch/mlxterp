#!/usr/bin/env python3
"""
Test logit lens plotting functionality.

This test verifies that the logit_lens method can generate visualizations
when plot=True is specified.
"""

from mlxterp import InterpretableModel
from mlx_lm import load

# Load model
print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Test text
text = "The Eiffel Tower is located in the city of"

print(f"\n{'='*60}")
print(f"Testing logit lens with visualization")
print(f"{'='*60}\n")

# Run logit lens with plotting
print(f"Input: '{text}'")
print("\nGenerating logit lens visualization...")
print("This will show:")
print("  - X-axis: Input token positions")
print("  - Y-axis: Model layers")
print("  - Cell values: Top predicted token at each (layer, position)")
print()

results = model.logit_lens(
    text,
    top_k=1,  # Just get top prediction per position
    layers=list(range(0, 16)),  # All layers
    plot=True,
    max_display_tokens=15,  # Show last 15 tokens
    figsize=(16, 10),
    cmap='viridis'
)

print("\n✅ Plotting test complete!")
print("\nYou should see a heatmap showing:")
print("  - Input tokens along the bottom (x-axis)")
print("  - Layers on the left (y-axis)")
print("  - Predicted tokens in each cell")
print("  - Colors distinguishing different predictions")
