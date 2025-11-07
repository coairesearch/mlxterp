#!/usr/bin/env python3
"""
Test the activation_patching() helper function.
"""

from mlxterp import InterpretableModel
from mlx_lm import load

print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print("\n" + "="*60)
print("Testing activation_patching() Helper Function")
print("="*60 + "\n")

# Test 1: Simple MLP patching
print("Test 1: MLP patching with plot")
print("-" * 60)

results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True,
    figsize=(14, 6)
)

print("\nResults:")
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nTop 5 most important layers:")
for layer_idx, recovery in sorted_results[:5]:
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

print("\nTop 5 layers encoding corruption:")
for layer_idx, recovery in sorted_results[-5:]:
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

# Test 2: Attention patching
print("\n" + "="*60)
print("Test 2: Attention patching")
print("-" * 60)

results_attn = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="self_attn",
    layers=[0, 5, 10, 15],  # Test subset
    plot=False
)

print("\nAttention results:")
for layer_idx, recovery in sorted(results_attn.items()):
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

# Test 3: Sub-component patching
print("\n" + "="*60)
print("Test 3: MLP gate projection patching")
print("-" * 60)

results_gate = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp.gate_proj",
    layers=[0, 8, 15],
    plot=False
)

print("\nGate projection results:")
for layer_idx, recovery in sorted(results_gate.items()):
    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

print("\n" + "="*60)
print("✅ All tests completed!")
print("="*60)

print("\nUsage summary:")
print("  - Simply call: model.activation_patching(clean, corrupted, plot=True)")
print("  - Component options: 'mlp', 'self_attn', 'mlp.gate_proj', etc.")
print("  - Returns dict of {layer_idx: recovery%}")
print("  - Positive % = important, Negative % = encodes corruption")
