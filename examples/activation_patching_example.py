#!/usr/bin/env python3
"""
Simple example of activation patching to find important layers.

This demonstrates a classic interpretability technique:
1. Get activations from a "clean" input
2. Patch them into a "corrupted" input at different layers
3. Measure how much this recovers the clean output
4. Layers with high recovery are most important for the task
"""

import mlx.core as mx
from mlxterp import InterpretableModel, interventions as iv
from mlx_lm import load

# Load model
print("Loading model...")
base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print("\n" + "="*60)
print("Activation Patching Example")
print("="*60 + "\n")

# Define clean vs corrupted inputs
clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

print(f"Clean: {clean_text}")
print(f"Corrupted: {corrupted_text}\n")

# Get baseline outputs
with model.trace(clean_text):
    clean_output = model.output.save()

with model.trace(corrupted_text):
    corrupted_output = model.output.save()

mx.eval(clean_output, corrupted_output)

# Helper function
def l2_distance(a, b):
    return float(mx.sqrt(mx.sum((a - b) ** 2)))

baseline = l2_distance(corrupted_output[0, -1], clean_output[0, -1])
print(f"Baseline L2 distance: {baseline:.2f}\n")

# Test patching each layer's MLP
print("Patching MLP at each layer...")
results = {}

for layer_idx in range(len(model.layers)):
    # Get clean MLP activation
    with model.trace(clean_text) as trace:
        clean_mlp = trace.activations[f"model.model.layers.{layer_idx}.mlp"]

    mx.eval(clean_mlp)

    # Patch into corrupted
    with model.trace(corrupted_text,
                    interventions={f"layers.{layer_idx}.mlp": iv.replace_with(clean_mlp)}):
        patched_output = model.output.save()

    mx.eval(patched_output)

    # Measure recovery
    dist = l2_distance(patched_output[0, -1], clean_output[0, -1])
    recovery = (baseline - dist) / baseline * 100
    results[layer_idx] = (dist, recovery)

    print(f"  Layer {layer_idx:2d}: {recovery:6.1f}% recovery")

# Summarize most important layers
print("\n" + "="*60)
print("Most important layers (highest recovery):")
print("="*60)

sorted_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)
for layer_idx, (dist, recovery) in sorted_results[:5]:
    print(f"  Layer {layer_idx:2d}: {recovery:5.1f}% recovery (L2={dist:.2f})")

print("\nInterpretation:")
print("  - High recovery = layer is critical for the task")
print("  - Negative recovery = layer encodes the corruption")
print("  - Layer 0 often important (early processing)")
print("  - Final layers often important (output formation)")
