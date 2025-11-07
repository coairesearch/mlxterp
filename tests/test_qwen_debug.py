#!/usr/bin/env python3
"""
Debug script to investigate the NaN issue with Qwen model.
"""

from mlxterp import InterpretableModel
from mlx_lm import load
import mlx.core as mx

print("Loading Qwen model...")
base_model, tokenizer = load('mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"Model has {len(model.layers)} layers")

# Test basic forward pass
print("\n" + "="*60)
print("Test 1: Basic forward passes")
print("="*60)

clean_text = "Paris is the capital of France"
corrupted_text = "London is the capital of France"

print(f"\nClean text: '{clean_text}'")
with model.trace(clean_text):
    clean_output = model.output.save()

print(f"Clean output shape: {clean_output.shape}")
print(f"Clean output dtype: {clean_output.dtype}")
print(f"Clean output stats:")
print(f"  min: {mx.min(clean_output)}")
print(f"  max: {mx.max(clean_output)}")
print(f"  mean: {mx.mean(clean_output)}")
print(f"  contains inf: {mx.any(mx.isinf(clean_output))}")
print(f"  contains nan: {mx.any(mx.isnan(clean_output))}")

print(f"\nCorrupted text: '{corrupted_text}'")
with model.trace(corrupted_text):
    corrupted_output = model.output.save()

print(f"Corrupted output shape: {corrupted_output.shape}")
print(f"Corrupted output dtype: {corrupted_output.dtype}")
print(f"Corrupted output stats:")
print(f"  min: {mx.min(corrupted_output)}")
print(f"  max: {mx.max(corrupted_output)}")
print(f"  mean: {mx.mean(corrupted_output)}")
print(f"  contains inf: {mx.any(mx.isinf(corrupted_output))}")
print(f"  contains nan: {mx.any(mx.isnan(corrupted_output))}")

# Check distance calculation
print("\n" + "="*60)
print("Test 2: Distance calculations")
print("="*60)

mx.eval(clean_output, corrupted_output)

# Last token comparison
clean_last = clean_output[0, -1]
corrupted_last = corrupted_output[0, -1]

print(f"\nLast token hidden states:")
print(f"  Clean shape: {clean_last.shape}")
print(f"  Corrupted shape: {corrupted_last.shape}")

# Calculate difference
diff = clean_last - corrupted_last
print(f"\nDifference stats:")
print(f"  min: {mx.min(diff)}")
print(f"  max: {mx.max(diff)}")
print(f"  mean: {mx.mean(diff)}")
print(f"  contains inf: {mx.any(mx.isinf(diff))}")
print(f"  contains nan: {mx.any(mx.isnan(diff))}")

# Try different distance metrics
print("\n" + "="*60)
print("Test 3: Different distance metrics")
print("="*60)

# L2 distance
diff_squared = (clean_last - corrupted_last) ** 2
print(f"\nSquared differences:")
print(f"  min: {mx.min(diff_squared)}")
print(f"  max: {mx.max(diff_squared)}")
print(f"  sum: {mx.sum(diff_squared)}")
print(f"  contains inf: {mx.any(mx.isinf(diff_squared))}")

sum_squared = mx.sum(diff_squared)
print(f"\nSum of squared diffs: {sum_squared}")

l2_dist = mx.sqrt(sum_squared)
print(f"L2 distance: {l2_dist}")

# Try clipping to prevent overflow
print("\n" + "="*60)
print("Test 4: Clipped distance")
print("="*60)

# Clip extreme values
clean_clipped = mx.clip(clean_last, -1e6, 1e6)
corrupted_clipped = mx.clip(corrupted_last, -1e6, 1e6)

diff_clipped = (clean_clipped - corrupted_clipped) ** 2
l2_clipped = float(mx.sqrt(mx.sum(diff_clipped)))
print(f"L2 distance (clipped to ±1e6): {l2_clipped}")

# Try normalizing
print("\n" + "="*60)
print("Test 5: Normalized distance")
print("="*60)

clean_norm = mx.sqrt(mx.sum(clean_last ** 2))
corrupted_norm = mx.sqrt(mx.sum(corrupted_last ** 2))

print(f"Clean norm: {clean_norm}")
print(f"Corrupted norm: {corrupted_norm}")

# Cosine distance
if clean_norm > 0 and corrupted_norm > 0:
    clean_normalized = clean_last / clean_norm
    corrupted_normalized = corrupted_last / corrupted_norm
    cosine_sim = mx.sum(clean_normalized * corrupted_normalized)
    cosine_dist = 1.0 - float(cosine_sim)
    print(f"Cosine distance: {cosine_dist}")
else:
    print("Cannot compute cosine distance (zero norm)")

print("\n" + "="*60)
print("Diagnosis complete!")
print("="*60)
