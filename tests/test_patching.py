"""
Test if __call__ patching works with MLX modules.
"""

import mlx.core as mx
import mlx.nn as nn

# Create a simple linear layer
layer = nn.Linear(10, 10)

# Store original __call__
original_call = layer.__call__

# Patch it
call_count = [0]

def wrapped_call(*args, **kwargs):
    call_count[0] += 1
    print(f"Wrapped call invoked! Count: {call_count[0]}")
    return original_call(*args, **kwargs)

layer.__call__ = wrapped_call

# Test it
x = mx.random.normal((2, 10))
result = layer(x)

print(f"\nResult shape: {result.shape}")
print(f"Call count: {call_count[0]}")

if call_count[0] > 0:
    print("✅ Patching __call__ works!")
else:
    print("❌ Patching __call__ doesn't work!")
