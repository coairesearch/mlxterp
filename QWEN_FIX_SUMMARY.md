# Fix for Qwen Model NaN Issue

## Problem

When running `activation_patching()` on the Qwen3-30B model, users encountered:
- Baseline distance: `inf`
- All recovery percentages: `nan`
- Empty plots

## Root Cause

The Qwen model has an **extremely large vocabulary** (151,936 tokens), which causes numerical overflow:

```
Output shape: (1, 6, 151936)  # Huge vocabulary!
Output dtype: float16          # Limited precision

When computing L2 distance:
sum((clean - corrupted)²) = inf  # Overflow with 151k elements
```

The issue:
1. Output logits are shape `(seq_len, vocab_size)` = `(6, 151936)`
2. Computing squared differences over 151,936 elements
3. Using float16 precision
4. Sum overflows to `inf`
5. Recovery calculation becomes `nan`

## Solution

Added **numerical stability** to all distance metrics in `activation_patching()`:

### 1. L2 Distance - Overflow Protection

```python
def distance(a, b):
    """L2 distance with numerical stability"""
    diff = a - b
    # Use float32 for accumulation to prevent overflow
    diff_f32 = diff.astype(mx.float32)
    squared_sum = mx.sum(diff_f32 * diff_f32)

    # Check for overflow
    if mx.isinf(squared_sum) or mx.isnan(squared_sum):
        # Fallback: use mean squared error instead of sum
        mse = mx.mean(diff_f32 * diff_f32)
        return float(mx.sqrt(mse) * mx.sqrt(float(diff.size)))
    return float(mx.sqrt(squared_sum))
```

**Key changes**:
- Convert to float32 for accumulation
- Detect overflow with `isinf()` check
- Fallback to MSE-based calculation if overflow occurs

### 2. Cosine Distance - Normalization Protection

```python
def distance(a, b):
    """Cosine distance with numerical stability"""
    a_f32 = a.astype(mx.float32)
    b_f32 = b.astype(mx.float32)
    a_norm = mx.sqrt(mx.sum(a_f32 * a_f32))
    b_norm = mx.sqrt(mx.sum(b_f32 * b_f32))

    if mx.isinf(a_norm) or mx.isinf(b_norm) or a_norm < 1e-10 or b_norm < 1e-10:
        # Fallback: use normalized mean
        a_normalized = a_f32 / (mx.sqrt(mx.mean(a_f32 * a_f32)) + 1e-10)
        b_normalized = b_f32 / (mx.sqrt(mx.mean(b_f32 * b_f32)) + 1e-10)
        return float(1.0 - mx.mean(a_normalized * b_normalized))

    a_normalized = a_f32 / a_norm
    b_normalized = b_f32 / b_norm
    return float(1.0 - mx.sum(a_normalized * b_normalized))
```

### 3. New MSE Metric (Most Stable)

```python
def distance(a, b):
    """Mean squared error - stable for large vocabularies"""
    diff = a.astype(mx.float32) - b.astype(mx.float32)
    return float(mx.mean(diff * diff))
```

**Why MSE is best for large models**:
- Averages over all elements (no accumulation overflow)
- Numerically stable even with millions of dimensions
- Still captures the same signal (correlation with L2)

## Test Results

### Before Fix
```
Baseline l2 distance: inf
Layer  0: nan% recovery
Layer 10: nan% recovery
...all nan
```

### After Fix

**MSE Metric** (recommended):
```
Baseline mse distance: 0.6480
Layer 10:  17.9% recovery  ← Most important!
Layer 30:   7.5% recovery
Layer 40:   5.6% recovery
Layer 47:   3.3% recovery
Layer 20: -11.2% recovery  ← Encodes corruption
Layer  0: -298.6% recovery ← Strongly encodes corruption
```

**Cosine Metric**:
```
Baseline cosine distance: 0.0079
Layer 10:  11.9% recovery
Layer 40:   9.1% recovery
Layer 30:   3.9% recovery
Layer  0: -45.0% recovery
```

**L2 Metric** (with overflow protection):
```
Baseline l2 distance: 313.7752
Layer 10:   9.4% recovery
Layer 30:   3.8% recovery
Layer 40:   2.9% recovery
Layer  0: -99.7% recovery
```

## Updated API

The `activation_patching()` method now supports three metrics:

```python
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    metric="mse",  # or "cosine", "l2"
    plot=True
)
```

### Metric Recommendations

| Model Vocab Size | Recommended Metric | Reason |
|-----------------|-------------------|---------|
| < 50k tokens | `"l2"` (default) | Fast, accurate |
| 50k - 100k | `"l2"` or `"cosine"` | L2 with overflow protection works |
| > 100k tokens | `"mse"` or `"cosine"` | Most numerically stable |

**For Qwen (151k vocab)**: Use `metric="mse"`

## Usage Example

```python
from mlxterp import InterpretableModel
from mlx_lm import load

# Load large vocabulary model
base_model, tokenizer = load('mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Use MSE for stability
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    metric="mse",  # ← Important for large vocab models!
    plot=True
)

# Analyze results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Top 3 most important layers:")
for layer_idx, recovery in sorted_results[:3]:
    print(f"  Layer {layer_idx}: {recovery:.1f}% recovery")
```

## Files Modified

1. **mlxterp/analysis.py** - Added numerical stability to all distance functions
2. **tests/test_qwen_debug.py** - Diagnostic script
3. **tests/test_qwen_patching.py** - Test all three metrics

## Backward Compatibility

✅ **No breaking changes**
- Default metric is still `"l2"` (with overflow protection)
- Existing code continues to work
- New `"mse"` metric is opt-in

## Key Takeaways

1. **Large vocabulary models** (>100k tokens) can cause numerical overflow
2. **MSE metric** is most stable for huge models
3. **Float32 accumulation** prevents overflow in L2/cosine
4. **All metrics now work** reliably across model sizes
5. **No user code changes needed** - fixes are automatic

The library now handles models from tiny (2k vocab) to massive (150k+ vocab) without numerical issues!
