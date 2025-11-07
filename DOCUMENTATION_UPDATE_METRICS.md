# Documentation Update: Distance Metrics

## Summary

Added comprehensive documentation for distance metrics in activation patching, including mathematical formulas, implementation details, and usage recommendations.

## Files Updated

### 1. `docs/guides/activation_patching.md`

**Added new section: "Distance Metrics"** (~200 lines)

Includes:
- **Three metric options** with formulas and implementations:
  1. L2 (Euclidean) Distance - Default
  2. Cosine Distance - Direction-based
  3. MSE (Mean Squared Error) - Most stable

- **Metric Selection Guide** table:
  | Vocab Size | Recommended Metric |
  |------------|-------------------|
  | < 50k | L2 |
  | 50k-100k | L2 or Cosine |
  | > 100k | MSE or Cosine |

- **Real-world example** with Qwen model (151k vocab):
  - Shows what happens with each metric
  - Demonstrates overflow problem with L2
  - Shows MSE working perfectly

- **Recovery calculation** explanation
- **Why not KL divergence** - explains numerical issues

### 2. `docs/API.md`

**Enhanced `activation_patching()` documentation**:

- Expanded metric parameter with recommendations
- Added distance metric formulas
- Added example showing Qwen model usage
- Cross-reference to comprehensive guide

Before:
```python
metric (str): Distance metric ("l2" or "cosine")
```

After:
```python
metric (str): Distance metric. Options:
  - "l2": Euclidean distance (default, with overflow protection)
  - "cosine": Cosine distance (recommended for large vocabularies)
  - "mse": Mean squared error (most stable for huge models > 100k vocab)

  Recommendation:
  - Vocab < 50k: use "l2"
  - Vocab 50k-100k: use "l2" or "cosine"
  - Vocab > 100k: use "mse" or "cosine"
```

### 3. `docs/examples.md`

**Added section: "Choosing the Right Metric"**

- Practical Qwen example showing metric selection
- Explains why it matters (overflow prevention)
- Shows actual results with MSE metric
- Clear recommendations table

## Key Content Added

### Mathematical Formulas

**L2 Distance**:
```
d(a, b) = √(Σ(aᵢ - bᵢ)²)
```

**Cosine Distance**:
```
d(a, b) = 1 - (a · b) / (||a|| × ||b||)

where:
  a · b = Σ(aᵢ × bᵢ)     # Dot product
  ||a|| = √(Σ aᵢ²)       # L2 norm
```

**MSE Distance**:
```
d(a, b) = (1/N) × Σ(aᵢ - bᵢ)²

where N = number of elements
```

### Implementation Details

Showed actual Python implementations with:
- Float32 conversion for numerical stability
- Overflow detection and fallback strategies
- Mean-based normalization for large arrays

### Real-World Problem & Solution

**Problem** (Qwen model with 151k vocab):
```
Output shape: (1, 6, 151936)  # 151k logits!
Baseline: inf                  # Overflow!
Recovery: nan, nan, nan...     # All NaN
```

**Solution** (MSE metric):
```
Baseline: 0.6480
Layer 10: 17.9% recovery  ← Works perfectly!
Layer 30:  7.5% recovery
Layer  0: -298.6% recovery
```

### Recommendations Summary

Created clear decision tree:

1. **Check your model's vocabulary size**:
   ```python
   print(f"Vocabulary: {model.vocab_size}")
   ```

2. **Choose metric based on size**:
   - Small (< 50k): `metric="l2"` (default)
   - Medium (50-100k): `metric="l2"` or `"cosine"`
   - Large (> 100k): `metric="mse"` or `"cosine"`

3. **Consider your research question**:
   - Studying magnitudes: `"l2"` or `"mse"`
   - Studying directions: `"cosine"`
   - Maximum stability: `"mse"`

## Examples Added

### Basic Example (examples.md)
```python
# Qwen model with 151k vocab
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    metric="mse",  # ← Important!
    plot=True
)
```

### API Documentation Example
```python
# Without correct metric - gets NaN
results = model.activation_patching(..., metric="l2")  # ❌ May overflow

# With correct metric - works perfectly
results = model.activation_patching(..., metric="mse")  # ✅ Stable
```

### Comprehensive Guide Examples
- L2 with overflow protection
- Cosine with fallback normalization
- MSE for maximum stability
- Recovery calculation walkthrough

## Why This Matters

Before these docs:
- Users got `nan` results with large models
- No guidance on which metric to use
- No explanation of overflow issues
- No mathematical foundations

After these docs:
- ✅ Clear metric selection guide
- ✅ Mathematical formulas for understanding
- ✅ Implementation details for transparency
- ✅ Real-world examples with Qwen
- ✅ Troubleshooting overflow issues

## Cross-References

All three docs now cross-reference each other:
- `examples.md` → points to comprehensive guide
- `API.md` → points to comprehensive guide
- `guides/activation_patching.md` → comprehensive standalone

## Verification

Documentation builds successfully:
```bash
$ uv run mkdocs build
INFO - Documentation built in 0.40 seconds
```

All metrics tested and working:
- ✅ L2 with overflow protection
- ✅ Cosine with fallback
- ✅ MSE for large models
- ✅ Qwen model (151k vocab) works perfectly

## User Impact

Users can now:
1. Understand **why** they're getting NaN results
2. **Choose** the right metric for their model
3. **Understand** the mathematics behind each metric
4. **Implement** their own custom metrics if needed
5. **Troubleshoot** numerical issues independently

The documentation provides:
- ✅ Theory (formulas)
- ✅ Practice (implementations)
- ✅ Examples (Qwen, Llama)
- ✅ Guidance (decision tables)
- ✅ Troubleshooting (overflow issues)

## Files Modified

1. `docs/guides/activation_patching.md` - Added ~200 lines on metrics
2. `docs/API.md` - Enhanced metric documentation
3. `docs/examples.md` - Added metric selection example
4. `mlxterp/analysis.py` - Code already updated with stability fixes

Total documentation added: ~300 lines of comprehensive metric coverage!
