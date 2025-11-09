# mlxterp Troubleshooting Guide

## SAE Training Issues

### Slow Training Speed (< 10 it/s)

**Symptoms:**
- Training shows very slow iteration speed (e.g., 1-5 it/s instead of 40-100 it/s)
- High memory usage (> 20 GB)
- Progress bar shows very high estimated time

**Common Causes:**

1. **Wrong component being trained**
   ```bash
   # Check activation dimension in training output
   # Should see: "✓ Activation dimension: 4096" (or similar reasonable size)
   # NOT: "✓ Activation dimension: 12288" (this indicates wrong component)
   ```

2. **text_batch_size too small**
   ```python
   # In your SAEConfig:
   text_batch_size=32  # Good - processes 32 texts at once
   # NOT text_batch_size=4  # Too small - causes many model traces
   ```

3. **Missing streaming optimization**
   - Check training output shows: "(streaming mode)"
   - Memory should stay constant, not grow during training

**Solutions:**
- Verify component name is correct (use examples as reference)
- Increase `text_batch_size` to 32-64 in SAEConfig
- Ensure you're using the latest version of the library

---

### High Dead Features (> 95%)

**Symptoms:**
- Training shows `dead=99.xx%` in progress bar
- SAE doesn't learn meaningful features

**Common Causes:**

1. **k parameter too low**
   ```python
   # For expansion_factor=4 (16K features):
   k=64   # Good - 0.4% sparsity
   k=32   # Too low - 0.2% sparsity, causes 99%+ dead features

   # For expansion_factor=8 (32K features):
   k=128  # Good
   k=256  # Better for larger SAEs
   ```

2. **Dataset too small or not diverse**
   ```python
   # Minimum recommended:
   NUM_SAMPLES = 5000   # For quick tests
   NUM_SAMPLES = 10000  # For development
   NUM_SAMPLES = 20000  # For production
   ```

**Solutions:**
- Increase k parameter (rule of thumb: k ≈ expansion_factor × 16)
- Use larger, more diverse dataset
- Normal dead features range: 60-80% (not 99%!)

---

### NaN Losses

**Symptoms:**
- Training shows `loss=nan`
- Dead features jump to 100%

**Common Causes:**

1. **Gradient explosion**
   ```python
   # In SAEConfig:
   gradient_clip=0.5  # Add gradient clipping
   ```

2. **Normalization issues**
   ```python
   # Try disabling normalization:
   normalize_input=False
   ```

3. **Learning rate too high**
   ```python
   learning_rate=1e-4  # Good default
   # NOT learning_rate=1e-2  # Too high
   ```

**Solutions:**
- Add or reduce gradient clipping
- Try disabling normalization
- Reduce learning rate
- Check for corrupted data in dataset

---

### Memory Issues

**Symptoms:**
- Training uses > 50 GB RAM
- System becomes unresponsive
- "Out of memory" errors

**Common Causes:**

1. **Large activation dimension**
   ```bash
   # Check the activation dimension in training output:
   # 4096 dim: Normal for 7-8B models
   # 8192 dim: Normal for 13-14B models
   # 12288+ dim: May indicate wrong component
   ```

2. **Accumulating activations (memory leak)**
   - Should be fixed in latest version
   - Memory should stay constant during training

**Solutions:**
- Use smaller model or different layer
- Reduce batch_size (e.g., from 256 to 128)
- Reduce expansion_factor (e.g., from 16 to 8 or 4)
- Ensure using latest library version with streaming fixes

---

## Component Name Reference

### Finding Available Components

Run this to see all available components for your model:

```python
from mlx_lm import load
from mlxterp import InterpretableModel

model, tokenizer = load("your-model-name")
interp = InterpretableModel(model, tokenizer=tokenizer)

with interp.trace("Test") as trace:
    pass

# Find components for layer 23:
for key in sorted(trace.activations.keys()):
    if "layers.23" in key:
        print(f"{key}: {trace.activations[key].shape}")
```

### Common Component Names

**For standard Transformer models:**
- `"mlp"` - MLP output (residual stream after MLP)
- `"self_attn"` or `"attn"` - Attention output
- `"mlp.down_proj"` - MLP down projection
- `"mlp.gate_proj"` - MLP gate projection (larger dimension)
- `"mlp.up_proj"` - MLP up projection (larger dimension)

**For Mixture of Experts (MoE) models:**
- `"mlp"` - Still the main MLP output
- `"mlp.switch_mlp"` - MoE-specific component
- `"mlp.gate"` - Expert routing gate

**Residual Stream:**
- The main layer output (usually `"mlp"` or `"self_attn"`)
- Represents the accumulated model state

---

## Configuration Guidelines

### Quick Test Configuration

For fast iteration during development:

```python
config = SAEConfig(
    sae_type="batchtopk",  # Modern BatchTopK (recommended)
    expansion_factor=4,
    k=64,
    learning_rate=3e-4,    # SAELens-validated
    num_epochs=1,          # Just 1 epoch for testing
    batch_size=64,
    text_batch_size=32,
    warmup_steps=100,
    lr_scheduler="cosine", # Cosine decay after warmup
    validation_split=0.1,
    normalize_input=True,
    use_ghost_grads=True,  # Reduce dead features
    gradient_clip=1.0,
    use_wandb=False,       # Disable W&B for testing
)
```

### Production Configuration (SAELens-validated)

For final training with all modern features:

```python
config = SAEConfig(
    # Architecture (SAELens defaults)
    sae_type="batchtopk",      # Modern BatchTopK (recommended over "topk")
    expansion_factor=32,        # SAELens-validated (increased from 16)
    k=128,

    # Optimization
    learning_rate=3e-4,         # SAELens-validated (increased from 1e-4)
    num_epochs=3,
    batch_size=256,
    text_batch_size=32,

    # Learning rate schedule
    warmup_steps=1000,
    lr_scheduler="cosine",      # Cosine decay after warmup
    lr_decay_steps=None,        # Auto: uses total training steps

    # Sparsity warmup (reduces dead features early)
    sparsity_warm_up_steps=None,  # Auto: set to total_steps

    # Ghost gradients (critical for reducing dead features)
    use_ghost_grads=True,       # Apply ghost grads to dead features
    feature_sampling_window=1000,
    dead_feature_window=5000,

    # Other settings
    validation_split=0.05,
    normalize_input=True,
    gradient_clip=1.0,
    use_wandb=True,
    wandb_project="your-project",
    checkpoint_every=5000,
)
```

**Expected improvements with new config:**
- Dead features: 95% → 70% (ghost gradients)
- Reconstruction loss: 0.27 → 0.10-0.15 (cosine decay + sparsity warmup)
- Training stability: Much improved (BatchTopK + LR schedule)

---

## Performance Expectations

### Training Speed

| Model Size | Expected Speed | Notes |
|------------|---------------|-------|
| 1-3B params | 80-200 it/s | Very fast |
| 7-8B params | 40-100 it/s | Good speed |
| 13-14B params | 20-50 it/s | Slower but acceptable |
| 30B+ params | 10-30 it/s | Depends on activation size |

**Note:** Speed depends more on activation dimension than total model parameters!

### Memory Usage

| SAE Size | Expected RAM | Notes |
|----------|-------------|-------|
| 4K → 16K (4x) | 2-5 GB | Efficient |
| 4K → 32K (8x) | 4-8 GB | Standard |
| 4K → 64K (16x) | 8-15 GB | Large |
| 8K → 64K (8x) | 8-16 GB | Large model |

**Memory should stay constant** during training - if it grows, there's a bug!

### Dead Features

| Percentage | Status | Action |
|------------|--------|--------|
| 60-80% | ✓ Normal | No action needed |
| 80-90% | ⚠ High | Consider increasing k or dataset |
| 90-95% | ❌ Too high | Increase k parameter |
| 95-100% | ❌ Critical | Check configuration |

---

## Common Error Messages

### "Could not find activations for layer X, component 'Y'"

**Cause:** Component name doesn't exist for this model.

**Solution:** Run the component discovery script (see "Finding Available Components" above).

### "ValueError: expansion_factor must be >= 1"

**Cause:** Invalid SAEConfig parameter.

**Solution:** Check all config parameters are valid (expansion_factor ≥ 1, k ≥ 1, etc.).

### "RuntimeError: Gradient computation failed"

**Cause:** Usually numerical instability.

**Solution:**
1. Add gradient clipping: `gradient_clip=0.5`
2. Reduce learning rate: `learning_rate=5e-5`
3. Disable normalization: `normalize_input=False`

---

## Understanding SAE Metrics

### L0 vs L1 vs Sparsity

**L0 (Feature Count):**
- Number of active (non-zero) features per sample
- For TopK with k=128: L0 ≈ 128
- Useful for checking if TopK is working

**L0 Sparsity (Fraction):**
- Fraction of features that are active: L0 / d_hidden
- For k=128, d_hidden=16,384: L0 sparsity ≈ 0.0078 (0.78%)
- More intuitive than raw count

**L1 Magnitude:**
- Average absolute value of feature activations
- Measures feature strength, NOT sparsity!
- Typical values: 0.01-0.05
- Higher = stronger feature activations

**Example:**
```
SAE with 16,384 features, k=128:
- L0: 125 features        (close to k=128 ✓)
- L0 sparsity: 0.0076     (125/16384 = 0.76%)
- L1 magnitude: 0.023     (features have avg strength 0.023)
```

**Common confusion:** L1 magnitude of 0.02 is NOT "2% sparsity"!
- Sparsity is measured by L0 / d_hidden
- L1 magnitude measures feature strength

---

## Getting Help

If you encounter issues not covered here:

1. Check the examples in `examples/` directory
2. Verify you're using the latest version
3. File an issue with:
   - Full error message
   - Model name and size
   - SAEConfig parameters
   - Training output (first 50 lines)
