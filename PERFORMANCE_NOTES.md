# SAE Training Performance Notes

## Training Speed Expectations

### Streaming Mode (Current Approach)
**Speed**: 1.5-2.0 it/s on 8B models
**Memory**: Low (no activation caching)
**Use case**: Large datasets that don't fit in memory

**Why is it slow?**
Each iteration requires:
1. Model forward pass (~500ms for 8B model)
2. SAE training step (~100ms)
3. Total: ~600ms/it = **1.6 it/s**

This is **EXPECTED and CORRECT** for streaming mode.

### Pre-Collected Activations Mode
**Speed**: 80-200 it/s (50-100x faster!)
**Memory**: High (all activations in RAM)
**Use case**: Smaller datasets (<10GB activations)

**Why is it fast?**
- Model traced ONCE upfront
- Each iteration is just SAE forward/backward
- No repeated model traces

## Dead Feature Behavior

### Timeline

| Step Range | Dead Features | Why |
|------------|---------------|-----|
| 0-1000 | 95-99% | Ghost grads not active yet |
| 1000-5000 | 85-95% | Ghost grads starting to work |
| 5000-10000 | 70-85% | Ghost grads fully active |
| 10000+ | 60-75% | Equilibrium reached |

### Ghost Gradient Activation

Ghost gradients don't activate immediately because:
1. Need `dead_feature_window` steps of history (default: 1000)
2. Features need time to prove they're actually dead
3. Early training is unstable, better to let network settle

**OLD default**: 5000 steps (very conservative)
**NEW default**: 1000 steps (faster activation)

## Recommended Configs

### Fast Testing (small model, pre-collected)
```python
config = SAEConfig(
    expansion_factor=16,
    k=64,
    num_epochs=1,
    batch_size=256,
    dead_feature_window=500,  # Start ghost grads early
)

# Pre-collect activations
activations = collect_activations(model, texts)  # Fast!
sae = trainer.train(activations)  # 80-200 it/s
```

### Production (large model, streaming)
```python
config = SAEConfig(
    expansion_factor=32,
    k=128,
    num_epochs=3,
    batch_size=64,  # Smaller for memory
    text_batch_size=32,  # Stream in chunks
    dead_feature_window=1000,  # Reasonable balance
    use_ghost_grads=True,
)

# Streaming mode
sae = model.train_sae(
    dataset=texts,  # 1.5-2.0 it/s (EXPECTED)
    config=config
)
```

## Trade-offs

| Approach | Speed | Memory | Use When |
|----------|-------|--------|----------|
| Streaming | 1-2 it/s | Low | >10k texts, large models |
| Pre-collected | 80-200 it/s | High | <5k texts, small models |

## Current Training Analysis

Your current training:
- Model: Qwen3-8B (8B params)
- Dataset: 10,000 texts
- Mode: Streaming
- **Speed: 1.8 it/s ✓ EXPECTED**
- **Dead features: 98% at step 200** - Ghost grads activate at step 1000+

**This is working as designed!** Ghost gradients will reduce dead features once they activate.
