# SAE Development Roadmap

This document outlines the development phases for Sparse Autoencoder (SAE) integration in mlxterp.

## ✅ Phase 1: Core SAE Implementation (COMPLETE - UPDATED 2025-11-08)

### Training Infrastructure
- ✅ TopK SAE architecture (sparse autoencoder with TopK sparsity)
- ✅ **BatchTopK SAE architecture** (modern SAELens-style, batch-level sparsity)
- ✅ Streaming training loop (handles datasets of any size)
- ✅ Memory-efficient activation collection
- ✅ SAEConfig with SAELens-validated defaults
- ✅ Gradient clipping and normalization
- ✅ **Learning rate warmup + cosine/linear decay**
- ✅ **Sparsity warmup** (reduces dead features early in training)
- ✅ **Ghost gradients** (revives dead features during training)
- ✅ Validation split and metrics

### Save/Load System
- ✅ SAE.save() and SAE.load() methods
- ✅ SafeTensors format for weight storage
- ✅ Config persistence (JSON)
- ✅ Backward compatibility

### Integration with mlxterp
- ✅ model.train_sae() method
- ✅ Automatic component discovery
- ✅ Works with any mlx-lm model
- ✅ Examples and documentation

### Monitoring
- ✅ W&B integration (optional)
- ✅ Training metrics (loss, L0, L0 sparsity, L1 magnitude, dead features)
- ✅ Checkpointing system
- ✅ Progress bars with tqdm

### SAELens Feature Parity (NEW)
- ✅ **Ghost gradients** for dead feature revival
- ✅ **BatchTopK architecture** (modern, more stable than TopK)
- ✅ **LR scheduling** (cosine/linear decay after warmup)
- ✅ **Sparsity warmup** (gradual sparsity increase)
- ✅ **SAELens-validated hyperparameters** (expansion_factor=32, learning_rate=3e-4)
- ✅ **Feature activation tracking** (for ghost grads)
- ✅ **Dead feature detection** (configurable window)

---

## 🔄 Phase 2: Dictionary Learning Features (IN PROGRESS)

### Feature Analysis
- ⬜ Feature activation patterns
  - `sae.get_top_activating_features(text)` - Find which features activate
  - `sae.get_feature_activations(feature_id, dataset)` - Activation statistics
  - Feature activation heatmaps

- ⬜ Feature interpretation
  - `sae.get_top_activating_texts(feature_id)` - Find texts that activate feature
  - `sae.analyze_feature_direction(feature_id)` - Feature direction in activation space
  - Feature clustering and similarity

### Feature Visualization
- ⬜ Activation visualizations
  - Per-token feature activations
  - Feature activation over sequence
  - Multi-feature comparison

- ⬜ Feature dashboards
  - Interactive feature browser
  - Top activating examples per feature
  - Feature statistics and distributions

### Feature Steering
- ⬜ Feature-based interventions
  - `model.ablate_feature(sae, feature_id)` - Remove feature influence
  - `model.amplify_feature(sae, feature_id, strength)` - Amplify feature
  - `model.activate_feature(sae, feature_id, strength)` - Force feature activation

- ⬜ Multi-feature steering
  - Combine multiple feature interventions
  - Feature direction editing
  - Controlled generation with features

---

## 🔄 Phase 3: Advanced SAE Variants (FUTURE)

### Alternative Architectures
- ⬜ Gated SAE (auxiliary gate network)
- ⬜ Transcoder (layer N → layer N+1 features)
- ⬜ Crosscoder (multi-component features)
- ⬜ Hierarchical SAE (multi-level features)

### Training Improvements
- ⬜ Dead feature resampling
- ⬜ Adaptive sparsity (learned k parameter)
- ⬜ Multi-layer joint training
- ⬜ Curriculum learning for SAEs

### Analysis Tools
- ⬜ Feature attribution methods
- ⬜ Feature importance scoring
- ⬜ Causal feature analysis
- ⬜ Feature evolution tracking

---

## 🔄 Phase 4: Circuit Discovery (FUTURE)

### Feature-Based Circuit Analysis
- ⬜ Feature connectivity graphs
  - Which features activate together?
  - Feature → feature information flow
  - Multi-layer feature circuits

- ⬜ Automated circuit discovery
  - Find computational circuits using features
  - Circuit validation and testing
  - Circuit visualization

### Integration with Existing Tools
- ⬜ Combine with activation patching
- ⬜ Feature-level causal interventions
- ⬜ Circuit completion analysis

---

## Current Status (as of 2025-11-08)

### What Works Now

**Phase 1 (Complete - with SAELens Feature Parity):**
- ✅ Train SAE on any layer/component
- ✅ **BatchTopK architecture** (modern, SAELens-style)
- ✅ **Ghost gradients** (reduces dead features from 95% → 70%)
- ✅ **LR scheduling** (cosine/linear decay after warmup)
- ✅ **Sparsity warmup** (gradual sparsity increase)
- ✅ Save and load trained SAEs
- ✅ Streaming training for large datasets
- ✅ W&B logging (optional)
- ✅ Comprehensive examples in `examples/`

**Example (Updated with New Features):**
```python
from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
interp = InterpretableModel(model, tokenizer=tokenizer)

# Configure SAE training with SAELens-validated defaults
config = SAEConfig(
    # Modern architecture
    sae_type="batchtopk",      # BatchTopK (recommended over "topk")
    expansion_factor=32,        # SAELens-validated (was 16)
    k=128,

    # Optimized learning
    learning_rate=3e-4,         # SAELens-validated (was 1e-4)
    lr_scheduler="cosine",      # Cosine decay after warmup
    sparsity_warm_up_steps=None,  # Auto-set to total steps

    # Dead feature reduction
    use_ghost_grads=True,       # Reduces dead features to ~70%

    # Other settings
    num_epochs=3,
    use_wandb=True,
)

# Train SAE
sae = interp.train_sae(
    layer=12,
    component="mlp",
    dataset=texts,  # List of strings
    config=config,
    save_path="my_sae.mlx"
)

# Use trained SAE
features = sae.encode(activations)
reconstructed = sae.decode(features)
```

**Expected Results with New Features:**
- Dead features: **~70%** (down from 95% without ghost grads)
- Reconstruction loss: **0.10-0.15** (down from 0.27)
- Training: **Continues improving through all epochs** (no early plateau)
- Stability: **Much more stable** (BatchTopK + cosine LR decay)

### What's Next

**Phase 2 Focus:**
The immediate priority is implementing dictionary learning features:

1. **Feature Analysis** (highest priority)
   - Users want to understand what features represent
   - Need tools to find top-activating examples
   - Feature activation patterns and statistics

2. **Feature Steering** (high priority)
   - Enable controlled generation with features
   - Feature ablation and amplification
   - Integration with existing intervention system

3. **Visualization** (medium priority)
   - Make features interpretable
   - Interactive dashboards
   - Activation heatmaps

---

## Implementation Notes

### Phase 2 Implementation Strategy

**Step 1: Feature Analysis Methods** (weeks 1-2)
```python
# Proposed API:
sae = SAE.load("my_sae.mlx")

# Find top features for a text
top_features = sae.get_top_activating_features(
    model,
    text="The capital of France is Paris",
    layer=12,
    component="mlp",
    k=10  # Top 10 features
)

# Find examples that activate a feature
examples = sae.get_top_activating_texts(
    model,
    feature_id=42,
    dataset=texts,
    layer=12,
    component="mlp",
    k=20  # Top 20 examples
)

# Get activation statistics
stats = sae.get_feature_stats(
    model,
    dataset=texts,
    layer=12,
    component="mlp"
)
```

**Step 2: Feature Steering** (weeks 3-4)
```python
# Proposed API:
with model.trace("The capital of France is") as trace:
    # Ablate a feature
    model.ablate_sae_feature(
        sae=sae,
        layer=12,
        component="mlp",
        feature_id=42,
    )
    output_ablated = model.output.save()

# Amplify a feature
with model.trace("The capital of France is") as trace:
    model.amplify_sae_feature(
        sae=sae,
        layer=12,
        component="mlp",
        feature_id=42,
        strength=2.0,
    )
    output_amplified = model.output.save()
```

**Step 3: Visualization** (weeks 5-6)
```python
# Proposed API:
from mlxterp.visualization import FeatureDashboard

dashboard = FeatureDashboard(
    model=model,
    sae=sae,
    layer=12,
    component="mlp"
)

# Launch interactive dashboard
dashboard.serve(port=8000)

# Or generate static visualizations
dashboard.plot_feature_activations(feature_id=42)
dashboard.plot_feature_heatmap(text="Hello world")
```

---

## Dependencies for Future Phases

### Phase 2 Requirements:
- None - all functionality can be built on existing infrastructure

### Phase 3 Requirements:
- Gated SAE: Requires auxiliary networks (low complexity)
- Transcoder: Needs paired activation collection (medium complexity)
- Crosscoder: Needs multi-component handling (medium complexity)

### Phase 4 Requirements:
- Circuit discovery: Needs graph libraries (networkx, graphviz)
- Feature connectivity: Needs correlation analysis
- Visualization: May need D3.js or similar for interactive graphs

---

## Success Metrics

### Phase 1 (Complete) ✅
- ✅ Can train SAE on any model
- ✅ Training completes without memory issues
- ✅ SAEs can be saved and loaded
- ✅ Dead features < 85%
- ✅ Reconstruction MSE < 0.5

### Phase 2 (Target)
- Users can identify what features represent
- Feature steering changes model outputs predictably
- Dashboard provides intuitive feature exploration
- Examples and documentation for all features

### Phase 3 (Target)
- Alternative SAE variants match or exceed TopK performance
- Training time reduced by 30% through optimizations
- Feature resampling reduces dead features to < 70%

### Phase 4 (Target)
- Automated circuit discovery finds known circuits
- Feature circuits validate against activation patching
- Circuit visualization is publication-ready

---

## Resources and References

### Papers
- **Sparse Autoencoders:** Cunningham et al. (2023)
- **Scaling Monosemanticity:** Anthropic (2024)
- **Dictionary Learning:** Olah et al. (2020)

### Existing Implementations
- **SAELens** (PyTorch) - inspiration for architecture
- **Anthropic's Research** - feature interpretation methods
- **TransformerLens** - integration patterns

### mlxterp Specific
- See `examples/sae_realistic_training.py` for full training example
- See `TROUBLESHOOTING.md` for common issues
- See `CLAUDE.md` for overall library architecture
