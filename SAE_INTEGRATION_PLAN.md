# SAE/Transcoder/Crosscoder Integration Plan

## Executive Summary

This plan outlines how to integrate Sparse Autoencoders (SAEs), Transcoders, and Crosscoders into mlxterp with a dead-simple API that hides all complexity from users.

**Goal**: One-line commands to train, use, and analyze interpretable features.

## Terminology

### What are these?

1. **SAE (Sparse Autoencoder)**
   - Learns interpretable features from a **single layer's activations**
   - Input: Layer N activations → Output: Sparse features → Reconstruct: Layer N activations
   - Use case: Understand what Layer 10 is computing

2. **Transcoder**
   - Learns features that **transform** between layers
   - Input: Layer N activations → Sparse features → Output: Layer N+1 activations
   - Use case: Understand how information flows from Layer 10 → Layer 11

3. **Crosscoder**
   - Learns features from **multiple components simultaneously**
   - Input: [MLP output, Attention output] → Shared sparse features → Reconstruct both
   - Use case: Find features shared between MLP and attention

## API Design Philosophy

### Core Principles

1. **Hide complexity**: User shouldn't need to understand MLX, optimizers, or loss functions
2. **Sensible defaults**: Works out-of-the-box with minimal configuration
3. **Progressive disclosure**: Simple API for beginners, advanced options for experts
4. **Integrate naturally**: Feels like a native part of mlxterp

### Design Pattern

```python
# Pattern 1: Train and save
sae = model.train_sae(layer=10, dataset=texts)
sae.save("layer10_sae.mlx")

# Pattern 2: Load and use
sae = model.load_sae("layer10_sae.mlx")
features = sae.encode(activation)

# Pattern 3: Analyze
top_features = sae.find_top_features(text="The capital of France")
sae.visualize_feature(42)
```

## Proposed API

### 1. Training SAEs

#### Simple API (Recommended)

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# One-line SAE training
sae = model.train_sae(
    layer=10,
    component="mlp",           # "mlp", "attn", "residual"
    dataset=text_samples,      # List of strings
    save_path="sae_layer10.mlx"
)

# Automatic progress bar, checkpointing, dead neuron handling
```

**Behind the scenes**:
- Collects activations using existing `batch_get_activations`
- Trains with sensible defaults (16x expansion, TopK, auto λ)
- Saves model + metadata (layer, component, hyperparams)
- Shows progress bar with metrics (loss, sparsity, dead neurons)

#### Advanced API (For researchers)

```python
from mlxterp.sae import SAEConfig, SAETrainer

config = SAEConfig(
    expansion_factor=32,        # Default: 16
    k=100,                      # TopK sparsity
    learning_rate=1e-4,
    batch_size=256,
    num_epochs=10,
    dead_neuron_threshold=0.01,
    checkpoint_every=1000,
    normalize_activations=True,
    tied_weights=False          # Tie encoder/decoder weights
)

sae = SAETrainer.train(
    model=model,
    layer=10,
    component="mlp",
    dataset=text_samples,
    config=config,
    validation_split=0.1
)
```

### 2. Training Transcoders

```python
# Simple: Learn layer N → N+1 transformation
transcoder = model.train_transcoder(
    from_layer=10,
    to_layer=11,
    component="mlp",            # Same component across layers
    dataset=text_samples,
    save_path="transcoder_10to11.mlx"
)

# Advanced: Cross-component transcoding
transcoder = model.train_transcoder(
    from_layer=10,
    from_component="mlp",
    to_layer=11,
    to_component="attn",        # MLP → Attention transformation
    dataset=text_samples
)
```

**Use case**: Understand how MLP in layer 10 influences attention in layer 11.

### 3. Training Crosscoders

```python
# Learn shared features between MLP and Attention
crosscoder = model.train_crosscoder(
    layer=10,
    components=["mlp", "attn"],  # Train on both simultaneously
    dataset=text_samples,
    save_path="crosscoder_layer10.mlx"
)

# Or across layers
crosscoder = model.train_crosscoder(
    layers=[8, 10, 12],
    component="mlp",              # Same component, different layers
    dataset=text_samples
)
```

**Use case**: Find features that appear in both MLP and attention (circuit discovery).

## Feature Analysis API

### 1. Encoding Activations

```python
# Load trained SAE
sae = model.load_sae("sae_layer10.mlx")

# Encode a specific prompt
with model.trace("The capital of France is") as trace:
    activation = trace.activations["model.model.layers.10.mlp"]

features = sae.encode(activation)
# features shape: (seq_len, num_features)

# See which features fired at last token
active = features[0, -1] > 0.1
print(f"Active features: {mx.where(active)[0]}")
```

### 2. Feature Analysis

#### Find Top Features for a Text

```python
# What features activate on this text?
result = sae.analyze_text(
    model=model,
    text="The capital of France is Paris",
    top_k=10
)

print(result)
# Output:
# Feature 142: 0.85 (token 'Paris')
# Feature 23:  0.72 (token 'capital')
# Feature 891: 0.65 (token 'France')
```

#### Visualize a Feature

```python
# What does feature 142 represent?
sae.visualize_feature(
    model=model,
    feature_id=142,
    dataset=validation_texts,  # Find top activating examples
    top_k=20
)

# Output:
# Feature 142: "Geographic Capitals"
#
# Top activating examples:
# 1. "Paris is the capital of France" (activation: 0.95)
# 2. "London is the capital of England" (activation: 0.91)
# 3. "Tokyo is the capital of Japan" (activation: 0.87)
# ...
#
# Decoded weights (top tokens):
# - capital: 0.42
# - city: 0.31
# - located: 0.28
```

#### Feature Dashboard

```python
# Interactive dashboard (if matplotlib available)
sae.dashboard(
    model=model,
    validation_dataset=texts,
    port=8080  # Launches local web server
)

# Shows:
# - Feature activation heatmap
# - Top activating examples per feature
# - Feature co-occurrence matrix
# - Dead neuron tracker
```

### 3. Feature Steering

```python
# Activate a specific feature to control model behavior
with model.trace("The capital of Germany is") as trace:
    activation = trace.activations["model.model.layers.10.mlp"]

# Manually activate "capital city" feature
steered = sae.steer(
    activation=activation,
    feature_id=142,
    strength=2.0  # Amplify feature
)

# Use steered activation in the model
with model.trace("The capital of Germany is",
                interventions={"layers.10.mlp": lambda x: steered}):
    output = model.output.save()

# Decode
prediction = model.get_token_predictions(output[0, -1], top_k=5)
print([model.token_to_str(t) for t in prediction])
# Likely: ["Berlin", "Munich", "Hamburg", ...]
```

### 4. Feature Arithmetic

```python
# Find "Paris - France + Germany" in feature space
paris_features = sae.encode_text(model, "Paris")
france_features = sae.encode_text(model, "France")
germany_features = sae.encode_text(model, "Germany")

# Feature arithmetic
result_features = paris_features - france_features + germany_features

# Decode back to activation space
result_activation = sae.decode(result_features)

# What would this produce?
sae.interpret_features(result_features, top_k=5)
# Might show: "Berlin", "German capital", etc.
```

## File Structure

```
mlxterp/
├── __init__.py
├── model.py                    # Add train_sae, train_transcoder, etc.
├── tokenization.py
├── analysis.py
├── sae/                        # New module
│   ├── __init__.py
│   ├── base.py                 # Base SAE class
│   ├── sae.py                  # Standard SAE
│   ├── transcoder.py           # Transcoder
│   ├── crosscoder.py           # Crosscoder
│   ├── config.py               # SAEConfig, TranscoderConfig
│   ├── trainer.py              # Training logic
│   ├── analysis.py             # Feature analysis tools
│   ├── visualization.py        # Plotting, dashboard
│   └── utils.py                # Helper functions
└── ...
```

## Implementation Phases

### Phase 1: Core SAE (Week 1-2)

**Goal**: Basic SAE training and encoding works

- [ ] `mlxterp/sae/base.py` - Abstract base class
- [ ] `mlxterp/sae/sae.py` - Standard SAE with TopK
- [ ] `mlxterp/sae/config.py` - Configuration dataclass
- [ ] `mlxterp/sae/trainer.py` - Training loop with progress bar
- [ ] `model.train_sae()` - Simple API
- [ ] `model.load_sae()` - Load saved SAE
- [ ] Tests with Llama-3.2-1B

**Deliverable**:
```python
sae = model.train_sae(layer=10, dataset=texts)
features = sae.encode(activation)
```

### Phase 2: Feature Analysis (Week 3)

**Goal**: Users can understand what features represent

- [ ] `sae.analyze_text()` - Find top features for a prompt
- [ ] `sae.visualize_feature()` - Show top activating examples
- [ ] `sae.interpret_features()` - Decode feature to tokens
- [ ] Dead neuron detection and resampling
- [ ] Feature activation statistics

**Deliverable**:
```python
sae.visualize_feature(142, dataset=validation_texts)
# Shows: "Feature 142: Geographic Capitals"
```

### Phase 3: Transcoders (Week 4)

**Goal**: Understand layer-to-layer transformations

- [ ] `mlxterp/sae/transcoder.py` - Transcoder class
- [ ] `model.train_transcoder()` - Simple API
- [ ] Layer pair validation
- [ ] Transcoder-specific analysis tools

**Deliverable**:
```python
transcoder = model.train_transcoder(from_layer=10, to_layer=11)
```

### Phase 4: Crosscoders (Week 5)

**Goal**: Find shared features across components

- [ ] `mlxterp/sae/crosscoder.py` - Crosscoder class
- [ ] Multi-component batching
- [ ] Shared feature visualization
- [ ] Component interaction analysis

**Deliverable**:
```python
crosscoder = model.train_crosscoder(layer=10, components=["mlp", "attn"])
```

### Phase 5: Advanced Features (Week 6+)

**Goal**: Research-grade capabilities

- [ ] Feature steering API
- [ ] Feature arithmetic
- [ ] Interactive dashboard (matplotlib-based)
- [ ] Feature clustering and similarity
- [ ] Circuit discovery with features
- [ ] Automatic feature labeling (using another LLM)
- [ ] Export to standard formats (for sharing)

## Training Data Management

### Dataset Collection API

```python
# Helper to collect diverse activations
from mlxterp.sae import ActivationDataset

# Collect from a dataset
dataset = ActivationDataset.from_texts(
    model=model,
    texts=text_samples,
    layer=10,
    component="mlp",
    cache_dir="./activation_cache",  # Disk caching for large datasets
    max_samples=1_000_000,
    shuffle=True
)

# Or from HuggingFace dataset
dataset = ActivationDataset.from_huggingface(
    model=model,
    dataset_name="openwebtext",
    layer=10,
    component="mlp",
    max_samples=10_000_000,
    streaming=True  # Don't load all at once
)

# Dataset is iterator-like, memory efficient
for batch in dataset.batches(batch_size=256):
    # batch shape: (256, hidden_dim)
    pass
```

### Automatic Dataset Curation

```python
# Smart dataset selection for better SAE training
dataset = ActivationDataset.curate(
    model=model,
    source="pile",               # Use The Pile
    layer=10,
    diversity_metric="activation_variance",  # Select diverse activations
    max_samples=1_000_000,
    min_activation_norm=0.1,     # Filter low-norm activations
    deduplicate=True
)
```

## Saving and Loading

### File Format

```
sae_layer10.mlx/
├── config.json              # Hyperparameters, metadata
├── encoder.safetensors      # Encoder weights
├── decoder.safetensors      # Decoder weights
├── encoder_bias.safetensors # Biases
├── statistics.json          # Training stats, dead neurons
└── metadata.json            # Layer, component, model info
```

### API

```python
# Save
sae.save("sae_layer10.mlx")

# Load
sae = model.load_sae("sae_layer10.mlx")

# Or load standalone (without model)
from mlxterp.sae import load_sae
sae = load_sae("sae_layer10.mlx")

# Verify compatibility
if sae.is_compatible(model, layer=10, component="mlp"):
    features = sae.encode(activation)
else:
    print("SAE was trained on different model/layer!")
```

## Configuration Defaults

### SAE Defaults (Sensible for most users)

```python
DEFAULT_SAE_CONFIG = {
    "expansion_factor": 16,      # Hidden dim = 16 × input dim
    "k": 100,                    # TopK (keep top 100 features)
    "learning_rate": 1e-4,
    "batch_size": 256,
    "num_epochs": 10,
    "lambda_sparse": 0.0,        # Using TopK, not L1
    "normalize_input": True,     # Normalize activations
    "tied_weights": False,       # Separate encoder/decoder
    "dead_neuron_threshold": 1e-6,
    "resample_dead_every": 10000,  # Steps between resampling
    "warmup_steps": 1000,
    "gradient_clip": 1.0,
    "checkpoint_every": 5000,
    "validation_split": 0.05
}
```

### Transcoder Defaults

```python
DEFAULT_TRANSCODER_CONFIG = {
    "expansion_factor": 8,       # Smaller than SAE (less reconstruction needed)
    "k": 50,
    "learning_rate": 5e-5,       # Slower learning for stability
    "batch_size": 256,
    "num_epochs": 15,            # More epochs (harder task)
    # ... similar to SAE
}
```

### Crosscoder Defaults

```python
DEFAULT_CROSSCODER_CONFIG = {
    "expansion_factor": 24,      # Larger (combining multiple components)
    "k": 150,
    "learning_rate": 1e-4,
    "batch_size": 128,           # Smaller (multiple components = more memory)
    "num_epochs": 12,
    "component_weights": None,   # Equal weight to all components
    # ... similar to SAE
}
```

## User Experience Examples

### Example 1: Beginner - Understand Layer 10

```python
from mlxterp import InterpretableModel

# Load model
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Collect some text
texts = ["Paris is the capital of France", ...]  # 10k samples

# Train SAE (one line!)
sae = model.train_sae(layer=10, dataset=texts, save_path="sae.mlx")

# What features activate on this text?
sae.visualize_feature(142, dataset=texts)
# Shows: "Feature 142: Geographic Capitals"
```

**User experience**: Dead simple. No ML knowledge needed.

### Example 2: Intermediate - Feature Steering

```python
# Load trained SAE
sae = model.load_sae("sae.mlx")

# Find "capital city" feature
capital_feature = sae.find_feature_by_examples([
    "Paris is the capital of France",
    "London is the capital of England",
    "Tokyo is the capital of Japan"
])

print(f"Capital feature: {capital_feature}")  # e.g., 142

# Use it to steer the model
def activate_capital_feature(activation):
    features = sae.encode(activation)
    features[:, :, capital_feature] *= 2.0  # Amplify
    return sae.decode(features)

with model.trace("The capital of Germany is",
                interventions={"layers.10.mlp": activate_capital_feature}):
    output = model.output.save()

# More likely to say "Berlin"
```

**User experience**: Some understanding of features, but API guides them.

### Example 3: Advanced - Circuit Discovery

```python
# Train SAEs on multiple layers
sae_8 = model.train_sae(layer=8, dataset=texts)
sae_10 = model.train_sae(layer=10, dataset=texts)
sae_12 = model.train_sae(layer=12, dataset=texts)

# Find which features in layer 8 influence features in layer 12
from mlxterp.sae import find_feature_circuits

circuit = find_feature_circuits(
    model=model,
    saes={8: sae_8, 10: sae_10, 12: sae_12},
    source_layer=8,
    target_layer=12,
    dataset=texts,
    top_k=50  # Top 50 strongest connections
)

# Visualize the circuit
circuit.visualize(
    feature_labels={
        8: {42: "French words", 15: "Geographic locations"},
        12: {91: "Capital cities"}
    }
)

# Shows graph: Feature 42 (L8) → Feature 91 (L12)
#              Feature 15 (L8) → Feature 91 (L12)
```

**User experience**: Research-grade tools, but with helper functions.

## Documentation Structure

```
docs/
├── sae/
│   ├── quickstart.md           # 5-minute guide
│   ├── training.md             # How to train SAEs
│   ├── analysis.md             # Analyzing features
│   ├── steering.md             # Using features for control
│   ├── transcoders.md          # Transcoder guide
│   ├── crosscoders.md          # Crosscoder guide
│   ├── circuits.md             # Circuit discovery
│   ├── api.md                  # Complete API reference
│   ├── examples.md             # Cookbook recipes
│   └── troubleshooting.md      # Common issues
```

## Performance Targets

### Training Time (M2 Max, 32GB)

| Model | Layer | Samples | Expansion | Time |
|-------|-------|---------|-----------|------|
| Llama-3.2-1B | 10 | 100k | 16x | ~10 min |
| Llama-3.2-1B | 10 | 1M | 16x | ~1.5 hr |
| Llama-3.2-1B | 10 | 10M | 16x | ~15 hr |

### Memory Requirements

| Model | Expansion | Activations | Peak Memory |
|-------|-----------|-------------|-------------|
| Llama-3.2-1B (2048d) | 16x | 1M samples | ~8 GB |
| Llama-3.2-1B (2048d) | 32x | 1M samples | ~10 GB |
| Llama-3-8B (4096d) | 16x | 1M samples | ~16 GB |

## Success Metrics

### Phase 1 (Core SAE)
- [ ] Can train SAE in < 3 lines of code
- [ ] Training completes without OOM on M2 Max
- [ ] Reconstruction loss < 0.01
- [ ] Sparsity (L0) < 100 active features
- [ ] < 5% dead neurons after training

### Phase 2 (Analysis)
- [ ] Can identify interpretable features (qualitative)
- [ ] Feature visualization shows meaningful patterns
- [ ] Top activating examples make semantic sense

### Phase 3+ (Advanced)
- [ ] Transcoders achieve < 0.05 reconstruction loss
- [ ] Crosscoders find shared features (validated manually)
- [ ] Feature steering changes model outputs as expected

## Open Questions

1. **Evaluation**: How do we automatically evaluate SAE quality?
   - Reconstruction loss (easy)
   - Sparsity (easy)
   - Interpretability (hard - needs human evaluation)
   - Downstream task performance (medium)

2. **Feature labeling**: Should we auto-label features using another LLM?
   ```python
   sae.auto_label_features(
       labeler_model="claude-3-5-sonnet",
       num_examples=20
   )
   ```

3. **Sharing**: Should we create a repository for trained SAEs?
   - HuggingFace model hub integration?
   - `model.load_sae("username/llama-3.2-1b-layer10-sae")`

4. **Compatibility**: How to handle model updates?
   - Version SAEs with model checkpoints
   - Warn if model/SAE mismatch

## Next Steps

1. **Review this plan** - Get feedback
2. **Phase 1 implementation** - Start with basic SAE
3. **Test with real use case** - Train SAE on Llama-3.2-1B layer 10
4. **Iterate on API** - Refine based on usability
5. **Documentation** - Write guides as we build

## Conclusion

This plan provides:
- ✅ Simple API for beginners (one-line training)
- ✅ Powerful tools for researchers (full control)
- ✅ Natural integration with existing mlxterp
- ✅ Progressive disclosure (simple → advanced)
- ✅ Clear implementation roadmap

The key insight: **Hide the complexity, expose the power.**

Users should be able to train and analyze SAEs without understanding optimization, loss functions, or even what "sparse" means. But experts should have access to every knob if needed.
