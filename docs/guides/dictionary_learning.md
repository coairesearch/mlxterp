# Dictionary Learning with Sparse Autoencoders

## Overview

Dictionary learning aims to decompose neural network activations into interpretable, sparse features using **Sparse Autoencoders (SAEs)**. This technique helps researchers understand what individual neurons or circuits in a model are computing.

### What are Sparse Autoencoders?

SAEs learn an **overcomplete** dictionary of features from layer activations:

- **Input**: Dense activations from a layer (e.g., MLP outputs)
- **Output**: Sparse feature representation (typically 8-32x more features than inputs)
- **Goal**: Each feature should represent a monosemantic concept or pattern

### Why Dictionary Learning?

Neural networks often exhibit **polysemanticity** - individual neurons respond to multiple unrelated concepts. SAEs help address this by:

1. **Decomposing polysemantic neurons** into monosemantic features
2. **Discovering interpretable circuits** by analyzing feature activations
3. **Enabling model steering** by manipulating specific features
4. **Understanding model behavior** through feature analysis

## Quick Start

### Basic Training

Train an SAE on a model layer with just one line:

```python
from mlxterp import InterpretableModel

# Load model
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Train SAE
sae = model.train_sae(
    layer=10,
    dataset=["Sample text 1", "Sample text 2", ...],
    save_path="my_sae.mlx"
)
```

### Using Pre-trained SAEs

```python
# Load saved SAE
sae = model.load_sae("my_sae.mlx")

# Check compatibility
if sae.is_compatible(model, layer=10, component="mlp"):
    print("✓ SAE is compatible!")

# Encode activations to features
with model.trace("Test input") as trace:
    pass

activation = trace.activations["model.layers.10.mlp"]
features = sae.encode(activation)  # Sparse features
reconstructed = sae.decode(features)  # Reconstructed activation
```

## Training SAEs

### Dataset Requirements

For meaningful feature learning, use a **large, diverse dataset**:

- **Minimum**: 10,000-20,000 text samples
- **Recommended**: 50,000+ samples
- **Quality**: Diverse topics and writing styles

#### Loading HuggingFace Datasets

```python
from datasets import load_dataset
from mlxterp import InterpretableModel, SAEConfig

# Load dataset
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

# Prepare texts
texts = []
for item in dataset:
    text = item['text'].strip()
    if len(text) > 50:  # Filter short texts
        texts.append(text[:512])  # Truncate long texts
    if len(texts) >= 20000:
        break

# Train SAE
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
sae = model.train_sae(
    layer=10,
    dataset=texts,
    save_path="sae_wikitext_20k.mlx"
)
```

See `examples/sae_realistic_training.py` for a complete example.

### Configuration

Customize SAE training with `SAEConfig`:

```python
from mlxterp import SAEConfig

config = SAEConfig(
    expansion_factor=32,    # 32x more features than input dimension
    k=128,                  # Keep top-128 features active (sparsity level)
    learning_rate=3e-4,     # Learning rate
    num_epochs=5,           # Training epochs
    batch_size=64,          # Batch size
    warmup_steps=500,       # Learning rate warmup steps
    validation_split=0.05,  # 5% validation set
    normalize_input=True,   # Normalize activations before encoding
    tied_weights=False,     # Whether to tie encoder/decoder weights
    gradient_clip=1.0,      # Gradient clipping threshold
)

sae = model.train_sae(layer=10, dataset=texts, config=config)
```

### Training Parameters

#### Expansion Factor
- **Definition**: Ratio of hidden features to input dimension
- **Typical values**: 8x to 64x
- **Trade-offs**:
  - Higher = More fine-grained features, but more computation and memory
  - Lower = Faster training, but may miss subtle features

#### Sparsity (k)
- **Definition**: Number of top features to keep active per sample
- **Typical values**: 50-200
- **Guidelines**:
  - k ≈ 0.5% to 2% of hidden dimension
  - For d_hidden=16384, try k=64-256

#### Learning Rate
- **Typical values**: 1e-4 to 5e-4
- **With warmup**: Start with 500-1000 warmup steps
- **Recommendation**: 3e-4 with 500-step warmup works well

#### Dataset Size
- **Minimum**: 10,000 samples for basic features
- **Recommended**: 50,000+ samples for robust features
- **Large-scale**: 100,000+ samples for production SAEs

### Training Monitoring

During training, monitor these metrics:

```
Training: 100%|████████| 15625/15625 [05:23, loss=0.245, l0=128.0, dead=8.5%]
```

- **loss**: Reconstruction loss (should decrease)
- **l0**: Average number of active features (should match k)
- **dead**: Percentage of features that never activate

**Healthy training**:
- ✅ Loss decreases over time
- ✅ L0 ≈ k (target sparsity)
- ✅ Dead fraction < 20%

**Warning signs**:
- ❌ `loss=nan` - Numerical instability
- ❌ `l0=0.0` - No features activating
- ❌ `dead=80%+` - Most features unused

## SAE Architecture

### TopK Sparse Autoencoder

mlxterp implements TopK SAEs, which enforce exact sparsity:

```
Input (d_model)
    ↓
Normalize (optional)
    ↓
Encoder: Linear + ReLU
    ↓
TopK(k) - Keep only top-k features
    ↓
Features (d_hidden) - SPARSE
    ↓
Decoder: Linear
    ↓
Denormalize (optional)
    ↓
Reconstruction (d_model)
```

### Forward Pass

```python
# Encode: activation → sparse features
features = sae.encode(activation)
# Shape: (batch, seq_len, d_hidden)
# Most values are zero (sparse!)

# Decode: sparse features → reconstruction
reconstructed = sae.decode(features)
# Shape: (batch, seq_len, d_model)
```

### Loss Function

SAEs are trained to minimize reconstruction error:

```python
loss = MSE(input, reconstruction)
```

Optional L1 sparsity penalty (not typically used with TopK):
```python
loss = MSE(input, reconstruction) + λ * L1(features)
```

## Using Trained SAEs

### Encoding Activations

```python
# Get activation from model
with model.trace("The capital of France is") as trace:
    pass

activation = trace.activations["model.layers.10.mlp"]

# Encode to sparse features
features = sae.encode(activation)

# Check sparsity
num_active = int(mx.sum(features != 0))
print(f"Active features: {num_active}/{sae.d_hidden}")  # e.g., 128/16384

# Get active feature indices
active_indices = mx.where(features[0, -1] != 0)[0]  # Last token
print(f"Active feature IDs: {active_indices.tolist()}")
```

### Reconstruction Quality

```python
# Decode back to activation space
reconstructed = sae.decode(features)

# Measure reconstruction error
mse = float(mx.mean((activation - reconstructed) ** 2))
print(f"Reconstruction MSE: {mse:.6f}")

# Good reconstruction: MSE < 0.01
# Poor reconstruction: MSE > 0.1
```

### Save and Load

```python
# Save trained SAE
sae.save("my_sae.mlx")

# Load later
sae = model.load_sae("my_sae.mlx")

# Check metadata
print(sae.metadata)
# {'layer': 10, 'component': 'mlp', 'model_name': '...', ...}
```

## Experiment Tracking with Weights & Biases

mlxterp supports automatic logging to [Weights & Biases](https://wandb.ai) for experiment tracking and visualization.

### Setup

```bash
# Install wandb
pip install wandb

# Login (first time only)
wandb login
```

### Basic Usage

Enable W&B logging in your SAE configuration:

```python
from mlxterp import InterpretableModel, SAEConfig

config = SAEConfig(
    expansion_factor=16,
    k=100,
    learning_rate=3e-4,

    # Enable W&B logging
    use_wandb=True,
    wandb_project="my-sae-experiments",
    wandb_name="sae_layer10_mlp",
    wandb_tags=["sae", "layer10", "mlp"],
)

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
sae = model.train_sae(
    layer=10,
    dataset=texts,
    config=config,
)
```

### Logged Metrics

W&B automatically logs:

**Training metrics** (every step):
- `train/loss` - Reconstruction loss
- `train/recon_loss` - MSE reconstruction error
- `train/l0` - Average active features per sample
- `train/dead_fraction` - Percentage of dead features
- `train/l1_loss` - L1 sparsity penalty (if used)
- `train/learning_rate` - Current learning rate

**Validation metrics** (per epoch):
- `val/loss` - Validation reconstruction loss
- `val/l0` - Validation feature activation
- `val/dead_fraction` - Validation dead features

**Configuration** (logged once):
- All SAEConfig parameters
- Model metadata (d_model, d_hidden, layer, component)
- Dataset size and activation counts

### Example

See `examples/sae_with_wandb.py` for a complete example.

### Tips for Experiment Tracking

1. **Organize with projects**: Use different projects for different models or research questions
```python
config = SAEConfig(
    use_wandb=True,
    wandb_project="llama-sae-research",  # One project per model family
)
```

2. **Use descriptive names**: Include layer, component, and key hyperparameters
```python
config = SAEConfig(
    use_wandb=True,
    wandb_name=f"sae_l{layer}_{component}_k{k}_exp{expansion}",
)
```

3. **Tag for filtering**: Add tags to group related experiments
```python
config = SAEConfig(
    use_wandb=True,
    wandb_tags=["baseline", "layer-10", "high-sparsity"],
)
```

4. **Compare runs**: Use W&B's comparison features to analyze:
   - How expansion factor affects reconstruction quality
   - Impact of sparsity (k) on feature interpretability
   - Training stability across different learning rates

### Visualizations in W&B

W&B provides automatic visualizations:
- **Loss curves**: Track training and validation loss
- **Feature activation**: Monitor l0 (number of active features)
- **Dead neurons**: Track feature utilization over time
- **Learning rate**: Visualize warmup and scheduling
- **Hyperparameter importance**: Compare runs to find optimal settings

## Advanced Topics

### Component Selection

Train SAEs on different components:

```python
# MLP output (recommended starting point)
sae_mlp = model.train_sae(layer=10, component="mlp", dataset=texts)

# Attention output
sae_attn = model.train_sae(layer=10, component="attn", dataset=texts)

# Residual stream (after layer)
sae_resid = model.train_sae(layer=10, component="output", dataset=texts)
```

### Layer Selection

Choose layers based on your research question:

- **Early layers** (0-5): Low-level features (syntax, simple patterns)
- **Middle layers** (6-15): Mid-level concepts and facts
- **Late layers** (16+): High-level reasoning and task-specific features

```python
# Train SAEs on multiple layers
for layer in [6, 12, 18, 24]:
    sae = model.train_sae(
        layer=layer,
        dataset=texts,
        save_path=f"sae_layer{layer}.mlx"
    )
```

### Normalization

Input normalization stabilizes training:

```python
# With normalization (recommended)
config = SAEConfig(normalize_input=True)
sae = model.train_sae(layer=10, dataset=texts, config=config)

# During training, SAE computes running mean/std
# These are used to normalize inputs during encoding
```

### Tied Weights

Reduce parameters by tying encoder and decoder weights:

```python
config = SAEConfig(
    tied_weights=True,  # decoder = encoder^T
)

sae = model.train_sae(layer=10, dataset=texts, config=config)

# Benefits:
# - Fewer parameters (2x reduction)
# - Faster training
# - May improve interpretability

# Trade-offs:
# - Slightly worse reconstruction
# - Less flexible decoder
```

## Performance Considerations

### Training Speed

Training time depends on:
- Dataset size
- Model dimension
- Expansion factor
- Batch size
- Hardware

**Typical speeds** (M1 Max):
- Small dataset (10k samples): 1-2 minutes
- Medium dataset (50k samples): 5-10 minutes
- Large dataset (200k samples): 30-60 minutes

### Memory Usage

```python
# SAE parameters
params = d_model * d_hidden * 2  # encoder + decoder weights

# Example: d_model=2048, expansion=32x
# d_hidden = 2048 * 32 = 65,536
# params = 2048 * 65,536 * 2 = 268M parameters
# memory ≈ 1GB (float32)
```

**Memory optimization**:
- Use smaller expansion factors (8x-16x)
- Train on smaller batches
- Process dataset in chunks
- Use MLX's lazy evaluation

### MLX Optimizations

mlxterp leverages MLX's features:

1. **Unified Memory**: No CPU↔GPU transfers
2. **Lazy Evaluation**: Compute only what's needed
3. **Efficient Kernels**: Optimized Metal operations
4. **Batch Processing**: Vectorized operations

## Best Practices

### Dataset Preparation

1. **Diversity**: Include varied topics and styles
2. **Quality**: Filter out low-quality or corrupted text
3. **Length**: Use texts with 50-500 tokens
4. **Size**: Aim for 20,000+ samples minimum

```python
def prepare_dataset(raw_texts, min_len=50, max_len=512):
    """Prepare high-quality dataset for SAE training."""
    filtered = []
    for text in raw_texts:
        # Remove very short texts
        if len(text) < min_len:
            continue
        # Truncate long texts
        if len(text) > max_len:
            text = text[:max_len]
        # Remove duplicates
        if text not in filtered:
            filtered.append(text.strip())
    return filtered
```

### Hyperparameter Selection

Start with these defaults:

```python
config = SAEConfig(
    expansion_factor=16,   # Good balance
    k=100,                 # ~0.5-1% sparsity
    learning_rate=3e-4,    # Stable learning rate
    num_epochs=5,          # Sufficient for convergence
    batch_size=64,         # Good for most GPUs
    warmup_steps=500,      # Stabilizes early training
)
```

Adjust based on results:
- **Poor reconstruction** → Increase expansion factor or decrease k
- **Too slow** → Decrease expansion factor or dataset size
- **Training unstable** → Decrease learning rate, increase warmup
- **High dead fraction** → Increase dataset diversity

### Validation

Check SAE quality:

```python
# 1. Reconstruction quality
test_texts = ["Sample 1", "Sample 2", ...]
with model.trace(test_texts) as trace:
    pass

activation = trace.activations["model.layers.10.mlp"]
reconstructed = sae.decode(sae.encode(activation))
mse = float(mx.mean((activation - reconstructed) ** 2))

# Target: MSE < 0.05

# 2. Feature activation distribution
features = sae.encode(activation)
activation_counts = mx.sum(features != 0, axis=(0, 1))

# Check for dead features
dead_features = int(mx.sum(activation_counts == 0))
print(f"Dead features: {dead_features}/{sae.d_hidden}")

# Target: < 20% dead
```

## Coming Soon: Phase 2 Features

Future functionality for feature analysis:

### Feature Analysis
```python
# Find top features for text (Phase 2)
top_features = sae.analyze_text(
    model,
    prompt="Artificial intelligence is transforming society",
    top_k=10
)

# Find texts that activate a feature (Phase 2)
activating_texts = sae.get_top_activating_texts(
    feature_idx=1234,
    dataset=texts,
    top_k=20
)
```

### Feature Visualization
```python
# Visualize feature activations (Phase 2)
sae.visualize_feature(
    feature_idx=1234,
    model=model,
    dataset=texts
)

# Interactive feature dashboard (Phase 2)
sae.launch_dashboard(model=model, dataset=texts)
```

### Model Steering
```python
# Steer model behavior with features (Phase 2)
with model.trace("Generate text about...") as trace:
    sae.steer(
        model=model,
        layer=10,
        feature_idx=1234,
        strength=2.0  # Amplify this feature
    )
    output = model.output.save()
```

## Resources

### Examples
- `examples/sae_quickstart.py` - Basic SAE training
- `examples/sae_realistic_training.py` - Training with HuggingFace datasets

### Related Research
- [Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) - Anthropic
- [Sparse Autoencoders Find Highly Interpretable Features](https://arxiv.org/abs/2309.08600)
- [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) - Anthropic

### API Reference
- [SAE API Documentation](../API.md#sae)
- [SAEConfig Documentation](../API.md#saeconfig)
- [InterpretableModel.train_sae()](../API.md#train_sae)

## Troubleshooting

### NaN Losses

**Symptoms**: `loss=nan` during training

**Causes**:
1. Learning rate too high
2. Numerical instability
3. Bad initialization

**Solutions**:
```python
config = SAEConfig(
    learning_rate=1e-4,      # Lower learning rate
    warmup_steps=1000,       # More warmup
    gradient_clip=0.5,       # Stricter clipping
    normalize_input=True,    # Enable normalization
)
```

### No Active Features

**Symptoms**: `l0=0.0` or very low

**Causes**:
1. k too small
2. Poor initialization
3. Dataset issues

**Solutions**:
```python
config = SAEConfig(
    k=100,  # Increase k
)
# Use more diverse dataset
```

### High Dead Fraction

**Symptoms**: `dead=50%+`

**Causes**:
1. Dataset not diverse enough
2. Expansion factor too high
3. Not enough training

**Solutions**:
- Increase dataset size and diversity
- Reduce expansion factor
- Train for more epochs
- Consider dead neuron resampling (Phase 2)

### Slow Training

**Symptoms**: Training takes hours

**Causes**:
1. Dataset too large
2. Model too large
3. Expansion factor too high

**Solutions**:
```python
config = SAEConfig(
    batch_size=128,        # Larger batches
    expansion_factor=8,    # Reduce expansion
)
# Use subset of dataset for faster iteration
```

## Summary

Dictionary learning with SAEs enables:

1. ✅ **Interpretable features** from dense activations
2. ✅ **Monosemantic decomposition** of polysemantic neurons
3. ✅ **Circuit discovery** through feature analysis
4. ✅ **Model steering** via feature manipulation

**Key takeaways**:
- Use large, diverse datasets (20k+ samples)
- Start with expansion=16x, k=100
- Monitor loss, l0, and dead fraction
- Validate reconstruction quality
- Experiment with different layers and components

**Next steps**:
- Train your first SAE with `examples/sae_realistic_training.py`
- Explore different hyperparameters
- Prepare for Phase 2: feature analysis and visualization
