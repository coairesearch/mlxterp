# Advanced SAE Features: Profi Mode & Circuit Discovery

## Part 1: Profi Mode API

### Philosophy

The plan has three API levels:

1. **Beginner**: One-line magic (`model.train_sae(...)`)
2. **Intermediate**: Config objects (`SAEConfig(...)`)
3. **Profi**: Full control over every aspect

### Profi Mode Design

#### Complete Control Over Training

```python
from mlxterp.sae import ProfiSAETrainer
from mlxterp.sae.callbacks import *
import mlx.core as mx
import mlx.optimizers as optim

# Profi mode: You control EVERYTHING
trainer = ProfiSAETrainer(
    model=model,
    layer=10,
    component="mlp"
)

# Custom architecture
class CustomSAE(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        # Your custom architecture here
        self.encoder = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.LayerNorm(d_hidden),
            CustomActivation(),  # Your custom activation
            nn.Dropout(0.1)
        )
        self.decoder = nn.Linear(d_hidden, d_model)

    def __call__(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

sae = CustomSAE(d_model=2048, d_hidden=32768)

# Custom loss function
def custom_loss(x, x_recon, z, step):
    # Your custom loss
    recon = mx.mean((x - x_recon) ** 2)

    # Custom sparsity penalty (e.g., hoyer sparsity)
    l1 = mx.mean(mx.abs(z))
    l2 = mx.sqrt(mx.mean(z ** 2))
    hoyer = l1 / (l2 + 1e-8)

    # Adaptive lambda based on training step
    lambda_sparse = 1e-3 * (1 + 0.1 * mx.log(1 + step / 1000))

    total = recon + lambda_sparse * (1 - hoyer)

    return total, {
        'recon': recon,
        'hoyer': hoyer,
        'lambda': lambda_sparse
    }

# Custom optimizer with custom schedule
def lr_schedule(step):
    warmup_steps = 1000
    if step < warmup_steps:
        return 1e-4 * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (100000 - warmup_steps)
        return 1e-4 * 0.5 * (1 + mx.cos(mx.pi * progress))

optimizer = optim.AdamW(
    learning_rate=lr_schedule,
    betas=[0.9, 0.999],
    eps=1e-8,
    weight_decay=0.01
)

# Custom data preprocessing
def preprocess(batch):
    # Center and scale
    batch = batch - mx.mean(batch, axis=-1, keepdims=True)
    batch = batch / (mx.std(batch, axis=-1, keepdims=True) + 1e-8)

    # Add noise for robustness
    noise = mx.random.normal(batch.shape) * 0.01
    return batch + noise

# Custom callbacks
callbacks = [
    # Log to Weights & Biases
    WandbCallback(project="sae-experiments", name="custom-sae-v1"),

    # Save checkpoints
    CheckpointCallback(save_every=5000, keep_best=3),

    # Early stopping
    EarlyStoppingCallback(patience=10, metric='val_recon'),

    # Dead neuron resampling with custom strategy
    DeadNeuronCallback(
        threshold=1e-6,
        resample_every=10000,
        resample_strategy='high_loss_examples'  # Resample from high-loss examples
    ),

    # Custom callback
    class MyCustomCallback:
        def on_batch_end(self, step, loss, metrics):
            if step % 100 == 0:
                print(f"Step {step}: Custom metric = {metrics.get('hoyer', 0):.4f}")

        def on_epoch_end(self, epoch, metrics):
            # Save feature visualizations
            pass
]

# Train with full control
trainer.train(
    sae=sae,
    loss_fn=custom_loss,
    optimizer=optimizer,
    dataset=activation_dataset,
    preprocess_fn=preprocess,
    callbacks=callbacks,
    num_steps=100000,
    batch_size=256,
    gradient_accumulation_steps=4,  # Simulate larger batch
    mixed_precision=True,            # Use float16 for speed
    gradient_checkpointing=False,
    validation_dataset=val_dataset,
    validate_every=1000,
    log_every=100,
    profile=False,                   # MLX profiling
    seed=42
)
```

### Profi Mode: Custom Activation Functions

```python
from mlxterp.sae.activations import BaseActivation

class TopKJumprelu(BaseActivation):
    """
    JumpReLU with TopK: ReLU + learned threshold + TopK

    Combines benefits of:
    - TopK: enforced sparsity
    - JumpReLU: learned per-neuron thresholds
    """

    def __init__(self, k: int = 100, init_threshold: float = 0.1):
        super().__init__()
        self.k = k
        self.threshold = mx.array(init_threshold)  # Learnable

    def __call__(self, x):
        # Apply learned threshold
        x_thresh = mx.maximum(x - self.threshold, 0.0)

        # TopK selection
        topk_vals, topk_indices = mx.topk(x_thresh, k=self.k, axis=-1)

        # Create sparse output
        output = mx.zeros_like(x)
        batch_indices = mx.arange(x.shape[0])[:, None]
        output[batch_indices, topk_indices] = topk_vals

        return output

    def parameters(self):
        return {'threshold': self.threshold}

# Use in SAE
sae = CustomSAE(
    d_model=2048,
    d_hidden=32768,
    activation=TopKJumprelu(k=100)
)
```

### Profi Mode: Custom Weight Initialization

```python
class CustomInitializer:
    """Smart initialization for SAE weights"""

    @staticmethod
    def initialize_decoder_from_pca(decoder, activation_dataset):
        """
        Initialize decoder with PCA components

        This gives SAE a good starting point by initializing
        with principal components of the data.
        """
        # Compute PCA
        data = mx.concatenate([batch for batch in activation_dataset], axis=0)
        data_centered = data - mx.mean(data, axis=0)

        # SVD to get principal components
        U, S, Vt = mx.linalg.svd(data_centered, full_matrices=False)

        # Initialize decoder with top components
        d_hidden = decoder.weight.shape[0]
        decoder.weight[:, :] = Vt[:d_hidden, :]

        print(f"Initialized decoder with top {d_hidden} principal components")

    @staticmethod
    def initialize_encoder_as_transpose(encoder, decoder):
        """Initialize encoder as transpose of decoder (tied initialization)"""
        encoder.weight[:, :] = decoder.weight.T
        print("Initialized encoder as decoder transpose")

# Use custom initialization
sae = CustomSAE(d_model=2048, d_hidden=32768)
CustomInitializer.initialize_decoder_from_pca(sae.decoder, activation_dataset)
CustomInitializer.initialize_encoder_as_transpose(sae.encoder, sae.decoder)
```

### Profi Mode: Advanced Architectures

```python
class GatedSAE(nn.Module):
    """
    Gated SAE: Separate gating mechanism for feature selection

    Instead of TopK, use a learned gate to select features.
    """

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.encoder = nn.Linear(d_model, d_hidden)
        self.gate = nn.Linear(d_model, d_hidden)  # Separate gate
        self.decoder = nn.Linear(d_hidden, d_model)

    def __call__(self, x):
        # Magnitude
        mag = mx.relu(self.encoder(x))

        # Gate (what to activate)
        gate_logits = self.gate(x)
        gate = mx.sigmoid(gate_logits)

        # Gated features
        z = mag * gate

        # Decode
        x_recon = self.decoder(z)

        return x_recon, z, gate

    def loss(self, x, lambda_gate=1e-3):
        x_recon, z, gate = self(x)

        # Reconstruction
        recon = mx.mean((x - x_recon) ** 2)

        # Gate sparsity (encourage few active gates)
        gate_sparsity = mx.mean(gate)  # Encourage low average

        total = recon + lambda_gate * gate_sparsity

        return total, {
            'recon': recon,
            'gate_sparsity': gate_sparsity,
            'active_features': mx.mean(mx.sum(z > 0.01, axis=-1))
        }

class HierarchicalSAE(nn.Module):
    """
    Two-level SAE: Coarse features + fine features

    Learn both high-level and low-level features.
    """

    def __init__(self, d_model, d_coarse, d_fine):
        super().__init__()
        # Coarse level (low dimensional, captures high-level features)
        self.coarse_encoder = nn.Linear(d_model, d_coarse)
        self.coarse_decoder = nn.Linear(d_coarse, d_model)

        # Fine level (high dimensional, captures details)
        self.fine_encoder = nn.Linear(d_model, d_fine)
        self.fine_decoder = nn.Linear(d_fine, d_model)

    def __call__(self, x):
        # Coarse encoding
        z_coarse = mx.relu(self.coarse_encoder(x))
        x_coarse_recon = self.coarse_decoder(z_coarse)

        # Fine encoding (on residual)
        residual = x - x_coarse_recon
        z_fine = topk_activation(self.fine_encoder(residual), k=100)
        x_fine_recon = self.fine_decoder(z_fine)

        # Combined reconstruction
        x_recon = x_coarse_recon + x_fine_recon

        return x_recon, z_coarse, z_fine
```

### Profi Mode: Experiment Tracking

```python
from mlxterp.sae import Experiment

# Set up experiment
exp = Experiment(
    name="sae-ablation-study",
    track_with=["wandb", "tensorboard", "local"],  # Multiple backends
    config={
        "model": "llama-3.2-1b",
        "layer": 10,
        "expansion_factor": 16,
        "k": 100,
        "hypothesis": "TopK better than L1 for interpretability"
    }
)

# Run multiple trials
for k in [50, 100, 150, 200]:
    exp.start_run(name=f"topk_{k}")

    sae = CustomSAE(d_model=2048, d_hidden=32768, k=k)

    trainer.train(
        sae=sae,
        callbacks=[exp.get_callback()],  # Auto-logs to experiment
        num_steps=50000
    )

    # Evaluate
    metrics = evaluate_sae(sae, test_dataset)
    exp.log_metrics(metrics)

    # Log feature visualizations
    for feature_id in range(10):
        fig = sae.visualize_feature(feature_id)
        exp.log_figure(f"feature_{feature_id}", fig)

    exp.end_run()

# Compare results
exp.compare_runs(metric='recon_loss')
```

---

## Part 2: Circuit Discovery

### What is a Circuit?

A **circuit** in a neural network is a computational subgraph:
- **Nodes**: Features (from SAEs at different layers)
- **Edges**: Causal connections between features

**Example**: "Indirect Object Identification" circuit in GPT-2
```
Layer 5: "Previous token" feature
    ↓
Layer 6: "Subject-verb agreement" feature
    ↓
Layer 7: "Indirect object" feature
    ↓
Output: Predicts the indirect object
```

### How to Identify Circuits

There are several methods:

---

### Method 1: Activation Patching Between Features

**Idea**: If patching feature A into feature B's layer changes B's activation, then A→B is a circuit edge.

```python
from mlxterp.sae import circuit_discovery

# Train SAEs on multiple layers
saes = {
    5: model.train_sae(layer=5, dataset=texts),
    6: model.train_sae(layer=6, dataset=texts),
    7: model.train_sae(layer=7, dataset=texts)
}

# Discover circuits
circuit = circuit_discovery.activation_patching_method(
    model=model,
    saes=saes,
    dataset=test_texts,
    method="mean_ablation"  # or "resample", "zero"
)

# Algorithm:
# For each feature f_i in layer L:
#   1. Get baseline activation of feature f_j in layer L+1
#   2. Ablate feature f_i (set to 0)
#   3. Measure change in f_j activation
#   4. If change > threshold: f_i → f_j is an edge

print(circuit)
# Output:
# Circuit with 234 edges:
#   Feature 42 (L5) → Feature 91 (L6): strength 0.85
#   Feature 91 (L6) → Feature 123 (L7): strength 0.72
#   ...
```

**Implementation**:

```python
def find_circuit_edges_activation_patching(
    model,
    sae_source,
    sae_target,
    source_layer,
    target_layer,
    dataset,
    threshold=0.1
):
    """
    Find edges from source_layer features to target_layer features
    using activation patching.
    """
    edges = []

    for text in dataset:
        # Get baseline activations
        with model.trace(text) as trace:
            source_act = trace.activations[f"model.model.layers.{source_layer}.mlp"]
            target_act = trace.activations[f"model.model.layers.{target_layer}.mlp"]

        # Encode to features
        source_features = sae_source.encode(source_act)
        baseline_target_features = sae_target.encode(target_act)

        # For each source feature
        for source_idx in range(source_features.shape[-1]):
            # Ablate source feature
            ablated_source_features = source_features.copy()
            ablated_source_features[:, :, source_idx] = 0

            # Decode back to activation space
            ablated_source_act = sae_source.decode(ablated_source_features)

            # Patch into model and get target
            with model.trace(text, interventions={
                f"layers.{source_layer}.mlp": iv.replace_with(ablated_source_act)
            }) as trace:
                ablated_target_act = trace.activations[f"model.model.layers.{target_layer}.mlp"]

            # Encode to target features
            ablated_target_features = sae_target.encode(ablated_target_act)

            # Measure change for each target feature
            delta = baseline_target_features - ablated_target_features

            # Find significant changes
            for target_idx in range(delta.shape[-1]):
                change = float(mx.mean(mx.abs(delta[:, :, target_idx])))

                if change > threshold:
                    edges.append({
                        'source': (source_layer, source_idx),
                        'target': (target_layer, target_idx),
                        'strength': change
                    })

    return edges
```

---

### Method 2: Attribution Patching

**Idea**: Use gradients to measure how much each source feature contributes to each target feature.

```python
def find_circuit_edges_attribution(
    model,
    sae_source,
    sae_target,
    source_layer,
    target_layer,
    dataset
):
    """
    Use gradients to find which source features affect target features.
    """
    edges = []

    for text in dataset:
        # Forward pass with gradient tracking
        with model.trace(text) as trace:
            source_act = trace.activations[f"model.model.layers.{source_layer}.mlp"]
            target_act = trace.activations[f"model.model.layers.{target_layer}.mlp"]

        # Encode
        source_features = sae_source.encode(source_act)
        target_features = sae_target.encode(target_act)

        # For each target feature
        for target_idx in range(target_features.shape[-1]):
            # Gradient of target feature w.r.t. source features
            grad_fn = lambda src: sae_target.encode(
                model_forward(sae_source.decode(src))
            )[:, :, target_idx]

            grads = mx.grad(grad_fn)(source_features)

            # Find source features with high gradients
            mean_grads = mx.mean(mx.abs(grads), axis=(0, 1))  # Average over batch, seq

            for source_idx in range(len(mean_grads)):
                if mean_grads[source_idx] > threshold:
                    edges.append({
                        'source': (source_layer, source_idx),
                        'target': (target_layer, target_idx),
                        'strength': float(mean_grads[source_idx])
                    })

    return edges
```

---

### Method 3: Feature Co-Activation

**Idea**: Features that activate together are likely connected.

```python
def find_circuit_edges_coactivation(
    model,
    sae_source,
    sae_target,
    source_layer,
    target_layer,
    dataset
):
    """
    Find features that co-activate (correlate).
    """
    # Collect feature activations
    source_activations = []
    target_activations = []

    for text in dataset:
        with model.trace(text) as trace:
            source_act = trace.activations[f"model.model.layers.{source_layer}.mlp"]
            target_act = trace.activations[f"model.model.layers.{target_layer}.mlp"]

        source_features = sae_source.encode(source_act)
        target_features = sae_target.encode(target_act)

        source_activations.append(source_features[0, -1])  # Last token
        target_activations.append(target_features[0, -1])

    # Stack into matrices
    source_matrix = mx.stack(source_activations, axis=0)  # (dataset_size, d_source)
    target_matrix = mx.stack(target_activations, axis=0)  # (dataset_size, d_target)

    # Compute correlation matrix
    correlation = mx.corrcoef(source_matrix.T, target_matrix.T)

    # Extract source-target correlations
    d_source = source_matrix.shape[-1]
    source_target_corr = correlation[:d_source, d_source:]

    # Find high correlations
    edges = []
    for source_idx in range(d_source):
        for target_idx in range(target_matrix.shape[-1]):
            corr = source_target_corr[source_idx, target_idx]

            if abs(corr) > threshold:
                edges.append({
                    'source': (source_layer, source_idx),
                    'target': (target_layer, target_idx),
                    'strength': float(abs(corr))
                })

    return edges
```

---

### Method 4: Path Patching (Most Rigorous)

**Idea**: Test if a specific path through features is actually used.

```python
def verify_circuit_path(
    model,
    saes,  # Dict: {layer: sae}
    path,  # List of (layer, feature_idx)
    dataset,
    task_metric  # Function to measure task performance
):
    """
    Verify that a hypothesized path is actually a circuit.

    Path example: [(5, 42), (6, 91), (7, 123)]
    Means: Feature 42@L5 → Feature 91@L6 → Feature 123@L7
    """

    # Baseline performance
    baseline_performance = []
    for text in dataset:
        with model.trace(text):
            output = model.output.save()
        baseline_performance.append(task_metric(output))

    baseline = mx.mean(mx.array(baseline_performance))

    # Ablate the path
    interventions = {}

    for layer, feature_idx in path:
        sae = saes[layer]

        def ablate_feature(activation):
            features = sae.encode(activation)
            features[:, :, feature_idx] = 0  # Ablate this feature
            return sae.decode(features)

        interventions[f"layers.{layer}.mlp"] = ablate_feature

    # Measure performance with ablated path
    ablated_performance = []
    for text in dataset:
        with model.trace(text, interventions=interventions):
            output = model.output.save()
        ablated_performance.append(task_metric(output))

    ablated = mx.mean(mx.array(ablated_performance))

    # If performance drops significantly, path is important
    importance = (baseline - ablated) / baseline

    return {
        'path': path,
        'baseline': float(baseline),
        'ablated': float(ablated),
        'importance': float(importance),
        'is_circuit': importance > 0.1  # Threshold
    }
```

---

### User-Friendly Circuit Discovery API

```python
from mlxterp.sae import CircuitFinder

# Train SAEs on multiple layers
saes = {
    layer: model.train_sae(layer=layer, dataset=texts)
    for layer in [5, 6, 7, 8, 9, 10]
}

# Automatic circuit discovery
finder = CircuitFinder(model=model, saes=saes)

# Find circuits for a specific task
circuit = finder.discover(
    dataset=task_dataset,
    method="hybrid",  # Combine multiple methods
    task="ioi",       # Indirect object identification
    min_edge_strength=0.1,
    max_edges=500,
    prune_weak_edges=True
)

# Visualize
circuit.visualize(
    layout="hierarchical",  # or "force-directed", "circular"
    show_feature_labels=True,
    highlight_path=[(5, 42), (6, 91), (7, 123)],  # Highlight specific path
    save="circuit_ioi.png"
)

# Interactive exploration
circuit.explore_interactive(port=8080)
# Opens browser with interactive graph
# - Click nodes to see feature examples
# - Click edges to see activation flow
# - Filter by edge strength
```

---

### Advanced: Automatic Circuit Interpretation

```python
# Automatically interpret discovered circuits
interpreted = circuit.auto_interpret(
    model=model,
    dataset=texts,
    labeler="claude-3-5-sonnet",  # Use LLM to label features
    num_examples=20
)

print(interpreted)
# Output:
# Circuit: "Indirect Object Identification"
#
# Layer 5, Feature 42: "Previous token (subject)"
#   Top examples: "When John and Mary went", "After Alice called"
#   ↓ (strength: 0.85)
#
# Layer 6, Feature 91: "Subject-verb agreement"
#   Top examples: "went to", "called to"
#   ↓ (strength: 0.72)
#
# Layer 7, Feature 123: "Indirect object (person)"
#   Top examples: "to Mary", "to Bob"
#   ↓
# Output: Predicts indirect object name
```

---

### Circuit Analysis Tools

```python
# 1. Circuit statistics
print(f"Nodes: {circuit.num_nodes}")
print(f"Edges: {circuit.num_edges}")
print(f"Density: {circuit.density:.2%}")
print(f"Longest path: {circuit.longest_path_length}")

# 2. Find bottleneck features
bottlenecks = circuit.find_bottlenecks(top_k=10)
print("Bottleneck features (most connected):")
for feature, connections in bottlenecks:
    print(f"  {feature}: {connections} connections")

# 3. Compare circuits for different tasks
circuit_ioi = finder.discover(dataset=ioi_dataset, task="ioi")
circuit_factual = finder.discover(dataset=factual_dataset, task="factual")

shared_features = circuit_ioi.shared_features(circuit_factual)
print(f"Shared features: {len(shared_features)}")

# 4. Circuit ablation study
ablation_results = circuit.ablation_study(
    model=model,
    dataset=test_dataset,
    ablate_fractions=[0.1, 0.2, 0.5, 0.8, 1.0],
    metric=task_accuracy
)

# Shows how performance degrades as circuit is ablated
ablation_results.plot()
```

---

## Profi Mode Summary

The "Profi Mode" gives researchers:

1. **Full control** over architecture, loss, optimization
2. **Custom callbacks** for experiment tracking
3. **Advanced architectures** (Gated, Hierarchical, etc.)
4. **Circuit discovery** with multiple methods
5. **Automatic interpretation** of circuits

But still maintains simplicity for beginners:
```python
# Beginner
sae = model.train_sae(layer=10, dataset=texts)

# Profi
trainer = ProfiSAETrainer(...)
trainer.train(sae=CustomSAE(), loss_fn=custom_loss, ...)
```

Both APIs coexist peacefully!
