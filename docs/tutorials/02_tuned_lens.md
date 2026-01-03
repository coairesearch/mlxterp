# Tutorial 2: The Tuned Lens

**Paper**: [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) by Belrose et al. (NeurIPS 2023)

**GitHub**: [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens)

**Difficulty**: Beginner-Intermediate | **Time**: 3-4 hours

---

## Learning Objectives

By the end of this tutorial, you will:

1. Understand why the raw logit lens is limited
2. Learn how the tuned lens improves upon it
3. Train a tuned lens for your model
4. Compare tuned lens vs. logit lens predictions
5. Interpret the learned translators

---

## Introduction

### Motivation: Limitations of the Logit Lens

In [Tutorial 1](01_logit_lens.md), we learned that the logit lens projects intermediate hidden states through the final layer norm and unembedding matrix. However, this approach has a fundamental limitation:

!!! warning "The Coordinate System Problem"
    The final layer norm and unembedding are trained only on the **final layer's output**. Intermediate layers may use different "coordinate systems" - their representations could be rotated, scaled, or shifted compared to the final layer.

This means the logit lens gives us **noisy, biased** predictions at early layers.

### The Tuned Lens Solution

The tuned lens addresses this by learning a **layer-specific affine transformation** for each layer:

```
Logit Lens:  prediction = unembed(layer_norm(h_layer))
Tuned Lens:  prediction = unembed(layer_norm(W_layer @ h + b_layer))
```

Where `W_layer` and `b_layer` are learned to map each layer's hidden states into the final layer's coordinate system.

### Key Insight

The affine translator for each layer is trained to minimize **KL divergence** between:
- **Prediction**: The tuned lens prediction at layer `i`
- **Target**: The model's final output distribution

This ensures that the tuned lens predictions match what the model "actually thinks" at each layer.

---

## Part 1: Using mlxterp's Built-in Tuned Lens

mlxterp provides a complete tuned lens implementation with three key methods:

### Loading a Pre-trained Tuned Lens

If you have pre-trained tuned lens weights:

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Load pre-trained tuned lens
tuned_lens = model.load_tuned_lens("path/to/tuned_lens")
```

### Training a New Tuned Lens

To train your own tuned lens:

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Prepare training data - needs enough text to exceed max_seq_len when tokenized
# Rule of thumb: total tokens > max_seq_len (e.g., ~500+ words for max_seq_len=512)
texts = [
    "The capital of France is Paris, known for the Eiffel Tower and Louvre Museum.",
    "Machine learning is a subset of artificial intelligence that enables learning.",
    "Water is composed of hydrogen and oxygen atoms, forming H2O molecules.",
    # ... add many more diverse texts (recommend 100+ samples)
]

# Train tuned lens
tuned_lens = model.train_tuned_lens(
    dataset=texts,
    num_steps=250,          # Paper recommends ~250 steps
    learning_rate=1.0,      # SGD with high learning rate
    momentum=0.9,           # Nesterov momentum
    max_seq_len=512,        # Keep lower than total tokens in dataset
    gradient_clip=1.0,      # Gradient clipping
    save_path="my_tuned_lens",  # Saves .npz and .json files
    verbose=True
)
```

!!! warning "Dataset Size Requirement"
    The training requires enough text so that total tokens > `max_seq_len`. With too little data, training will fail.

!!! note "Demo vs Paper Settings"
    The example scripts use reduced settings (50 steps, small dataset) for fast execution. For paper-accurate results:

    **From the paper:**

    - **Optimizer**: SGD with Nesterov momentum
    - **Learning rate**: 1.0
    - **Momentum**: 0.9
    - **Steps**: 250
    - **LR decay**: Linear to 0

    **Pragmatic recommendations** (not from paper):

    - Use 1000+ diverse text samples for robust training
    - Gradient clipping at 1.0 helps stability

### Applying the Tuned Lens

```python
# Apply tuned lens (same API as logit_lens!)
results = model.tuned_lens(
    "The capital of France is",
    tuned_lens,
    layers=[0, 4, 8, 12, 15],
    top_k=3
)

# Print results
for layer_idx in sorted(results.keys()):
    print(f"\nLayer {layer_idx}:")
    for token_id, score, token_str in results[layer_idx][-1][:3]:
        print(f"  '{token_str}': {score:.4f}")
```

---

## Part 2: Understanding the Implementation

### The TunedLens Class

The core of the tuned lens is a set of affine translators:

```python
from mlxterp.tuned_lens import TunedLens

# Create a tuned lens (initialized to identity)
tuned_lens = TunedLens(
    num_layers=16,    # Number of transformer layers
    hidden_dim=2048   # Model hidden dimension
)

# Each translator is a linear layer: W @ h + b
# Initially: W = identity matrix, b = zeros
```

### Identity Initialization

The translators are initialized close to identity:
- Weight matrix `W = I` (identity)
- Bias vector `b = 0`

This means an untrained tuned lens behaves exactly like the logit lens! Training then adjusts the translators to correct for coordinate system mismatches.

### Training Objective

The training minimizes KL divergence:

```python
# For each layer:
translated = translator(hidden_state)  # W @ h + b
normalized = layer_norm(translated)
predicted_logits = unembed(normalized)
predicted_probs = softmax(predicted_logits)

# Target: model's final output
target_probs = softmax(final_output)

# Loss: KL divergence
loss = KL(target_probs || predicted_probs)
```

---

## Part 3: Reproducing Paper Results

### Experiment 1: Tuned Lens vs. Logit Lens Comparison

The paper's Figure 2 shows that tuned lens predictions are more accurate than logit lens, especially in early layers:

```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Train or load tuned lens
tuned_lens = model.load_tuned_lens("my_tuned_lens")  # Or train fresh

# Compare on the same prompt
prompt = "The capital of France is"

# Get both predictions
logit_results = model.logit_lens(prompt, top_k=5)
tuned_results = model.tuned_lens(prompt, tuned_lens, top_k=5)

print("Layer | Logit Lens    | Tuned Lens")
print("-" * 45)

for layer_idx in sorted(logit_results.keys()):
    if layer_idx in tuned_results:
        logit_pred = logit_results[layer_idx][-1][0][2]  # Top prediction
        tuned_pred = tuned_results[layer_idx][-1][0][2]
        match = "=" if logit_pred == tuned_pred else ""
        print(f"{layer_idx:5d} | {logit_pred:12s} | {tuned_pred:12s} {match}")
```

**Expected Result** (with full training): In early layers, the tuned lens often predicts more coherent tokens than the raw logit lens.

!!! note "Demo Caveat"
    With reduced demo settings (50 steps, small dataset), you may not observe clear improvements. The paper's results require 250 steps and diverse training data.

### Experiment 2: Early Layer Improvement

```python
# Focus on early layers where tuned lens helps most
early_layers = [0, 1, 2, 3, 4]

test_prompts = [
    "The Eiffel Tower is located in",
    "Water is made of hydrogen and",
    "Albert Einstein developed the theory of",
]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 50)

    logit_results = model.logit_lens(prompt, layers=early_layers, top_k=1)
    tuned_results = model.tuned_lens(prompt, tuned_lens, layers=early_layers, top_k=1)

    for layer in early_layers:
        logit_pred = logit_results[layer][-1][0][2]
        tuned_pred = tuned_results[layer][-1][0][2]
        print(f"  Layer {layer}: Logit='{logit_pred}' | Tuned='{tuned_pred}'")
```

### Experiment 3: Visualizing Both

```python
# Side-by-side visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Note: Use plot=True to display a visualization directly
# For custom plotting, use plot=False and work with the results

prompt = "The quick brown fox jumps"

logit_results = model.logit_lens(prompt, plot=False)
tuned_results = model.tuned_lens(prompt, tuned_lens, plot=False)

# ... custom visualization code ...
```

---

## Part 4: Training Details from the Paper

### Hyperparameters from the Paper

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | SGD + Nesterov | Stability with high learning rate |
| Learning Rate | 1.0 | High for fast convergence |
| Momentum | 0.9 | Standard Nesterov momentum |
| Steps | 250 | Sufficient for convergence |
| LR Decay | Linear to 0 | Gradual reduction over training |

### Pragmatic Additions (Not from Paper)

| Parameter | Value | Reason |
|-----------|-------|--------|
| Gradient Clip | 1.0 | Helps prevent training instability |
| Dataset Size | 1000+ samples | Ensures robust translator learning |

### Why SGD Instead of Adam?

The paper found that SGD with Nesterov momentum works better than Adam for this task:
- The affine translators are simple (no need for adaptive learning rates)
- High learning rate (1.0) enables fast convergence
- Linear decay ensures stable final convergence

### Training Data

The paper trained on diverse text. For robust translators, use varied sources:
- Wikipedia articles
- News articles
- Code snippets
- Conversational text

More diversity = better generalization.

---

## Part 5: Advanced Usage

### Custom Final Norm Override

If your model has a non-standard final layer norm:

```python
results = model.tuned_lens(
    prompt,
    tuned_lens,
    final_norm=my_custom_norm,  # Override auto-detected norm
)
```

### Skip Normalization

For models without final layer norm:

```python
results = model.tuned_lens(
    prompt,
    tuned_lens,
    skip_norm=True
)
```

### Training with Callbacks

Monitor training progress:

```python
losses = []

def my_callback(step, loss):
    losses.append(loss)
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss:.4f}")

tuned_lens = model.train_tuned_lens(
    dataset=texts,
    num_steps=250,
    callback=my_callback
)

# Plot loss curve
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("KL Divergence Loss")
plt.title("Tuned Lens Training")
plt.show()
```

---

## Part 6: Exercises

### Exercise 1: Train Your Own Tuned Lens

1. Collect 1000+ diverse text samples
2. Train a tuned lens with the default hyperparameters
3. Compare predictions with the raw logit lens
4. Save and reload your trained tuned lens

### Exercise 2: Layer-wise Analysis

Which layers benefit most from tuning?

```python
# Compare average prediction quality across layers
improvements = {}

for layer_idx in range(len(model.layers)):
    # Measure how much tuned lens improves over logit lens
    # ... your analysis code ...
    pass
```

### Exercise 3: Different Model Sizes

Train tuned lenses for different model sizes (1B, 3B, 8B) and compare:
- Training time
- Improvement magnitude
- Layer-wise patterns

---

## Summary

In this tutorial, you learned:

1. **The Coordinate System Problem**: Why the logit lens gives noisy early-layer predictions
2. **The Tuned Lens Solution**: Learned affine translators that correct for layer-specific coordinate systems
3. **Training**: How to train a tuned lens using KL divergence minimization
4. **Usage**: How to apply the tuned lens and compare with logit lens
5. **Paper Results**: The tuned lens significantly improves early-layer predictions

---

## Next Steps

- **Tutorial 3: Causal Tracing (ROME)** - Localize where factual knowledge is stored
- **Tutorial 4: Steering Vectors** - Control model behavior with activation interventions

---

## References

1. Belrose, N., et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). NeurIPS 2023.

2. nostalgebraist (2020). [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). LessWrong.

3. GitHub: [AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens)
