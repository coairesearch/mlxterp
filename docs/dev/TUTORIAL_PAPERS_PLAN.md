# Tutorial Papers Planning Document

This document outlines a plan for creating tutorials that reimplement key mechanistic interpretability papers using mlxterp. The goal is to demonstrate the library's capabilities through concrete, educational examples.

## Library Capabilities Summary

Before selecting papers, here's what mlxterp currently supports:

| Capability | Implementation | Status |
|------------|----------------|--------|
| Activation Tracing | `with model.trace("text") as trace:` | ✅ Full |
| Fine-grained Access | ~196 activations (Q/K/V, MLP, attention, etc.) | ✅ Full |
| Logit Lens | `model.logit_lens()` | ✅ Built-in |
| Activation Patching | `model.activation_patching()` | ✅ Built-in |
| Token Predictions | `model.get_token_predictions()` | ✅ Built-in |
| Steering Vectors | `interventions.add_vector()` | ✅ Full |
| Zero Ablation | `interventions.zero_out` | ✅ Full |
| Scaling | `interventions.scale()` | ✅ Full |
| Activation Replacement | `interventions.replace_with()` | ✅ Full |
| Noise Injection | `interventions.noise()` | ✅ Full |
| SAE Training | `model.train_sae()` | ✅ Full |
| SAE Feature Analysis | `get_top_features_for_text()` | ✅ Full |

---

## Recommended Papers for Tutorials

### Paper 1: The Logit Lens (Beginner) and Tuned Lens

**Original Work:** [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist (2020) 

**Why This Paper:**
- **Simplicity:** Conceptually straightforward - just project intermediate layers through the unembedding matrix
- **Already Implemented:** mlxterp has `model.logit_lens()` built-in, making verification easy
- **Educational Value:** Introduces the key concept that transformers iteratively refine predictions
- **Visual:** Produces intuitive heatmap visualizations

**Core Concept:**
The logit lens reveals that intermediate hidden states, when projected through the output embedding matrix, produce sensible token distributions that progressively refine toward the final prediction.

**mlxterp Implementation:**
```python
from mlxterp import InterpretableModel

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Built-in logit lens with visualization
results = model.logit_lens(
    "The capital of France is",
    layers=[0, 4, 8, 12, 15],
    plot=True
)

# Show how predictions evolve across layers
for layer_idx in [0, 4, 8, 12, 15]:
    top_pred = results[layer_idx][-1][0][2]  # Top token at last position
    print(f"Layer {layer_idx}: '{top_pred}'")
```

**Tutorial Structure:**
1. Introduction to the residual stream concept
2. Manual implementation using `get_token_predictions()`
3. Comparison with built-in `logit_lens()`
4. Visualization and interpretation
5. Exercises: Try different prompts, observe when predictions "crystallize"

**Estimated Complexity:** ⭐ (Beginner)

---

### Paper 1b: The Tuned Lens (Beginner-Intermediate)

**Original Work:** [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112) by Belrose et al. (NeurIPS 2023)

**GitHub:** https://github.com/AlignmentResearch/tuned-lens

**Why This Paper:**
- **Direct Extension:** Natural follow-up to the Logit Lens tutorial
- **Addresses Limitations:** Fixes the "brittleness" of raw logit lens
- **Trainable Component:** Introduces the concept of learned probes for interpretability
- **Practical Improvement:** More accurate layer-wise predictions

**Core Concept:**
The raw logit lens assumes all layers use the same "coordinate system" as the final layer. In reality, representations are rotated, shifted, and stretched between layers. The tuned lens learns a small affine transformation (Wx + b) for each layer that maps hidden states to the final layer's coordinate system before unembedding.

**Key Insight:**
```
Logit Lens:  prediction = unembed(layer_norm(h_layer))           # Assumes same coords
Tuned Lens:  prediction = unembed(layer_norm(W_layer @ h + b))   # Learns correction
```

The affine "translator" for each layer is trained to minimize KL divergence between its prediction and the model's final output distribution.

**mlxterp Implementation:**

```python
import mlx.core as mx
import mlx.nn as nn
from mlxterp import InterpretableModel

class TunedLens(nn.Module):
    """Tuned Lens: learned affine probes for each layer."""

    def __init__(self, num_layers: int, hidden_dim: int):
        super().__init__()
        # One affine translator per layer: h -> W @ h + b
        self.translators = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]

        # Initialize close to identity for stable training
        for translator in self.translators:
            translator.weight = mx.eye(hidden_dim)
            translator.bias = mx.zeros((hidden_dim,))

    def __call__(self, hidden_state: mx.array, layer_idx: int) -> mx.array:
        """Apply the tuned lens translator for a specific layer."""
        return self.translators[layer_idx](hidden_state)


def train_tuned_lens(
    model: InterpretableModel,
    texts: list,
    num_steps: int = 250,
    learning_rate: float = 1.0,
    chunk_size: int = 2048
) -> TunedLens:
    """
    Train tuned lens translators on a dataset.

    Each translator learns to minimize KL divergence between:
    - Its prediction: softmax(unembed(layer_norm(translator(h_layer))))
    - Target: softmax(unembed(layer_norm(h_final)))
    """
    num_layers = len(model.layers)
    hidden_dim = model.model.model.embed_tokens.weight.shape[1]

    tuned_lens = TunedLens(num_layers, hidden_dim)

    # Get final layer norm for projection
    final_norm = model.model.model.norm

    def loss_fn(tuned_lens, hidden_states, final_logits):
        """KL divergence between tuned predictions and final output."""
        total_loss = 0.0

        for layer_idx in range(num_layers):
            h = hidden_states[layer_idx]

            # Apply translator and layer norm
            h_translated = tuned_lens(h, layer_idx)
            h_normed = final_norm(h_translated)

            # Get logits via unembedding
            translated_logits = model.get_token_predictions(
                h_normed[0, -1, :],
                top_k=model.vocab_size,
                return_scores=True
            )

            # Compute KL divergence
            # KL(final || translated) = sum(final * log(final / translated))
            final_probs = mx.softmax(final_logits[0, -1, :])
            translated_probs = mx.softmax(translated_logits)
            kl = mx.sum(final_probs * (mx.log(final_probs + 1e-10) - mx.log(translated_probs + 1e-10)))

            total_loss += kl

        return total_loss / num_layers

    # Training loop with SGD + Nesterov momentum
    optimizer = nn.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)

    for step in range(num_steps):
        text = texts[step % len(texts)]

        with model.trace(text) as trace:
            final_logits = trace.activations['__model_output__']
            hidden_states = [
                trace.activations[f'model.model.layers.{i}']
                for i in range(num_layers)
            ]

        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(tuned_lens, hidden_states, final_logits)
        optimizer.update(tuned_lens, grads)

        if step % 50 == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

    return tuned_lens


def tuned_lens(
    model: InterpretableModel,
    tuned_lens: TunedLens,
    text: str,
    layers: list = None,
    top_k: int = 5
) -> dict:
    """
    Apply tuned lens to get improved intermediate predictions.

    Returns dict mapping layer_idx -> list of (token_id, score, token_str) tuples
    """
    final_norm = model.model.model.norm

    with model.trace(text) as trace:
        pass

    if layers is None:
        layers = list(range(len(model.layers)))

    results = {}
    for layer_idx in layers:
        h = trace.activations[f'model.model.layers.{layer_idx}']

        # Apply tuned lens translator
        h_translated = tuned_lens(h, layer_idx)

        # Apply final layer norm
        h_normed = final_norm(h_translated[0, -1, :])

        # Get predictions
        predictions = model.get_token_predictions(h_normed, top_k=top_k, return_scores=True)
        predictions_with_str = [
            (token_id, score, model.token_to_str(token_id))
            for token_id, score in predictions
        ]
        results[layer_idx] = predictions_with_str

    return results


# Example usage
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Train tuned lens (or load pre-trained)
texts = ["Sample text 1...", "Sample text 2...", ...]  # Training data
tuned_lens = train_tuned_lens(model, texts, num_steps=250)

# Compare raw vs. tuned logit lens
text = "The capital of France is"

raw_results = model.logit_lens(text, layers=[0, 5, 10, 15])
tuned_results = tuned_lens(model, tuned_lens, text, layers=[0, 5, 10, 15])

print("Layer | Raw Logit Lens | Tuned Lens")
print("-" * 45)
for layer in [0, 5, 10, 15]:
    raw_pred = raw_results[layer][-1][0][2]
    tuned_pred = tuned_results[layer][0][2]
    print(f"  {layer:2d}  | {raw_pred:14s} | {tuned_pred}")
```

**Tutorial Structure:**
1. Limitations of the raw logit lens (coordinate system mismatch)
2. The tuned lens solution: learned affine translators
3. Training objective: minimizing KL divergence
4. Implementing the TunedLens class in MLX
5. Training loop with SGD + momentum
6. Comparing raw vs. tuned predictions
7. Applications: malicious input detection, understanding prediction trajectories

**Key Experiments to Reproduce:**
- Figure 2: Comparison of logit lens vs. tuned lens accuracy
- Prediction trajectory visualization
- Layer-wise KL divergence analysis

**Estimated Complexity:** ⭐⭐ (Beginner-Intermediate)

**Note:** This paper requires implementing a new feature in mlxterp. See GitHub Issue #2 for the implementation plan.

---

### Paper 2: Locating and Editing Factual Associations in GPT (Intermediate)

**Original Work:** [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262) by Meng et al. (NeurIPS 2022)

**Project Page:** https://rome.baulab.info/

**Why This Paper:**
- **Foundational:** One of the most cited mech interp papers
- **Clear Methodology:** Causal tracing has a well-defined experimental setup
- **Direct Match:** mlxterp's `activation_patching()` implements the core technique
- **Practical Impact:** Shows where factual knowledge is stored in transformers

**Core Concept:**
Factual associations (e.g., "Eiffel Tower is in Paris") are stored in specific MLP modules at middle layers, specifically when processing the subject's last token. This is discovered through "causal tracing" - corrupting inputs with noise, then patching clean activations back in to see which restore the correct answer.

**mlxterp Implementation:**
```python
from mlxterp import InterpretableModel, interventions as iv
import mlx.core as mx

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Causal tracing: Find which layers store "Paris" for "Eiffel Tower"
# Step 1: Get clean and corrupted baselines
clean_text = "The Eiffel Tower is located in the city of"
corrupted_text = "The Eiffel Tower is located in the city of"  # Will add noise

# Step 2: Use activation patching to identify critical layers
results = model.activation_patching(
    clean_text="The Eiffel Tower is located in the city of",
    corrupted_text="The Louvre Museum is located in the city of",  # Different subject
    component="mlp",
    plot=True
)

# Step 3: Identify the "causal" layers
sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("Most important MLP layers for factual recall:")
for layer, recovery in sorted_layers[:5]:
    print(f"  Layer {layer}: {recovery:.1f}% recovery")
```

**Tutorial Structure:**
1. Background: Where is knowledge stored in LLMs?
2. The causal tracing methodology
3. Implementing clean vs. corrupted runs
4. Using `activation_patching()` for MLP and attention
5. Visualizing the localization of factual associations
6. Discussion: Implications for model editing

**Key Experiments to Reproduce:**
- Figure 2: Causal tracing showing MLP importance at subject's last token
- Comparison of MLP vs. attention contributions
- Layer-wise importance profile

**Estimated Complexity:** ⭐⭐ (Intermediate)

---

### Paper 3: Steering Llama 2 via Contrastive Activation Addition (Intermediate)

**Original Work:** [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681) by Rimsky et al. (ACL 2024)

**Why This Paper:**
- **Practical:** Directly applicable technique for controlling model behavior
- **Simple Core Idea:** Subtract negative from positive activations → steering vector
- **Perfect Match:** mlxterp's `add_vector` intervention is designed for this
- **Immediate Results:** Effects are visible in model outputs

**Core Concept:**
Create a "steering vector" by computing the difference between activations on positive examples (e.g., honest responses) and negative examples (e.g., deceptive responses). Adding this vector during inference steers the model toward the desired behavior.

**mlxterp Implementation:**
```python
from mlxterp import InterpretableModel, interventions as iv
import mlx.core as mx

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Step 1: Collect activations from contrastive examples
positive_prompts = [
    "I think this is great because",
    "I love the way this works since",
    "This makes me happy because",
]

negative_prompts = [
    "I hate this because",
    "This is terrible since",
    "This makes me angry because",
]

# Step 2: Compute mean activations
layer = 10  # Middle layer typically works well
positive_acts = []
negative_acts = []

for prompt in positive_prompts:
    with model.trace(prompt) as trace:
        act = trace.activations[f'model.model.layers.{layer}']
        positive_acts.append(act[0, -1, :])  # Last token

for prompt in negative_prompts:
    with model.trace(prompt) as trace:
        act = trace.activations[f'model.model.layers.{layer}']
        negative_acts.append(act[0, -1, :])

# Step 3: Compute steering vector
positive_mean = mx.mean(mx.stack(positive_acts), axis=0)
negative_mean = mx.mean(mx.stack(negative_acts), axis=0)
steering_vector = positive_mean - negative_mean

# Step 4: Apply steering
test_prompt = "This movie is"

# Without steering
with model.trace(test_prompt) as trace:
    normal_output = trace.activations['__model_output__']

# With positive steering (scale factor for strength)
with model.trace(test_prompt,
                 interventions={f'model.model.layers.{layer}': iv.add_vector(steering_vector * 2.0)}):
    steered_output = model.output.save()

# Compare predictions
print("Normal:", model.get_token_predictions(normal_output[0, -1, :], top_k=5))
print("Steered:", model.get_token_predictions(steered_output[0, -1, :], top_k=5))
```

**Tutorial Structure:**
1. Introduction to representation engineering
2. The contrastive activation addition method
3. Collecting activations from contrastive pairs
4. Computing and normalizing steering vectors
5. Applying steering with different strengths
6. Evaluating behavioral changes
7. Exercises: Create your own steering vectors (honesty, creativity, formality)

**Key Experiments to Reproduce:**
- Sycophancy steering
- Corrigibility/refusal steering
- Multi-layer steering comparison

**Estimated Complexity:** ⭐⭐ (Intermediate)

---

### Paper 4: In-Context Learning and Induction Heads (Intermediate-Advanced)

**Original Work:** [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/) by Olsson et al. (Anthropic, 2022)

**Why This Paper:**
- **Mechanistic:** One of the clearest examples of reverse-engineering a specific capability
- **Foundational:** Induction heads are a fundamental building block in transformers
- **Testable:** Clear algorithm ([A][B]...[A] → [B]) that can be verified
- **Attention Focus:** Demonstrates how to analyze attention patterns

**Core Concept:**
Induction heads are attention heads that implement a simple pattern-completion algorithm: given a sequence [A][B]...[A], they predict [B]. They work through a two-step process involving a "previous token head" and the induction head itself.

**mlxterp Implementation:**
```python
from mlxterp import InterpretableModel
import mlx.core as mx

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Test prompt with repeating pattern
test_prompt = "The cat sat on the mat. The cat"
# Expected: model should predict tokens that followed "The cat" earlier

# Step 1: Trace and get attention patterns
with model.trace(test_prompt) as trace:
    pass

# Get tokens for analysis
tokens = model.encode(test_prompt)
print("Tokens:", [model.token_to_str(t) for t in tokens])

# Step 2: Find induction heads by checking attention patterns
# An induction head attends to positions where the current token appeared before
# and copies what came after

# Analyze attention at different layers
for layer_idx in range(len(model.layers)):
    attn_key = f'model.model.layers.{layer_idx}.self_attn'
    if attn_key in trace.activations:
        attn_output = trace.activations[attn_key]
        print(f"Layer {layer_idx} attention shape: {attn_output.shape}")

# Step 3: Measure induction score
# The induction score measures how much the model's loss decreases
# at repeated tokens (indicating it's using the pattern)
def measure_induction_score(model, text_with_repetition):
    """Measure how well the model predicts repeated sequences."""
    tokens = model.encode(text_with_repetition)

    # Find repeated subsequences
    # Compare loss at first occurrence vs. repeated occurrence
    with model.trace(text_with_repetition) as trace:
        logits = trace.activations['__model_output__']

    # Compute per-token log probabilities
    # Higher at repeated positions = induction behavior
    return logits

# Step 4: Ablation study - knock out potential induction heads
for layer_idx in [4, 5, 6]:  # Common locations for induction heads
    with model.trace(test_prompt,
                     interventions={f'model.model.layers.{layer_idx}.self_attn': iv.zero_out}):
        ablated_output = model.output.save()

    predictions = model.get_token_predictions(ablated_output[0, -1, :], top_k=3)
    print(f"Layer {layer_idx} ablated: {[model.token_to_str(p) for p in predictions]}")
```

**Tutorial Structure:**
1. What are induction heads and why do they matter?
2. The [A][B]...[A] → [B] pattern
3. How induction heads compose with previous token heads
4. Measuring induction behavior in language models
5. Finding induction heads through attention analysis
6. Ablation studies to confirm causal role
7. Connection to in-context learning

**Key Experiments to Reproduce:**
- Induction score computation
- Layer-wise ablation to find induction heads
- Attention pattern visualization showing diagonal offset pattern

**Estimated Complexity:** ⭐⭐⭐ (Intermediate-Advanced)

---

### Paper 5: Towards Monosemanticity - Sparse Autoencoders (Advanced)

**Original Work:** [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/) by Anthropic (2023)

**Why This Paper:**
- **State of the Art:** SAEs are the current frontier of mech interp research
- **Built-in Support:** mlxterp has SAE training and analysis built-in
- **Feature Discovery:** Shows how to find interpretable features in superposition
- **Practical:** Enables feature-level understanding of model behavior

**Core Concept:**
Individual neurons are often polysemantic (responding to multiple unrelated concepts) due to superposition. Sparse autoencoders learn to decompose activations into a larger set of more interpretable "features" that correspond to single concepts.

**mlxterp Implementation:**
```python
from mlxterp import InterpretableModel, SAEConfig
from datasets import load_dataset

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Step 1: Collect training data
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
texts = [item["text"] for item in dataset if len(item["text"]) > 100][:5000]

# Step 2: Configure and train SAE
config = SAEConfig(
    expansion_factor=32,  # 32x overcomplete dictionary
    k=64,                 # Top-k sparsity
    learning_rate=3e-4,
    num_epochs=3
)

sae = model.train_sae(
    layer=10,
    dataset=texts,
    component="mlp",
    config=config,
    save_path="tutorial_sae_layer10.mlx"
)

# Step 3: Analyze learned features
# Find what features activate for specific concepts
test_texts = [
    "Paris is the capital of France",
    "Tokyo is a city in Japan",
    "The president signed the bill",
    "def fibonacci(n): return n if n < 2 else",
]

for text in test_texts:
    top_features = model.get_top_features_for_text(
        text=text,
        sae=sae,
        layer=10,
        component="mlp",
        top_k=5
    )
    print(f"\n'{text[:50]}...'")
    for feat_id, activation in top_features:
        print(f"  Feature {feat_id}: {activation:.3f}")

# Step 4: Interpret a specific feature
# Find what texts maximize activation of a feature
feature_to_analyze = top_features[0][0]
examples = model.get_top_texts_for_feature(
    feature_id=feature_to_analyze,
    sae=sae,
    texts=texts[:1000],  # Search through dataset
    layer=10,
    component="mlp",
    top_k=10
)

print(f"\nTexts that activate feature {feature_to_analyze}:")
for text, activation, pos in examples[:5]:
    print(f"  [{activation:.3f}] {text[:80]}...")
```

**Tutorial Structure:**
1. The superposition hypothesis and why neurons are polysemantic
2. Sparse autoencoders: architecture and training objective
3. Training an SAE on model activations
4. Identifying and interpreting learned features
5. Feature steering: using SAE features for targeted interventions
6. Limitations and future directions

**Key Experiments to Reproduce:**
- Training an SAE on a specific layer
- Feature identification and interpretation
- Finding maximally activating examples for features
- Comparing feature sparsity across different configurations

**Estimated Complexity:** ⭐⭐⭐⭐ (Advanced)

---

## Implementation Priority

Based on educational value and difficulty progression:

| Priority | Paper | Complexity | Estimated Time | Requires New Feature |
|----------|-------|------------|----------------|---------------------|
| 1 | Logit Lens | ⭐ Beginner | 1-2 hours | No |
| 1b | Tuned Lens | ⭐⭐ Beginner-Int | 3-4 hours | Yes (Issue #2) |
| 2 | ROME / Causal Tracing | ⭐⭐ Intermediate | 3-4 hours | No |
| 3 | Steering Vectors (CAA) | ⭐⭐ Intermediate | 2-3 hours | No |
| 4 | Induction Heads | ⭐⭐⭐ Int-Advanced | 4-5 hours | No |
| 5 | Sparse Autoencoders | ⭐⭐⭐⭐ Advanced | 5-6 hours | No |

## Tutorial Format

Each tutorial should include:

1. **Introduction**
   - Paper summary and motivation
   - Prerequisites and background
   - Learning objectives

2. **Conceptual Overview**
   - Key ideas explained simply
   - Diagrams and visualizations
   - Connection to broader mech interp themes

3. **Step-by-Step Implementation**
   - Working code with explanations
   - Expected outputs shown
   - Common pitfalls noted

4. **Experiments**
   - Reproduce key paper results
   - Try variations and extensions
   - Compare with paper figures

5. **Exercises**
   - Hands-on challenges
   - Questions for deeper understanding
   - Suggestions for further exploration

6. **References**
   - Original paper and related work
   - Additional resources
   - mlxterp documentation links

## File Organization

```
examples/tutorials/
├── 01_logit_lens/
│   ├── tutorial.md
│   ├── logit_lens_tutorial.py
│   └── figures/
├── 01b_tuned_lens/
│   ├── tutorial.md
│   ├── tuned_lens_tutorial.py
│   └── figures/
├── 02_causal_tracing/
│   ├── tutorial.md
│   ├── causal_tracing_tutorial.py
│   └── figures/
├── 03_steering_vectors/
│   ├── tutorial.md
│   ├── steering_tutorial.py
│   └── figures/
├── 04_induction_heads/
│   ├── tutorial.md
│   ├── induction_heads_tutorial.py
│   └── figures/
└── 05_sparse_autoencoders/
    ├── tutorial.md
    ├── sae_tutorial.py
    └── figures/
```

## References

### Papers

1. **Logit Lens**
   - nostalgebraist. (2020). [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

1b. **Tuned Lens**
   - Belrose, N., et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). NeurIPS 2023.
   - GitHub: https://github.com/AlignmentResearch/tuned-lens
   - Documentation: https://tuned-lens.readthedocs.io/

2. **ROME / Causal Tracing**
   - Meng, K., et al. (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). NeurIPS 2022.
   - Project page: https://rome.baulab.info/

3. **Contrastive Activation Addition**
   - Rimsky, N., et al. (2024). [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). ACL 2024.

4. **Induction Heads**
   - Olsson, C., et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.

5. **Sparse Autoencoders**
   - Anthropic. (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/).

### Additional Resources

- [Mechanistic Interpretability for AI Safety — A Review](https://arxiv.org/abs/2404.14082)
- [TransformerLens Documentation](https://github.com/TransformerLensOrg/TransformerLens)
- [nnsight Documentation](https://nnsight.net)
- [Anthropic's Transformer Circuits Thread](https://transformer-circuits.pub/)
