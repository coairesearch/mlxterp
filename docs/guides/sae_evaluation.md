# SAE Evaluation and Feature Analysis

This guide shows how to evaluate trained SAEs and analyze the features they have learned.

## Overview

After training an SAE, you need to:
1. **Evaluate quality** - How well does it reconstruct activations?
2. **Analyze features** - What concepts do the features represent?
3. **Validate utility** - Is it useful for interpretability research?

---

## Quick Start

### Evaluate Your SAE

Run the comprehensive evaluation script:

```bash
python examples/evaluate_sae.py
```

This produces:
- Reconstruction quality metrics
- Sparsity analysis
- Feature interpretability assessment
- Overall quality score (0-5)
- Detailed JSON report

### Analyze Features

Explore what your SAE has learned:

```bash
python examples/sae_feature_analysis.py
```

This demonstrates:
- Finding top features for any text
- Finding top texts for any feature
- Understanding feature representations

---

## Evaluation Metrics

### 1. Reconstruction Quality (0-2 points)

**Measures**: How well the SAE can reconstruct the original activations

**Metrics:**
- **MSE (Mean Squared Error)**: Lower is better
  - Excellent: < 0.01
  - Good: < 0.05
  - Fair: < 0.10
  - Poor: > 0.10

- **Cosine Similarity**: Higher is better
  - Excellent: > 0.95 ✅
  - Good: > 0.90 ✅
  - Fair: > 0.85 ⚠️
  - Poor: < 0.85 ❌

- **Explained Variance**: Percentage of variance captured
  - Excellent: > 90%
  - Good: > 80%
  - Fair: > 70%
  - Poor: < 70%

**Interpretation:**
```python
# Good reconstruction means:
# - SAE captures most important information
# - Features represent meaningful patterns
# - Safe to use for interpretability

# Poor reconstruction means:
# - Information loss is too high
# - May miss important features
# - Consider retraining with larger expansion factor
```

### 2. Sparsity Metrics (0-2 points)

**Measures**: How efficiently the SAE uses its features

**Metrics:**
- **L0 (Active Features)**: Average features active per sample
  - Should match target `k` value
  - Example: If k=128, expect L0 ≈ 128

- **Dead Features**: Features that never activate
  - Excellent: < 30% ✅
  - Good: < 50% ✅
  - Fair: < 70% ⚠️
  - Poor: > 70% ❌

**With ghost gradients**: Expect ~60-75% dead features (improvement from 95% without)

**Dead Feature Timeline:**
```
Training Steps    Dead Features    Why
─────────────────────────────────────────────────────
0-1000           95-99%           Ghost grads not active yet
1000-5000        85-95%           Ghost grads starting to work
5000-10000       70-85%           Ghost grads fully active
10000+           60-75%           Equilibrium reached
```

**Interpretation:**
```python
# Dead features are normal and expected
# - Caused by feature specialization
# - Batch-level vs sample-level activation mismatch
# - Some features only activate on rare patterns

# Target: 60-75% dead with ghost gradients
# - This is considered good performance
# - Matches SAELens benchmarks
# - Better than 95%+ without ghost grads
```

### 3. Feature Interpretability (0-1 point)

**Measures**: Can humans understand what features represent?

**Metrics:**
- **Feature Diversity**: Number of interpretable features found
  - Good: ≥ 8 distinct features
  - Fair: ≥ 5 distinct features
  - Poor: < 5 distinct features

**Example:**
```python
# Good interpretability:
Feature 1234: Activates on mathematical equations
  - "2 + 2 = 4"
  - "E = mc²"
  - "ax² + bx + c = 0"

Feature 5678: Activates on geographic locations
  - "Paris, France"
  - "Tokyo, Japan"
  - "New York City"

# Poor interpretability:
Feature 9999: No clear pattern
  - Random mix of unrelated texts
  - No semantic coherence
```

### Overall Quality Score

**Formula**: `score = reconstruction (0-2) + sparsity (0-2) + interpretability (0-1)`

**Rating:**
- **4.5-5.0**: Excellent - Ready for production research ✅
- **3.5-4.5**: Good - Suitable for interpretability work ✅
- **2.5-3.5**: Fair - Consider retraining with different hyperparameters ⚠️
- **< 2.5**: Poor - Retraining strongly recommended ❌

---

## Feature Analysis

### Finding Top Features for Text

**Goal**: Discover which features activate when processing specific input

```python
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp.sae import BatchTopKSAE
from examples.sae_feature_analysis import get_top_activating_features

# Load SAE and model
sae = BatchTopKSAE.load("my_sae.mlx")
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
interp = InterpretableModel(model, tokenizer=tokenizer)

# Analyze a text
text = "Machine learning models learn patterns from data"
top_features = get_top_activating_features(
    sae, interp, text,
    layer=12, component="mlp",
    top_k=10
)

print(f"Top features for: '{text}'")
for feat_id, activation in top_features:
    print(f"  Feature {feat_id:6d}: {activation:.4f}")
```

**Output:**
```
Top features for: 'Machine learning models learn patterns from data'
  Feature   4251: 0.8523
  Feature  12089: 0.7341
  Feature   8765: 0.6892
  ...
```

**Use cases:**
- Understanding model's internal representation of concepts
- Discovering unexpected feature activations
- Validating feature interpretability
- Finding relevant features for interventions

### Finding Top Texts for Feature

**Goal**: Understand what a specific feature represents

```python
from examples.sae_feature_analysis import get_top_activating_texts

# Pick a feature to investigate
feature_id = 4251

# Find texts that activate it strongly
examples = get_top_activating_texts(
    sae, interp,
    feature_id=feature_id,
    texts=dataset_texts,  # Your dataset
    layer=12, component="mlp",
    top_k=20
)

print(f"Feature {feature_id} activates most on:")
for text, activation, pos in examples[:5]:
    print(f"  [{activation:.4f}] {text[:100]}...")
```

**Output:**
```
Feature 4251 activates most on:
  [0.9234] Machine learning algorithms can identify patterns...
  [0.8876] Neural networks learn hierarchical representations...
  [0.8654] Deep learning has revolutionized AI research...
  [0.8432] Supervised learning requires labeled training data...
  [0.8210] Artificial intelligence systems use statistical models...
```

**Interpretation**: Feature 4251 appears to represent "machine learning/AI concepts"

**Workflow:**
1. Find top activating texts
2. Read and analyze the examples
3. Look for common patterns/themes
4. Form hypothesis about feature meaning
5. Validate with additional examples
6. Test with feature ablation/amplification

### Feature Interpretation Example

**Complete analysis of a feature:**

```python
# Step 1: Find a feature of interest
text = "The quick brown fox jumps over the lazy dog"
top_features = get_top_activating_features(
    sae, interp, text, layer=12, component="mlp", top_k=5
)

# Step 2: Pick the top feature
feature_id = top_features[0][0]
print(f"Analyzing feature {feature_id}...")

# Step 3: Find many examples
examples = get_top_activating_texts(
    sae, interp, feature_id,
    texts=large_dataset,  # Use large dataset
    layer=12, component="mlp",
    top_k=50  # Get many examples
)

# Step 4: Analyze patterns
print(f"\nTop 10 activating texts:")
for i, (text, activation, pos) in enumerate(examples[:10], 1):
    print(f"{i:2d}. [{activation:.4f}] {text[:150]}...")

# Step 5: Form hypothesis
# Look for:
# - Common words or phrases
# - Shared semantic themes
# - Syntactic patterns
# - Domain-specific vocabulary

# Step 6: Give feature a name
# Example: "animal-related" or "physical movement" or "common phrases"
```

---

## Troubleshooting

### Poor Reconstruction Quality

**Symptoms:**
- Cosine similarity < 0.85
- High MSE (> 0.10)
- Low explained variance (< 70%)

**Possible causes:**
1. Expansion factor too small
2. k (sparsity) too low
3. Not enough training epochs
4. Poor training data quality

**Solutions:**
```python
# Increase expansion factor
config = SAEConfig(
    expansion_factor=64,  # Was 32
    k=256,  # Proportionally increase k
)

# Or decrease sparsity (keep more features active)
config = SAEConfig(
    expansion_factor=32,
    k=200,  # Was 128
)

# Or train longer
config = SAEConfig(
    num_epochs=10,  # Was 5
)
```

### Too Many Dead Features

**Symptoms:**
- Dead features > 80%
- Most features never activate

**Possible causes:**
1. Dataset not diverse enough
2. Training stopped before ghost grads activated
3. Expansion factor too high
4. k (sparsity) too low

**Solutions:**
```python
# Use more diverse dataset
texts = load_diverse_dataset(size=50000)

# Enable ghost grads with earlier activation
config = SAEConfig(
    use_ghost_grads=True,
    dead_feature_window=500,  # Start early (was 1000)
    sparsity_warm_up_steps=2000,
)

# Or reduce expansion factor
config = SAEConfig(
    expansion_factor=16,  # Was 32
    k=64,  # Proportionally reduce k
)
```

### Features Not Interpretable

**Symptoms:**
- Top activating texts seem random
- No clear semantic patterns
- Features activate on unrelated concepts

**Possible causes:**
1. SAE not well-trained
2. Looking at wrong layer/component
3. Dataset too homogeneous
4. Need more examples to see pattern

**Solutions:**
```python
# Get more examples (patterns may emerge)
examples = get_top_activating_texts(
    sae, interp, feature_id,
    texts=dataset,
    top_k=100,  # Was 20
)

# Try different layers
for layer in [6, 12, 18, 24]:
    sae_layer = BatchTopKSAE.load(f"sae_layer{layer}.mlx")
    # Analyze features...

# Use more diverse dataset
diverse_texts = load_multiple_sources([
    "wikipedia", "books", "news", "code", "conversations"
])
```

---

## Best Practices

### 1. Evaluation Workflow

```python
# Step 1: Quick evaluation
python examples/evaluate_sae.py

# Check quality score:
# - If score ≥ 3.5: Proceed to feature analysis
# - If score < 3.5: Consider retraining

# Step 2: Feature analysis (if quality is good)
python examples/sae_feature_analysis.py

# Step 3: Detailed investigation of interesting features
# (Use the functions in sae_feature_analysis.py)

# Step 4: Document findings
# - Which features are interpretable?
# - What concepts do they represent?
# - Are they useful for your research question?
```

### 2. Feature Naming Convention

When you identify interpretable features, document them:

```python
# Create a feature registry
feature_registry = {
    1234: {
        "name": "mathematical_equations",
        "description": "Activates on mathematical formulas and equations",
        "examples": ["2+2=4", "E=mc²", "ax²+bx+c=0"],
        "confidence": "high"
    },
    5678: {
        "name": "geographic_locations",
        "description": "Activates on city and country names",
        "examples": ["Paris, France", "Tokyo, Japan"],
        "confidence": "high"
    },
    9999: {
        "name": "unknown_pattern",
        "description": "No clear pattern identified yet",
        "examples": [...],
        "confidence": "low"
    }
}

# Save registry
import json
with open("sae_features.json", "w") as f:
    json.dump(feature_registry, f, indent=2)
```

### 3. Iterative Analysis

Feature interpretation is iterative:

```
1. Get initial examples (top 10-20)
2. Form hypothesis about feature meaning
3. Get more examples (top 50-100) to validate
4. Refine hypothesis based on all examples
5. Test hypothesis with ablation/steering
6. Document findings
```

### 4. Multiple Perspectives

Analyze features from different angles:

```python
# Perspective 1: What activates the feature?
examples = get_top_activating_texts(...)

# Perspective 2: When does it activate in context?
# (Look at token positions in examples)

# Perspective 3: What happens when you remove it?
# (Feature ablation - Phase 2.5)

# Perspective 4: What happens when you amplify it?
# (Feature steering - Phase 2.5)
```

---

## Example Analysis Session

Here's a complete example of analyzing an SAE:

```python
from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp.sae import BatchTopKSAE
from examples.sae_feature_analysis import (
    get_top_activating_features,
    get_top_activating_texts
)
from datasets import load_dataset

# Setup
sae = BatchTopKSAE.load("sae_layer12_mlp.mlx")
model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
interp = InterpretableModel(model, tokenizer=tokenizer)

# Load test dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [item["text"] for item in dataset if len(item["text"]) > 50][:1000]

print("="*80)
print("SAE FEATURE ANALYSIS SESSION")
print("="*80)

# Analysis 1: Pick interesting texts
test_texts = [
    "Python is a programming language used for data science",
    "The Eiffel Tower is located in Paris, France",
    "Photosynthesis converts sunlight into chemical energy",
]

for text in test_texts:
    print(f"\nText: '{text}'")

    # Find top features
    features = get_top_activating_features(
        sae, interp, text,
        layer=12, component="mlp",
        top_k=3
    )

    print("Top 3 features:")
    for feat_id, activation in features:
        print(f"  Feature {feat_id}: {activation:.4f}")

        # Get examples for this feature
        examples = get_top_activating_texts(
            sae, interp, feat_id,
            texts=texts[:200],  # Search subset for speed
            layer=12, component="mlp",
            top_k=3
        )

        print(f"    Top examples:")
        for ex_text, ex_act, pos in examples:
            print(f"      [{ex_act:.4f}] {ex_text[:60]}...")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
```

---

## Next Steps

### Immediate Actions

1. **Run evaluation** on your trained SAE:
   ```bash
   python examples/evaluate_sae.py
   ```

2. **Analyze features** if quality is good:
   ```bash
   python examples/sae_feature_analysis.py
   ```

3. **Document findings** in a feature registry

### Future Capabilities (Phase 2.5)

- **Feature steering**: Ablate or amplify features during generation
- **Visualization dashboard**: Interactive feature browser
- **Automated clustering**: Find similar features automatically
- **Circuit discovery**: Map feature interactions across layers

---

## Resources

- [Dictionary Learning Guide](dictionary_learning.md) - How to train SAEs
- [SAE Examples](../../examples/) - Complete code examples
- [SAE Roadmap](../../SAE_ROADMAP.md) - Development status
- [Troubleshooting](../../TROUBLESHOOTING.md) - Common issues and solutions

## References

- **Scaling Monosemanticity** - Anthropic (2024)
- **Sparse Autoencoders Find Highly Interpretable Features** - Cunningham et al. (2023)
- **SAELens** - Production SAE framework
- **Toy Models of Superposition** - Anthropic (2022)
