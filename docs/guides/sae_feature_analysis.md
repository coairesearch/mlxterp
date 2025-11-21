# SAE Feature Analysis (Phase 2)

This guide shows how to analyze SAE features to understand what concepts they represent.

## Overview

After training an SAE, you can use Phase 2 tools to:

1. **Find top features for text** - Which features activate when processing specific inputs
2. **Find top texts for features** - What examples activate a specific feature most strongly
3. **Visualize feature activations** - See activations highlighted in context

---

## Quick Start

### Basic Feature Analysis

```python
from mlxterp import InterpretableModel
from mlx_lm import load

# Load model and SAE
mlx_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
model = InterpretableModel(mlx_model, tokenizer=tokenizer)
sae = model.load_sae("sae_layer10.mlx")

# Example 1: What features activate for this text?
top_features = model.get_top_features_for_text(
    text="Paris is the capital of France",
    sae=sae,
    layer=10,
    component="mlp",
    top_k=10
)

print("Top features:")
for feature_id, activation in top_features:
    print(f"  Feature {feature_id}: {activation:.3f}")

# Example 2: What texts activate feature #1234?
from datasets import load_dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [item["text"] for item in dataset if len(item["text"]) > 50][:1000]

examples = model.get_top_texts_for_feature(
    feature_id=1234,
    sae=sae,
    texts=texts,
    layer=10,
    component="mlp",
    top_k=20
)

print(f"\nFeature 1234 activates most on:")
for text, activation, pos in examples[:5]:
    print(f"  [{activation:.3f}] {text[:100]}...")
```

---

## API Reference

### `get_top_features_for_text()`

Find which SAE features activate most strongly for a given text.

**Signature:**
```python
model.get_top_features_for_text(
    text: str,
    sae: Union[SAE, BatchTopKSAE, str],
    layer: int,
    component: str = "mlp",
    top_k: int = 10
) -> List[Tuple[int, float]]
```

**Parameters:**
- `text` - Input text to analyze
- `sae` - SAE instance or path to saved SAE
- `layer` - Layer number where SAE was trained
- `component` - Component name ("mlp", "attn", etc.)
- `top_k` - Number of top features to return

**Returns:**
- List of `(feature_id, activation_value)` tuples, sorted by activation strength

**Example:**
```python
# Analyze what the model "thinks about" when processing this text
features = model.get_top_features_for_text(
    "The Eiffel Tower is in Paris",
    sae=sae,
    layer=10,
    component="mlp",
    top_k=5
)

# Output:
# Feature 42: 0.856  (might represent "landmarks")
# Feature 128: 0.743  (might represent "France/French")
# Feature 91: 0.621  (might represent "locations")
```

### `get_top_texts_for_feature()`

Find texts where a specific SAE feature activates most strongly.

**Signature:**
```python
model.get_top_texts_for_feature(
    feature_id: int,
    sae: Union[SAE, BatchTopKSAE, str],
    texts: List[str],
    layer: int,
    component: str = "mlp",
    top_k: int = 10
) -> List[Tuple[str, float, int]]
```

**Parameters:**
- `feature_id` - The feature index to analyze
- `sae` - SAE instance or path to saved SAE
- `texts` - Dataset of texts to search through
- `layer` - Layer number where SAE was trained
- `component` - Component name ("mlp", "attn", etc.)
- `top_k` - Number of top examples to return

**Returns:**
- List of `(text, activation_value, token_position)` tuples

**Example:**
```python
# What does feature 1234 represent?
examples = model.get_top_texts_for_feature(
    feature_id=1234,
    sae=sae,
    texts=dataset_texts,
    layer=10,
    component="mlp",
    top_k=20
)

# Examine examples to understand the feature
for text, activation, pos in examples[:5]:
    print(f"[{activation:.3f}] {text}")

# If they all relate to "geography", then feature 1234
# likely represents geographic concepts
```

### `visualize_feature_activations()` *(Coming Soon)*

Visualize feature activations highlighted in text (Neuronpedia-style).

```python
# Future API
model.visualize_feature_activations(
    text="Paris is the capital of France",
    sae=sae,
    layer=10,
    component="mlp",
    feature_ids=[42, 128, 91]
)
```

This will show an interactive visualization with tokens colored by activation strength.

---

## Complete Workflow Example

Here's a complete workflow for understanding what your SAE has learned:

```python
from mlxterp import InterpretableModel
from mlx_lm import load
from datasets import load_dataset

# 1. Setup
print("Loading model and SAE...")
mlx_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
model = InterpretableModel(mlx_model, tokenizer=tokenizer)
sae = model.load_sae("sae_layer10.mlx")

# 2. Load dataset for searching
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [item["text"] for item in dataset if len(item["text"].strip()) > 50][:1000]

# 3. Pick an interesting text to analyze
test_text = "The Eiffel Tower is a famous landmark in Paris, France"

# 4. Find top features
print(f"\nAnalyzing: '{test_text}'")
top_features = model.get_top_features_for_text(
    text=test_text,
    sae=sae,
    layer=10,
    component="mlp",
    top_k=10
)

# 5. For each top feature, understand what it represents
for feature_id, activation in top_features[:3]:  # Analyze top 3
    print(f"\n{'='*80}")
    print(f"Feature {feature_id} (activation: {activation:.3f})")
    print('='*80)

    # Find examples where this feature activates
    examples = model.get_top_texts_for_feature(
        feature_id=feature_id,
        sae=sae,
        texts=texts,
        layer=10,
        component="mlp",
        top_k=10
    )

    print("\nTop activating texts:")
    for i, (text, act, pos) in enumerate(examples[:5], 1):
        clean_text = ' '.join(text.split())
        print(f"  {i}. [{act:.3f}] {clean_text[:100]}...")

    # Form hypothesis about feature meaning
    print("\nHypothesis: (examine examples above)")
    # If examples all relate to landmarks: "This feature represents landmarks"
    # If examples all relate to France: "This feature represents French things"
    # etc.
```

---

## Feature Interpretation Tips

### 1. Look for Patterns

When analyzing top activating texts for a feature, look for:

- **Common words/phrases** - Does "Paris" appear in all examples?
- **Semantic themes** - Are they all about geography? Science? Politics?
- **Syntactic patterns** - Do they share grammatical structure?
- **Domain-specific terms** - Medical terms? Technical jargon?

### 2. Use Multiple Perspectives

Understand a feature by:

1. **Forward analysis**: Text → Features (what activates for this text?)
2. **Backward analysis**: Feature → Texts (what activates this feature?)
3. **Comparative analysis**: How does this feature relate to others?

### 3. Validate Hypotheses

Once you form a hypothesis about a feature:

1. Find more examples (increase `top_k`)
2. Test with custom texts
3. Look for counter-examples
4. Compare to similar features

### 4. Document Findings

Create a feature registry:

```python
feature_registry = {
    1234: {
        "name": "geographic_locations",
        "description": "Activates on city and country names",
        "examples": ["Paris, France", "Tokyo, Japan", "New York City"],
        "confidence": "high",
        "notes": "Especially strong for European locations"
    },
    5678: {
        "name": "mathematical_notation",
        "description": "Activates on equations and formulas",
        "examples": ["E=mc²", "ax²+bx+c=0"],
        "confidence": "medium"
    }
}

import json
with open("sae_features.json", "w") as f:
    json.dump(feature_registry, f, indent=2)
```

---

## Performance Considerations

### Memory Usage

When analyzing many texts:

```python
# Instead of loading all at once
texts = load_all_texts()  # Might use too much memory

# Process in batches
batch_size = 500
for i in range(0, len(all_texts), batch_size):
    batch = all_texts[i:i+batch_size]
    examples = model.get_top_texts_for_feature(
        feature_id=feature_id,
        sae=sae,
        texts=batch,
        layer=10,
        component="mlp"
    )
    # Process examples...
```

### Speed Optimization

For large-scale analysis:

1. **Pre-compute activations** - If analyzing many features on same dataset
2. **Use smaller dataset** - 500-1000 texts often sufficient for understanding
3. **Parallel processing** - Analyze multiple features concurrently
4. **Cache results** - Save feature→text mappings for reuse

---

## Common Use Cases

### Use Case 1: Understanding Model Behavior

**Goal**: Why did the model generate this specific output?

```python
# 1. Get the output
output = model.generate("Paris is")

# 2. Find which features were active
features = model.get_top_features_for_text(
    "Paris is",
    sae=sae,
    layer=10,
    component="mlp"
)

# 3. Understand what those features represent
for feat_id, act in features[:3]:
    examples = model.get_top_texts_for_feature(
        feat_id, sae, texts, layer=10, component="mlp"
    )
    print(f"Feature {feat_id}:", [ex[0][:50] for ex in examples[:3]])
```

### Use Case 2: Feature Discovery

**Goal**: What concepts has the SAE learned?

```python
# Sample random features and understand them
import random
all_features = range(sae.d_hidden)
sample_features = random.sample(all_features, 50)

for feat_id in sample_features:
    examples = model.get_top_texts_for_feature(
        feat_id, sae, texts, layer=10, component="mlp", top_k=5
    )

    if examples:  # If feature activates
        print(f"\nFeature {feat_id}:")
        for text, act, pos in examples:
            print(f"  [{act:.3f}] {text[:80]}")
```

### Use Case 3: Circuit Discovery

**Goal**: Find features that work together

```python
# For a specific behavior, find active features
behavior_text = "The capital of France is Paris"
features_for_behavior = model.get_top_features_for_text(
    behavior_text, sae, layer=10, component="mlp", top_k=20
)

# Hypothesis: These features form a "geographic knowledge" circuit
circuit_features = [f[0] for f in features_for_behavior]

# Validate by checking if they co-activate
for test_text in ["London is in England", "Tokyo is in Japan"]:
    active = model.get_top_features_for_text(
        test_text, sae, layer=10, component="mlp", top_k=20
    )
    active_ids = [f[0] for f in active]
    overlap = set(circuit_features) & set(active_ids)
    print(f"{test_text}: {len(overlap)}/{len(circuit_features)} features overlap")
```

---

## Next Steps

- **Evaluation**: See [SAE Evaluation Guide](sae_evaluation.md) for quality metrics
- **Training**: See [Dictionary Learning Guide](dictionary_learning.md) for training SAEs
- **Examples**: Check `examples/phase2_feature_analysis.py` for complete code
- **API Reference**: See [API Documentation](../API.md) for full details

---

## References

- **Neuronpedia** - Interactive feature visualization platform
- **Anthropic's Scaling Monosemanticity** - Feature interpretation methodology
- **SAELens** - Production SAE framework with analysis tools
- **Toy Models of Superposition** - Understanding feature representations
