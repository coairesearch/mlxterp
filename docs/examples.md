# Examples

This page provides practical examples for using mlxterp.

## Basic Tracing

### Simple Model Tracing

```python
from mlxterp import InterpretableModel
import mlx.core as mx
import mlx.nn as nn

# Create a simple model
class SimpleTransformer(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        self.layers = [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = nn.relu(x)
        return x

# Wrap with InterpretableModel
base_model = SimpleTransformer()
model = InterpretableModel(base_model)

# Create input
input_data = mx.random.normal((1, 10, 64))  # (batch, seq, hidden)

# Trace execution
with model.trace(input_data) as trace:
    layer_0 = model.layers[0].output.save()
    layer_2 = model.layers[2].output.save()

print(f"Layer 0 shape: {layer_0.shape}")
print(f"Layer 2 shape: {layer_2.shape}")
```

### Loading Real Models

```python
from mlxterp import InterpretableModel

# Load from model hub (automatically loads tokenizer)
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Use with text input
with model.trace("The capital of France is") as trace:
    # Access attention outputs
    for i in [3, 6, 9]:
        attn = model.layers[i].attn.output.save()
        print(f"Layer {i} attention shape: {attn.shape}")

    # Get final output
    logits = model.output.save()

print(f"Output shape: {logits.shape}")
```

## Working with Tokenizers

### Encoding and Decoding Text

```python
from mlxterp import InterpretableModel

# Load model with tokenizer
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Encode text to tokens
text = "Hello world"
tokens = model.encode(text)
print(f"Text: '{text}'")
print(f"Tokens: {tokens}")
# Output: Tokens: [128000, 9906, 1917]

# Decode tokens back to text
decoded = model.decode(tokens)
print(f"Decoded: '{decoded}'")
# Output: Decoded: '<|begin_of_text|>Hello world'

# Get vocabulary size
print(f"Vocabulary size: {model.vocab_size}")
# Output: Vocabulary size: 128000
```

### Analyzing Individual Tokens

```python
# Inspect each token
text = "The capital of France is Paris"
tokens = model.encode(text)

print("Token breakdown:")
for i, token_id in enumerate(tokens):
    token_str = model.token_to_str(token_id)
    print(f"  Position {i}: {token_id:6d} -> '{token_str}'")

# Output:
# Position 0: 128000 -> '<|begin_of_text|>'
# Position 1:    791 -> 'The'
# Position 2:   6864 -> ' capital'
# Position 3:    315 -> ' of'
# Position 4:   9822 -> ' France'
# Position 5:    374 -> ' is'
# Position 6:  12366 -> ' Paris'
```

### Token-Position Specific Analysis

```python
# Analyze activation at specific token position
text = "The capital of France is"
tokens = model.encode(text)

with model.trace(text) as trace:
    # Get layer 8 output
    layer_8_output = model.layers[8].output.save()

# Find position of "France" token
for i, token_id in enumerate(tokens):
    if "France" in model.token_to_str(token_id):
        france_pos = i
        print(f"'France' is at position {i}")
        break

# Extract activation at "France" position
france_activation = layer_8_output[0, france_pos, :]
print(f"Activation at 'France': {france_activation.shape}")
```

### Batch Encoding

```python
# Encode multiple texts
texts = ["Hello", "World", "Test"]
token_lists = model.encode_batch(texts)

for text, tokens in zip(texts, token_lists):
    print(f"'{text}' -> {tokens}")

# Output:
# 'Hello' -> [128000, 9906]
# 'World' -> [128000, 10343]
# 'Test' -> [128000, 2323]
```

### Working with Token Arrays

```python
import mlx.core as mx

# Create token array manually
tokens = mx.array([[128000, 9906, 1917]])  # [BOS, "Hello", " world"]

# Trace with token array
with model.trace(tokens) as trace:
    output = model.output.save()

# Decode the tokens
text = model.decode(tokens[0])  # Pass 1D array or list
print(f"Input text: '{text}'")
```

### Custom Token Sequences

```python
# Build custom token sequence
bos_token = 128000
tokens = model.encode("The answer is")

# Add specific continuation tokens
tokens.extend([220, 2983])  # " 42" tokens

# Run model on custom sequence
with model.trace(tokens) as trace:
    output = model.output.save()

# Verify what we created
full_text = model.decode(tokens)
print(f"Custom sequence: '{full_text}'")
```

## Logit Lens and Predictions

### Decoding Hidden States to Tokens

Convert any hidden state to token predictions:

```python
# Get predictions from layer 6
with model.trace("The capital of France is") as trace:
    layer_6 = trace.activations["model.model.layers.6"]

# Get last token's hidden state
last_token_hidden = layer_6[0, -1, :]

# Get top 10 predictions
predictions = model.get_token_predictions(last_token_hidden, top_k=10)

print("Layer 6 predicts:")
for i, token_id in enumerate(predictions, 1):
    token_str = model.token_to_str(token_id)
    print(f"  {i}. '{token_str}'")

# Output:
#   1. ' Paris'
#   2. ' Brussels'
#   3. ' Lyon'
#   ...
```

### Getting Prediction Scores

Include confidence scores with predictions:

```python
with model.trace("Hello, how are") as trace:
    layer_10 = trace.activations["model.model.layers.10"]

hidden = layer_10[0, -1, :]

# Get predictions with scores
predictions = model.get_token_predictions(
    hidden,
    top_k=5,
    return_scores=True
)

print("Top predictions with scores:")
for token_id, score in predictions:
    token_str = model.token_to_str(token_id)
    print(f"  '{token_str}': {score:.2f}")

# Output:
#   ' you': 18.45
#   ' ya': 14.23
#   ' doing': 12.87
#   ...
```

### Logit Lens Analysis

The logit lens technique shows what each layer predicts at each token position in the input sequence. This reveals how the model builds up its understanding layer by layer.

#### Understanding the Technique

```python
# The logit lens analyzes predictions at EVERY position
text = "The capital of France is"
results = model.logit_lens(text)

# Results structure: {layer_idx: [[pos_0_preds], [pos_1_preds], ...]}
# Each layer has predictions for each input position

# Get tokens for reference
tokens = model.encode(text)
token_strs = [model.token_to_str(t) for t in tokens]
print(f"Input tokens: {token_strs}")
```

#### Show Predictions at Last Position

See how predictions for the last token evolve through layers:

```python
text = "The capital of France is"
results = model.logit_lens(text, layers=[0, 5, 10, 15])

print("What each layer predicts for the LAST position:\n")
for layer_idx in [0, 5, 10, 15]:
    # Get prediction at last position
    last_pos_pred = results[layer_idx][-1][0][2]  # Top token string
    print(f"Layer {layer_idx:2d}: '{last_pos_pred}'")

# Output:
# Layer  0: ' the'
# Layer  5: ' a'
# Layer 10: ' Paris'
# Layer 15: ' Paris'
```

#### Show All Positions for a Specific Layer

See what a single layer predicts at each position:

```python
text = "The capital of France"
results = model.logit_lens(text, layers=[10])
tokens = model.encode(text)

print("Layer 10 predictions at each position:\n")
for pos_idx, predictions in enumerate(results[10]):
    input_token = model.token_to_str(tokens[pos_idx])
    pred_token = predictions[0][2]  # Top prediction
    print(f"  Position {pos_idx} ('{input_token}') -> predicts: '{pred_token}'")

# Output:
#   Position 0 ('The') -> predicts: ' capital'
#   Position 1 (' capital') -> predicts: ' of'
#   Position 2 (' of') -> predicts: ' France'
#   Position 3 (' France') -> predicts: ' is'
```

#### Get Multiple Predictions per Position

```python
text = "Machine learning is"
results = model.logit_lens(text, top_k=3, layers=[10, 15])

for layer_idx in [10, 15]:
    print(f"\nLayer {layer_idx} - last position top 3:")
    last_pos_preds = results[layer_idx][-1]  # All predictions at last position

    for i, (token_id, score, token_str) in enumerate(last_pos_preds, 1):
        print(f"  {i}. '{token_str}' (score: {score:.2f})")
```

### Visualizing with Heatmaps

Generate a visual heatmap showing what each layer predicts at each position:

```python
# Create visualization
results = model.logit_lens(
    "The Eiffel Tower is located in the city of",
    plot=True,
    max_display_tokens=15,  # Show last 15 tokens
    figsize=(16, 10)
)

# The heatmap shows:
#   - X-axis: Input token positions (bottom row shows input tokens)
#   - Y-axis: Model layers (Layer 0 to Layer N)
#   - Cell values: Top predicted token at each (layer, position)
#   - Colors: Different predictions shown with different colors
```

**How to read the visualization:**

1. **Bottom row (x-axis labels)**: The actual input tokens
2. **Left column (y-axis labels)**: Layer numbers
3. **Each cell**: What that layer predicts after seeing tokens up to that position
4. **Last column**: Shows how the final prediction evolves through layers
5. **Colors**: Same color = same prediction across different positions/layers

**Customization options:**

```python
# Show all layers
results = model.logit_lens(
    "Machine learning is fascinating",
    plot=True,
    layers=list(range(16)),  # All 16 layers
    figsize=(18, 12),
    cmap='viridis'  # Try: 'plasma', 'inferno', 'cividis', etc.
)

# Focus on specific layers
results = model.logit_lens(
    "The quick brown fox",
    plot=True,
    layers=[0, 4, 8, 12, 15],  # Early, middle, late layers
    max_display_tokens=10,  # Show last 10 tokens only
    figsize=(14, 8)
)

# Handle long sequences
results = model.logit_lens(
    "Very long input text that has many tokens...",
    plot=True,
    max_display_tokens=20,  # Automatically shows last 20 tokens
    figsize=(20, 10)
)
```

**Note:** Requires matplotlib: `pip install matplotlib`

### Analyzing Different Token Positions

Look at predictions for specific positions in the sequence:

```python
text = "The capital of France is Paris"
tokens = model.encode(text)

# Analyze what the model predicts at "France"
for i, token_id in enumerate(tokens):
    if "France" in model.token_to_str(token_id):
        france_position = i
        break

# Run logit lens at the France position
results = model.logit_lens(
    text,
    position=france_position,
    top_k=3,
    layers=[10, 12, 14, 15]
)

print(f"What model predicts at 'France' position ({france_position}):\n")
for layer_idx, predictions in results.items():
    print(f"Layer {layer_idx}:")
    for token_id, score, token_str in predictions:
        print(f"  '{token_str}': {score:.2f}")
```

### Comparing Predictions Across Prompts

Analyze how different prompts affect layer predictions:

```python
prompts = [
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Italy is"
]

for prompt in prompts:
    results = model.logit_lens(prompt, layers=[15], top_k=1)

    # Get top prediction from last layer
    token_id, score, token_str = results[15][0]
    print(f"'{prompt}' -> '{token_str}' ({score:.1f})")

# Output:
# 'The capital of France is' -> ' Paris' (19.0)
# 'The capital of Germany is' -> ' Berlin' (18.5)
# 'The capital of Italy is' -> ' Rome' (17.8)
```

## Activation Collection

### Collecting Specific Layers

```python
from mlxterp import get_activations

# Collect activations from multiple prompts
prompts = [
    "The quick brown fox",
    "Hello world",
    "Machine learning is"
]

activations = get_activations(
    model,
    prompts=prompts,
    layers=[3, 8, 12],
    positions=-1  # Last token position
)

# Access activations
print(f"Layer 3 activations: {activations['layer_3'].shape}")
# Output: (3, hidden_dim) - batch of 3 prompts
```

### Multiple Token Positions

```python
# Get first and last token activations
activations = get_activations(
    model,
    prompts="Hello world",
    layers=[5],
    positions=[0, -1]  # First and last tokens
)

print(f"Shape: {activations['layer_5'].shape}")
# Output: (1, 2, hidden_dim) - 1 prompt, 2 positions
```

### Batch Processing Large Datasets

```python
from mlxterp import batch_get_activations

# Process 1000 prompts efficiently
large_dataset = [f"Sample prompt {i}" for i in range(1000)]

activations = batch_get_activations(
    model,
    prompts=large_dataset,
    layers=[3, 8, 12],
    batch_size=32  # Process 32 at a time
)

print(f"Total activations: {activations['layer_3'].shape}")
# Output: (1000, hidden_dim)
```

## Interventions

### Basic Interventions

```python
from mlxterp import interventions as iv

# Zero out a layer
with model.trace("Test input", interventions={"layers.4": iv.zero_out}):
    output = model.output.save()

# Scale activations
with model.trace("Test input", interventions={"layers.3": iv.scale(0.5)}):
    output = model.output.save()

# Clamp values
with model.trace("Test input", interventions={"layers.5": iv.clamp(-1.0, 1.0)}):
    output = model.output.save()

# Add noise
with model.trace("Test input", interventions={"layers.2": iv.noise(std=0.1)}):
    output = model.output.save()
```

### Steering Vectors

```python
import mlx.core as mx
from mlxterp import interventions as iv

# Create a steering vector
hidden_dim = 4096  # Model hidden dimension
steering_vector = mx.random.normal((hidden_dim,))

# Apply steering to multiple layers
interventions_dict = {
    "layers.5": iv.add_vector(steering_vector * 1.0),
    "layers.6": iv.add_vector(steering_vector * 1.5),
    "layers.7": iv.add_vector(steering_vector * 2.0),
}

with model.trace("Original prompt", interventions=interventions_dict):
    steered_output = model.output.save()
```

### Custom Interventions

```python
import mlx.core as mx

def my_custom_intervention(activation: mx.array) -> mx.array:
    """Apply custom transformation"""
    # Example: ReLU clipping
    return mx.maximum(activation, 0.0)

with model.trace("Test", interventions={"layers.3": my_custom_intervention}):
    output = model.output.save()
```

### Composed Interventions

```python
from mlxterp import interventions as iv

# Chain multiple interventions
combined = iv.compose() \
    .add(iv.scale(0.8)) \
    .add(iv.noise(std=0.05)) \
    .add(iv.clamp(-3.0, 3.0)) \
    .build()

with model.trace("Test", interventions={"layers.4": combined}):
    output = model.output.save()
```

## Activation Patching

!!! info "Complete Guide Available"
    For a comprehensive guide on activation patching including theory, interpretation, and common pitfalls, see the [Activation Patching Guide](guides/activation_patching.md).

### Simple One-Line Activation Patching

Use the built-in helper to automatically test all layers:

```python
# Find which MLPs are important - that's it!
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True
)

# Results show:
# Layer  0: +43.1% recovery  ← Very important!
# Layer 10: -23.5% recovery  ← Encodes corruption
```

**Result interpretation:**
- **Positive %** (e.g., +40%): Layer is important - patching helps recover clean output
- **Negative %** (e.g., -20%): Layer encodes the corruption - patching makes it worse
- **Near 0%**: Layer not relevant for this task

### Test Different Components

```python
# Test MLPs
mlp_results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp"
)

# Test attention
attn_results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="self_attn"
)

# Test specific sub-components
gate_results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp.gate_proj"  # or "mlp.up_proj", "self_attn.q_proj", etc.
)

# Get most important layer
sorted_layers = sorted(mlp_results.items(), key=lambda x: x[1], reverse=True)
print(f"Most important: Layer {sorted_layers[0][0]} ({sorted_layers[0][1]:.1f}% recovery)")
```

### Choosing the Right Metric

For large vocabulary models (> 100k tokens), use the `mse` metric for numerical stability:

```python
from mlxterp import InterpretableModel
from mlx_lm import load

# Load large vocabulary model (e.g., Qwen with 151k tokens)
base_model, tokenizer = load('mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit')
model = InterpretableModel(base_model, tokenizer=tokenizer)

print(f"Vocabulary size: {model.vocab_size}")  # 151,643 tokens!

# Use MSE metric for stability
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    metric="mse",  # ← Important for large vocab models!
    layers=[0, 10, 20, 30, 40, 47],  # Test subset
    plot=True
)

# Results:
# Layer 10: 17.9% recovery  ← Most important!
# Layer 30:  7.5% recovery
# Layer  0: -298.6% recovery  ← Encodes corruption
```

**Metric recommendations**:
- **< 50k vocab**: Use `metric="l2"` (default)
- **50k - 100k vocab**: Use `metric="l2"` or `metric="cosine"`
- **> 100k vocab**: Use `metric="mse"` or `metric="cosine"`

**Why it matters**: With 150k vocabulary, computing L2 distance over 150k logits can overflow to `inf`, resulting in `nan` recovery percentages. MSE averages instead of summing, preventing overflow.

### Manual Patching (Advanced)

If you need more control, you can still do manual patching:

```python
import mlx.core as mx
from mlxterp import interventions as iv

# Get clean activation
with model.trace("Paris is the capital of France") as trace:
    clean_mlp = trace.activations["model.model.layers.10.mlp"]

mx.eval(clean_mlp)

# Patch into corrupted run
with model.trace("London is the capital of France",
                interventions={"layers.10.mlp": iv.replace_with(clean_mlp)}):
    patched_output = model.output.save()
```

### Available Components

You can patch any captured component:

- **Full components**: `"mlp"`, `"self_attn"`
- **MLP sub-components**: `"mlp.gate_proj"`, `"mlp.up_proj"`, `"mlp.down_proj"`
- **Attention sub-components**: `"self_attn.q_proj"`, `"self_attn.k_proj"`, `"self_attn.v_proj"`, `"self_attn.o_proj"`

## Advanced Patterns

### Attention Pattern Analysis

```python
with model.trace("Analyze this text") as trace:
    # Collect all attention outputs
    attention_outputs = []
    for i in range(len(model.layers)):
        if hasattr(model.layers[i], 'attn'):
            attn = model.layers[i].attn.output.save()
            attention_outputs.append(attn)

# Analyze attention patterns
for i, attn in enumerate(attention_outputs):
    print(f"Layer {i} attention: {attn.shape}")
```

### Residual Stream Analysis

```python
# Trace the residual stream through the network
with model.trace("Input text") as trace:
    residuals = []
    for i in range(len(model.layers)):
        # Save layer output (which includes residual)
        layer_out = model.layers[i].output.save()
        residuals.append(layer_out)

# Analyze how information flows
for i in range(len(residuals) - 1):
    diff = mx.linalg.norm(residuals[i+1] - residuals[i])
    print(f"Change from layer {i} to {i+1}: {diff:.4f}")
```

### Gradient-Based Analysis

```python
import mlx.core as mx
import mlx.nn as nn

# Get activations
with model.trace("Sample text") as trace:
    target_activation = model.layers[8].output.save()

# Compute gradient w.r.t. input
def loss_fn(x):
    with model.trace(x):
        act = model.layers[8].output.save()
    return mx.mean((act - target_activation) ** 2)

# Analyze gradient
grad_fn = mx.grad(loss_fn)
input_grad = grad_fn(input_tokens)
print(f"Input gradient shape: {input_grad.shape}")
```

## Working with Different Model Architectures

### GPT-Style Models

```python
# GPT-2 uses 'h' for layers
gpt2_model = load_gpt2()  # Your loading function
model = InterpretableModel(gpt2_model, layer_attr="h")

with model.trace(input_text):
    layer_3 = model.layers[3].output.save()
```

### Custom Layer Names

```python
# Model with nested structure
custom_model = load_custom_model()
model = InterpretableModel(custom_model, layer_attr="transformer.blocks")

with model.trace(input_text):
    block_5 = model.layers[5].output.save()
```

## Debugging and Inspection

### Inspecting Model Structure

```python
# List all modules in the model
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")

# Check number of layers
print(f"Total layers: {len(model.layers)}")

# Access model parameters
params = model.parameters()
print(f"Total parameters: {sum(p.size for p in params.values())}")
```

### Tracking Activation Statistics

```python
import mlx.core as mx

with model.trace("Test input") as trace:
    stats = {}
    for i in range(len(model.layers)):
        act = model.layers[i].output.save()
        stats[f"layer_{i}"] = {
            "mean": float(mx.mean(act)),
            "std": float(mx.std(act)),
            "max": float(mx.max(act)),
            "min": float(mx.min(act)),
        }

# Print statistics
for layer, stat in stats.items():
    print(f"{layer}: mean={stat['mean']:.3f}, std={stat['std']:.3f}")
```

## Performance Tips

### Batch Processing

```python
# Good: Process in batches
from mlxterp import batch_get_activations

activations = batch_get_activations(
    model,
    prompts=large_list,
    layers=[3, 8],
    batch_size=32
)

# Avoid: Loading everything at once
# with model.trace(large_list):  # May run out of memory
#     acts = model.layers[3].output.save()
```

### Selective Activation Saving

```python
# Good: Only save what you need
with model.trace(input_text):
    important_layers = [3, 8, 12]
    acts = {i: model.layers[i].output.save() for i in important_layers}

# Avoid: Saving everything
# with model.trace(input_text):
#     all_acts = [model.layers[i].output.save() for i in range(100)]
```

### Memory Management

```python
import mlx.core as mx

# Process in chunks and clear cache
results = []
for chunk in chunks(large_dataset, size=100):
    acts = get_activations(model, chunk, layers=[8])
    results.append(acts)
    mx.eval(acts)  # Force evaluation
    # Results are computed, can process them

# Concatenate at the end
final_results = mx.concatenate(results, axis=0)
```

## See Also

- [API Reference](API.md) - Complete API documentation
- [Quick Start](QUICKSTART.md) - Getting started guide
- [GitHub Examples](https://github.com/yourusername/mlxterp/tree/main/examples) - More examples
