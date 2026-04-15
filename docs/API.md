# mlxterp API Documentation

Complete API reference for mlxterp.

## Table of Contents

1. [InterpretableModel](#interpretablemodel)
2. [Tracing](#tracing)
3. [Interventions](#interventions)
4. [Utilities](#utilities)
5. [Core Components](#core-components)
6. [Metrics](#metrics)
7. [Analysis Results](#analysis-results)
8. [Causal Interpretability](#causal-interpretability)
9. [Text Generation](#text-generation)
10. [Conversation Analysis](#conversation-analysis)
11. [Visualization](#visualization-patching--dashboards)

---

## InterpretableModel

### Class: `InterpretableModel`

Main entry point for wrapping MLX models to add interpretability features.

#### Constructor

```python
InterpretableModel(
    model: Union[nn.Module, str],
    tokenizer: Optional[Any] = None,
    layer_attr: str = "layers",
    embedding_path: Optional[str] = None,
    norm_path: Optional[str] = None,
    lm_head_path: Optional[str] = None
)
```

**Parameters:**

- **model** (`nn.Module` or `str`): Either:
  - An MLX `nn.Module` instance to wrap
  - A model name/path string (attempts to load via `mlx_lm.load()`)

- **tokenizer** (`Optional[Any]`): Tokenizer for processing text inputs. If `None` and model is loaded from string, attempts to load tokenizer automatically.

- **layer_attr** (`str`, default: `"layers"`): Name of the attribute containing the model's transformer layers. Common values:
  - `"layers"` (Llama, Mistral)
  - `"h"` (GPT-2)
  - `"transformer.h"` (some GPT variants)

- **embedding_path** (`Optional[str]`, default: `None`): Override path for the token embedding layer. Used for weight-tied output projection in `get_token_predictions`. Auto-detected if not specified. Tried paths: `model.embed_tokens`, `model.model.embed_tokens`, `embed_tokens`, `tok_embeddings`, `wte`.

- **norm_path** (`Optional[str]`, default: `None`): Override path for the final layer normalization. Used in `logit_lens` for projecting intermediate activations. Auto-detected if not specified. Tried paths: `model.norm`, `model.model.norm`, `norm`, `ln_f`, `model.ln_f`.

- **lm_head_path** (`Optional[str]`, default: `None`): Override path for the output projection layer. If not found, falls back to weight-tied embedding. Tried paths: `lm_head`, `model.lm_head`, `model.model.lm_head`, `output`, `head`.

**Returns:** `InterpretableModel` instance

**Example:**

```python
from mlxterp import InterpretableModel

# Load from model name (auto-detection works)
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Wrap existing model
import mlx.nn as nn
base_model = nn.Module()  # Your model
model = InterpretableModel(base_model, tokenizer=my_tokenizer)

# Custom layer attribute
model = InterpretableModel(gpt2_model, layer_attr="h")

# Custom model with non-standard attribute names
model = InterpretableModel(
    custom_model,
    tokenizer=my_tokenizer,
    embedding_path="my_custom_embeddings",
    norm_path="my_final_norm",
    lm_head_path="my_output_projection"
)
```

---

#### Method: `trace`

Create a tracing context for capturing activations and applying interventions.

```python
model.trace(
    inputs: Union[str, List[str], mx.array, List[int]],
    interventions: Optional[Dict[str, Callable]] = None
) -> Trace
```

**Parameters:**

- **inputs**: Input data in various formats:
  - `str`: Single text prompt (requires tokenizer)
  - `List[str]`: Batch of text prompts (requires tokenizer)
  - `mx.array`: Token array, shape `(batch, seq_len)`
  - `List[int]`: Single sequence of token IDs

- **interventions** (`Optional[Dict[str, Callable]]`): Dictionary mapping module names to intervention functions. Module names use dot notation (e.g., `"layers.3.self_attn"` for mlx-lm models).

**Returns:** `Trace` context manager

**Example:**

```python
# Basic tracing
with model.trace("Hello world"):
    output = model.output.save()

# With interventions
from mlxterp import interventions as iv
with model.trace("Test", interventions={"layers.3": iv.scale(0.5)}):
    output = model.output.save()

# Batch inputs
with model.trace(["Hello", "World"]):
    acts = model.layers[3].output.save()
```

---

#### Method: `encode`

Encode text to token IDs using the model's tokenizer.

```python
model.encode(text: str) -> List[int]
```

**Parameters:**

- **text** (`str`): Text string to encode

**Returns:** List of token IDs

**Raises:** `ValueError` if no tokenizer is available

**Example:**

```python
tokens = model.encode("Hello world")
print(tokens)  # [128000, 9906, 1917]
```

---

#### Method: `decode`

Decode token IDs to text using the model's tokenizer.

```python
model.decode(tokens: Union[List[int], mx.array]) -> str
```

**Parameters:**

- **tokens** (`List[int]` or `mx.array`): Token IDs to decode

**Returns:** Decoded text string

**Raises:** `ValueError` if no tokenizer is available

**Example:**

```python
text = model.decode([128000, 9906, 1917])
print(text)  # "<|begin_of_text|>Hello world"

# Also works with mx.array
import mlx.core as mx
tokens_array = mx.array([128000, 9906, 1917])
text = model.decode(tokens_array)
```

---

#### Method: `encode_batch`

Encode multiple texts to token IDs.

```python
model.encode_batch(texts: List[str]) -> List[List[int]]
```

**Parameters:**

- **texts** (`List[str]`): List of text strings to encode

**Returns:** List of token ID lists

**Raises:** `ValueError` if no tokenizer is available

**Example:**

```python
token_lists = model.encode_batch(["Hello", "World", "Test"])
print(token_lists)
# [[128000, 9906], [128000, 10343], [128000, 2323]]
```

---

#### Method: `token_to_str`

Convert a single token ID to its string representation.

```python
model.token_to_str(token_id: int) -> str
```

**Parameters:**

- **token_id** (`int`): Token ID to decode

**Returns:** String representation of the token

**Raises:** `ValueError` if no tokenizer is available

**Example:**

```python
# Decode individual tokens
tokens = model.encode("Hello world")
for i, token_id in enumerate(tokens):
    token_str = model.token_to_str(token_id)
    print(f"Token {i}: {token_id} -> '{token_str}'")

# Output:
# Token 0: 128000 -> '<|begin_of_text|>'
# Token 1: 9906 -> 'Hello'
# Token 2: 1917 -> ' world'
```

---

#### Property: `vocab_size`

Get the vocabulary size of the tokenizer.

```python
model.vocab_size -> Optional[int]
```

**Returns:** Vocabulary size, or `None` if no tokenizer is available

**Example:**

```python
print(f"Vocabulary size: {model.vocab_size}")  # 128000
```

---

#### Attribute: `tokenizer`

Direct access to the underlying tokenizer for advanced operations.

**Type:** Tokenizer object (varies by model)

**Example:**

```python
# Access tokenizer directly for advanced features
tokenizer = model.tokenizer

# Use tokenizer-specific methods
if hasattr(tokenizer, 'special_tokens'):
    print(tokenizer.special_tokens)
```

---

#### Method: `get_token_predictions`

Decode hidden states to token predictions using the model's output projection.

```python
model.get_token_predictions(
    hidden_state: mx.array,
    top_k: int = 10,
    return_scores: bool = False,
    embedding_layer: Optional[Any] = None,
    lm_head: Optional[Any] = None
) -> Union[List[int], List[tuple]]
```

**Parameters:**

- **hidden_state** (`mx.array`): Hidden state tensor, shape `(hidden_dim,)` or `(batch, hidden_dim)`
- **top_k** (`int`, default: `10`): Number of top predictions to return
- **return_scores** (`bool`, default: `False`): If True, return `(token_id, score)` tuples
- **embedding_layer** (`Optional[Any]`, default: `None`): Override embedding layer for weight-tied projection. If provided, uses this layer's weights for output projection.
- **lm_head** (`Optional[Any]`, default: `None`): Override lm_head layer. If provided, uses this layer directly. Takes precedence over `embedding_layer`.

**Returns:** List of token IDs or `(token_id, score)` tuples

**Example:**

```python
# Get predictions from a specific layer
with model.trace("The capital of France is") as trace:
    layer_6 = trace.activations["model.model.layers.6"]

# Get last token's hidden state
last_token_hidden = layer_6[0, -1, :]

# Get top predictions
predictions = model.get_token_predictions(last_token_hidden, top_k=5)

# Decode to words
for token_id in predictions:
    print(model.token_to_str(token_id))

# With scores
predictions_with_scores = model.get_token_predictions(
    last_token_hidden,
    top_k=5,
    return_scores=True
)
for token_id, score in predictions_with_scores:
    token_str = model.token_to_str(token_id)
    print(f"{token_str}: {score:.2f}")

# Custom model with override at call time
predictions = model.get_token_predictions(
    hidden,
    top_k=5,
    lm_head=custom_model.my_lm_head
)
```

**Notes:**
- Automatically handles weight-tied models (uses embedding weights transposed)
- Works with quantized embeddings (dequantizes automatically)
- Model-agnostic: auto-detects embedding/lm_head paths for various architectures
- Useful for analyzing what any layer "thinks" at a specific position

---

#### Method: `logit_lens`

Apply logit lens technique to see what each layer predicts at each token position.

The logit lens projects each layer's hidden states through the final layer norm and embedding matrix to see what tokens each layer predicts at each position in the input sequence.

```python
model.logit_lens(
    text: str,
    top_k: int = 1,
    layers: Optional[List[int]] = None,
    position: Optional[int] = None,
    plot: bool = False,
    max_display_tokens: int = 15,
    figsize: tuple = (16, 10),
    cmap: str = 'viridis',
    font_family: Optional[str] = None,
    final_norm: Optional[Any] = None,
    skip_norm: bool = False
) -> Dict[int, List[List[tuple]]]
```

**Parameters:**

- **text** (`str`): Input text to analyze
- **top_k** (`int`, default: `1`): Number of top predictions to return per position
- **layers** (`Optional[List[int]]`, default: `None`): Specific layers to analyze (None = all)
- **position** (`Optional[int]`, default: `None`): Specific position to analyze (None = all). Supports negative indexing (-1 = last position).
- **plot** (`bool`, default: `False`): If True, display a heatmap visualization showing predictions
- **max_display_tokens** (`int`, default: `15`): Maximum number of tokens to show in visualization (from the end)
- **figsize** (`tuple`, default: `(16, 10)`): Figure size for plot (width, height)
- **cmap** (`str`, default: `'viridis'`): Colormap for heatmap
- **font_family** (`Optional[str]`, default: `None`): Font for plot (use 'Arial Unicode MS' for CJK support)
- **final_norm** (`Optional[Any]`, default: `None`): Override final layer normalization module. If provided, uses this module instead of auto-detected norm layer.
- **skip_norm** (`bool`, default: `False`): If True, skip final layer normalization entirely. Useful for models without a final norm layer.

**Returns:** Dict mapping `layer_idx` -> list of positions -> list of `(token_id, score, token_str)` tuples

**Structure:** `{layer_idx: [[pos_0_predictions], [pos_1_predictions], ...]}`

**Model Compatibility:** This method automatically detects model structure and works with:
- mlx-lm models (`model.model.norm`, `model.model.embed_tokens`)
- GPT-2 style models (`ln_f`, `wte`)
- Custom models (use `norm_path` constructor arg or `final_norm`/`skip_norm` parameters)

**Example:**

```python
# Get predictions at all positions for all layers
results = model.logit_lens("The capital of France is")

# Access predictions for layer 10 at position 3
layer_10_predictions = results[10]
pos_3_top_pred = layer_10_predictions[3][0]  # (token_id, score, token_str)
print(f"Layer 10, Position 3: {pos_3_top_pred[2]}")

# Show what each layer predicts at the LAST position
text = "The capital of France is"
results = model.logit_lens(text, layers=[0, 5, 10, 15])

for layer_idx in [0, 5, 10, 15]:
    # Get prediction at last position
    last_pos_pred = results[layer_idx][-1][0][2]
    print(f"Layer {layer_idx}: '{last_pos_pred}'")

# Output:
# Layer  0: ' the'
# Layer  5: ' a'
# Layer 10: ' Paris'
# Layer 15: ' Paris'

# Show predictions at each position for a specific layer
text = "The capital of France"
results = model.logit_lens(text, layers=[10])
tokens = model.encode(text)

for pos_idx, predictions in enumerate(results[10]):
    input_token = model.token_to_str(tokens[pos_idx])
    pred_token = predictions[0][2]  # Top prediction
    print(f"Position {pos_idx} ('{input_token}') -> '{pred_token}'")

# Visualize with heatmap
results = model.logit_lens(
    "The Eiffel Tower is located in the city of",
    plot=True,
    max_display_tokens=15,
    figsize=(16, 10)
)
# Displays a heatmap with:
#   - X-axis: Input token positions
#   - Y-axis: Model layers
#   - Cell values: Top predicted token at each (layer, position)
#   - Colors: Different predictions shown with different colors

# Model without final normalization
results = model.logit_lens("Hello world", skip_norm=True)

# Custom final norm at call time
results = model.logit_lens(
    "Hello world",
    final_norm=custom_model.my_final_norm
)
```

**Note:** Plotting requires matplotlib: `pip install matplotlib`

**Use Cases:**
- Understand how model predictions evolve through layers
- Debug model behavior at intermediate layers
- Visualize progressive refinement of predictions
- Identify where in the model certain facts are computed

---

#### Method: `tuned_lens`

Apply tuned lens for improved layer-wise predictions.

The tuned lens technique (Belrose et al., 2023) uses learned affine transformations for each layer to correct for coordinate system mismatches between layers, producing more accurate intermediate predictions than the standard logit lens.

```python
model.tuned_lens(
    text: str,
    tuned_lens: TunedLens,
    top_k: int = 1,
    layers: Optional[List[int]] = None,
    position: Optional[int] = None,
    plot: bool = False,
    max_display_tokens: int = 15,
    figsize: tuple = (16, 10),
    cmap: str = 'viridis',
    font_family: Optional[str] = None,
    final_norm: Any = None,
    skip_norm: bool = False
) -> Dict[int, List[List[tuple]]]
```

**Parameters:**

- **text** (`str`): Input text to analyze
- **tuned_lens** (`TunedLens`): Trained TunedLens instance with layer translators
- **top_k** (`int`, default: `1`): Number of top predictions to return per position
- **layers** (`Optional[List[int]]`, default: `None`): Specific layers to analyze (None = all)
- **position** (`Optional[int]`, default: `None`): Specific position to analyze (None = all). Supports negative indexing.
- **plot** (`bool`, default: `False`): If True, display a heatmap visualization
- **max_display_tokens** (`int`, default: `15`): Maximum number of tokens to show in visualization
- **figsize** (`tuple`, default: `(16, 10)`): Figure size for plot
- **cmap** (`str`, default: `'viridis'`): Colormap for heatmap
- **font_family** (`Optional[str]`): Font for plot (auto-detected if None)
- **final_norm** (`Any`, default: `None`): Override for final layer norm. Pass a callable to use a custom norm.
- **skip_norm** (`bool`, default: `False`): If True, skip final layer normalization (for models without it)

**Returns:** Dict mapping `layer_idx` -> list of positions -> list of `(token_id, score, token_str)` tuples

**Example:**

```python
from mlxterp import InterpretableModel, TunedLens

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Option 1: Train new tuned lens
tuned_lens = model.train_tuned_lens(
    dataset=["Sample text 1", "Sample text 2", ...],
    num_steps=250,
    save_path="tuned_lens_llama.npz"
)

# Option 2: Load pre-trained tuned lens
tuned_lens = model.load_tuned_lens("tuned_lens_llama.npz")

# Apply tuned lens
results = model.tuned_lens(
    "The capital of France is",
    tuned_lens,
    layers=[0, 5, 10, 15],
    plot=True
)

# Compare with regular logit lens
regular_results = model.logit_lens("The capital of France is", layers=[0, 5, 10, 15])
```

**Reference:** Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (https://arxiv.org/abs/2303.08112)

---

#### Method: `train_tuned_lens`

Train a tuned lens for this model.

The tuned lens technique trains small affine transformations for each layer to correct for coordinate system mismatches, producing more accurate intermediate predictions.

```python
model.train_tuned_lens(
    dataset: List[str],
    num_steps: int = 250,
    learning_rate: float = 1.0,
    momentum: float = 0.9,
    max_seq_len: int = 2048,
    gradient_clip: float = 1.0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None
) -> TunedLens
```

**Parameters:**

- **dataset** (`List[str]`): List of text strings for training
- **num_steps** (`int`, default: `250`): Number of training steps
- **learning_rate** (`float`, default: `1.0`): Initial learning rate (uses linear decay)
- **momentum** (`float`, default: `0.9`): Nesterov momentum coefficient
- **max_seq_len** (`int`, default: `2048`): Maximum sequence length for training chunks
- **gradient_clip** (`float`, default: `1.0`): Gradient clipping norm
- **save_path** (`Optional[str]`): Path to save trained weights
- **verbose** (`bool`, default: `True`): Print training progress
- **callback** (`Optional[Callable]`): Callback function called with `(step, loss)` after each step

**Returns:** Trained `TunedLens` instance

**Raises:**

- `ValueError`: If dataset is empty or contains only whitespace
- `ValueError`: If dataset has fewer tokens than `max_seq_len`
- `ValueError`: If `max_seq_len` is less than 10 tokens
- `ValueError`: If model hidden dimension cannot be determined

**Training Details (from paper):**
- Optimizer: SGD with Nesterov momentum (0.9)
- Learning rate: 1.0 with linear decay over training steps
- Gradient clipping: norm 1.0
- Loss: KL divergence between translator prediction and final output

**Example:**

```python
# Load sample texts for training
texts = [
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    # ... more training texts
]

# Train tuned lens
tuned_lens = model.train_tuned_lens(
    dataset=texts,
    num_steps=250,
    save_path="my_tuned_lens.npz",
    verbose=True
)
```

---

#### Method: `load_tuned_lens`

Load a pre-trained tuned lens from a file.

```python
model.load_tuned_lens(path: str) -> TunedLens
```

**Parameters:**

- **path** (`str`): Path to the saved tuned lens weights (expects `.npz` and `.json` files)

**Returns:** Loaded `TunedLens` instance

**Example:**

```python
tuned_lens = model.load_tuned_lens("tuned_lens_llama.npz")
results = model.tuned_lens("Hello world", tuned_lens)
```

---

#### Method: `activation_patching`

Automated activation patching to identify important layers for a task.

This helper method performs activation patching across all (or specified) layers to determine which components are critical for a specific task. It automates the boilerplate of running clean/corrupted inputs, patching activations, and measuring recovery.

```python
model.activation_patching(
    clean_text: str,
    corrupted_text: str,
    component: str = "mlp",
    layers: Optional[List[int]] = None,
    metric: str = "l2",
    plot: bool = False,
    figsize: tuple = (12, 8),
    cmap: str = "RdBu_r"
) -> Dict[int, float]
```

**Parameters:**

- **clean_text** (`str`): Clean/correct input text
- **corrupted_text** (`str`): Corrupted input text (differs in the aspect you're studying)
- **component** (`str`, default: `"mlp"`): Component to patch. Options:
  - `"mlp"` - Full MLP block
  - `"self_attn"` - Full attention block
  - `"mlp.gate_proj"` - MLP gate projection
  - `"mlp.up_proj"` - MLP up projection
  - `"mlp.down_proj"` - MLP down projection
  - `"self_attn.q_proj"` - Query projection
  - `"self_attn.k_proj"` - Key projection
  - `"self_attn.v_proj"` - Value projection
  - `"self_attn.o_proj"` - Output projection
- **layers** (`Optional[List[int]]`, default: `None`): Specific layers to test (None = all layers)
- **metric** (`str`, default: `"l2"`): Distance metric. Options:
  - `"l2"`: Euclidean distance (default, with overflow protection)
  - `"cosine"`: Cosine distance (recommended for large vocabularies)
  - `"mse"`: Mean squared error (most stable for huge models > 100k vocab)

  **Recommendation**:
  - Vocab < 50k: use `"l2"`
  - Vocab 50k-100k: use `"l2"` or `"cosine"`
  - Vocab > 100k: use `"mse"` or `"cosine"`
- **plot** (`bool`, default: `False`): If True, display a bar chart of recovery percentages
- **figsize** (`tuple`, default: `(12, 8)`): Figure size for plot
- **cmap** (`str`, default: `"RdBu_r"`): Colormap for plot (blue = positive, red = negative)

**Returns:** Dict mapping `layer_idx` -> recovery percentage

**Recovery Interpretation:**
- **Positive %** (e.g., +40%): Layer is important for the task
- **Negative %** (e.g., -20%): Layer encodes the corruption
- **~0%**: Layer is not relevant to this task

**Example:**

```python
# Find which MLP layers are important for factual recall
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    plot=True
)

# Analyze results
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
print("\nMost important layers:")
for layer_idx, recovery in sorted_results[:3]:
    print(f"  Layer {layer_idx}: {recovery:.1f}% recovery")

# Output:
# Layer  0: +43.1% recovery  ← Very important!
# Layer 15: +24.2% recovery  ← Important
# Layer  6: +17.6% recovery  ← Somewhat important

# Test specific layers only
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="self_attn",
    layers=[0, 5, 10, 15],
    plot=False
)

# Test MLP sub-components
results = model.activation_patching(
    clean_text="The cat sits on the mat",
    corrupted_text="The cat sit on the mat",
    component="mlp.gate_proj",
    layers=[3, 8, 12]
)

# For large vocabulary models (> 100k tokens), use MSE metric
results = model.activation_patching(
    clean_text="Paris is the capital of France",
    corrupted_text="London is the capital of France",
    component="mlp",
    metric="mse",  # Most stable for Qwen, GPT-4 scale models
    plot=True
)
```

**Note:** Plotting requires matplotlib: `pip install matplotlib`

**Distance Metrics**:

The method supports three distance metrics for measuring output differences:

1. **L2 (Euclidean)** - Default, best for models with < 50k vocab
   ```
   d(a,b) = √(Σ(aᵢ-bᵢ)²)
   ```

2. **Cosine** - Best for direction-based similarity, good for 50k-150k vocab
   ```
   d(a,b) = 1 - (a·b)/(||a||×||b||)
   ```

3. **MSE (Mean Squared Error)** - Most stable for very large models (> 100k vocab)
   ```
   d(a,b) = (1/N)×Σ(aᵢ-bᵢ)²
   ```

**Example with large vocabulary model** (Qwen: 151k tokens):
```python
# Without correct metric - gets NaN
results = model.activation_patching(..., metric="l2")  # ❌ May overflow

# With correct metric - works perfectly
results = model.activation_patching(..., metric="mse")  # ✅ Stable
```

**See Also:** [Activation Patching Guide](guides/activation_patching.md) for comprehensive coverage including metric selection and numerical details

---

#### Method: `named_modules`

Iterator over all modules in the wrapped model with their names.

```python
model.named_modules() -> Iterator[Tuple[str, nn.Module]]
```

**Returns:** Iterator yielding `(name, module)` tuples

**Example:**

```python
for name, module in model.named_modules():
    print(f"{name}: {type(module).__name__}")
```

---

#### Attribute: `layers`

Indexed access to model layers via `LayerListProxy`.

**Type:** `LayerListProxy`

**Example:**

```python
# Access specific layer
layer_3 = model.layers[3]

# Iterate over layers
for layer in model.layers:
    print(layer)

# Get number of layers
num_layers = len(model.layers)
```

---

#### Attribute: `output`

Access to model output within a trace context. Returns an `OutputProxy` that can be saved.

**Type:** `OutputProxy`

**Example:**

```python
with model.trace(input_text):
    final_output = model.output.save()
```

---

## Tracing

### Class: `Trace`

Context manager for tracing model execution. Created by `InterpretableModel.trace()`.

#### Attributes

- **output** (`mx.array`): Model output after trace completes
- **saved_values** (`Dict[str, mx.array]`): Values saved with `.save()`
- **activations** (`Dict[str, mx.array]`): All captured activations by module name

#### Methods

##### `get(name: str) -> Optional[mx.array]`

Get a saved value by name.

```python
with model.trace(input) as trace:
    model.layers[3].output.save()

activation = trace.get("layers.3.output")
```

##### `get_activation(name: str) -> Optional[mx.array]`

Get an activation by module name.

```python
with model.trace(input) as trace:
    pass  # Activations captured automatically

# Note: get_activation requires the full key (not normalized)
attn_act = trace.get_activation("model.model.layers.3.self_attn")
```

---

### Class: `OutputProxy`

Wraps module outputs to provide `.save()` functionality.

#### Method: `save() -> Any`

Save the wrapped value to the current trace context and return the unwrapped value.

**Returns:** The underlying value (usually `mx.array`)

**Example:**

```python
with model.trace(input):
    # Save returns the actual array (use self_attn for mlx-lm models)
    attn = model.layers[3].self_attn.output.save()
    print(attn.shape)  # Can use immediately
```

---

## Interventions

Intervention functions modify activations during forward passes.

### Module: `mlxterp.interventions`

Namespace containing pre-built intervention functions.

```python
from mlxterp import interventions as iv

# Use intervention functions
with model.trace(input, interventions={"layers.3": iv.scale(0.5)}):
    output = model.output.save()
```

---

### Function: `zero_out`

Set activations to zero.

```python
zero_out(x: mx.array) -> mx.array
```

**Example:**

```python
with model.trace(input, interventions={"layers.4": iv.zero_out}):
    output = model.output.save()
```

---

### Function: `scale`

Multiply activations by a constant factor.

```python
scale(factor: float) -> Callable[[mx.array], mx.array]
```

**Parameters:**
- **factor** (`float`): Scaling factor

**Example:**

```python
# Reduce by 50%
with model.trace(input, interventions={"layers.3": iv.scale(0.5)}):
    output = model.output.save()

# Amplify
with model.trace(input, interventions={"layers.3": iv.scale(2.0)}):
    output = model.output.save()
```

---

### Function: `add_vector`

Add a steering vector to activations.

```python
add_vector(vector: mx.array) -> Callable[[mx.array], mx.array]
```

**Parameters:**
- **vector** (`mx.array`): Vector to add (must be broadcastable to activation shape)

**Example:**

```python
import mlx.core as mx

# Create steering vector
steering = mx.random.normal((hidden_dim,))

with model.trace(input, interventions={"layers.5": iv.add_vector(steering)}):
    steered_output = model.output.save()
```

---

### Function: `replace_with`

Replace activations with a fixed value.

```python
replace_with(value: Union[mx.array, float]) -> Callable[[mx.array], mx.array]
```

**Parameters:**
- **value**: Replacement value (array or scalar)

**Example:**

```python
# Replace with zeros
with model.trace(input, interventions={"layers.3": iv.replace_with(0.0)}):
    output = model.output.save()

# Replace with custom array
custom = mx.ones((batch, seq_len, hidden_dim))
with model.trace(input, interventions={"layers.3": iv.replace_with(custom)}):
    output = model.output.save()
```

---

### Function: `clamp`

Clamp activation values to a range.

```python
clamp(min_val: float = None, max_val: float = None) -> Callable[[mx.array], mx.array]
```

**Parameters:**
- **min_val** (`Optional[float]`): Minimum value
- **max_val** (`Optional[float]`): Maximum value

**Example:**

```python
# Clamp to [-1, 1]
with model.trace(input, interventions={"layers.3": iv.clamp(-1.0, 1.0)}):
    output = model.output.save()

# Only maximum
with model.trace(input, interventions={"layers.3": iv.clamp(max_val=10.0)}):
    output = model.output.save()
```

---

### Function: `noise`

Add Gaussian noise to activations.

```python
noise(std: float = 0.1) -> Callable[[mx.array], mx.array]
```

**Parameters:**
- **std** (`float`, default: `0.1`): Standard deviation of noise

**Example:**

```python
with model.trace(input, interventions={"layers.3": iv.noise(std=0.2)}):
    output = model.output.save()
```

---

### Class: `InterventionComposer`

Compose multiple interventions into a single function.

#### Method: `add`

Add an intervention to the composition.

```python
add(fn: Callable[[mx.array], mx.array]) -> InterventionComposer
```

**Returns:** `self` for chaining

#### Method: `build`

Build the composed intervention function.

```python
build() -> Callable[[mx.array], mx.array]
```

**Returns:** Composed intervention function

**Example:**

```python
from mlxterp import interventions as iv

# Compose multiple interventions
combined = iv.compose() \
    .add(iv.scale(0.8)) \
    .add(iv.noise(std=0.1)) \
    .add(iv.clamp(-5.0, 5.0)) \
    .build()

with model.trace(input, interventions={"layers.3": combined}):
    output = model.output.save()
```

---

### Custom Interventions

Create your own intervention functions:

```python
import mlx.core as mx

def my_intervention(activation: mx.array) -> mx.array:
    """Custom activation modification"""
    # Your logic here
    return mx.tanh(activation)

with model.trace(input, interventions={"layers.3": my_intervention}):
    output = model.output.save()
```

**Requirements:**
- Function signature: `(mx.array) -> mx.array`
- Must return array with same shape as input
- Can use any MLX operations

---

## Utilities

### Function: `get_activations`

Collect activations for specified layers and token positions.

```python
get_activations(
    model: InterpretableModel,
    prompts: Union[str, List[str]],
    layers: Optional[List[int]] = None,
    positions: Union[int, List[int]] = -1
) -> Dict[str, mx.array]
```

**Parameters:**

- **model**: InterpretableModel instance
- **prompts**: Single prompt or list of prompts
- **layers**: Layer indices to collect (None = all layers)
- **positions**: Token position(s) to extract
  - `-1`: Last token
  - `0`: First token
  - `[0, -1]`: First and last tokens

**Returns:** Dict mapping `"layer_{i}"` to activation arrays

**Shapes:**
- Single position: `(batch_size, hidden_dim)`
- Multiple positions: `(batch_size, num_positions, hidden_dim)`

**Example:**

```python
from mlxterp import get_activations

# Single prompt, multiple layers
acts = get_activations(model, "Hello world", layers=[3, 8, 12])
print(acts["layer_3"].shape)  # (1, hidden_dim)

# Batch prompts
acts = get_activations(
    model,
    ["Hello", "World", "Test"],
    layers=[5],
    positions=-1
)
print(acts["layer_5"].shape)  # (3, hidden_dim)

# Multiple positions
acts = get_activations(
    model,
    "Test prompt",
    layers=[3],
    positions=[0, -1]  # First and last token
)
print(acts["layer_3"].shape)  # (1, 2, hidden_dim)
```

---

### Function: `batch_get_activations`

Memory-efficient batch processing for large datasets.

```python
batch_get_activations(
    model: InterpretableModel,
    prompts: List[str],
    layers: Optional[List[int]] = None,
    positions: Union[int, List[int]] = -1,
    batch_size: int = 8
) -> Dict[str, mx.array]
```

**Parameters:**

- **model**: InterpretableModel instance
- **prompts**: List of prompts
- **layers**: Layer indices to collect
- **positions**: Token position(s) to extract
- **batch_size**: Number of prompts per batch

**Returns:** Dict mapping `"layer_{i}"` to concatenated activation arrays

**Example:**

```python
from mlxterp import batch_get_activations

# Process 1000 prompts efficiently
large_dataset = [f"Prompt {i}" for i in range(1000)]

acts = batch_get_activations(
    model,
    prompts=large_dataset,
    layers=[3, 8, 12],
    batch_size=32
)

print(acts["layer_3"].shape)  # (1000, hidden_dim)
```

---

### Function: `collect_activations`

Direct activation collection with caching.

```python
collect_activations(
    model: InterpretableModel,
    inputs: Any,
    layers: Optional[List[str]] = None
) -> ActivationCache
```

**Parameters:**

- **model**: InterpretableModel instance
- **inputs**: Input data
- **layers**: List of layer names to cache (None = all)

**Returns:** `ActivationCache` object

**Example:**

```python
from mlxterp import collect_activations

cache = collect_activations(
    model,
    "Test input",
    layers=["layers.3", "layers.8"]
)

# Access cached activations
act_3 = cache.get("layers.3")
print(f"Cached {len(cache)} activations")
print(f"Available keys: {cache.keys()}")
```

---

## Core Components

Advanced usage: Direct access to core components.

### Class: `ModuleProxy`

Wraps `nn.Module` to intercept forward passes. Created automatically by `InterpretableModel`.

**Attributes:**
- `output`: OutputProxy for the module's output

**Example:**

```python
# Access through InterpretableModel.layers
proxy = model.layers[3]  # Returns ModuleProxy
print(type(proxy))  # ModuleProxy

with model.trace(input):
    act = proxy.output.save()
```

---

### Class: `LayerListProxy`

Provides indexed access to model layers.

**Methods:**
- `__getitem__(idx)`: Get layer at index
- `__len__()`: Number of layers
- `__iter__()`: Iterate over layers

**Example:**

```python
# Created automatically
layers = model.layers

# Access
layer_3 = layers[3]

# Length
print(len(layers))  # 12

# Iterate
for i, layer in enumerate(layers):
    print(f"Layer {i}: {layer}")
```

---

### Class: `ActivationCache`

Storage for cached activations.

**Attributes:**
- **activations** (`Dict[str, mx.array]`): Activation storage
- **metadata** (`Optional[Dict]`): Additional information

**Methods:**
- `get(name)`: Get activation by name
- `keys()`: List all cached names
- `__contains__(name)`: Check if activation exists
- `__len__()`: Number of cached activations

**Example:**

```python
from mlxterp import collect_activations

cache = collect_activations(model, input)

# Access
act = cache.get("layers.3")

# Check existence
if "layers.3" in cache:
    print("Found!")

# List all
for name in cache.keys():
    print(name)
```

---

### Class: `ModuleResolver`

Generic module resolution for different MLX model architectures. Automatically finds embedding, final norm, and lm_head modules using fallback chains.

**Constructor:**

```python
ModuleResolver(
    model: nn.Module,
    embedding_path: Optional[str] = None,
    norm_path: Optional[str] = None,
    lm_head_path: Optional[str] = None
)
```

**Methods:**

- `get_embedding_layer()`: Get token embedding layer
- `get_final_norm()`: Get final layer normalization
- `get_lm_head()`: Get output projection layer
- `get_output_projection()`: Get output projection with weight-tied detection. Returns `(module, path, is_weight_tied)`
- `clear_cache()`: Clear resolved module cache (call after modifying model structure)

**Fallback Chains:**

| Component | Resolution Order |
|-----------|-----------------|
| Embedding | `model.embed_tokens`, `model.model.embed_tokens`, `embed_tokens`, `tok_embeddings`, `wte` |
| Final Norm | `model.norm`, `model.model.norm`, `norm`, `ln_f`, `model.ln_f` |
| LM Head | `lm_head`, `model.lm_head`, `model.model.lm_head`, `output`, `head` (falls back to embedding if not found) |

**Example:**

```python
from mlxterp.core import ModuleResolver

# Create resolver for a model
resolver = ModuleResolver(model)

# Get components
embedding = resolver.get_embedding_layer()
norm = resolver.get_final_norm()
proj, path, is_tied = resolver.get_output_projection()

if is_tied:
    print("Model uses weight-tied embedding for output")

# With custom paths
resolver = ModuleResolver(
    model,
    embedding_path="my_embed",
    norm_path="my_norm"
)

# Cache invalidation (after modifying model structure)
resolver.clear_cache()  # Force re-resolution on next access
```

---

### Class: `TunedLens`

Learned affine translators for each layer, implementing the Tuned Lens technique from Belrose et al. (2023).

The tuned lens uses layer-specific affine transformations (Wx + b) to map hidden states from each layer into a space where the final output projection can make accurate predictions. This corrects for coordinate system mismatches between layers.

```python
from mlxterp import TunedLens

tuned_lens = TunedLens(num_layers=32, hidden_dim=4096)
```

**Parameters:**

- **num_layers** (`int`): Number of transformer layers in the model
- **hidden_dim** (`int`): Dimension of hidden states

**Attributes:**

- `num_layers`: Number of layers
- `hidden_dim`: Hidden dimension
- `translators`: List of linear layers, one per transformer layer

**Methods:**

| Method | Description |
|--------|-------------|
| `__call__(h, layer_idx)` | Apply translator for a specific layer |
| `save(path)` | Save weights and config to files (.npz and .json) |
| `load(path)` | Load tuned lens from saved files (classmethod) |

**Example:**

```python
from mlxterp import TunedLens

# Create tuned lens
tuned_lens = TunedLens(num_layers=32, hidden_dim=4096)

# Apply to hidden state from layer 10
translated = tuned_lens(hidden_state, layer_idx=10)

# Save and load
tuned_lens.save("my_tuned_lens")
loaded = TunedLens.load("my_tuned_lens")
```

**Reference:** Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens" (https://arxiv.org/abs/2303.08112)

---

### Function: `normalize_layer_key`

Normalize activation keys by removing model prefixes.

```python
normalize_layer_key(key: str) -> str
```

**Example:**

```python
from mlxterp.core import normalize_layer_key

normalize_layer_key("model.model.layers.0")  # "layers.0"
normalize_layer_key("model.layers.5.self_attn")  # "layers.5.self_attn"
```

---

### Function: `find_layer_key_pattern`

Find the correct activation key pattern for a layer index.

```python
find_layer_key_pattern(
    activations: dict,
    layer_idx: int,
    component: Optional[str] = None
) -> Optional[str]
```

**Example:**

```python
from mlxterp.core import find_layer_key_pattern

# Find layer 5's key in activations dict
key = find_layer_key_pattern(trace.activations, 5)
# Returns "model.model.layers.5" or "layers.5" etc.

# Find specific component
key = find_layer_key_pattern(trace.activations, 5, "self_attn")
```

---

## Type Annotations

Common types used in mlxterp:

```python
from typing import Union, List, Dict, Callable, Optional, Any
import mlx.core as mx
import mlx.nn as nn

# Input types
InputType = Union[str, List[str], mx.array, List[int]]

# Intervention function type
InterventionFn = Callable[[mx.array], mx.array]

# Interventions dict
InterventionsDict = Dict[str, InterventionFn]

# Model type
ModelType = Union[nn.Module, str]
```

---

## Error Handling

### Common Exceptions

#### `ValueError`

Raised when:
- String input provided without tokenizer
- Invalid input format
- Model cannot be loaded from string

```python
# Will raise ValueError
model = InterpretableModel(base_model)  # No tokenizer
with model.trace("text input"):  # Needs tokenizer!
    pass
```

**Solution:** Provide tokenizer

```python
model = InterpretableModel(base_model, tokenizer=my_tokenizer)
```

#### `AttributeError`

Raised when:
- Accessing non-existent module attribute
- Layer attribute doesn't exist

```python
# Will raise AttributeError
model = InterpretableModel(custom_model, layer_attr="transformer")
# If custom_model doesn't have 'transformer' attribute
```

**Solution:** Specify correct layer attribute

```python
model = InterpretableModel(custom_model, layer_attr="layers")
```

---

## Best Practices

### 1. Always Use Context Managers

```python
# ✅ Good
with model.trace(input):
    act = model.layers[3].output.save()

# ❌ Avoid
trace = model.trace(input)
# Missing context manager!
```

### 2. Save Early, Access Later

```python
# ✅ Good
with model.trace(input) as t:
    model.layers[3].output.save()

# Access after trace completes
act = t.get("layers.3.output")

# ❌ Avoid trying to access during trace
with model.trace(input) as t:
    act = t.get("layers.3.output")  # Not saved yet!
```

### 3. Use Utility Functions for Common Tasks

```python
# ✅ Good - Use get_activations
from mlxterp import get_activations
acts = get_activations(model, prompts, layers=[3, 8])

# ❌ Avoid manual loops
acts = {}
for layer_idx in [3, 8]:
    with model.trace(prompts):
        acts[layer_idx] = model.layers[layer_idx].output.save()
```

### 4. Batch Large Datasets

```python
# ✅ Good - Use batching
from mlxterp import batch_get_activations
acts = batch_get_activations(model, large_list, batch_size=32)

# ❌ Avoid loading everything at once
with model.trace(large_list):  # May run out of memory
    acts = model.layers[3].output.save()
```

---

## Visualization Module

The visualization module provides tools for attention pattern analysis and visualization.

```python
from mlxterp.visualization import (
    # Attention extraction and visualization
    get_attention_patterns,
    attention_heatmap,
    attention_from_trace,
    AttentionVisualizationConfig,

    # Pattern detection
    AttentionPatternDetector,
    detect_head_types,
    detect_induction_heads,
    induction_score,
    previous_token_score,
    first_token_score,
    copying_score,
)
```

---

### Function: `get_attention_patterns`

Extract attention weight patterns from a trace.

```python
get_attention_patterns(
    trace: Trace,
    layers: Optional[List[int]] = None
) -> Dict[int, np.ndarray]
```

**Parameters:**

- **trace** (`Trace`): Completed trace context with captured attention weights
- **layers** (`Optional[List[int]]`, default: `None`): Specific layers to extract (None = all)

**Returns:** Dict mapping `layer_idx` -> attention array of shape `(batch, heads, seq_len, seq_len)`

**Example:**

```python
from mlxterp import InterpretableModel
from mlxterp.visualization import get_attention_patterns

model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")

with model.trace("The capital of France is") as trace:
    pass

# Get all attention patterns
patterns = get_attention_patterns(trace)
print(f"Found {len(patterns)} layers")
print(f"Shape: {patterns[0].shape}")  # (1, 32, 6, 6)

# Get specific layers
patterns = get_attention_patterns(trace, layers=[0, 5, 10])
```

---

### Function: `attention_heatmap`

Create a heatmap visualization of attention patterns.

```python
attention_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    head_idx: int = 0,
    title: Optional[str] = None,
    colorscale: str = "Blues",
    backend: str = "matplotlib",
    mask_upper_tri: bool = True,
    figsize: tuple = (8, 6)
) -> Any
```

**Parameters:**

- **attention** (`np.ndarray`): Attention weights, shape `(batch, heads, seq_q, seq_k)`
- **tokens** (`List[str]`): Token strings for axis labels
- **head_idx** (`int`, default: `0`): Which attention head to visualize
- **title** (`Optional[str]`): Plot title
- **colorscale** (`str`, default: `"Blues"`): Colormap name
- **backend** (`str`, default: `"matplotlib"`): Backend (`"matplotlib"`, `"plotly"`, `"circuitsviz"`)
- **mask_upper_tri** (`bool`, default: `True`): Mask future positions (for causal attention)
- **figsize** (`tuple`, default: `(8, 6)`): Figure size for matplotlib

**Returns:** Figure object (type depends on backend)

**Example:**

```python
from mlxterp.visualization import get_attention_patterns, attention_heatmap

with model.trace("Hello world") as trace:
    pass

patterns = get_attention_patterns(trace, layers=[5])
tokens = model.to_str_tokens("Hello world")

# Create heatmap for head 0
fig = attention_heatmap(
    patterns[5],
    tokens,
    head_idx=0,
    title="Layer 5, Head 0",
    backend="matplotlib"
)
```

---

### Function: `attention_from_trace`

High-level function to visualize attention patterns from a trace.

```python
attention_from_trace(
    trace: Trace,
    tokens: List[str],
    layers: Optional[List[int]] = None,
    heads: Optional[List[int]] = None,
    mode: str = "single",
    head_notation: str = "LH",
    config: Optional[AttentionVisualizationConfig] = None
) -> Any
```

**Parameters:**

- **trace** (`Trace`): Completed trace context
- **tokens** (`List[str]`): Token strings for labels
- **layers** (`Optional[List[int]]`): Layers to visualize (default: first layer)
- **heads** (`Optional[List[int]]`): Heads to visualize (default: first head)
- **mode** (`str`, default: `"single"`): Visualization mode:
  - `"single"`: One heatmap
  - `"grid"`: Grid of multiple heatmaps
- **head_notation** (`str`, default: `"LH"`): Notation for titles (`"LH"` = L5H3, `"dot"` = 5.3)
- **config** (`Optional[AttentionVisualizationConfig]`): Custom configuration

**Returns:** Figure object

**Example:**

```python
from mlxterp.visualization import attention_from_trace, AttentionVisualizationConfig

with model.trace("The quick brown fox") as trace:
    pass

tokens = model.to_str_tokens("The quick brown fox")

# Single heatmap
fig = attention_from_trace(trace, tokens, layers=[5], heads=[0], mode="single")

# Grid of multiple heads
config = AttentionVisualizationConfig(backend="matplotlib", colorscale="Blues")
fig = attention_from_trace(
    trace, tokens,
    layers=[0, 5, 10],
    heads=[0, 1, 2, 3],
    mode="grid",
    config=config
)
```

---

### Class: `AttentionVisualizationConfig`

Configuration for attention visualization.

```python
AttentionVisualizationConfig(
    colorscale: str = "Blues",
    mask_upper_tri: bool = True,
    backend: str = "auto",
    figsize: tuple = (10, 8),
    show_colorbar: bool = True,
    font_size: int = 10
)
```

**Parameters:**

- **colorscale** (`str`): Colormap name (default: `"Blues"`)
- **mask_upper_tri** (`bool`): Mask future positions (default: `True`)
- **backend** (`str`): Visualization backend (default: `"auto"` - tries circuitsviz, then plotly, then matplotlib)
- **figsize** (`tuple`): Figure size (default: `(10, 8)`)
- **show_colorbar** (`bool`): Show colorbar (default: `True`)
- **font_size** (`int`): Font size for labels (default: `10`)

---

### Function: `induction_score`

Compute induction head score for an attention pattern.

Induction heads implement the pattern `[A][B]...[A] -> predict [B]` by attending from position `i` to position `i - seq_len + 1` in repeated sequences.

```python
induction_score(
    attention_pattern: np.ndarray,
    seq_len: int
) -> float
```

**Parameters:**

- **attention_pattern** (`np.ndarray`): Attention weights, shape `(seq_q, seq_k)`
- **seq_len** (`int`): Length of the repeated subsequence

**Returns:** Induction score (0-1, higher = more induction-like)

**Example:**

```python
import numpy as np
from mlxterp.visualization import induction_score

# Create pattern from repeated sequence "ABC ABC"
# In a true induction head, position 4 (second A) attends to position 1 (first B)
seq_len = 3
total_len = 6

# Synthetic perfect induction pattern
pattern = np.zeros((total_len, total_len))
for i in range(seq_len, total_len):
    pattern[i, i - seq_len + 1] = 1.0

score = induction_score(pattern, seq_len)
print(f"Induction score: {score:.3f}")  # ~1.0
```

---

### Function: `previous_token_score`

Compute previous token head score.

Previous token heads attend strongly to position `i-1` (the immediately preceding token).

```python
previous_token_score(attention_pattern: np.ndarray) -> float
```

**Parameters:**

- **attention_pattern** (`np.ndarray`): Attention weights, shape `(seq_q, seq_k)`

**Returns:** Previous token score (0-1, higher = attends more to previous position)

**Example:**

```python
from mlxterp.visualization import previous_token_score

# Perfect previous token pattern
pattern = np.zeros((5, 5))
for i in range(1, 5):
    pattern[i, i-1] = 1.0

score = previous_token_score(pattern)
print(f"Previous token score: {score:.3f}")  # ~1.0
```

---

### Function: `first_token_score`

Compute first token (BOS) head score.

First token heads attend strongly to position 0 (typically BOS or first token).

```python
first_token_score(attention_pattern: np.ndarray) -> float
```

**Parameters:**

- **attention_pattern** (`np.ndarray`): Attention weights, shape `(seq_q, seq_k)`

**Returns:** First token score (0-1, higher = attends more to position 0)

**Example:**

```python
from mlxterp.visualization import first_token_score

# Perfect first token pattern
pattern = np.zeros((5, 5))
pattern[:, 0] = 1.0  # All positions attend to first

score = first_token_score(pattern)
print(f"First token score: {score:.3f}")  # ~1.0
```

---

### Function: `copying_score`

Compute copying head score from OV circuit.

Copying heads increase the logit of the attended-to token.

```python
copying_score(
    ov_circuit: np.ndarray,
    unembedding: Optional[np.ndarray] = None
) -> float
```

**Parameters:**

- **ov_circuit** (`np.ndarray`): OV circuit matrix `W_V @ W_O`, shape `(d_model, d_model)`
- **unembedding** (`Optional[np.ndarray]`): Optional unembedding matrix for full analysis

**Returns:** Copying score (higher = more copying behavior)

---

### Class: `AttentionPatternDetector`

Detector for classifying attention head types.

```python
AttentionPatternDetector(
    induction_threshold: float = 0.4,
    previous_token_threshold: float = 0.5,
    first_token_threshold: float = 0.3,
    current_token_threshold: float = 0.3
)
```

**Parameters:**

- **induction_threshold** (`float`): Threshold for induction head classification
- **previous_token_threshold** (`float`): Threshold for previous token head
- **first_token_threshold** (`float`): Threshold for first token head
- **current_token_threshold** (`float`): Threshold for current token (self-attention) head

**Methods:**

#### `analyze_head`

Compute all pattern scores for a single attention head.

```python
analyze_head(
    attention_pattern: np.ndarray,
    seq_len_for_induction: Optional[int] = None
) -> Dict[str, float]
```

**Returns:** Dict of pattern type -> score

#### `classify_head`

Classify an attention head into one or more types.

```python
classify_head(
    attention_pattern: np.ndarray,
    seq_len_for_induction: Optional[int] = None
) -> List[str]
```

**Returns:** List of head type labels that exceed thresholds

**Example:**

```python
from mlxterp.visualization import AttentionPatternDetector

detector = AttentionPatternDetector(
    previous_token_threshold=0.5,
    first_token_threshold=0.3
)

# Analyze a head
scores = detector.analyze_head(attention_pattern)
print(f"Scores: {scores}")
# {'previous_token': 0.85, 'first_token': 0.1, 'current_token': 0.05}

# Classify
types = detector.classify_head(attention_pattern)
print(f"Classification: {types}")  # ['previous_token']
```

---

### Function: `detect_head_types`

Detect attention head types across a model.

```python
detect_head_types(
    model: InterpretableModel,
    text: str,
    threshold: float = 0.4,
    layers: Optional[List[int]] = None
) -> Dict[str, List[Tuple[int, int]]]
```

**Parameters:**

- **model** (`InterpretableModel`): Model to analyze
- **text** (`str`): Input text for analysis
- **threshold** (`float`, default: `0.4`): Score threshold for classification
- **layers** (`Optional[List[int]]`): Specific layers to analyze (None = all)

**Returns:** Dict mapping head type to list of `(layer, head)` tuples

**Example:**

```python
from mlxterp.visualization import detect_head_types

head_types = detect_head_types(
    model,
    "The quick brown fox jumps over the lazy dog",
    threshold=0.3,
    layers=[0, 5, 10, 15]
)

print(f"Previous token heads: {len(head_types['previous_token'])}")
print(f"First token heads: {len(head_types['first_token'])}")

# Print specific heads
for layer, head in head_types['previous_token'][:5]:
    print(f"  L{layer}H{head}")
```

---

### Function: `detect_induction_heads`

Detect induction heads using repeated random token sequences.

```python
detect_induction_heads(
    model: InterpretableModel,
    n_random_tokens: int = 50,
    n_repeats: int = 2,
    threshold: float = 0.4,
    layers: Optional[List[int]] = None,
    seed: int = 42
) -> List[HeadScore]
```

**Parameters:**

- **model** (`InterpretableModel`): Model to analyze
- **n_random_tokens** (`int`, default: `50`): Number of random tokens in subsequence
- **n_repeats** (`int`, default: `2`): Number of times to repeat the subsequence
- **threshold** (`float`, default: `0.4`): Score threshold for detection
- **layers** (`Optional[List[int]]`): Specific layers to analyze (None = all)
- **seed** (`int`, default: `42`): Random seed for reproducibility

**Returns:** List of `HeadScore` objects for heads above threshold, sorted by score descending

**Example:**

```python
from mlxterp.visualization import detect_induction_heads

# Find induction heads
induction_heads = detect_induction_heads(
    model,
    n_random_tokens=50,
    threshold=0.3,
    layers=[0, 5, 10, 15]
)

print(f"Found {len(induction_heads)} induction heads")
for head in induction_heads[:10]:
    print(f"  L{head.layer}H{head.head}: {head.score:.3f}")
```

---

### Class: `HeadScore`

Score for an attention head (dataclass).

```python
@dataclass
class HeadScore:
    layer: int
    head: int
    score: float
    head_type: Optional[str] = None
```

---

### Method: `InterpretableModel.to_str_tokens`

Convert text or token IDs to a list of token strings.

```python
model.to_str_tokens(
    input: Union[str, List[int], mx.array],
    prepend_bos: bool = False
) -> List[str]
```

**Parameters:**

- **input**: Text string, list of token IDs, or mx.array of tokens
- **prepend_bos** (`bool`, default: `False`): Whether to prepend BOS token (only for string input)

**Returns:** List of token strings

**Example:**

```python
# From text
tokens = model.to_str_tokens("Hello world")
print(tokens)  # ['<|begin_of_text|>', 'Hello', ' world']

# From token IDs
token_ids = model.encode("Hello world")
tokens = model.to_str_tokens(token_ids)
print(tokens)  # ['<|begin_of_text|>', 'Hello', ' world']
```

---

---

## Metrics

Module: `mlxterp.metrics`

Standard metrics for causal interpretability experiments. All metrics follow a uniform signature so they can be passed to any patching or attribution function.

### Metric Signature

Every metric function accepts:

```python
def metric(
    patched_logits: mx.array,    # Logits after intervention
    clean_logits: mx.array,      # Logits from clean/correct input
    corrupted_logits: mx.array,  # Logits from corrupted input
    **kwargs,                    # Metric-specific parameters
) -> float:
```

Returns a scalar where **higher = more recovery of clean behavior**.

### Function: `logit_diff`

Measures recovery of the logit difference between correct and incorrect tokens. The standard metric for Indirect Object Identification (IOI) and factual recall tasks.

```python
from mlxterp.metrics import logit_diff

effect = logit_diff(
    patched_logits, clean_logits, corrupted_logits,
    correct_token=tokenizer.encode(" Paris")[0],
    incorrect_token=tokenizer.encode(" Rome")[0],
)
# Returns: 1.0 = fully recovered, 0.0 = no change, <0 = worse
```

**Required kwargs:**

- **correct_token** (`int`): Token ID for the correct completion
- **incorrect_token** (`int`): Token ID for the incorrect completion

### Function: `kl_divergence`

KL divergence between patched and clean output distributions. Returns negative KL so higher = better (consistent with other metrics).

```python
from mlxterp.metrics import kl_divergence

effect = kl_divergence(patched_logits, clean_logits, corrupted_logits)
# Returns: 0.0 when distributions match, negative when they diverge
```

### Function: `cross_entropy_diff`

Difference in cross-entropy loss: `CE(corrupted) - CE(patched)`. Positive means patching reduced the loss.

```python
from mlxterp.metrics import cross_entropy_diff

effect = cross_entropy_diff(
    patched_logits, clean_logits, corrupted_logits,
    target_token=42,  # Optional: auto-detected from clean argmax if omitted
)
```

### Function: `l2_distance`

Normalized L2 recovery: what fraction of the clean-corrupted distance was recovered.

```python
from mlxterp.metrics import l2_distance

effect = l2_distance(patched_logits, clean_logits, corrupted_logits)
# Returns: 1.0 = patched == clean, 0.0 = patched == corrupted
```

### Function: `cosine_distance`

Cosine similarity recovery between patched and clean outputs.

```python
from mlxterp.metrics import cosine_distance

effect = cosine_distance(patched_logits, clean_logits, corrupted_logits)
```

### Function: `get_metric`

Look up a metric by name string, or pass through a callable.

```python
from mlxterp.metrics import get_metric

fn = get_metric("logit_diff")  # Returns the logit_diff function
fn = get_metric("l2")          # Alias for l2_distance
fn = get_metric(my_custom_fn)  # Returns my_custom_fn unchanged
```

**Available names:** `"logit_diff"`, `"kl_divergence"` / `"kl"`, `"cross_entropy_diff"` / `"ce_diff"`, `"l2_distance"` / `"l2"`, `"cosine_distance"` / `"cosine"`

---

## Analysis Results

Module: `mlxterp.results`

Every analysis function returns a structured result object with `.data`, `.summary()`, `.to_json()`, `.to_markdown()`, and `.plot()` methods. This enables both human consumption and agent/programmatic processing.

### Class: `AnalysisResult`

Base class for all results.

```python
from mlxterp.results import AnalysisResult

result.data          # Dict of structured results
result.metadata      # Dict of parameters and timing
result.result_type   # String tag (e.g., "patching", "dla")
result.summary()     # One-line human-readable summary
result.to_json()     # JSON serialization (handles mx.array)
result.to_markdown() # Markdown report
result.plot()        # Visualization (subclass-specific)
```

### Class: `PatchingResult`

Returned by `activation_patching()` and `path_patching()`.

```python
from mlxterp.causal import activation_patching

result = activation_patching(model, clean, corrupted, component="mlp")

result.effect_matrix   # mx.array: (n_layers,) or (n_layers, n_heads)
result.layers          # List of layer indices tested
result.component       # "mlp", "attn", "resid_post", "attn_head"
result.metric_name     # Name of the metric used

# Get top components by effect size
top = result.top_components(k=5)
# Returns: [(layer_idx, effect_score), ...]

# Visualize
result.plot()  # Heatmap or bar chart depending on dimensions
```

### Class: `AttributionResult`

Returned by `attribution_patching()`.

```python
result.attribution_scores  # mx.array of scores
result.layers              # Layers analyzed
result.component           # Component analyzed
result.method              # "finite_diff" or "gradient"
```

### Class: `DLAResult`

Returned by `direct_logit_attribution()`.

```python
result.head_contributions  # mx.array: per-attention-layer contributions
result.mlp_contributions   # mx.array: per-MLP contributions
result.target_token        # Token ID being attributed
result.target_token_str    # String representation
```

### Class: `GenerationResult`

Returned by `model.generate()`.

```python
result.text           # Generated text string
result.tokens         # List of token IDs
result.token_logits   # Per-token logit distributions (optional)
result.prompt         # Original prompt
```

### Class: `ConversationResult`

Returned by `ConversationTrace.to_result()`.

```python
result.turns                 # List of turn metadata dicts
result.cross_turn_attention  # Turn x turn attention matrix
```

### Class: `CircuitResult`

Returned by `acdc()` and `feature_circuit()`.

```python
result.nodes      # List of component names in the circuit
result.edges      # List of (sender, receiver, weight) tuples
result.threshold  # Pruning threshold used

# Export as graph
graph = result.to_graph()
# Returns: {"nodes": [{"id": ...}], "edges": [{"source": ..., "target": ..., "weight": ...}]}
```

---

## Causal Interpretability

Module: `mlxterp.causal`

The causal package provides tools for understanding which model components cause specific behaviors, from activation patching to automated circuit discovery.

### Function: `activation_patching`

The core causal analysis tool. Patches clean activations into corrupted forward passes to identify important components.

```python
from mlxterp.causal import activation_patching

# Layer-level patching: which layers matter?
result = activation_patching(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    component="mlp",        # "resid_post", "attn", "mlp", "attn_head"
    metric="l2",            # or "logit_diff", "kl", "cosine", "ce_diff"
    layers=range(16),       # None = all layers
)

print(result.summary())
result.plot()  # Heatmap of effects

# Position-level: which token positions matter?
result = activation_patching(
    model, clean, corrupted,
    component="mlp",
    positions=[3, 4, 5],   # Only patch these positions
)

# Head-level: which attention heads matter?
result = activation_patching(
    model, clean, corrupted,
    component="attn_head",  # Returns (n_layers, n_heads) matrix
    metric="logit_diff",
    metric_kwargs={"correct_token": 123, "incorrect_token": 456},
)
top = result.top_components(k=10)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `InterpretableModel` | required | Model to analyze |
| `clean` | `str` or `mx.array` | required | Clean/correct input |
| `corrupted` | `str` or `mx.array` | required | Corrupted/counterfactual input |
| `layers` | `List[int]` | `None` (all) | Layer indices to test |
| `component` | `str` | `"resid_post"` | Component to patch |
| `metric` | `str` or `Callable` | `"l2"` | Metric function |
| `positions` | `List[int]` | `None` (all) | Token positions to patch |
| `metric_kwargs` | `Dict` | `None` | Extra metric arguments |
| `verbose` | `bool` | `False` | Print progress |

### Class: `CausalTrace`

Declarative context manager for paired clean/corrupted analysis. Reduces boilerplate for multi-component patching experiments.

```python
from mlxterp.causal import CausalTrace

# Using model.causal_trace() shortcut:
with model.causal_trace(clean_text, corrupted_text) as ct:
    # Register patches declaratively
    ct.patch("layers.5.mlp")
    ct.patch("layers.7.self_attn", positions=[3, 4])

    # Compute metric with all patches applied at once
    effect = ct.metric("l2")
    print(f"Combined patching effect: {effect:.4f}")

# Access clean activations
with model.causal_trace(clean, corrupted) as ct:
    act = ct.get_clean_activation("mlp", layer=5)
    print(f"Clean MLP output shape: {act.shape}")

# Use canonical names with layer parameter
with model.causal_trace(clean, corrupted) as ct:
    ct.patch("mlp", layer=0)
    ct.patch("attn", layer=3)
    effect = ct.metric("logit_diff", correct_token=100, incorrect_token=200)
```

**Methods:**

- `ct.patch(component, positions=None, layer=None)` - Register a component to be patched from clean into corrupted
- `ct.metric(metric_fn, **kwargs)` - Apply all registered patches and compute metric
- `ct.get_clean_activation(component, layer=None)` - Retrieve a specific clean activation

### Class: `ResidualStreamAccessor`

Structured access to the residual stream at each layer, decomposed into attention and MLP contributions.

```python
from mlxterp.causal import ResidualStreamAccessor

with model.trace("The capital of France is") as trace:
    pass

rs = ResidualStreamAccessor(trace.activations)

# Access residual stream at each layer
pre = rs.resid_pre(5)            # Input to layer 5
post = rs.resid_post(5)          # Output of layer 5
mid = rs.resid_mid(5)            # Between attn and MLP (= pre + attn)

# Component contributions (raw, before residual add)
attn_out = rs.attn_contribution(5)   # What attention added
mlp_out = rs.mlp_contribution(5)     # What MLP added
total = rs.layer_contribution(5)     # Total: post - pre

# Discover available layers
layers = rs.available_layers()       # [0, 1, 2, ..., 15]
```

**Key invariants:**

- `resid_pre[i] == resid_post[i-1]` (layer i's input is layer i-1's output)
- `resid_post[i] == resid_pre[i] + attn_contribution[i] + mlp_contribution[i]` (approximately, for standard transformer blocks)
- `resid_mid[i] == resid_pre[i] + attn_contribution[i]`

### Function: `direct_logit_attribution`

Decomposes the final logit prediction into per-component contributions by projecting each component's output through the unembedding matrix.

```python
from mlxterp.causal import direct_logit_attribution

result = direct_logit_attribution(
    model,
    text="The capital of France is",
    target_token=None,     # Auto-detect from argmax, or specify token ID
    position=-1,           # Token position to analyze (-1 = last)
    layers=None,           # None = all layers
)

print(f"Target: '{result.target_token_str}' (id={result.target_token})")
print(f"Attention contributions: {result.head_contributions}")
print(f"MLP contributions: {result.mlp_contributions}")
print(result.summary())
```

### Function: `attribution_patching`

Fast approximation of activation patching using gradient-based finite differences. ~100x faster than brute-force patching.

```python
from mlxterp.causal import attribution_patching

result = attribution_patching(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    component="resid_post",  # "resid_post", "attn", "mlp"
    metric="l2",
    n_projections=1,         # More projections = more accurate, slower
    eps=1e-3,                # Perturbation size
)

print(result.summary())
# Scores approximate brute-force patching effects
print(f"Attribution scores: {result.attribution_scores}")
```

### Function: `path_patching`

Measures the causal effect along a specific sender-receiver path by freezing all other components.

```python
from mlxterp.causal import path_patching

result = path_patching(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    sender="layers.7.self_attn",    # Source component
    receiver="layers.9.self_attn",  # Target component
    metric="l2",
)

print(f"Path effect: {result.data['effect']:.4f}")
print(f"Components frozen: {result.data['n_frozen']}")
```

### Function: `acdc`

Automated Circuit Discovery. Iteratively tests each component's importance and prunes those below a threshold.

```python
from mlxterp.causal import acdc

circuit = acdc(
    model,
    clean="The Eiffel Tower is in",
    corrupted="The Colosseum is in",
    threshold=0.01,           # Minimum effect to keep a node
    components=["attn", "mlp"],  # Component types to test
    layers=range(16),
)

print(circuit.summary())
# "Circuit: 8 nodes, 12 edges (threshold=0.01)"

print("Important components:", circuit.nodes)
graph = circuit.to_graph()  # For visualization
```

### Feature-Level Analysis

#### Function: `feature_patching`

Compute causal effect of individual SAE features via ablation.

```python
from mlxterp.causal.feature_circuits import feature_patching

effects = feature_patching(
    model, sae,
    text="The capital of France is",
    layer=10,
    component="mlp",
    top_k=20,          # Test top-20 most active features
)
# Returns: {feature_id: effect_score, ...}

# Or test specific features
effects = feature_patching(
    model, sae, text,
    layer=10,
    feature_ids=[42, 123, 456],
)
```

#### Function: `feature_circuit`

Discover which SAE features are causally important.

```python
from mlxterp.causal.feature_circuits import feature_circuit

circuit = feature_circuit(
    model, sae, text,
    layer=10,
    component="mlp",
    threshold=0.01,
    top_k=50,
)

print(circuit.nodes)  # ["L10.mlp.f42", "L10.mlp.f123", ...]
```

---

## Text Generation

Module: `mlxterp.generation`

Generate text token-by-token with optional interventions applied at each step.

### Function: `generate` / `model.generate()`

```python
# Basic generation
result = model.generate(
    "The capital of France is",
    max_tokens=20,
    temperature=0.0,     # 0.0 = greedy (deterministic)
)
print(result.text)       # " Paris. Paris is..."
print(result.tokens)     # [3681, 13, 3681, ...]

# Generation with interventions (steering)
from mlxterp.core.intervention import add_vector, scale

result = model.generate(
    "I think the movie was",
    max_tokens=30,
    temperature=0.7,
    interventions={"layers.5": add_vector(positive_sentiment_vec)},
)

# Top-k and top-p sampling
result = model.generate(
    prompt,
    max_tokens=50,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
)

# Callback for custom stopping or logging
def my_callback(step, token_id, logits):
    print(f"Step {step}: token {token_id}")
    return step >= 20  # Return True to stop

result = model.generate(
    prompt,
    max_tokens=100,
    callback=my_callback,
    stop_tokens=[tokenizer.eos_token_id],
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str`, `mx.array`, `List[int]` | required | Input prompt |
| `max_tokens` | `int` | `50` | Maximum tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature (0 = greedy) |
| `top_k` | `int` | `0` | Top-k filtering (0 = disabled) |
| `top_p` | `float` | `1.0` | Nucleus sampling (1.0 = disabled) |
| `interventions` | `Dict[str, Callable]` | `None` | Interventions applied at each step |
| `stop_tokens` | `List[int]` | `None` | Token IDs that stop generation |
| `callback` | `Callable` | `None` | `callback(step, token_id, logits) -> bool` |

**Returns:** `GenerationResult`

---

## Conversation Analysis

Module: `mlxterp.conversation`

Tools for analyzing multi-turn conversations: turn detection, per-turn activation slicing, and cross-turn attention analysis.

### Class: `Turn`

Represents a single turn in a conversation with token position tracking.

```python
from mlxterp.conversation import Turn

turn = Turn(
    index=0,
    role="user",
    full_start=0,       # Including template tokens
    full_end=15,
    content_start=3,    # Content only (no role markers)
    content_end=12,
)

turn.content_positions    # slice(3, 12)
turn.full_positions       # slice(0, 15)
turn.n_content_tokens     # 9
turn.n_total_tokens       # 15
```

### Class: `TurnList`

Container with indexing, slicing, and role-based filtering.

```python
from mlxterp.conversation import TurnList

turns[0]                        # First turn
turns[0:2]                      # First two turns (returns TurnList)
turns.by_role("user")           # All user turns
turns.by_role("assistant")      # All assistant turns
turns.roles                     # ["user", "assistant", "user"]
turns.content_positions()       # All content positions across turns
```

### Function: `detect_turns`

Detect turn boundaries in a tokenized conversation using the chat template.

```python
from mlxterp.conversation import detect_turns

messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you!"},
    {"role": "user", "content": "What's my name?"},
]

turns = detect_turns(tokenizer, messages)
print(turns)
# [Turn(0, role='user', content=5:12, full=1:13),
#  Turn(1, role='assistant', content=16:22, full=13:23),
#  Turn(2, role='user', content=26:31, full=23:32)]
```

Supports Llama 3, ChatML (Qwen), Gemma, and custom templates.

### Class: `ConversationTrace`

Context manager for multi-turn conversation analysis. Uses `model.conversation_trace()`.

```python
conversation = [
    {"role": "user", "content": "The capital of France is Paris."},
    {"role": "assistant", "content": "That's correct!"},
    {"role": "user", "content": "What's the capital of France?"},
]

with model.conversation_trace(conversation) as ct:
    # Access turns
    print(ct.turns)  # TurnList with 3 turns

    # Get activations for a specific turn
    turn0_act = ct.get_turn_activation(0, "layers.5")
    turn2_act = ct.get_turn_activation(2, "layers.5")

    # Cross-turn attention: how much does each turn attend to others?
    cross_attn = ct.cross_turn_attention(layer=5, head=0)
    # Returns: (3, 3) matrix — entry [i,j] = avg attention from turn i to turn j

    # Convert to structured result
    result = ct.to_result()
    print(result.summary())
```

**Methods:**

| Method | Description |
|--------|-------------|
| `ct.get_turn_activation(turn_idx, component, content_only=True)` | Get activations sliced to a turn |
| `ct.cross_turn_attention(layer, head)` | Turn x turn attention aggregation |
| `ct.to_result()` | Convert to `ConversationResult` |

---

## Visualization (Patching & Dashboards)

### Patching Visualization

Module: `mlxterp.visualization.patching`

```python
from mlxterp.visualization.patching import plot_patching_result, plot_patching_comparison

# Plot a single result (auto-detects bar chart vs heatmap)
result = activation_patching(model, clean, corrupted, component="mlp")
fig = result.plot()  # Calls plot_patching_result internally

# Or call directly with options
fig = plot_patching_result(result, figsize=(14, 6), cmap="viridis", title="MLP Patching")

# Compare multiple results side by side
results = [
    activation_patching(model, clean, corrupted, component="mlp"),
    activation_patching(model, clean, corrupted, component="attn"),
]
fig = plot_patching_comparison(results, title="MLP vs Attention")
```

### Feature Dashboards

Module: `mlxterp.visualization.dashboards`

Generate standalone HTML dashboards for SAE feature analysis.

```python
from mlxterp.visualization.dashboards import (
    max_activating_examples,
    feature_activation_histogram,
    generate_feature_dashboard_html,
)

# Find texts that maximally activate a feature
examples = max_activating_examples(
    model, sae,
    feature_id=42,
    texts=dataset_texts,
    layer=10,
    component="mlp",
    top_k=10,
)
# Returns: [{"text": "...", "activation_value": 0.95, "token_position": 7}, ...]

# Compute activation distribution
histogram = feature_activation_histogram(
    model, sae,
    feature_id=42,
    texts=dataset_texts,
    layer=10,
)
# Returns: {"bin_edges": [...], "counts": [...], "mean": 0.12, "std": 0.3, "sparsity": 0.85}

# Generate HTML dashboard
html = generate_feature_dashboard_html(42, examples, histogram)
with open("feature_42_dashboard.html", "w") as f:
    f.write(html)
```

### Intervention: `replace_at_positions`

Position-level activation patching (added to `mlxterp.core.intervention`).

```python
from mlxterp.core.intervention import replace_at_positions

# Only replace positions 3 and 4, leaving all others unchanged
with model.trace(input, interventions={
    "layers.5.mlp": replace_at_positions(clean_activation, [3, 4])
}):
    output = model.output.save()
```

### Function: `resolve_component`

Cross-architecture component name resolution (in `mlxterp.core.module_resolver`).

```python
from mlxterp.core.module_resolver import resolve_component

# Find the actual activation key for "mlp" at layer 5
key = resolve_component("mlp", 5, trace.activations)
# Returns: "model.model.layers.5.mlp" (Llama) or "model.h.5.mlp" (GPT-2)

# Canonical names: "resid_post", "resid_pre", "attn", "mlp", "attn_head"
```

---

## Version History

### 0.1.0 (Current)

- Initial release
- Core tracing functionality
- Intervention system
- Basic utility functions
- Support for any MLX model
- **New**: Attention visualization module
  - Attention weight capture during tracing
  - Pattern extraction and visualization
  - Pattern detection (induction, previous token, first token heads)
  - Multiple backends (matplotlib, plotly, circuitsviz)
