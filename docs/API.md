# mlxterp API Documentation

Complete API reference for mlxterp.

## Table of Contents

1. [InterpretableModel](#interpretablemodel)
2. [Tracing](#tracing)
3. [Interventions](#interventions)
4. [Utilities](#utilities)
5. [Core Components](#core-components)

---

## InterpretableModel

### Class: `InterpretableModel`

Main entry point for wrapping MLX models to add interpretability features.

#### Constructor

```python
InterpretableModel(
    model: Union[nn.Module, str],
    tokenizer: Optional[Any] = None,
    layer_attr: str = "layers"
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

**Returns:** `InterpretableModel` instance

**Example:**

```python
from mlxterp import InterpretableModel

# Load from model name
model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")

# Wrap existing model
import mlx.nn as nn
base_model = nn.Module()  # Your model
model = InterpretableModel(base_model, tokenizer=my_tokenizer)

# Custom layer attribute
model = InterpretableModel(gpt2_model, layer_attr="h")
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

- **interventions** (`Optional[Dict[str, Callable]]`): Dictionary mapping module names to intervention functions. Module names use dot notation (e.g., `"layers.3.attn"`).

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
    return_scores: bool = False
) -> Union[List[int], List[tuple]]
```

**Parameters:**

- **hidden_state** (`mx.array`): Hidden state tensor, shape `(hidden_dim,)` or `(batch, hidden_dim)`
- **top_k** (`int`, default: `10`): Number of top predictions to return
- **return_scores** (`bool`, default: `False`): If True, return `(token_id, score)` tuples

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
```

**Notes:**
- Automatically handles weight-tied models (uses embedding weights transposed)
- Works with quantized embeddings (dequantizes automatically)
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
    plot: bool = False,
    max_display_tokens: int = 15,
    figsize: tuple = (16, 10),
    cmap: str = 'viridis'
) -> Dict[int, List[List[tuple]]]
```

**Parameters:**

- **text** (`str`): Input text to analyze
- **top_k** (`int`, default: `1`): Number of top predictions to return per position
- **layers** (`Optional[List[int]]`, default: `None`): Specific layers to analyze (None = all)
- **plot** (`bool`, default: `False`): If True, display a heatmap visualization showing predictions
- **max_display_tokens** (`int`, default: `15`): Maximum number of tokens to show in visualization (from the end)
- **figsize** (`tuple`, default: `(16, 10)`): Figure size for plot (width, height)
- **cmap** (`str`, default: `'viridis'`): Colormap for heatmap

**Returns:** Dict mapping `layer_idx` -> list of positions -> list of `(token_id, score, token_str)` tuples

**Structure:** `{layer_idx: [[pos_0_predictions], [pos_1_predictions], ...]}`

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
```

**Note:** Plotting requires matplotlib: `pip install matplotlib`

**Use Cases:**
- Understand how model predictions evolve through layers
- Debug model behavior at intermediate layers
- Visualize progressive refinement of predictions
- Identify where in the model certain facts are computed

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

attn_act = trace.get_activation("layers.3.attn")
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

## Version History

### 0.1.0 (Current)

- Initial release
- Core tracing functionality
- Intervention system
- Basic utility functions
- Support for any MLX model
