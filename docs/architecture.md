# Architecture

This document explains the design and implementation of mlxterp.

## Design Philosophy

mlxterp follows these core principles:

1. **Simplicity First**: Make mechanistic interpretability accessible with minimal code
2. **Clean API**: Context managers and direct attribute access (inspired by nnterp)
3. **Model Agnostic**: Generic wrapping works with any MLX model (inspired by nnsight)
4. **MLX Native**: Leverage MLX's unique features (lazy evaluation, unified memory)
5. **Minimal Abstractions**: Avoid unnecessary complexity and boilerplate

## System Architecture

```
┌─────────────────────────────────────────┐
│        InterpretableModel               │
│  (User-facing API)                      │
└──────────────┬──────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼─────┐  ┌─────▼──────┐
│   Trace    │  │LayerListProxy│
│(Context Mgr)│  │(Layer Access)│
└──────┬─────┘  └─────┬───────┘
       │               │
       └───────┬───────┘
               │
        ┌──────▼───────┐
        │ TraceContext │
        │(Global State)│
        └──────┬───────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼──┐  ┌───▼────┐
│Proxies│  │Cache│  │Intervene│
└───────┘  └─────┘  └────────┘
```

## Core Components

### 1. InterpretableModel

**File**: `mlxterp/model.py`

The main entry point for users. Provides:

- Model loading (from string or nn.Module)
- Layer discovery and wrapping
- Trace context creation
- Delegation to underlying model

**Key Methods**:
- `trace()`: Create tracing context
- `named_modules()`: Iterate over modules
- `parameters()`: Access parameters

**Design Decision**: Uses composition over inheritance - wraps models rather than subclassing.

### 2. Proxy System

**File**: `mlxterp/core/proxy.py`

Implements transparent wrapping for clean attribute access.

#### ModuleProxy

Wraps `nn.Module` instances to intercept forward passes:

```python
class ModuleProxy:
    def __call__(self, *args, **kwargs):
        # Call original module
        result = self._module(*args, **kwargs)

        # Capture in context
        ctx = TraceContext.current()
        if ctx:
            ctx.activations[self._name] = result

        return result
```

**Key Features**:
- Transparent wrapping (user doesn't see proxy)
- Automatic activation capture
- Intervention support
- Lazy submodule wrapping

#### OutputProxy

Provides `.save()` functionality:

```python
class OutputProxy:
    def save(self):
        ctx = TraceContext.current()
        ctx.save(self._name, self._value)
        return self._value
```

#### LayerListProxy

Provides indexed access to layers:

```python
model.layers[3]  # Returns ModuleProxy
model.layers[3].self_attn.output  # Returns OutputProxy (use self_attn for mlx-lm models)
```

### 3. Trace Context

**File**: `mlxterp/core/trace.py`

Context manager for tracing execution.

#### Lifecycle

1. **`__enter__`**: Setup context, push to global stack, execute forward pass
2. **User code**: Access activations, call `.save()` on outputs
3. **`__exit__`**: Copy saved values and activations, restore layers, pop context

**Design Decision**: Forward pass executes in `__enter__` so activations are immediately available inside the context. Saved values and activations are copied in `__exit__` so they remain available after the context.

#### Global State Management

Uses a stack to support nested traces (though typically only one active):

```python
class TraceContext:
    _stack = []

    @classmethod
    def current(cls):
        return cls._stack[-1] if cls._stack else None
```

### 4. Intervention System

**File**: `mlxterp/core/intervention.py`

Interventions are simply functions: `(mx.array) -> mx.array`

**Design Decision**: Functional approach over class hierarchies for simplicity.

#### Pre-built Interventions

- `zero_out`: Set to zeros
- `scale(factor)`: Multiply by constant
- `add_vector(vec)`: Add steering vector
- `replace_with(value)`: Replace with fixed value
- `clamp(min, max)`: Clamp to range
- `noise(std)`: Add Gaussian noise

#### Composition

```python
combined = InterventionComposer() \
    .add(scale(0.5)) \
    .add(noise(0.1)) \
    .build()
```

### 5. Caching System

**File**: `mlxterp/core/cache.py`

Simple dictionary-based caching with utility functions:

- `ActivationCache`: Storage container
- `collect_activations()`: Helper for common patterns

**Design Decision**: Simple dict storage rather than complex cache invalidation - rely on MLX's lazy evaluation.

## MLX Integration

### Leveraging MLX Features

#### 1. Lazy Evaluation

MLX arrays are lazy - computation deferred until `mx.eval()`:

```python
# These don't execute immediately
a = mx.random.normal((100, 100))
b = mx.matmul(a, a)
c = mx.sum(b)

# Execute now
mx.eval(c)
```

**How mlxterp uses it**:
- Activations captured as lazy arrays
- Only materialized when accessed via `.save()`
- Efficient memory usage

#### 2. Unified Memory

MLX arrays live in shared memory - no CPU/GPU transfers:

```python
# Automatically available on both CPU and GPU
arr = mx.array([1, 2, 3])
# No need for .to('cuda') or .cpu()
```

**How mlxterp uses it**:
- Zero-copy activation caching
- Efficient intervention application
- No device management needed

#### 3. Dynamic Graphs

Computation graphs built dynamically (like PyTorch eager mode):

```python
# Graph constructed on-the-fly
for layer in model.layers:
    x = layer(x)  # Each call adds to graph
```

**How mlxterp uses it**:
- Easy to intercept operations
- No need for graph tracing hooks
- Simple proxy-based wrapping works

#### 4. Module System

MLX modules are dict-based:

```python
class Module(dict):
    def __init__(self):
        self._no_grad = set()
        self._training = True
```

**How mlxterp uses it**:
- Easy parameter inspection via `tree_flatten()`
- Natural module traversal via `named_modules()`
- Simple attribute access patterns

## Implementation Patterns

### Pattern 1: Proxy-Based Interception

Instead of monkey-patching or hooks:

```python
# MLX module
original_module = nn.Linear(128, 128)

# Wrap with proxy
proxy = ModuleProxy(original_module, "layer_name")

# Calls are intercepted transparently
output = proxy(input)  # Captured in trace context
```

**Benefits**:
- No modification of original module
- Type-safe (proxy delegates to original)
- Easy to add/remove

### Pattern 2: Context Manager Pattern

Clean setup/teardown:

```python
class Trace:
    def __enter__(self):
        # Setup: Create and push context
        self.context = TraceContext()
        TraceContext.push(self.context)

        # Execute: Run model forward pass immediately
        # This allows users to access activations inside the with block
        self.output = self.model_forward(self.inputs)
        return self

    def __exit__(self, *args):
        # Copy saved_values from context (for values saved inside the block)
        self.saved_values = self.context.saved_values.copy()

        # Cleanup: Pop context
        TraceContext.pop()
```

**Benefits**:
- Guaranteed cleanup
- Clear lifecycle
- Pythonic API

### Pattern 3: Functional Interventions

Simple functions over classes:

```python
# Intervention is just a function
def scale(factor: float):
    def _scale(x: mx.array) -> mx.array:
        return x * factor
    return _scale

# Use it
interventions = {"layers.3": scale(0.5)}
```

**Benefits**:
- Easy to compose
- No inheritance complexity
- Clear semantics

### Pattern 4: Lazy Proxy Creation

Proxies created on-demand:

```python
class ModuleProxy:
    def __getattr__(self, name):
        if name not in self._subproxies:
            # Create proxy only when accessed
            attr = getattr(self._module, name)
            if isinstance(attr, nn.Module):
                self._subproxies[name] = ModuleProxy(attr, f"{self._name}.{name}")
        return self._subproxies[name]
```

**Benefits**:
- Efficient memory usage
- Automatic submodule discovery
- No upfront cost

## Comparison with Other Libraries

### vs TransformerLens

| Aspect | mlxterp | TransformerLens |
|--------|---------|-----------------|
| Approach | Generic wrapping | Model-specific classes |
| Models | Any MLX model | Specific architectures |
| API | Context managers | Direct methods |
| Framework | MLX | PyTorch |
| Lines of Code | ~600 | ~10,000+ |

**Trade-offs**:
- mlxterp: More flexible, less feature-complete
- TransformerLens: More features, less flexible

### vs nnsight

| Aspect | mlxterp | nnsight |
|--------|---------|---------|
| Approach | Similar (generic) | Generic wrapping |
| API Style | Context managers | Context managers |
| Framework | MLX | PyTorch |
| Focus | Simplicity | Completeness |

**Inspiration**: mlxterp closely follows nnsight's design philosophy.

### vs nnterp

| Aspect | mlxterp | nnterp |
|--------|---------|---------|
| Approach | Similar | Standardized naming |
| API | Very similar | Clean context managers |
| Framework | MLX | PyTorch/nnsight |
| Focus | Apple Silicon | Model unification |

**Inspiration**: mlxterp adopts nnterp's clean API design.

## Performance Considerations

### Memory Efficiency

1. **Lazy Evaluation**: Activations not computed until needed
2. **Selective Saving**: Only save what you explicitly `.save()`
3. **Unified Memory**: No device transfers
4. **Batch Processing**: Utilities for large datasets

### Computational Efficiency

1. **Minimal Overhead**: Proxy calls are lightweight
2. **Metal Acceleration**: Inherits MLX's Metal optimizations
3. **Zero-Copy Operations**: Interventions don't copy data unnecessarily

### Best Practices

```python
# ✅ Good: Only save needed activations
with model.trace(input):
    important = [3, 8, 12]
    acts = {i: model.layers[i].output.save() for i in important}

# ❌ Avoid: Saving everything
with model.trace(input):
    all_acts = [model.layers[i].output.save() for i in range(100)]
```

## Extension Points

### Custom Interventions

Easy to add:

```python
def my_intervention(x: mx.array) -> mx.array:
    # Your logic
    return transformed_x

# Use immediately
with model.trace(input, interventions={"layers.3": my_intervention}):
    output = model.output.save()
```

### Custom Utilities

Built on core components:

```python
from mlxterp.core import collect_activations

def my_analysis_function(model, prompts):
    cache = collect_activations(model, prompts)
    # Your analysis
    return results
```

### Integration with Other Tools

Compatible with MLX ecosystem:

```python
# Use with mlx_lm
from mlx_lm import load
model_base, tokenizer = load("model-name")
model = InterpretableModel(model_base, tokenizer)

# Use with mlx training
# Access underlying model
model.model.train()
optimizer = optim.Adam(model.parameters())
```

## Future Enhancements

Potential additions (maintaining simplicity):

1. **Visualization Tools**: Attention pattern plotting
2. **Circuit Discovery**: Automated path finding
3. **Sparse Autoencoders**: Feature extraction support
4. **Logit Lens**: Intermediate decoding utilities
5. **Benchmarks**: Standard interpretability tasks

## Summary

mlxterp achieves its goals through:

- **Clean abstraction layers**: Each component has clear responsibility
- **MLX integration**: Leverages framework strengths
- **Simple patterns**: Proxies, context managers, functions
- **Minimal code**: ~600 lines of core functionality
- **Extensible design**: Easy to add custom behavior

The architecture prioritizes **simplicity** and **usability** over feature completeness, making mechanistic interpretability accessible to researchers on Apple Silicon.
