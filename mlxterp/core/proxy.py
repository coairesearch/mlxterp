"""
Proxy objects for transparent module wrapping and attribute access.

This module provides the proxy system that enables the clean API:
    model.layers[3].self_attn.output.save()

Key classes:
    - ModuleProxy: Wraps nn.Module to intercept __call__ and provide attribute access
    - OutputProxy: Wraps outputs to provide .save() method
    - LayerListProxy: Provides indexed access to layers
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Any, Dict, Optional, Callable
from weakref import ref


class TraceContext:
    """
    Global context for tracking the current trace operation.

    Uses a stack to support nested traces (though typically only one active).
    """
    _stack = []

    @classmethod
    def current(cls) -> Optional['TraceContext']:
        """Get the currently active trace context"""
        return cls._stack[-1] if cls._stack else None

    @classmethod
    def push(cls, ctx: 'TraceContext'):
        """Push a new context onto the stack"""
        cls._stack.append(ctx)

    @classmethod
    def pop(cls):
        """Pop the current context"""
        if cls._stack:
            cls._stack.pop()

    def __init__(self):
        self.activations: Dict[str, mx.array] = {}
        self.interventions: Dict[str, Callable] = {}
        self.saved_values: Dict[str, mx.array] = {}

    def save(self, name: str, value: mx.array):
        """Save a value for later retrieval"""
        self.saved_values[name] = value

    def should_intervene(self, name: str) -> bool:
        """Check if an intervention is registered for this module"""
        # Try multiple naming schemes to match user-provided names
        # E.g., user might specify "layers.5" but internal name is "model.model.layers.5"

        # 1. Exact match
        if name in self.interventions:
            return True

        # 2. Try removing "model." prefix
        if name.startswith("model."):
            short_name = name[6:]  # Remove "model."
            if short_name in self.interventions:
                return True

        # 3. Try removing "model.model." prefix (for mlx-lm models)
        if name.startswith("model.model."):
            short_name = name[12:]  # Remove "model.model."
            if short_name in self.interventions:
                return True

        return False

    def apply_intervention(self, name: str, value: mx.array) -> mx.array:
        """Apply the intervention function to the value"""
        # Try multiple naming schemes (same as should_intervene)

        # 1. Exact match
        if name in self.interventions:
            return self.interventions[name](value)

        # 2. Try removing "model." prefix
        if name.startswith("model."):
            short_name = name[6:]
            if short_name in self.interventions:
                return self.interventions[short_name](value)

        # 3. Try removing "model.model." prefix
        if name.startswith("model.model."):
            short_name = name[12:]
            if short_name in self.interventions:
                return self.interventions[short_name](value)

        return value


class OutputProxy:
    """
    Wraps a module output to provide .save() functionality.

    Example:
        output = model.layers[3].attn.output.save()
    """

    def __init__(self, value: Any, name: str):
        self._value = value
        self._name = name

    def save(self) -> Any:
        """
        Save this value to the current trace context.

        Returns:
            The unwrapped value
        """
        ctx = TraceContext.current()
        if ctx:
            ctx.save(self._name, self._value)
        return self._value

    def __getattr__(self, name: str):
        """Forward attribute access to the wrapped value"""
        return getattr(self._value, name)

    def __repr__(self):
        return f"OutputProxy({self._name})"


class ModuleProxy:
    """
    Wraps an nn.Module to intercept forward passes and provide attribute access.

    This is the core of the proxy system. It:
    1. Wraps the module's __call__ to capture outputs
    2. Provides attribute access to submodules
    3. Returns OutputProxy for .save() functionality

    Example:
        proxy = ModuleProxy(model.layers[0], "layers.0")
        output = proxy(x)  # Captured in trace context
    """

    def __init__(self, module: nn.Module, name: str):
        # Use object.__setattr__ to avoid recursion
        object.__setattr__(self, '_module', module)
        object.__setattr__(self, '_name', name)
        object.__setattr__(self, '_subproxies', {})

    @property
    def output(self) -> OutputProxy:
        """
        Access to the output of this module.

        Returns an OutputProxy that can be .save()'d
        """
        ctx = TraceContext.current()
        if ctx:
            # Try multiple naming schemes to find the activation
            # 1. Exact name (e.g., "layers.5")
            if self._name in ctx.activations:
                return OutputProxy(ctx.activations[self._name], f"{self._name}.output")

            # 2. With "model." prefix (e.g., "model.layers.5")
            model_prefixed = f"model.{self._name}"
            if model_prefixed in ctx.activations:
                return OutputProxy(ctx.activations[model_prefixed], f"{self._name}.output")

            # 3. With "model.model." prefix for mlx-lm models (e.g., "model.model.layers.5")
            double_prefixed = f"model.model.{self._name}"
            if double_prefixed in ctx.activations:
                return OutputProxy(ctx.activations[double_prefixed], f"{self._name}.output")

        # If no context or no activation, return a proxy for the name
        return OutputProxy(None, f"{self._name}.output")

    def __call__(self, *args, **kwargs):
        """
        Forward pass with activation capture and intervention support.
        """
        ctx = TraceContext.current()

        # Call the original module
        result = self._module(*args, **kwargs)

        if ctx:
            # Apply intervention if registered
            if ctx.should_intervene(self._name):
                result = ctx.apply_intervention(self._name, result)

            # Store activation
            ctx.activations[self._name] = result

        return result

    def __getattr__(self, name: str):
        """
        Provide attribute access to submodules.

        This enables: model.layers[0].attn
        """
        if name.startswith('_'):
            # Avoid infinite recursion for private attributes
            return object.__getattribute__(self, name)

        # Get the actual module
        module = object.__getattribute__(self, '_module')
        module_name = object.__getattribute__(self, '_name')
        subproxies = object.__getattribute__(self, '_subproxies')

        # Check if it's a submodule
        if hasattr(module, name):
            attr = getattr(module, name)

            # Check if it's a SimpleWrapper (from tracing)
            if hasattr(attr, '_wrapped_layer') and hasattr(attr, '_layer_name'):
                # It's a SimpleWrapper, wrap it with ModuleProxy
                if name not in subproxies:
                    subproxies[name] = ModuleProxy(attr, f"{module_name}.{name}")
                return subproxies[name]

            if isinstance(attr, nn.Module):
                # Create or retrieve cached proxy
                if name not in subproxies:
                    subproxies[name] = ModuleProxy(attr, f"{module_name}.{name}")
                return subproxies[name]

            # Not a module, return as-is
            return attr

        raise AttributeError(f"'{type(module).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        """
        Allow setting attributes for interventions.

        Example:
            model.layers[3].attn.output = mx.zeros_like(...)
        """
        if name.startswith('_'):
            object.__setattr__(self, name, value)
        else:
            # This is used for interventions during trace
            ctx = TraceContext.current()
            if ctx and name == 'output':
                # Setting .output directly is an intervention
                ctx.interventions[self._name] = lambda x: value
            else:
                # Set on the underlying module
                setattr(self._module, name, value)

    def __repr__(self):
        return f"ModuleProxy({self._name})"


class LayerListProxy:
    """
    Provides indexed access to model layers.

    Example:
        model.layers[3]  # Returns ModuleProxy for layer 3
    """

    def __init__(self, layers: list, base_name: str = "layers", model_ref=None, attr_path: str = None):
        self._layers = layers
        self._base_name = base_name
        self._model_ref = model_ref  # Reference to the actual model
        self._attr_path = attr_path  # Path to access layers (e.g., "model.layers")
        self._proxies = {}

    def __getitem__(self, idx: int) -> ModuleProxy:
        """
        Get a layer by index.

        Returns a ModuleProxy for the layer.
        If we're inside a trace context, this will return a proxy to the wrapped layer.
        """
        # Get the CURRENT layer (might be wrapped during tracing)
        if self._model_ref is not None and self._attr_path:
            # Navigate to get the current layer
            current_obj = self._model_ref
            for attr in self._attr_path.split('.'):
                current_obj = getattr(current_obj, attr)
            current_layer = current_obj[idx]
        else:
            # Fallback to cached layers
            current_layer = self._layers[idx]

        layer_name = f"{self._base_name}.{idx}"
        # Always create a new proxy with the current layer
        # (during tracing, this will be the SimpleWrapper)
        return ModuleProxy(current_layer, layer_name)

    def __len__(self):
        return len(self._layers)

    def __iter__(self):
        for i in range(len(self._layers)):
            yield self[i]

    def __repr__(self):
        return f"LayerListProxy(length={len(self._layers)})"
