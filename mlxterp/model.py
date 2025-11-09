"""
Main InterpretableModel class - entry point for mlxterp.

Provides a clean, intuitive API for mechanistic interpretability on MLX models.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Dict, Callable, Any, Union, List

from .core import Trace, ModuleProxy, LayerListProxy
from .tokenization import TokenizerMixin
from .analysis import AnalysisMixin
from .sae_mixin import SAEMixin


class InterpretableModel(TokenizerMixin, AnalysisMixin, SAEMixin):
    """
    Wraps any MLX model to provide interpretability features.

    This is the main entry point for mlxterp. It provides:
    - Context manager-based tracing: `with model.trace(input):`
    - Direct attribute access to layers: `model.layers[3].attn.output`
    - Activation capture with `.save()`
    - Intervention support
    - Tokenization utilities (via TokenizerMixin)
    - Analysis methods: logit_lens, activation_patching (via AnalysisMixin)
    - SAE training and analysis: train_sae, load_sae (via SAEMixin)

    Example:
        >>> from mlxterp import InterpretableModel
        >>> model = InterpretableModel(base_model, tokenizer)
        >>>
        >>> with model.trace("Hello world"):
        >>>     attn_out = model.layers[3].attn.output.save()
        >>>     logits = model.output.save()

    Attributes:
        model: The wrapped MLX model
        tokenizer: Optional tokenizer for text inputs
        layers: LayerListProxy for indexed access to model layers
    """

    def __init__(
        self,
        model: Union[nn.Module, str],
        tokenizer: Optional[Any] = None,
        layer_attr: str = "layers",
    ):
        """
        Initialize an InterpretableModel.

        Args:
            model: Either an nn.Module instance or a model name/path string.
                   If string, attempts to load via mlx_lm or other loaders.
            tokenizer: Optional tokenizer for processing text inputs
            layer_attr: Name of the attribute containing model layers (default: "layers")

        Raises:
            ValueError: If model is a string but cannot be loaded
            AttributeError: If layer_attr doesn't exist on the model
        """
        # Handle model loading
        if isinstance(model, str):
            self.model = self._load_model(model)
            # If tokenizer not provided, try to load it with the model
            if tokenizer is None:
                tokenizer = self._try_load_tokenizer(model)
        else:
            self.model = model

        self.tokenizer = tokenizer
        self._layer_attr = layer_attr

        # Discover and wrap layers
        self._setup_layer_access()

        # Cache for module proxies
        self._module_proxies: Dict[str, ModuleProxy] = {}

    def _load_model(self, model_name: str) -> nn.Module:
        """
        Load a model from a name or path.

        Tries multiple loading strategies:
        1. mlx_lm.load() for language models
        2. Add more loaders as needed

        Args:
            model_name: Model name or path

        Returns:
            Loaded nn.Module

        Raises:
            ValueError: If model cannot be loaded
        """
        errors = []

        # Try mlx_lm
        try:
            from mlx_lm import load
            model, tokenizer_obj = load(model_name)
            # Store tokenizer if we loaded one
            if self.tokenizer is None:
                self.tokenizer = tokenizer_obj
            return model
        except ImportError as e:
            errors.append("mlx-lm not installed. Install with: uv add mlx-lm (or pip install mlx-lm)")
        except Exception as e:
            errors.append(f"mlx-lm failed to load model: {str(e)}")

        # Add other loading strategies here
        # - HuggingFace transformers
        # - Direct checkpoint loading
        # etc.

        error_msg = f"Could not load model '{model_name}'.\n"
        error_msg += "Tried the following strategies:\n"
        for i, err in enumerate(errors, 1):
            error_msg += f"  {i}. {err}\n"
        error_msg += "\nAlternatively, load the model manually and pass it to InterpretableModel(model)."

        raise ValueError(error_msg)

    def _try_load_tokenizer(self, model_name: str) -> Optional[Any]:
        """Try to load a tokenizer for the model"""
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(model_name)
        except:
            return None

    def _setup_layer_access(self):
        """
        Set up proxied access to model layers.

        Detects the layer structure and creates a LayerListProxy.
        """
        # Try to find layers, checking multiple possible paths
        layer_paths = [
            self._layer_attr,  # Direct attribute (e.g., "layers")
            f"model.{self._layer_attr}",  # Nested (e.g., "model.layers" for mlx-lm models)
        ]

        layers = None
        actual_path = None

        for path in layer_paths:
            try:
                obj = self.model
                parts = path.split('.')
                for part in parts:
                    obj = getattr(obj, part)
                layers = obj
                actual_path = path
                break
            except AttributeError:
                continue

        if layers is None:
            raise AttributeError(
                f"Could not find layers attribute. Tried: {layer_paths}. "
                f"Specify the correct path with layer_attr parameter."
            )

        # Check if it's a list/tuple or a single module with indexed access
        if isinstance(layers, (list, tuple)):
            self.layers = LayerListProxy(
                list(layers),
                base_name=self._layer_attr,
                model_ref=self.model,
                attr_path=actual_path
            )
        elif hasattr(layers, '__getitem__'):
            # It's something with __getitem__ (like nn.Sequential)
            # Convert to list
            layer_list = []
            i = 0
            while True:
                try:
                    layer_list.append(layers[i])
                    i += 1
                except (IndexError, KeyError):
                    break

            self.layers = LayerListProxy(
                layer_list,
                base_name=self._layer_attr,
                model_ref=self.model,
                attr_path=actual_path
            )
        else:
            raise AttributeError(
                f"Model's '{self._layer_attr}' attribute is not iterable"
            )

    def trace(
        self,
        inputs: Union[str, List[str], mx.array, List[int]],
        interventions: Optional[Dict[str, Callable]] = None,
    ) -> Trace:
        """
        Create a tracing context for the model.

        Usage:
            with model.trace("Hello world") as t:
                output = model.layers[3].attn.output.save()

            # After context exits, access saved values
            print(t.saved_values)

        Args:
            inputs: Input data - can be:
                - String: "Hello world" (requires tokenizer)
                - List of strings: ["Hello", "World"] (batch, requires tokenizer)
                - mx.array: Token array
                - List[int]: Token list
            interventions: Optional dict mapping module names to intervention functions
                Example: {"layers.3.attn": lambda x: x * 0.5}

        Returns:
            Trace context manager
        """
        return Trace(
            model_forward=self._forward,
            inputs=inputs,
            tokenizer=self.tokenizer,
            interventions=interventions,
            interpretable_model=self,
        )

    def _forward(self, inputs: mx.array) -> mx.array:
        """
        Execute the model forward pass.

        This is called by the Trace context manager.
        """
        return self.model(inputs)

    def __call__(self, *args, **kwargs):
        """
        Direct forward pass without tracing.

        For normal model usage without interpretability features.
        """
        return self.model(*args, **kwargs)

    def named_modules(self):
        """
        Iterator over all modules in the model.

        Yields:
            (name, module) tuples

        Example:
            for name, module in model.named_modules():
                print(name, type(module))
        """
        if hasattr(self.model, 'named_modules'):
            return self.model.named_modules()
        else:
            # Fallback: just yield the model itself
            return [("", self.model)]

    def parameters(self):
        """Get model parameters (delegates to wrapped model)"""
        return self.model.parameters()

    def trainable_parameters(self):
        """Get trainable parameters (delegates to wrapped model)"""
        if hasattr(self.model, 'trainable_parameters'):
            return self.model.trainable_parameters()
        return self.parameters()

    @property
    def output(self):
        """
        Access to model output (for use within trace context).

        Returns an OutputProxy that can be saved.

        Example:
            with model.trace(input):
                output = model.output.save()
        """
        from .core import OutputProxy, TraceContext
        # Get the current trace context and return its output
        ctx = TraceContext.current()
        if ctx and "__model_output__" in ctx.activations:
            return OutputProxy(value=ctx.activations["__model_output__"], name="__model_output__")
        else:
            # Not in trace context - return placeholder
            return OutputProxy(value=None, name="__model_output__")

    def __repr__(self):
        """String representation of the model"""
        model_type = type(self.model).__name__
        num_layers = len(self.layers) if hasattr(self, 'layers') else 'unknown'
        has_tokenizer = self.tokenizer is not None
        return (
            f"InterpretableModel(\n"
            f"  model={model_type},\n"
            f"  layers={num_layers},\n"
            f"  tokenizer={'✓' if has_tokenizer else '✗'}\n"
            f")"
        )
