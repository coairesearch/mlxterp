"""
Generic module resolution for different MLX model architectures.

This module provides utilities to resolve common model components (embedding,
final norm, lm_head) across different model architectures using fallback chains.
"""

import mlx.nn as nn
from typing import Optional, Tuple, Any
import warnings


class ModuleResolver:
    """
    Resolves model components across different architectures.

    Supports fallback chains for finding embedding layers, final normalization,
    and output projection (lm_head) across different model structures like
    mlx-lm models, GPT-style models, and custom architectures.

    Attributes:
        model: The wrapped model to resolve components from

    Example:
        >>> resolver = ModuleResolver(model)
        >>> embedding = resolver.get_embedding_layer()
        >>> norm = resolver.get_final_norm()
        >>> lm_head, is_tied = resolver.get_output_projection()
    """

    # Fallback chains for different components
    EMBEDDING_PATHS = [
        "model.embed_tokens",       # mlx-lm single-wrapped
        "model.model.embed_tokens", # mlx-lm double-wrapped
        "embed_tokens",             # direct
        "tok_embeddings",           # some Llama implementations
        "wte",                      # GPT-2 style
        "embeddings.word_embeddings", # BERT style
        "transformer.wte",          # GPT with transformer wrapper
    ]

    NORM_PATHS = [
        "model.norm",               # mlx-lm single-wrapped
        "model.model.norm",         # mlx-lm double-wrapped
        "norm",                     # direct
        "ln_f",                     # GPT-2 style
        "model.ln_f",               # GPT-2 with model wrapper
        "transformer.ln_f",         # GPT with transformer wrapper
        "final_layer_norm",         # Some models
    ]

    LM_HEAD_PATHS = [
        "lm_head",                  # Standard location
        "model.lm_head",            # With model wrapper
        "output",                   # Some Llama implementations
        "head",                     # Alternative naming
    ]

    def __init__(
        self,
        model: nn.Module,
        embedding_path: Optional[str] = None,
        norm_path: Optional[str] = None,
        lm_head_path: Optional[str] = None,
    ):
        """
        Initialize the module resolver.

        Args:
            model: The MLX model to resolve components from
            embedding_path: Override path for embedding layer (e.g., "my_embed")
            norm_path: Override path for final normalization (e.g., "my_norm")
            lm_head_path: Override path for output projection (e.g., "my_lm_head")
        """
        self.model = model
        self._embedding_path = embedding_path
        self._norm_path = norm_path
        self._lm_head_path = lm_head_path

        # Cache resolved modules
        self._embedding_cache: Optional[Tuple[Any, str]] = None
        self._norm_cache: Optional[Tuple[Any, str]] = None
        self._lm_head_cache: Optional[Tuple[Any, str, bool]] = None

    def _resolve_path(self, path: str) -> Optional[Any]:
        """
        Resolve a dotted path to a module.

        Args:
            path: Dotted path like "model.embed_tokens"

        Returns:
            The module at that path, or None if not found
        """
        try:
            obj = self.model
            for part in path.split('.'):
                obj = getattr(obj, part)
            return obj
        except AttributeError:
            return None

    def _find_module(
        self,
        override_path: Optional[str],
        fallback_paths: list,
        component_name: str,
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Find a module using override path or fallback chain.

        Args:
            override_path: User-specified path (takes priority)
            fallback_paths: List of paths to try
            component_name: Name for error messages

        Returns:
            Tuple of (module, resolved_path) or (None, None)
        """
        # Try override first
        if override_path is not None:
            module = self._resolve_path(override_path)
            if module is not None:
                return module, override_path
            warnings.warn(
                f"Override path '{override_path}' for {component_name} not found. "
                f"Falling back to automatic resolution."
            )

        # Try fallback chain
        for path in fallback_paths:
            module = self._resolve_path(path)
            if module is not None:
                return module, path

        return None, None

    def get_embedding_layer(self) -> Optional[nn.Module]:
        """
        Get the token embedding layer.

        Returns:
            The embedding module, or None if not found
        """
        if self._embedding_cache is not None:
            return self._embedding_cache[0]

        module, path = self._find_module(
            self._embedding_path,
            self.EMBEDDING_PATHS,
            "embedding"
        )

        if module is not None:
            self._embedding_cache = (module, path)

        return module

    def get_embedding_path(self) -> Optional[str]:
        """Get the resolved path to the embedding layer."""
        if self._embedding_cache is None:
            self.get_embedding_layer()
        return self._embedding_cache[1] if self._embedding_cache else None

    def get_final_norm(self) -> Optional[nn.Module]:
        """
        Get the final layer normalization.

        Returns:
            The norm module, or None if not found
        """
        if self._norm_cache is not None:
            return self._norm_cache[0]

        module, path = self._find_module(
            self._norm_path,
            self.NORM_PATHS,
            "final_norm"
        )

        if module is not None:
            self._norm_cache = (module, path)

        return module

    def get_final_norm_path(self) -> Optional[str]:
        """Get the resolved path to the final norm."""
        if self._norm_cache is None:
            self.get_final_norm()
        return self._norm_cache[1] if self._norm_cache else None

    def get_lm_head(self) -> Optional[nn.Module]:
        """
        Get the output projection (lm_head) layer.

        Returns:
            The lm_head module, or None if not found
        """
        _, _, _ = self.get_output_projection()
        if self._lm_head_cache is not None:
            return self._lm_head_cache[0]
        return None

    def get_output_projection(self) -> Tuple[Optional[nn.Module], Optional[str], bool]:
        """
        Get the output projection layer and determine if weights are tied.

        Many language models (like Llama) use weight tying where the embedding
        weights are reused (transposed) for the output projection. This method
        detects both cases.

        Returns:
            Tuple of (module, path, is_weight_tied)
            - module: The projection layer (lm_head or embedding)
            - path: Resolved path to the module
            - is_weight_tied: True if using embedding weights for projection
        """
        if self._lm_head_cache is not None:
            return self._lm_head_cache

        # First try to find explicit lm_head
        module, path = self._find_module(
            self._lm_head_path,
            self.LM_HEAD_PATHS,
            "lm_head"
        )

        if module is not None:
            self._lm_head_cache = (module, path, False)
            return module, path, False

        # Fall back to weight-tied embedding
        embedding = self.get_embedding_layer()
        if embedding is not None:
            embed_path = self.get_embedding_path()
            self._lm_head_cache = (embedding, embed_path, True)
            return embedding, embed_path, True

        self._lm_head_cache = (None, None, False)
        return None, None, False

    def clear_cache(self):
        """
        Clear the resolved module cache.

        Call this method after modifying the model's module structure
        (e.g., replacing layers, changing attributes) to force re-resolution
        on the next access.

        Example:
            >>> resolver = ModuleResolver(model)
            >>> embedding = resolver.get_embedding_layer()  # Cached
            >>>
            >>> # Modify model structure
            >>> model.my_new_embed = nn.Embedding(1000, 64)
            >>>
            >>> # Clear cache to pick up changes
            >>> resolver.clear_cache()
            >>> embedding = resolver.get_embedding_layer()  # Re-resolved
        """
        self._embedding_cache = None
        self._norm_cache = None
        self._lm_head_cache = None


# Utility function for layer key normalization
def normalize_layer_key(key: str) -> str:
    """
    Normalize an activation key by removing common model prefixes.

    Converts keys like "model.model.layers.0" to "layers.0" for
    consistent access across different model wrapping styles.

    Args:
        key: The activation key to normalize

    Returns:
        Normalized key without model prefixes

    Example:
        >>> normalize_layer_key("model.model.layers.0.self_attn")
        'layers.0.self_attn'
        >>> normalize_layer_key("model.layers.0.mlp")
        'layers.0.mlp'
        >>> normalize_layer_key("layers.0")
        'layers.0'
    """
    if key.startswith("model.model."):
        return key[12:]
    if key.startswith("model."):
        return key[6:]
    return key


def find_layer_key_pattern(
    activations: dict,
    layer_idx: int,
    component: Optional[str] = None,
) -> Optional[str]:
    """
    Find the correct activation key pattern for a layer.

    Uses two strategies:
    1. Try common patterns directly
    2. Normalize all keys and match on suffix (for deeply nested wrappers)

    Args:
        activations: Dict of activation keys
        layer_idx: Layer index to find
        component: Optional component suffix (e.g., "self_attn")

    Returns:
        The matching key, or None if not found
    """
    suffix = f".{component}" if component else ""
    target_suffix = f"layers.{layer_idx}{suffix}"

    # Strategy 1: Try common patterns directly
    patterns = [
        f"model.model.layers.{layer_idx}{suffix}",  # mlx-lm double-wrapped
        f"model.layers.{layer_idx}{suffix}",        # mlx-lm single-wrapped
        f"layers.{layer_idx}{suffix}",              # direct
    ]

    for pattern in patterns:
        if pattern in activations:
            return pattern

    # Strategy 2: Normalize keys and match on suffix
    # This handles deeply nested wrappers (model.model.model.layers.0, etc.)
    for key in activations:
        normalized = normalize_layer_key(key)
        if normalized == target_suffix:
            return key
        # Also try matching on suffix directly
        if key.endswith(target_suffix):
            return key

    return None
