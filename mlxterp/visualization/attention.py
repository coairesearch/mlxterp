"""
Attention Visualization Module.

Provides functions to visualize attention patterns from mlxterp traces.
Supports multiple backends: CircuitsVis (recommended), Plotly, and matplotlib.

Example:
    >>> from mlxterp import InterpretableModel
    >>> from mlxterp.visualization import attention_from_trace
    >>>
    >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
    >>>
    >>> with model.trace("The Eiffel Tower is in Paris") as trace:
    >>>     pass
    >>>
    >>> tokens = model.to_str_tokens("The Eiffel Tower is in Paris")
    >>> attention_from_trace(trace, tokens, layers=[0, 5, 10])
"""

import mlx.core as mx
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class AttentionVisualizationConfig:
    """Configuration for attention visualization."""

    # Display settings
    colorscale: str = "Blues"
    mask_upper_tri: bool = True  # Mask future tokens (causal)
    show_colorbar: bool = True
    figure_width: int = 800
    figure_height: int = 600

    # Head selection
    heads_per_row: int = 4  # For multi-head grid view

    # Interactivity
    hover_info: bool = True

    # Backend selection
    backend: str = "auto"  # "auto", "circuitsvis", "plotly", "matplotlib"


def get_attention_patterns(
    trace,
    layers: Optional[Union[int, List[int]]] = None,
) -> Dict[int, np.ndarray]:
    """
    Extract attention patterns from an mlxterp trace.

    Args:
        trace: mlxterp Trace object with captured activations
        layers: Layer index, list of indices, or None for all layers

    Returns:
        Dict mapping layer index to attention weights array.
        Each array has shape (batch, num_heads, seq_len, seq_len)

    Example:
        >>> with model.trace("Hello world") as trace:
        >>>     pass
        >>> patterns = get_attention_patterns(trace, layers=[0, 5, 10])
        >>> print(patterns[0].shape)  # (1, num_heads, seq_len, seq_len)
    """
    # Normalize layers to list
    if layers is None:
        # Find all layers with attention weights
        layer_set = set()
        for key in trace.activations.keys():
            if ".attention_weights" in key:
                # Extract layer number from key like "model.model.layers.5.self_attn.attention_weights"
                import re
                match = re.search(r'layers\.(\d+)', key)
                if match:
                    layer_set.add(int(match.group(1)))
        layers = sorted(layer_set)
    elif isinstance(layers, int):
        layers = [layers]

    patterns = {}

    for layer_idx in layers:
        # Find the attention weights key for this layer
        key = None
        for k in trace.activations.keys():
            if f"layers.{layer_idx}.self_attn.attention_weights" in k:
                key = k
                break
            # Also check for "attn" instead of "self_attn" (GPT-2 style)
            if f"layers.{layer_idx}.attn.attention_weights" in k:
                key = k
                break

        if key is None:
            continue

        weights = trace.activations[key]
        mx.eval(weights)
        patterns[layer_idx] = np.array(weights)

    return patterns


def attention_heatmap(
    attention: np.ndarray,
    tokens: List[str],
    head_idx: int = 0,
    title: Optional[str] = None,
    config: Optional[AttentionVisualizationConfig] = None,
    backend: str = "auto",
) -> Any:
    """
    Create a heatmap visualization of attention patterns.

    Args:
        attention: Attention weights array of shape (batch, num_heads, seq_len, seq_len)
                  or (num_heads, seq_len, seq_len) or (seq_len, seq_len)
        tokens: List of token strings
        head_idx: Which attention head to visualize (if multiple)
        title: Optional title for the visualization
        config: Visualization configuration
        backend: Visualization backend ("auto", "plotly", "matplotlib")

    Returns:
        Visualization object (Figure for plotly/matplotlib)

    Example:
        >>> patterns = get_attention_patterns(trace, layers=[5])
        >>> attention_heatmap(patterns[5], tokens, head_idx=7, title="Layer 5, Head 7")
    """
    config = config or AttentionVisualizationConfig()

    # Normalize attention shape
    if attention.ndim == 4:
        # (batch, heads, seq_q, seq_k) -> select first batch and specified head
        attn_2d = attention[0, head_idx]
    elif attention.ndim == 3:
        # (heads, seq_q, seq_k) -> select specified head
        attn_2d = attention[head_idx]
    elif attention.ndim == 2:
        # (seq_q, seq_k) -> use directly
        attn_2d = attention
    else:
        raise ValueError(f"Unexpected attention shape: {attention.shape}")

    # Apply causal mask visualization (show as white/zero)
    if config.mask_upper_tri:
        mask = np.triu(np.ones_like(attn_2d), k=1)
        attn_2d = np.where(mask, np.nan, attn_2d)

    # Determine backend
    if backend == "auto":
        backend = _detect_backend()

    if backend == "plotly":
        return _heatmap_plotly(attn_2d, tokens, title, config)
    elif backend == "matplotlib":
        return _heatmap_matplotlib(attn_2d, tokens, title, config)
    else:
        raise ValueError(f"Unknown backend: {backend}")


def attention_from_trace(
    trace,
    tokens: List[str],
    layers: Optional[Union[int, List[int]]] = None,
    heads: Optional[List[Tuple[int, int]]] = None,
    mode: str = "grid",
    head_notation: str = "LH",
    config: Optional[AttentionVisualizationConfig] = None,
) -> Any:
    """
    Visualize attention patterns directly from an mlxterp trace.

    This is the main entry point for attention visualization, similar to
    CircuitsVis's from_cache API.

    Args:
        trace: mlxterp Trace object from model.trace() context
        tokens: List of token strings (use model.to_str_tokens())
        layers: Layer index, list of indices, or None (all layers)
        heads: List of (layer, head) tuples to highlight specific heads
        mode: Visualization mode:
              - "grid": Grid of all heads for selected layers
              - "single": Single heatmap (first layer, first head)
              - "interactive": Interactive CircuitsVis-style (if available)
        head_notation: "LH" for L0H1 format, "dot" for 0.1 format
        config: Visualization configuration

    Returns:
        Visualization object (displays in Jupyter)

    Example:
        >>> with model.trace("The Eiffel Tower is in Paris") as trace:
        >>>     pass
        >>> tokens = model.to_str_tokens("The Eiffel Tower is in Paris")
        >>> attention_from_trace(trace, tokens, layers=[0, 5, 10])
    """
    config = config or AttentionVisualizationConfig()

    # Get attention patterns
    patterns = get_attention_patterns(trace, layers)

    if not patterns:
        raise ValueError(
            "No attention weights found in trace. "
            "Make sure attention modules are being traced correctly."
        )

    # Determine backend
    backend = config.backend
    if backend == "auto":
        backend = _detect_backend()

    # Try CircuitsVis first for interactive mode
    if mode == "interactive" and backend in ("auto", "circuitsvis"):
        try:
            return _attention_circuitsvis(patterns, tokens, heads, head_notation, config)
        except ImportError:
            # Fall back to plotly grid
            mode = "grid"
            backend = "plotly"

    if mode == "single":
        # Single heatmap of first layer, first head
        first_layer = min(patterns.keys())
        return attention_heatmap(
            patterns[first_layer],
            tokens,
            head_idx=0,
            title=f"Layer {first_layer}, Head 0",
            config=config,
            backend=backend,
        )

    elif mode == "grid":
        # Grid of all heads
        if backend == "plotly":
            return _attention_grid_plotly(patterns, tokens, heads, head_notation, config)
        else:
            return _attention_grid_matplotlib(patterns, tokens, heads, head_notation, config)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def _detect_backend() -> str:
    """Detect best available visualization backend."""
    # Try CircuitsVis first
    try:
        import circuitsvis
        return "circuitsvis"
    except ImportError:
        pass

    # Try Plotly
    try:
        import plotly
        return "plotly"
    except ImportError:
        pass

    # Fall back to matplotlib
    return "matplotlib"


def _heatmap_plotly(
    attention: np.ndarray,
    tokens: List[str],
    title: Optional[str],
    config: AttentionVisualizationConfig,
) -> Any:
    """Create attention heatmap with Plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly not installed. Install with: pip install plotly")

    fig = go.Figure(data=go.Heatmap(
        z=attention,
        x=tokens,
        y=tokens,
        colorscale=config.colorscale,
        showscale=config.show_colorbar,
        hoverongaps=False,
    ))

    fig.update_layout(
        title=title or "Attention Pattern",
        xaxis_title="Key Tokens",
        yaxis_title="Query Tokens",
        width=config.figure_width,
        height=config.figure_height,
        yaxis=dict(autorange="reversed"),  # Flip y-axis for attention convention
    )

    return fig


def _heatmap_matplotlib(
    attention: np.ndarray,
    tokens: List[str],
    title: Optional[str],
    config: AttentionVisualizationConfig,
) -> Any:
    """Create attention heatmap with matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(10, 8))

    # Use colorscale as-is (matplotlib is case-sensitive)
    im = ax.imshow(attention, cmap=config.colorscale)

    # Set ticks
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha='right')
    ax.set_yticklabels(tokens)

    # Labels
    ax.set_xlabel("Key Tokens")
    ax.set_ylabel("Query Tokens")
    ax.set_title(title or "Attention Pattern")

    if config.show_colorbar:
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


def _attention_grid_plotly(
    patterns: Dict[int, np.ndarray],
    tokens: List[str],
    heads: Optional[List[Tuple[int, int]]],
    head_notation: str,
    config: AttentionVisualizationConfig,
) -> Any:
    """Create grid of attention heatmaps with Plotly."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError("Plotly not installed. Install with: pip install plotly")

    # Determine which heads to show
    if heads is not None:
        head_list = heads
    else:
        # Show all heads from all layers
        head_list = []
        for layer_idx, attn in patterns.items():
            num_heads = attn.shape[1]
            for h in range(num_heads):
                head_list.append((layer_idx, h))

    # Limit to reasonable number
    max_heads = 16
    if len(head_list) > max_heads:
        head_list = head_list[:max_heads]

    # Calculate grid dimensions
    n_heads = len(head_list)
    cols = min(config.heads_per_row, n_heads)
    rows = (n_heads + cols - 1) // cols

    # Create subplot titles
    subplot_titles = []
    for layer_idx, head_idx in head_list:
        if head_notation == "LH":
            subplot_titles.append(f"L{layer_idx}H{head_idx}")
        else:
            subplot_titles.append(f"{layer_idx}.{head_idx}")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.05,
        vertical_spacing=0.1,
    )

    for i, (layer_idx, head_idx) in enumerate(head_list):
        row = i // cols + 1
        col = i % cols + 1

        attn = patterns[layer_idx][0, head_idx]  # First batch

        # Apply causal mask
        if config.mask_upper_tri:
            mask = np.triu(np.ones_like(attn), k=1)
            attn = np.where(mask, np.nan, attn)

        fig.add_trace(
            go.Heatmap(
                z=attn,
                x=tokens,
                y=tokens,
                colorscale=config.colorscale,
                showscale=(i == 0),  # Only show colorbar for first
                hoverongaps=False,
            ),
            row=row, col=col
        )

        # Flip y-axis
        fig.update_yaxes(autorange="reversed", row=row, col=col)

    fig.update_layout(
        title="Attention Patterns",
        width=config.figure_width * (cols / 2),
        height=config.figure_height * (rows / 2),
    )

    return fig


def _attention_grid_matplotlib(
    patterns: Dict[int, np.ndarray],
    tokens: List[str],
    heads: Optional[List[Tuple[int, int]]],
    head_notation: str,
    config: AttentionVisualizationConfig,
) -> Any:
    """Create grid of attention heatmaps with matplotlib."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib not installed. Install with: pip install matplotlib")

    # Determine which heads to show
    if heads is not None:
        head_list = heads
    else:
        head_list = []
        for layer_idx, attn in patterns.items():
            num_heads = attn.shape[1]
            for h in range(num_heads):
                head_list.append((layer_idx, h))

    max_heads = 16
    if len(head_list) > max_heads:
        head_list = head_list[:max_heads]

    n_heads = len(head_list)
    cols = min(config.heads_per_row, n_heads)
    rows = (n_heads + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))

    if n_heads == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    for i, (layer_idx, head_idx) in enumerate(head_list):
        row = i // cols
        col = i % cols
        ax = axes[row, col]

        attn = patterns[layer_idx][0, head_idx]

        if config.mask_upper_tri:
            mask = np.triu(np.ones_like(attn), k=1)
            attn = np.where(mask, np.nan, attn)

        im = ax.imshow(attn, cmap=config.colorscale)

        if head_notation == "LH":
            ax.set_title(f"L{layer_idx}H{head_idx}")
        else:
            ax.set_title(f"{layer_idx}.{head_idx}")

        # Only show labels on edge subplots
        if row == rows - 1:
            ax.set_xticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=6)
        else:
            ax.set_xticks([])

        if col == 0:
            ax.set_yticks(range(len(tokens)))
            ax.set_yticklabels(tokens, fontsize=6)
        else:
            ax.set_yticks([])

    # Hide empty subplots
    for i in range(n_heads, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')

    plt.tight_layout()
    return fig


def _attention_circuitsvis(
    patterns: Dict[int, np.ndarray],
    tokens: List[str],
    heads: Optional[List[Tuple[int, int]]],
    head_notation: str,
    config: AttentionVisualizationConfig,
) -> Any:
    """Create interactive attention visualization with CircuitsVis."""
    try:
        import circuitsvis as cv
    except ImportError:
        raise ImportError(
            "CircuitsVis not installed. Install with: pip install circuitsvis"
        )

    # For CircuitsVis, we need to stack attention patterns
    # Shape expected: (num_heads, seq_len, seq_len)

    if heads is not None:
        # Select specific heads
        selected_attns = []
        head_names = []
        for layer_idx, head_idx in heads:
            if layer_idx in patterns:
                selected_attns.append(patterns[layer_idx][0, head_idx])
                if head_notation == "LH":
                    head_names.append(f"L{layer_idx}H{head_idx}")
                else:
                    head_names.append(f"{layer_idx}.{head_idx}")

        attention = np.stack(selected_attns, axis=0)
    else:
        # Use first layer, all heads
        first_layer = min(patterns.keys())
        attention = patterns[first_layer][0]  # (num_heads, seq_len, seq_len)
        num_heads = attention.shape[0]
        if head_notation == "LH":
            head_names = [f"L{first_layer}H{h}" for h in range(num_heads)]
        else:
            head_names = [f"{first_layer}.{h}" for h in range(num_heads)]

    return cv.attention.attention_heads(
        attention=attention,
        tokens=tokens,
        attention_head_names=head_names,
        mask_upper_tri=config.mask_upper_tri,
    )
