"""
SAE Feature Visualization - Neuronpedia-style

Provides functions to visualize SAE feature activations in text,
similar to Neuronpedia's interface.
"""

from typing import List, Tuple, Optional, Union, Dict
import mlx.core as mx
from ..sae import SAE, BatchTopKSAE


def get_feature_activations_by_token(
    model,
    text: str,
    sae: Union[SAE, BatchTopKSAE],
    layer: int,
    component: str = "mlp",
    feature_ids: Optional[List[int]] = None,
    top_k_features: Optional[int] = None
) -> Tuple[List[str], mx.array, List[int]]:
    """Get token-level feature activations for visualization.

    Args:
        model: InterpretableModel instance
        text: Input text to analyze
        sae: SAE instance
        layer: Layer number
        component: Component name ("mlp", "attn", etc.)
        feature_ids: Specific features to track (optional)
        top_k_features: If set, return top-k features overall (optional)

    Returns:
        Tuple of (tokens, activations, feature_ids)
        - tokens: List of token strings
        - activations: Array of shape (n_tokens, n_features)
        - feature_ids: List of feature IDs included

    Example:
        >>> tokens, acts, feat_ids = get_feature_activations_by_token(
        ...     model, "The Eiffel Tower is in Paris",
        ...     sae, layer=10, component="mlp", top_k_features=5
        ... )
        >>> # tokens = ["The", "Eif", "fel", "Tower", ...]
        >>> # acts.shape = (n_tokens, 5)
    """
    # Get model activations
    with model.trace(text) as trace:
        pass

    # Find activation key
    activation_key = None
    for key in trace.activations.keys():
        if f"layers.{layer}" in key and key.endswith(f".{component}"):
            activation_key = key
            break

    if activation_key is None:
        raise ValueError(
            f"Could not find activations for layer {layer} component {component}"
        )

    activations = trace.activations[activation_key]  # (seq_len, d_model)

    # Tokenize to get tokens
    tokens = model.tokenizer.tokenize(text)

    # Add batch dimension for SAE: (seq_len, 1, d_model)
    activations_3d = activations[:, None, :]

    # Run through SAE
    _, features = sae(activations_3d)  # (seq_len, 1, d_hidden)
    features_2d = features[:, 0, :]  # (seq_len, d_hidden)

    # Select features
    if feature_ids is not None:
        # Use specified features
        selected_features = features_2d[:, feature_ids]
        selected_ids = feature_ids
    elif top_k_features is not None:
        # Find top-k features overall
        feature_max = mx.max(mx.abs(features_2d), axis=0)  # (d_hidden,)
        top_indices = mx.argsort(feature_max)[-top_k_features:]

        import numpy as np
        top_indices_np = np.array(top_indices).flatten()[::-1]
        selected_ids = top_indices_np.tolist()

        selected_features = features_2d[:, selected_ids]
    else:
        # Return all features
        selected_features = features_2d
        selected_ids = list(range(sae.d_hidden))

    return tokens, selected_features, selected_ids


def format_neuronpedia_html(
    tokens: List[str],
    activations: mx.array,
    feature_ids: List[int],
    show_negative: bool = True,
    show_positive: bool = True
) -> str:
    """Format activations as HTML for Jupyter notebook display.

    Args:
        tokens: List of token strings
        activations: Array of shape (n_tokens, n_features)
        feature_ids: List of feature IDs
        show_negative: Show negative activations in red
        show_positive: Show positive activations in blue

    Returns:
        HTML string for display

    Example:
        >>> html = format_neuronpedia_html(tokens, acts, feat_ids)
        >>> from IPython.display import HTML, display
        >>> display(HTML(html))
    """
    import numpy as np

    # Convert to numpy for easier handling
    acts_np = np.array(activations)

    # Build HTML
    html_parts = []
    html_parts.append("""
    <style>
        .feature-viz {
            font-family: monospace;
            margin: 10px 0;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 5px;
        }
        .feature-header {
            font-weight: bold;
            margin: 15px 0 5px 0;
            padding: 5px;
            background: #e0e0e0;
            border-radius: 3px;
        }
        .token {
            display: inline-block;
            padding: 2px 4px;
            margin: 2px;
            border-radius: 3px;
            white-space: pre;
        }
        .positive-strong { background: #1e40af; color: white; font-weight: bold; }
        .positive-medium { background: #3b82f6; color: white; }
        .positive-weak { background: #93c5fd; color: #1e293b; }
        .negative-strong { background: #991b1b; color: white; font-weight: bold; }
        .negative-medium { background: #dc2626; color: white; }
        .negative-weak { background: #fca5a5; color: #1e293b; }
        .neutral { background: transparent; }
    </style>
    <div class="feature-viz">
    """)

    for feat_idx, feature_id in enumerate(feature_ids):
        html_parts.append(f'<div class="feature-header">Feature {feature_id}</div>')
        html_parts.append('<div>')

        # Get activations for this feature
        feat_acts = acts_np[:, feat_idx]

        # Ensure 1D
        if feat_acts.ndim > 1:
            feat_acts = feat_acts.flatten()

        # Normalize for coloring (0-100 scale)
        max_abs = np.max(np.abs(feat_acts))
        if max_abs > 0:
            normalized = ((feat_acts / max_abs) * 100).flatten()
        else:
            normalized = np.zeros_like(feat_acts).flatten()

        # Build colored tokens
        for i, token in enumerate(tokens):
            norm_act = float(normalized[i])
            raw_act = float(feat_acts[i])

            # Escape HTML
            token_html = token.replace('<', '&lt;').replace('>', '&gt;')
            token_html = token_html.replace(' ', '&nbsp;')

            if norm_act > 0 and show_positive:
                # Positive activation (blue)
                if norm_act > 70:
                    css_class = "positive-strong"
                elif norm_act > 30:
                    css_class = "positive-medium"
                else:
                    css_class = "positive-weak"
                html_parts.append(
                    f'<span class="token {css_class}" title="{raw_act:.3f}">{token_html}</span>'
                )
            elif norm_act < 0 and show_negative:
                # Negative activation (red)
                if -norm_act > 70:
                    css_class = "negative-strong"
                elif -norm_act > 30:
                    css_class = "negative-medium"
                else:
                    css_class = "negative-weak"
                html_parts.append(
                    f'<span class="token {css_class}" title="{raw_act:.3f}">{token_html}</span>'
                )
            else:
                html_parts.append(f'<span class="token neutral">{token_html}</span>')

        html_parts.append('</div>')

    html_parts.append('</div>')
    return ''.join(html_parts)


def format_neuronpedia_style(
    tokens: List[str],
    activations: mx.array,
    feature_ids: List[int],
    show_negative: bool = True,
    show_positive: bool = True
) -> str:
    """Format activations in Neuronpedia-style colored text.

    Args:
        tokens: List of token strings
        activations: Array of shape (n_tokens, n_features)
        feature_ids: List of feature IDs
        show_negative: Show negative activations in red
        show_positive: Show positive activations in blue

    Returns:
        Formatted string with ANSI color codes

    Example:
        >>> formatted = format_neuronpedia_style(tokens, acts, feat_ids)
        >>> print(formatted)
        # Tokens will be colored based on activation strength
    """
    import numpy as np

    # Convert to numpy for easier handling
    acts_np = np.array(activations)

    # ANSI color codes
    RESET = "\033[0m"
    RED = "\033[91m"      # Negative activation
    BLUE = "\033[94m"     # Positive activation
    BOLD = "\033[1m"

    # For each feature, create a formatted view
    output_lines = []

    for feat_idx, feature_id in enumerate(feature_ids):
        output_lines.append(f"\n{BOLD}Feature {feature_id}:{RESET}")

        # Get activations for this feature
        feat_acts = acts_np[:, feat_idx]

        # Ensure it's 1D
        if feat_acts.ndim > 1:
            feat_acts = feat_acts.flatten()

        # Normalize for coloring (0-100 scale)
        max_abs = np.max(np.abs(feat_acts))
        if max_abs > 0:
            normalized = ((feat_acts / max_abs) * 100).flatten().tolist()
        else:
            normalized = np.zeros_like(feat_acts).flatten().tolist()

        # Build colored string
        colored_tokens = []
        for i, token in enumerate(tokens):
            norm_act = normalized[i]

            if norm_act > 0 and show_positive:
                # Positive activation (blue)
                intensity = int(min(norm_act, 100))
                # Darker blue for stronger activation
                if intensity > 70:
                    colored_tokens.append(f"{BOLD}{BLUE}{token}{RESET}")
                elif intensity > 30:
                    colored_tokens.append(f"{BLUE}{token}{RESET}")
                else:
                    colored_tokens.append(token)
            elif norm_act < 0 and show_negative:
                # Negative activation (red)
                intensity = int(min(-norm_act, 100))
                if intensity > 70:
                    colored_tokens.append(f"{BOLD}{RED}{token}{RESET}")
                elif intensity > 30:
                    colored_tokens.append(f"{RED}{token}{RESET}")
                else:
                    colored_tokens.append(token)
            else:
                colored_tokens.append(token)

        output_lines.append("  " + " ".join(colored_tokens))

    return "\n".join(output_lines)


def visualize_feature_activations(
    model,
    text: str,
    sae: Union[SAE, BatchTopKSAE],
    layer: int,
    component: str = "mlp",
    feature_ids: Optional[List[int]] = None,
    top_k_features: int = 5,
    show_values: bool = False,
    mode: str = "auto"
) -> None:
    """Visualize SAE feature activations in text (Neuronpedia-style).

    This function displays text with tokens colored by their activation
    strength for each feature, similar to Neuronpedia's interface.

    Args:
        model: InterpretableModel instance
        text: Input text to analyze
        sae: SAE instance or path
        layer: Layer number
        component: Component name
        feature_ids: Specific features to visualize (optional)
        top_k_features: Number of top features to show (default: 5)
        show_values: Show activation values alongside tokens
        mode: Display mode - "auto" (detect), "html" (Jupyter), "terminal" (ANSI colors)

    Example:
        >>> from mlxterp import InterpretableModel
        >>> from mlx_lm import load
        >>>
        >>> mlx_model, tokenizer = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
        >>> model = InterpretableModel(mlx_model, tokenizer=tokenizer)
        >>> sae = model.load_sae("sae_layer10.mlx")
        >>>
        >>> # Visualize top 5 features (auto-detects environment)
        >>> visualize_feature_activations(
        ...     model,
        ...     "The Eiffel Tower is located in Paris, France",
        ...     sae,
        ...     layer=10,
        ...     component="mlp"
        ... )
        >>>
        >>> # Force HTML mode for Jupyter
        >>> visualize_feature_activations(
        ...     model,
        ...     "Machine learning models process data",
        ...     sae,
        ...     layer=10,
        ...     component="mlp",
        ...     feature_ids=[42, 128, 256],
        ...     mode="html"
        ... )
    """
    # Load SAE if path provided
    if isinstance(sae, str):
        sae = model.load_sae(sae)

    # Get token-level activations
    tokens, activations, selected_ids = get_feature_activations_by_token(
        model, text, sae, layer, component,
        feature_ids=feature_ids,
        top_k_features=top_k_features if feature_ids is None else None
    )

    # Determine rendering mode
    render_mode = mode
    if mode == "auto":
        # Try to detect Jupyter environment
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None and 'IPKernelApp' in ipython.config:
                render_mode = "html"
            else:
                render_mode = "terminal"
        except (ImportError, AttributeError):
            render_mode = "terminal"

    # Render based on mode
    if render_mode == "html":
        # HTML rendering for Jupyter
        try:
            from IPython.display import HTML, display

            # Header
            display(HTML(f"""
                <div style="font-family: monospace; margin: 20px 0;">
                    <div style="border-bottom: 2px solid #333; padding: 10px 0; margin-bottom: 10px;">
                        <strong>SAE Feature Activations: Layer {layer} {component}</strong>
                    </div>
                    <div style="margin: 10px 0; color: #555;">
                        <strong>Input:</strong> {text}
                    </div>
                </div>
            """))

            # Feature visualization
            html_content = format_neuronpedia_html(
                tokens, activations, selected_ids
            )
            display(HTML(html_content))

            # Show values if requested
            if show_values:
                import numpy as np
                acts_np = np.array(activations)

                values_html = ['<div style="font-family: monospace; margin: 20px 0;">']
                values_html.append('<div style="border-bottom: 2px solid #333; padding: 10px 0; margin-bottom: 10px;"><strong>Activation Values</strong></div>')

                for feat_idx, feature_id in enumerate(selected_ids):
                    values_html.append(f'<div style="margin: 10px 0;"><strong>Feature {feature_id}:</strong></div>')
                    values_html.append('<div style="margin-left: 20px;">')

                    feat_acts = acts_np[:, feat_idx]

                    # Ensure 1D
                    if feat_acts.ndim > 1:
                        feat_acts = feat_acts.flatten()

                    for i, token in enumerate(tokens):
                        act = float(feat_acts[i])
                        if abs(act) > 0.01:
                            token_escaped = token.replace('<', '&lt;').replace('>', '&gt;').replace(' ', '&nbsp;')
                            values_html.append(f'<div>{token_escaped:15s} {act:7.3f}</div>')

                    values_html.append('</div>')

                values_html.append('</div>')
                display(HTML(''.join(values_html)))

        except ImportError:
            # Fall back to terminal if IPython not available
            render_mode = "terminal"

    if render_mode == "terminal":
        # Terminal rendering with ANSI colors
        formatted = format_neuronpedia_style(
            tokens, activations, selected_ids
        )

        print("=" * 80)
        print(f"SAE Feature Activations: Layer {layer} {component}")
        print("=" * 80)
        print(f"\nInput: {text}")
        print(formatted)

        if show_values:
            print("\n" + "=" * 80)
            print("Activation Values:")
            print("=" * 80)
            import numpy as np
            acts_np = np.array(activations)

            for feat_idx, feature_id in enumerate(selected_ids):
                print(f"\nFeature {feature_id}:")
                feat_acts = acts_np[:, feat_idx]

                # Ensure 1D
                if feat_acts.ndim > 1:
                    feat_acts = feat_acts.flatten()

                for i, token in enumerate(tokens):
                    act = float(feat_acts[i])
                    if abs(act) > 0.01:  # Only show non-zero
                        print(f"  {token:15s} {act:7.3f}")


def get_top_activating_tokens(
    model,
    text: str,
    sae: Union[SAE, BatchTopKSAE],
    layer: int,
    feature_id: int,
    component: str = "mlp",
    top_k: int = 10
) -> List[Tuple[str, float, int]]:
    """Get top tokens where a specific feature activates.

    Args:
        model: InterpretableModel instance
        text: Input text
        sae: SAE instance
        layer: Layer number
        component: Component name
        feature_id: Feature to analyze
        top_k: Number of top tokens to return

    Returns:
        List of (token, activation, position) tuples

    Example:
        >>> top_tokens = get_top_activating_tokens(
        ...     model, text, sae, layer=10, component="mlp",
        ...     feature_id=1234, top_k=5
        ... )
        >>> for token, act, pos in top_tokens:
        ...     print(f"Position {pos}: '{token}' = {act:.3f}")
    """
    tokens, activations, _ = get_feature_activations_by_token(
        model, text, sae, layer, component,
        feature_ids=[feature_id]
    )

    import numpy as np
    acts_np = np.array(activations).flatten()

    # Get top-k indices
    top_indices = np.argsort(acts_np)[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if acts_np[idx] > 0:
            results.append((tokens[idx], float(acts_np[idx]), int(idx)))

    return results
