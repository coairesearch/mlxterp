"""
Visualization utilities for patching results.

Provides heatmaps, bar charts, and interactive plots for
activation patching, attribution, and DLA results.
"""

from typing import Optional, List
import mlx.core as mx
import matplotlib.pyplot as plt


def plot_patching_result(result, figsize=(12, 6), cmap="RdBu_r", title=None, **kwargs):
    """
    Plot a patching result as a heatmap or bar chart.

    Automatically chooses the right plot type based on the effect matrix shape:
    - 1D: bar chart (layer-level effects)
    - 2D: heatmap (layer x position or layer x head)

    Args:
        result: PatchingResult instance
        figsize: Figure size tuple
        cmap: Colormap name
        title: Optional title override
        **kwargs: Passed to matplotlib

    Returns:
        matplotlib Figure
    """
    effects = result.effect_matrix
    if isinstance(effects, mx.array):
        effects = effects.tolist()

    if effects is None:
        raise ValueError("No effect_matrix in result")

    fig, ax = plt.subplots(figsize=figsize)

    if isinstance(effects[0], list):
        # 2D: heatmap
        import numpy as np
        data = np.array(effects)
        im = ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
        ax.set_xlabel("Head" if result.component == "attn_head" else "Position")
        ax.set_ylabel("Layer")
        ax.set_yticks(range(len(result.layers)))
        ax.set_yticklabels(result.layers)
        plt.colorbar(im, ax=ax, label=result.metric_name)
        default_title = f"Activation Patching: {result.component} ({result.metric_name})"
    else:
        # 1D: bar chart
        ax.bar(range(len(effects)), effects, color="steelblue", **kwargs)
        ax.set_xlabel("Layer")
        ax.set_ylabel(result.metric_name)
        ax.set_xticks(range(len(result.layers)))
        ax.set_xticklabels(result.layers, rotation=45 if len(result.layers) > 20 else 0)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        default_title = f"Activation Patching: {result.component} ({result.metric_name})"

    ax.set_title(title or default_title)
    plt.tight_layout()
    return fig


def plot_patching_comparison(results, figsize=(14, 6), title=None):
    """
    Plot multiple patching results side by side for comparison.

    Args:
        results: List of PatchingResult instances
        figsize: Figure size tuple
        title: Optional title

    Returns:
        matplotlib Figure
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        effects = result.effect_matrix
        if isinstance(effects, mx.array):
            effects = effects.tolist()

        if isinstance(effects[0], list):
            import numpy as np
            im = ax.imshow(np.array(effects), cmap="RdBu_r", aspect="auto")
            plt.colorbar(im, ax=ax, shrink=0.8)
        else:
            ax.bar(range(len(effects)), effects, color="steelblue")
            ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")

        ax.set_title(f"{result.component}\n({result.metric_name})")
        ax.set_xlabel("Layer")

    axes[0].set_ylabel("Effect")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    return fig
