"""Patching-result visualisation helpers.

Renderers for the dict / matrix shapes that mlxterp's patching APIs
return: bar charts for layer-keyed and component-keyed effects, and a
heatmap for ``(layer × position)`` matrices.

The interfaces are deliberately structural — they take plain dicts /
arrays so they work with whatever the user has, including results
from external workflows that match the same shape.

Example:

    >>> # Attribution patching scores → bar chart
    >>> attr = model.attribution_patching(...)
    >>> from mlxterp.visualization import plot_attribution_bar
    >>> fig = plot_attribution_bar(attr, title="My run")
    >>> fig.savefig("attr.png")

    >>> # Path patching sweep → grouped bar chart
    >>> from mlxterp.visualization import plot_path_effects_bar
    >>> effects = {(7, "self_attn"): 0.43, (7, "mlp"): -0.27, ...}
    >>> fig = plot_path_effects_bar(effects)

    >>> # DLA decomposition → bar chart with embed/attn/mlp groups
    >>> from mlxterp.visualization import plot_dla
    >>> dla = model.direct_logit_attribution(...)
    >>> fig = plot_dla(dla)

    >>> # (layer × position) patching matrix → heatmap
    >>> from mlxterp.visualization import plot_patching_heatmap
    >>> fig = plot_patching_heatmap(
    ...     matrix=patching_results,
    ...     x_labels=token_strings,
    ...     y_labels=[f"L{i}" for i in range(n_layers)],
    ... )
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np


def _maybe_get_ax(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    return fig, ax


def _signed_colors(values, cmap_name: str = "RdBu_r"):
    """Return per-bar colours: blue for positive, red for negative,
    matching the convention used by activation_patching's existing
    bar plot."""
    return ["#2166ac" if v >= 0 else "#b2182b" for v in values]


def plot_attribution_bar(
    scores: Mapping[int, float],
    *,
    ax=None,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Attribution patching by layer",
    xlabel: str = "Layer",
    ylabel: str = "Attribution score",
):
    """Bar chart of layer-keyed scalar scores.

    Designed for ``InterpretableModel.attribution_patching`` output
    but works for any ``dict[int, float]`` keyed by layer index.

    Args:
        scores: Mapping ``layer_idx -> attribution score``.
        ax: Optional pre-created Axes; if None, a new figure is made.
        figsize: Figure size when creating new.
        title / xlabel / ylabel: Self-explanatory.

    Returns:
        The matplotlib Figure.
    """
    if not scores:
        raise ValueError("plot_attribution_bar: empty scores dict.")

    fig, ax = _maybe_get_ax(ax, figsize)
    layers = sorted(scores)
    values = [scores[i] for i in layers]
    colors = _signed_colors(values)

    ax.bar(layers, values, color=colors, edgecolor="black", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_path_effects_bar(
    effects: Mapping[Tuple[int, str], float],
    *,
    ax=None,
    figsize: Tuple[float, float] = (12, 5),
    title: str = "Path-patching effect by component",
    component_order: Sequence[str] = ("self_attn", "mlp"),
):
    """Grouped bar chart of path-patching effects keyed by
    ``(layer_idx, component_label)``.

    Renders ``component_order``-many groups per layer (default:
    self_attn next to mlp). Designed for the dict produced by
    sweeping ``InterpretableModel.path_patching`` across components.

    Args:
        effects: Mapping ``(layer_idx, component_label) -> effect``.
        ax / figsize / title: Standard.
        component_order: Order of components within each group.

    Returns:
        The matplotlib Figure.
    """
    if not effects:
        raise ValueError("plot_path_effects_bar: empty effects dict.")

    fig, ax = _maybe_get_ax(ax, figsize)

    layers = sorted({k[0] for k in effects})
    n_components = len(component_order)
    width = 0.8 / max(n_components, 1)

    for i, comp in enumerate(component_order):
        comp_values = [effects.get((layer, comp), 0.0) for layer in layers]
        xs = [layer + (i - (n_components - 1) / 2) * width for layer in layers]
        ax.bar(
            xs, comp_values, width=width, label=comp,
            color=_signed_colors(comp_values), edgecolor="black", alpha=0.85,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Path effect (Δ logit_diff)")
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_dla(
    dla: Mapping[Tuple[str, int], float],
    *,
    ax=None,
    figsize: Tuple[float, float] = (12, 5),
    title: str = "Direct Logit Attribution",
    top_k: Optional[int] = None,
):
    """Bar chart of DLA contributions, sorted by signed contribution.

    Designed for the dict returned by
    ``InterpretableModel.direct_logit_attribution``: keys are
    ``(component_label, layer_idx)`` with ``component_label`` in
    ``{"embed", "self_attn", "mlp"}``. ``"embed"``'s layer index is
    ``-1`` and is rendered as a dedicated leftmost bar.

    Args:
        dla: The DLA dict (component label, layer index) → contribution.
        ax / figsize / title: Standard.
        top_k: If set, show only the top-k by |contribution|.

    Returns:
        The matplotlib Figure.
    """
    if not dla:
        raise ValueError("plot_dla: empty dla dict.")

    fig, ax = _maybe_get_ax(ax, figsize)

    items = sorted(dla.items(), key=lambda kv: kv[1])  # ascending
    if top_k is not None:
        items = sorted(dla.items(), key=lambda kv: -abs(kv[1]))[:top_k]
        items = sorted(items, key=lambda kv: kv[1])

    labels = [
        f"{comp}.{layer}" if layer >= 0 else comp
        for (comp, layer), _ in items
    ]
    values = [v for _, v in items]
    colors = _signed_colors(values)

    ax.barh(range(len(items)), values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(labels)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Contribution to logit_diff")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


def plot_patching_heatmap(
    matrix,
    *,
    x_labels: Optional[Sequence[str]] = None,
    y_labels: Optional[Sequence[str]] = None,
    ax=None,
    figsize: Tuple[float, float] = (10, 6),
    title: str = "Patching effect (layer × position)",
    xlabel: str = "Position",
    ylabel: str = "Layer",
    cmap: str = "RdBu_r",
    center_zero: bool = True,
    cbar_label: str = "effect",
):
    """Heatmap of a 2-D matrix (typically layer × position).

    Args:
        matrix: 2-D ndarray-like of shape (n_layers, n_positions). MLX
            arrays are converted to numpy via ``np.asarray``.
        x_labels / y_labels: Optional axis tick labels.
        ax / figsize / title / xlabel / ylabel / cmap / cbar_label:
            Standard.
        center_zero: If True, the colour scale is symmetric around 0
            (matches the patching convention where positive / negative
            both matter).

    Returns:
        The matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(
            f"plot_patching_heatmap: expected 2-D matrix, got shape {arr.shape}"
        )

    if center_zero:
        m = float(np.nanmax(np.abs(arr)))
        vmin, vmax = -m, m
    else:
        vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))

    fig, ax = _maybe_get_ax(ax, figsize)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if x_labels is not None:
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
    if y_labels is not None:
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return fig
