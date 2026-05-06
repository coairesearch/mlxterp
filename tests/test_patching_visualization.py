"""Smoke tests for the patching-visualisation helpers.

These don't try to verify pixels — they verify that each renderer
produces a non-trivial Figure with the right axes labels and bar /
image counts given a representative input shape. Heavy regression on
exact rendering is fragile and not the point.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend for CI

from mlxterp.visualization import (
    plot_attribution_bar,
    plot_path_effects_bar,
    plot_dla,
    plot_patching_heatmap,
)


def test_plot_attribution_bar_renders():
    scores = {0: -0.1, 1: 0.3, 2: 0.5, 3: -0.2}
    fig = plot_attribution_bar(scores, title="t")
    ax = fig.axes[0]
    assert len(ax.patches) == 4  # one bar per layer
    assert ax.get_title() == "t"


def test_plot_attribution_bar_rejects_empty():
    with pytest.raises(ValueError):
        plot_attribution_bar({})


def test_plot_path_effects_bar_renders():
    effects = {
        (0, "self_attn"): 0.1,
        (0, "mlp"): -0.2,
        (1, "self_attn"): 0.3,
        (1, "mlp"): 0.0,
    }
    fig = plot_path_effects_bar(effects, title="paths")
    ax = fig.axes[0]
    assert len(ax.patches) == 4
    assert ax.get_legend() is not None


def test_plot_dla_renders():
    dla = {
        ("embed", -1): 1.0,
        ("self_attn", 0): -0.1,
        ("mlp", 0): 0.5,
        ("self_attn", 1): 0.7,
        ("mlp", 1): 0.2,
    }
    fig = plot_dla(dla)
    ax = fig.axes[0]
    # One horizontal bar per entry.
    assert len(ax.patches) == len(dla)


def test_plot_dla_top_k_limits_bars():
    dla = {("self_attn", i): float(i) for i in range(10)}
    fig = plot_dla(dla, top_k=3)
    ax = fig.axes[0]
    assert len(ax.patches) == 3


def test_plot_patching_heatmap_renders():
    matrix = np.random.randn(8, 5)
    fig = plot_patching_heatmap(
        matrix,
        x_labels=[f"t{j}" for j in range(5)],
        y_labels=[f"L{i}" for i in range(8)],
        title="hm",
    )
    ax = fig.axes[0]
    images = ax.get_images()
    assert len(images) == 1
    assert images[0].get_array().shape == (8, 5)


def test_plot_patching_heatmap_rejects_non_2d():
    with pytest.raises(ValueError):
        plot_patching_heatmap(np.zeros((3, 3, 3)))
