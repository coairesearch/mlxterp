"""Render a heatmap and a couple of bar charts for synthetic patching
results.

This script doesn't load a model — it shows the visualisation API on
inputs that match the shapes returned by `attribution_patching`,
`path_patching`, `direct_logit_attribution`, and a hypothetical
``(layer × position)`` matrix. The point is to demonstrate the
shapes; plug your real results in and the figures land the same way.

Outputs four PNGs in the working directory:
  - attribution_bar.png
  - path_effects_bar.png
  - dla_bar.png
  - patching_heatmap.png
"""

from __future__ import annotations

import numpy as np
from mlxterp.visualization import (
    plot_attribution_bar,
    plot_path_effects_bar,
    plot_dla,
    plot_patching_heatmap,
)


def main():
    rng = np.random.default_rng(0)

    # 1) attribution_patching: dict[layer_idx -> float], 16 layers
    attr_scores = {
        i: float(rng.normal(loc=(i - 8) / 8, scale=0.3)) for i in range(16)
    }
    fig = plot_attribution_bar(
        attr_scores, title="Attribution patching (synthetic)"
    )
    fig.savefig("attribution_bar.png", dpi=120, bbox_inches="tight")
    print("wrote attribution_bar.png")

    # 2) path_patching sweep: dict[(layer_idx, comp) -> float]
    path_effects = {}
    for i in range(16):
        path_effects[(i, "self_attn")] = float(rng.normal(0, 0.3))
        path_effects[(i, "mlp")] = float(rng.normal(0, 0.3))
    path_effects[(15, "self_attn")] = 2.2  # the dominant component
    fig = plot_path_effects_bar(path_effects, title="Path effects (synthetic)")
    fig.savefig("path_effects_bar.png", dpi=120, bbox_inches="tight")
    print("wrote path_effects_bar.png")

    # 3) DLA: dict[(label, layer) -> contribution]
    dla = {("embed", -1): 1.06}
    for i in range(16):
        dla[("self_attn", i)] = float(rng.normal(0, 0.2))
        dla[("mlp", i)] = float(rng.normal(0, 0.4))
    dla[("mlp", 10)] = 1.28
    dla[("mlp", 11)] = 0.95
    fig = plot_dla(dla, title="DLA (synthetic, top 12)", top_k=12)
    fig.savefig("dla_bar.png", dpi=120, bbox_inches="tight")
    print("wrote dla_bar.png")

    # 4) (layer × position) patching matrix: simulate the
    #    activation_patching positions=… result.
    matrix = rng.normal(size=(16, 8)) * 0.2
    matrix[12:, 5:] += 0.8  # late-layer late-position structure
    matrix[2:5, 0] -= 0.6   # early-layer first-position dependence
    fig = plot_patching_heatmap(
        matrix,
        x_labels=[f"tok{j}" for j in range(8)],
        y_labels=[f"L{i}" for i in range(16)],
        title="Layer × position patching (synthetic)",
    )
    fig.savefig("patching_heatmap.png", dpi=120, bbox_inches="tight")
    print("wrote patching_heatmap.png")


if __name__ == "__main__":
    main()
