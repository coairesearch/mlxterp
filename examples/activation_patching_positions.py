#!/usr/bin/env python
"""Example: per-position activation patching with the logit_diff metric.

Demonstrates the new positions parameter and built-in causal metrics on
the canonical Paris/London-style task. Each cell of the output matrix
answers: 'how much does patching layer L at token position P recover
the clean run's preference for the clean target?'

Run with a small MLX-quantised Llama (4-bit) for fast iteration; swap
in any other MLX model as needed.
"""

from __future__ import annotations

from mlx_lm import load
from mlxterp import InterpretableModel


MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"


def main() -> None:
    print(f"Loading {MODEL_NAME} ...")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)
    print(f"Model has {len(model.layers)} decoder layers.\n")

    # The canonical Paris/London task. Both prompts must tokenise to the
    # same length for position-level patching; verify before running.
    clean = "Paris is the capital of France"
    corrupted = "London is the capital of France"
    clean_ids = tok.encode(clean)
    corrupted_ids = tok.encode(corrupted)
    assert len(clean_ids) == len(corrupted_ids), (
        f"Token-length mismatch: {len(clean_ids)} vs {len(corrupted_ids)}. "
        "Position-level patching requires same-length inputs."
    )
    print(f"Tokenised length: {len(clean_ids)}")
    print()

    # 1. Layer-only (the original API). Returns a dict.
    print("Layer-level patching (positions=None) — returns dict")
    layer_results = model.activation_patching(
        clean_text=clean,
        corrupted_text=corrupted,
        component="output",
        layers=list(range(0, len(model.layers), 4)),
        metric="l2",
    )
    print(f"  per-layer recovery: {layer_results}")
    print()

    # 2. Per-position with logit_diff metric. Returns an ndarray.
    print("Per-position patching (positions='all', metric='logit_diff')")
    print("  — returns (n_layers, n_positions) matrix")
    matrix = model.activation_patching(
        clean_text=clean,
        corrupted_text=corrupted,
        component="output",
        layers=list(range(0, len(model.layers), 4)),
        positions="all",
        metric="logit_diff",
        clean_target=" Paris",
        corrupted_target=" London",
        plot=False,  # set True to display a heatmap
    )
    print(f"  matrix shape: {matrix.shape}")
    print(f"  max recovery cell: {float(matrix.max()):.1f}%")
    print(f"  min recovery cell: {float(matrix.min()):.1f}%")
    print()

    # Pretty-print the matrix
    layer_list = list(range(0, len(model.layers), 4))
    print("  Recovery % per (layer, position):")
    print("    layer", "  ".join(f"pos{p:>2}" for p in range(len(clean_ids))))
    for li, layer_idx in enumerate(layer_list):
        row = matrix[li]
        cells = "  ".join(f"{v:+5.1f}" for v in row)
        print(f"    {layer_idx:>5}  {cells}")
    print()
    print("Tip: pass plot=True to render this as a heatmap.")


if __name__ == "__main__":
    main()
