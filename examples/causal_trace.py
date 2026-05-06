#!/usr/bin/env python
"""Example: clean/corrupted comparative tracing with the causal_trace API.

Demonstrates the ergonomic ``model.causal_trace`` wrapper for the
standard activation-patching pattern. Compare to the lower-level
two-trace pattern in ``activation_patching_example.py`` — same result,
much less boilerplate.
"""

from __future__ import annotations

from mlx_lm import load
from mlxterp import InterpretableModel


MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
CLEAN = "Paris is the capital of France"
CORRUPTED = "London is the capital of France"


def main() -> None:
    print(f"Loading {MODEL_NAME} ...")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)
    print(f"Model has {len(model.layers)} decoder layers.\n")

    # The verbose pattern we're replacing:
    print("# Pattern A: low-level trace + replace_with (still works)")
    from mlxterp import interventions as iv
    with model.trace(CLEAN) as t:
        clean_l5 = t.activations.get("model.model.layers.5")

    if clean_l5 is not None:
        with model.trace(CORRUPTED, interventions={"layers.5": iv.replace_with(clean_l5)}):
            print("  scheduled patch on layers.5; corrupted run ran with intervention")
    print()

    # The new ergonomic pattern:
    print("# Pattern B: model.causal_trace (Tier 1 #2)")
    with model.causal_trace(CLEAN, CORRUPTED) as ct:
        ct.patch("layers.5")
        ct.patch("layers.10.mlp")
        print(f"  scheduled patches: {ct.patches}")
        print(f"  clean output shape:    {ct.clean_output.shape}")
        print(f"  patched output shape:  {ct.output.shape}")  # triggers corrupted run

    print()
    print("# What changed")
    print("  - one context manager instead of two")
    print("  - patch by short name; the trace handles path-prefix discovery")
    print("  - clean_output captured up front; output runs on first access")
    print("  - errors loudly if you try to patch after reading output")


if __name__ == "__main__":
    main()
