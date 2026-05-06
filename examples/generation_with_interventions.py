#!/usr/bin/env python
"""Example: generate text under a persistent intervention.

Demonstrates the new ``InterpretableModel.generate(text, interventions=...)``
API. Loads a real MLX-quantised model, generates a baseline answer, then
generates the same prompt with a mid-network layer suppressed and shows
how the answer changes.

This is the standard mech-interp pattern for measuring causal importance:
install a hook, generate the continuation under the hook, observe how the
answer changes vs the unhooked baseline. Before this API existed in
mlxterp, callers had to use a ``with model.trace(text, interventions=...,
skip_forward=True): mlx_lm.generate(...)`` pattern; ``generate()`` is the
ergonomic version of the same thing.
"""

from __future__ import annotations

from mlx_lm import load
from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"  # small example model
PROMPT = "The capital of France is"
MAX_TOKENS = 20


def main() -> None:
    print(f"Loading {MODEL_NAME} ...")
    base, tok = load(MODEL_NAME)
    model = InterpretableModel(base, tokenizer=tok)
    print(f"Model has {len(model.layers)} decoder layers.\n")

    print(f"Prompt: {PROMPT!r}, max_tokens={MAX_TOKENS}")
    print()

    print("[baseline] no intervention")
    baseline = model.generate(PROMPT, max_tokens=MAX_TOKENS)
    print(f"  {baseline!r}\n")

    print("[ablation] zero out residual at layer 5")
    ablated = model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.5": iv.zero_out},
    )
    print(f"  {ablated!r}\n")

    print("[scaling] scale residual at layer 8 by 0.5")
    scaled = model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.8": iv.scale(0.5)},
    )
    print(f"  {scaled!r}\n")

    print("[post-check] generate again with no intervention")
    post = model.generate(PROMPT, max_tokens=MAX_TOKENS)
    print(f"  {post!r}")
    print(f"  matches baseline: {post == baseline}")


if __name__ == "__main__":
    main()
