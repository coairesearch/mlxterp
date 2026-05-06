"""Path-patching demo on Llama-3.2-1B-Instruct-4bit.

For each layer's self_attn and mlp output, computes the single-edge
path effect: how much the (target − foil) logit margin moves when this
component is set to its clean value while every other component is
frozen at its corrupted value.

This is the canonical "what does each component contribute to the
final answer, isolated from all other paths" sweep. On a factual
recall task you should expect a few mid/late-layer components to
dominate, with most components near zero.
"""

from __future__ import annotations

import time
import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel


MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
CLEAN = "Paris is the capital of France"
CORRUPTED = "London is the capital of France"


def main():
    print(f"Loading {MODEL}...")
    base, tok = load(MODEL)
    model = InterpretableModel(base, tokenizer=tok)
    n_layers = len(model.layers)
    print(f"Model has {n_layers} layers.\n")

    print(f"clean:     {CLEAN!r}")
    print(f"corrupted: {CORRUPTED!r}\n")

    print("Sweeping path effects across every layer's self_attn and mlp...\n")
    t0 = time.perf_counter()
    rows = []
    for i in range(n_layers):
        for suffix in ("self_attn", "mlp"):
            sender = f"layers.{i}.{suffix}"
            try:
                eff = model.path_patching(
                    clean_text=CLEAN,
                    corrupted_text=CORRUPTED,
                    sender=sender,
                    target_token=" Paris",
                    foil_token=" London",
                )
            except ValueError as e:
                print(f"  skip {sender}: {e}")
                continue
            rows.append((i, suffix, eff))
            print(f"  {sender:>22}: {eff:+.4f}")
    secs = time.perf_counter() - t0
    print(f"\nTotal sweep: {secs:.2f}s ({len(rows)} components, "
          f"{secs / max(len(rows),1):.2f}s/component)\n")

    print("Top 5 components by |path effect|:")
    rows_sorted = sorted(rows, key=lambda r: -abs(r[2]))
    for i, (layer, comp, eff) in enumerate(rows_sorted[:5], 1):
        print(f"  {i}. layers.{layer}.{comp}: {eff:+.4f}")
    print()
    print(
        "Interpretation: components with |effect| close to zero do not "
        "carry the contrastive signal through the residual stream when "
        "everything else is frozen. Large positive effects ⇒ this "
        "component is on a clean-aligned path. Large negative effects "
        "⇒ corrupted-aligned. This is the single-edge MVP — the "
        "receiver is the final logits. Per-receiver paths (e.g. "
        "layers.7.attn → layers.9.attn) are a follow-up."
    )


if __name__ == "__main__":
    main()
