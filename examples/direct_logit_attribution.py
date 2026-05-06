"""Direct Logit Attribution demo on Llama-3.2-1B-Instruct-4bit.

Decomposes the (target − foil) logit margin into per-component
contributions in residual space. With ``apply_final_norm=True`` this
is a *decomposition* — the contributions sum to the actual logit_diff
at the same position, exactly (up to float precision).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message=r".*LibreSSL.*", module="urllib3.*")
warnings.filterwarnings("ignore", message=r".*head_dim.*", module=".*")

import mlx.core as mx
from mlx_lm import load
from mlxterp import InterpretableModel


MODEL = "mlx-community/Llama-3.2-1B-Instruct-4bit"
TEXT = "Paris is the capital of France"
TARGET = " Paris"
FOIL = " London"


def main():
    print(f"Loading {MODEL}...")
    base, tok = load(MODEL)
    model = InterpretableModel(base, tokenizer=tok)
    print(f"Model has {len(model.layers)} layers.\n")

    print(f"prompt:  {TEXT!r}")
    print(f"target:  {TARGET!r}    foil:  {FOIL!r}\n")

    dla = model.direct_logit_attribution(
        text=TEXT,
        target_token=TARGET,
        foil_token=FOIL,
        apply_final_norm=True,
    )

    # Sum check: the with-norm decomposition equals the actual
    # logit_diff modulo float precision.
    with model.trace(TEXT):
        logits = model.output.save()
    mx.eval(logits)
    target_ids = tok.encode(TARGET, add_special_tokens=False)
    foil_ids = tok.encode(FOIL, add_special_tokens=False)
    if tok.bos_token_id is not None:
        if target_ids and target_ids[0] == tok.bos_token_id:
            target_ids = target_ids[1:]
        if foil_ids and foil_ids[0] == tok.bos_token_id:
            foil_ids = foil_ids[1:]
    actual = float(logits[0, -1, target_ids[0]] - logits[0, -1, foil_ids[0]])
    summed = sum(dla.values())

    print(f"{'component':>14} | {'contribution':>14}")
    print("-" * 33)
    for (comp, layer), v in sorted(dla.items(), key=lambda kv: -kv[1]):
        label = f"{comp}.{layer}" if layer >= 0 else comp
        print(f"{label:>14} | {v:>+14.4f}")
    print()
    print(f"Sum of DLA contributions:  {summed:+.4f}")
    print(f"Actual logit_diff at -1:    {actual:+.4f}")
    print(f"Identity gap:               {summed - actual:+.4f}  "
          f"(small ⇒ all hook points captured)")
    print()
    print("Top 5 components writing toward target:")
    for (comp, layer), v in sorted(dla.items(), key=lambda kv: -kv[1])[:5]:
        label = f"{comp}.{layer}" if layer >= 0 else comp
        print(f"  {label}: {v:+.4f}")
    print()
    print("Top 5 components writing toward foil:")
    for (comp, layer), v in sorted(dla.items(), key=lambda kv: kv[1])[:5]:
        label = f"{comp}.{layer}" if layer >= 0 else comp
        print(f"  {label}: {v:+.4f}")


if __name__ == "__main__":
    main()
