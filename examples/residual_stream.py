"""ResidualStream view demo: identity check + per-layer norms.

Two things this script demonstrates:

  1. The defining identity holds to float precision: at any layer i,
     resid_post[i] equals the sum of the embedding plus every
     component contribution (attn[0..i] + mlp[0..i]) up to layer i.

  2. The residual stream's L2 norm grows with depth — a standard
     signal that the model is accumulating information.

Both are useful checks before building DLA / path-patching analyses
on top of the residual stream.
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


def main():
    print(f"Loading {MODEL}...")
    base, tok = load(MODEL)
    model = InterpretableModel(base, tokenizer=tok)
    n = len(model.layers)

    print(f"prompt: {TEXT!r}\n")
    rs = model.residual_stream(TEXT)
    print(f"captured: {len(rs.layer_outputs)} layer outputs, "
          f"{len(rs.attn_outputs)} attn, {len(rs.mlp_outputs)} mlp, "
          f"embedding {'yes' if rs.embedding is not None else 'no'}\n")

    # Identity check at every layer.
    print(f"{'layer':>5} | {'max |Δ| post − sum(contribs)':>30}")
    print("-" * 40)
    if rs.embedding is None:
        print("  (embedding not captured — identity check skipped; "
              "comparing pre/post chain instead)")
        for i in range(1, n):
            pre = rs.pre(i)
            post_im1 = rs.post(i - 1)
            diff = mx.max(mx.abs(pre - post_im1)).astype(mx.float32)
            mx.eval(diff)
            print(f"{i:>5} | {float(diff):>30.6f}    (pre[i] vs post[i-1])")
    else:
        for i in range(n):
            contribs = rs.decompose(layer_idx=i)
            summed = None
            for v in contribs.values():
                summed = v if summed is None else summed + v
            actual = rs.post(i)
            diff = mx.max(mx.abs(summed - actual)).astype(mx.float32)
            mx.eval(diff)
            print(f"{i:>5} | {float(diff):>30.6f}")

    # Per-layer L2 norm of the residual stream at the last position.
    print(f"\n{'layer':>5} | {'||resid_post[i]||₂ at last token':>32}")
    print("-" * 42)
    for i in range(n):
        post = rs.post(i)[0, -1].astype(mx.float32)
        norm = mx.sqrt(mx.sum(post * post))
        mx.eval(norm)
        print(f"{i:>5} | {float(norm):>32.4f}")

    # Quick sanity: cumulative attn-only residual.
    print("\nFirst 5 layers — attn vs MLP contribution norms at last token:")
    for i in range(min(5, n)):
        attn = rs.attn_contribution(i)[0, -1].astype(mx.float32)
        mlp = rs.mlp_contribution(i)[0, -1].astype(mx.float32)
        a_norm = mx.sqrt(mx.sum(attn * attn))
        m_norm = mx.sqrt(mx.sum(mlp * mlp))
        mx.eval(a_norm, m_norm)
        print(f"  layer {i}: attn={float(a_norm):.3f}, mlp={float(m_norm):.3f}")


if __name__ == "__main__":
    main()
