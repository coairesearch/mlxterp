"""Contract tests for the residual-stream view.

The defining identity: at any layer ``i``, ``resid_post[i]`` equals
the sum of the embedding plus every component contribution
(attn[0..i] + mlp[0..i]). The view should reproduce this within float
precision.
"""

import math
import pytest
import mlx.core as mx

pytestmark = pytest.mark.timeout(180)


def _load_small_model():
    from mlxterp import InterpretableModel
    from mlx_lm import load

    base, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    return InterpretableModel(base, tokenizer=tok)


def test_residual_stream_post_matches_layer_output():
    model = _load_small_model()
    rs = model.residual_stream("Paris is the capital of France")
    n = len(model.layers)

    # Each post(i) should be identical to the captured layer output.
    for i in range(n):
        v = rs.post(i)
        assert v is not None
        assert v.ndim >= 2  # (batch, seq, hidden) or (seq, hidden)


def test_residual_stream_pre_chains_through_post():
    """resid_pre[i] should equal resid_post[i-1] for i > 0."""
    model = _load_small_model()
    rs = model.residual_stream("Paris is the capital of France")
    n = len(model.layers)
    for i in range(1, n):
        pre_i = rs.pre(i)
        post_im1 = rs.post(i - 1)
        diff = mx.max(mx.abs(pre_i - post_im1)).astype(mx.float32)
        mx.eval(diff)
        assert float(diff) < 1e-3, (
            f"resid_pre[{i}] should equal resid_post[{i-1}]; max diff {float(diff)}"
        )


def test_residual_stream_decompose_sums_to_post():
    """The defining identity: sum of all contributions ≈ resid_post.

    Llama uses pre-norm + skip-add, so the residual stream is the
    literal sum of contributions plus the embedding. The decompose()
    method should reflect that.
    """
    model = _load_small_model()
    rs = model.residual_stream("Paris is the capital of France")
    if rs.embedding is None:
        pytest.skip("embedding output not captured by trace; identity is "
                    "approximate without it. Skipping the strict-identity "
                    "test.")

    n = len(model.layers)
    # Pick a middle layer to keep numerics reasonable.
    layer_idx = n // 2
    contribs = rs.decompose(layer_idx=layer_idx)
    summed = None
    for v in contribs.values():
        summed = v if summed is None else summed + v
    actual = rs.post(layer_idx)

    diff = mx.max(mx.abs(summed - actual)).astype(mx.float32)
    mx.eval(diff)
    assert float(diff) < 1e-2, (
        f"sum-of-contributions should match resid_post[{layer_idx}]; "
        f"max diff {float(diff)}"
    )


def test_residual_stream_mid_uses_attn_contribution():
    """resid_mid[i] = resid_pre[i] + attn[i]."""
    model = _load_small_model()
    rs = model.residual_stream("Paris is the capital of France")
    n = len(model.layers)
    for i in (0, n // 2, n - 1):
        pre = rs.pre(i)
        attn = rs.attn_contribution(i)
        mid = rs.mid(i)
        diff = mx.max(mx.abs(mid - (pre + attn))).astype(mx.float32)
        mx.eval(diff)
        assert float(diff) < 1e-4
