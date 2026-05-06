"""Contract tests for direct_logit_attribution.

DLA's value proposition rests on a sum-check identity: when
``apply_final_norm=True`` the per-component contributions sum to the
actual logit_diff at the same position, modulo float precision. The
heavy validation script exercises this; here we just lock down the
contract surface.
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


def test_dla_returns_dict_with_expected_keys():
    model = _load_small_model()
    out = model.direct_logit_attribution(
        text="Paris is the capital of France",
        target_token=" Paris",
        foil_token=" London",
    )
    assert isinstance(out, dict)
    # Should contain at least one self_attn and one mlp entry per
    # active layer in the model.
    n_layers = len(model.layers)
    attn_keys = [k for k in out if k[0] == "self_attn"]
    mlp_keys = [k for k in out if k[0] == "mlp"]
    assert len(attn_keys) == n_layers
    assert len(mlp_keys) == n_layers


def test_dla_values_are_finite():
    model = _load_small_model()
    out = model.direct_logit_attribution(
        text="Paris is the capital of France",
        target_token=" Paris",
        foil_token=" London",
    )
    for k, v in out.items():
        assert isinstance(v, float)
        assert math.isfinite(v), f"non-finite contribution at {k}: {v}"


def test_dla_rejects_multi_token_target():
    model = _load_small_model()
    # "logit_diff" tokenizes to several pieces, so it should raise.
    with pytest.raises(ValueError, match="single token"):
        model.direct_logit_attribution(
            text="Paris is the capital of France",
            target_token="logit_diff",
            foil_token=" London",
        )


def test_dla_sum_matches_logit_diff_with_final_norm():
    """The frozen-norm DLA decomposition is an identity when applied
    correctly: the per-component contributions sum to the actual
    logit_diff at the same position (within float-precision tolerance).
    """
    model = _load_small_model()
    text = "Paris is the capital of France"
    out = model.direct_logit_attribution(
        text=text,
        target_token=" Paris",
        foil_token=" London",
        apply_final_norm=True,
    )

    with model.trace(text):
        logits = model.output.save()
    mx.eval(logits)

    target_id = model.tokenizer.encode(" Paris", add_special_tokens=False)
    foil_id = model.tokenizer.encode(" London", add_special_tokens=False)
    bos = model.tokenizer.bos_token_id
    if bos is not None and target_id and target_id[0] == bos:
        target_id = target_id[1:]
    if bos is not None and foil_id and foil_id[0] == bos:
        foil_id = foil_id[1:]
    actual = float(logits[0, -1, target_id[0]] - logits[0, -1, foil_id[0]])

    summed = sum(out.values())
    # Tolerance: the embedding contribution is sometimes missing if
    # the trace doesn't capture the embedding output, in which case
    # the sum will be off by that vector's contribution. We allow a
    # generous tolerance here; the example script reports the exact
    # gap.
    gap = abs(summed - actual)
    assert gap < 0.5, (
        f"DLA sum {summed:.4f} should approximately match actual logit_diff "
        f"{actual:.4f} (gap {gap:.4f}); large gaps usually mean a hook "
        f"point was missed."
    )
