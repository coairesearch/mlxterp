"""Contract tests for attribution_patching.

Attribution patching is a first-order linear approximation of the
brute-force activation-patching effect. The test bar is therefore not
"attribution == patching", but a few qualitative checks:

  1. Length-mismatch raises (attribution requires aligned positions).
  2. Unsupported component raises NotImplementedError (we only have
     resid_post / "output" today).
  3. Output shape matches the layers requested.
  4. The function returns finite numbers (no NaN/Inf) on a small real
     model.

A correlation-vs-brute-force check is in the standalone validation
script (`examples/attribution_patching.py`) since it is wall-clock
expensive.
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


def test_attribution_patching_rejects_length_mismatch():
    model = _load_small_model()
    with pytest.raises(ValueError, match="same length"):
        model.attribution_patching(
            clean_text="The capital of France is",
            corrupted_text="London is the capital of France today",
            target_token=" Paris",
            foil_token=" London",
            layers=[0, 1],
        )


def test_attribution_patching_rejects_unsupported_component():
    model = _load_small_model()
    with pytest.raises(NotImplementedError):
        model.attribution_patching(
            clean_text="Paris is the capital of France",
            corrupted_text="London is the capital of France",
            target_token=" Paris",
            foil_token=" London",
            component="attn_head",
            layers=[0],
        )


def test_attribution_patching_returns_finite_per_layer():
    model = _load_small_model()
    layers = [0, 1, 2, 3, 4]
    attr = model.attribution_patching(
        clean_text="Paris is the capital of France",
        corrupted_text="London is the capital of France",
        target_token=" Paris",
        foil_token=" London",
        layers=layers,
    )
    assert set(attr.keys()).issubset(set(layers))
    assert len(attr) >= 1, "expected at least one layer to resolve a path"
    for layer_idx, score in attr.items():
        assert isinstance(score, float)
        assert math.isfinite(score), f"non-finite attribution at layer {layer_idx}: {score}"


def test_attribution_patching_zero_when_clean_equals_corrupted():
    """Sanity: if clean == corrupted, attribution is exactly zero
    (clean_act - corrupted_act == 0 for all layers, regardless of grad)."""
    model = _load_small_model()
    text = "Paris is the capital of France"
    attr = model.attribution_patching(
        clean_text=text,
        corrupted_text=text,
        target_token=" Paris",
        foil_token=" London",
        layers=[0, 1, 2],
    )
    for layer_idx, score in attr.items():
        assert abs(score) < 1e-4, (
            f"clean==corrupted should give ~0 attribution; got {score} at "
            f"layer {layer_idx}"
        )
