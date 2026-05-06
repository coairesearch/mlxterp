"""Contract tests for path_patching.

Path patching with sender-only-clean / everything-else-frozen-at-corrupted
is a real causal effect on the metric. The contracts:

  1. Length-mismatch raises.
  2. Unresolvable sender raises with a useful message.
  3. With sender == something irrelevant (early layer's MLP on a tiny
     prompt), the effect is small in magnitude.
  4. Returns a finite number on a real model.
  5. The numeric is reproducible (deterministic forward, same seed).
"""

import math
import pytest

pytestmark = pytest.mark.timeout(180)


def _load_small_model():
    from mlxterp import InterpretableModel
    from mlx_lm import load

    base, tok = load("mlx-community/Llama-3.2-1B-Instruct-4bit")
    return InterpretableModel(base, tokenizer=tok)


def test_path_patching_rejects_length_mismatch():
    model = _load_small_model()
    with pytest.raises(ValueError, match="same length"):
        model.path_patching(
            clean_text="The capital of France is",
            corrupted_text="London is the capital of France today",
            sender="layers.5.mlp",
            target_token=" Paris",
            foil_token=" London",
        )


def test_path_patching_rejects_unresolvable_sender():
    model = _load_small_model()
    with pytest.raises(ValueError, match="not found"):
        model.path_patching(
            clean_text="Paris is the capital of France",
            corrupted_text="London is the capital of France",
            sender="layers.999.mlp",
            target_token=" Paris",
            foil_token=" London",
        )


def test_path_patching_returns_finite():
    model = _load_small_model()
    eff = model.path_patching(
        clean_text="Paris is the capital of France",
        corrupted_text="London is the capital of France",
        sender="layers.10.mlp",
        target_token=" Paris",
        foil_token=" London",
    )
    assert isinstance(eff, float)
    assert math.isfinite(eff)


def test_path_patching_clean_equals_corrupted_is_zero():
    """If clean == corrupted, the sender's clean value equals its
    corrupted value, so the freeze run is the corrupted run, and the
    effect is exactly zero (within numerical tolerance)."""
    model = _load_small_model()
    text = "Paris is the capital of France"
    eff = model.path_patching(
        clean_text=text,
        corrupted_text=text,
        sender="layers.10.mlp",
        target_token=" Paris",
        foil_token=" London",
    )
    assert abs(eff) < 1e-3, (
        f"clean==corrupted should give ~0 effect; got {eff}"
    )
