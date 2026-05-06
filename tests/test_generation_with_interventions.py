"""Tests for InterpretableModel.generate with persistent interventions.

These cover the basic contract:
- generate() with no interventions matches raw mlx_lm.generate
- generate() with interventions diverges from baseline
- patches are properly restored after generate returns
- the lower-level skip_forward=True trace path produces the same output
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from mlxterp import InterpretableModel
from mlxterp import interventions as iv
from mlxterp.core.trace import Trace


# Use a tiny MLX-quantised model so the test runs in reasonable time.
# Qwen3-4B works but takes ~5s; for CI we'd want something smaller. The
# test is parameterised on model name so the maintainer can swap to a
# smaller default.
MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
PROMPT = "Q: What is 2 + 2? A:"
MAX_TOKENS = 16


@pytest.fixture(scope="module")
def model_and_tokenizer():
    base, tok = mlx_lm.load(MODEL_NAME)
    return base, tok


@pytest.fixture(scope="module")
def interp_model(model_and_tokenizer):
    base, tok = model_and_tokenizer
    return InterpretableModel(base, tokenizer=tok)


def test_generate_no_interventions_matches_raw_mlx_lm(interp_model, model_and_tokenizer):
    """Generate with interventions=None should be a transparent passthrough."""
    base, tok = model_and_tokenizer
    raw = mlx_lm.generate(base, tok, PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    api = interp_model.generate(PROMPT, max_tokens=MAX_TOKENS)
    assert raw == api, (
        "generate() without interventions must match raw mlx_lm.generate "
        "exactly, otherwise patch lifecycle has a side effect"
    )


def test_generate_with_intervention_changes_output(interp_model):
    """Zeroing a mid-network layer should change the generated text."""
    baseline = interp_model.generate(PROMPT, max_tokens=MAX_TOKENS)
    ablated = interp_model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.5": iv.zero_out},
    )
    assert ablated != baseline, (
        "generate() with iv.zero_out on a mid-layer must produce different "
        "output. If output matched baseline, the patch never fired during "
        "the autoregressive loop."
    )


def test_generate_restores_patches_on_exit(interp_model):
    """Generation under intervention must not leak into subsequent calls."""
    baseline_before = interp_model.generate(PROMPT, max_tokens=MAX_TOKENS)
    _ = interp_model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.5": iv.zero_out},
    )
    baseline_after = interp_model.generate(PROMPT, max_tokens=MAX_TOKENS)
    assert baseline_before == baseline_after, (
        "Patches leaked between generate() calls. The intervention should be "
        "active only for the call it was passed to, then restored."
    )


def test_skip_forward_trace_matches_generate_api(interp_model, model_and_tokenizer):
    """The lower-level trace(skip_forward=True) + manual mlx_lm.generate
    pattern must produce identical output to InterpretableModel.generate.
    The high-level API is just sugar over this pattern."""
    base, tok = model_and_tokenizer
    api = interp_model.generate(
        PROMPT,
        max_tokens=MAX_TOKENS,
        interventions={"layers.5": iv.zero_out},
    )
    with Trace(
        model_forward=interp_model._forward,
        inputs=PROMPT,
        tokenizer=tok,
        interventions={"layers.5": iv.zero_out},
        interpretable_model=interp_model,
        skip_forward=True,
    ):
        manual = mlx_lm.generate(base, tok, PROMPT, max_tokens=MAX_TOKENS, verbose=False)
    assert api == manual
