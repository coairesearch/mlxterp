"""Tests for ``model.causal_trace``: clean/corrupted comparative tracing."""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
mlx_lm = pytest.importorskip("mlx_lm")

from mlxterp import InterpretableModel


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"
CLEAN = "Paris is the capital of France"
CORRUPTED = "London is the capital of France"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    base, tok = mlx_lm.load(MODEL_NAME)
    return base, tok


@pytest.fixture(scope="module")
def interp_model(model_and_tokenizer):
    base, tok = model_and_tokenizer
    return InterpretableModel(base, tokenizer=tok)


def test_clean_output_captured_on_enter(interp_model):
    """Clean run executes during __enter__; clean_output is available
    immediately, even before any patches are scheduled."""
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        assert ct.clean_output is not None
        # Shape: (batch=1, seq_len, vocab)
        assert ct.clean_output.ndim == 3


def test_corrupted_output_runs_lazily(interp_model):
    """Corrupted run is deferred until .output is read."""
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        # No patches; output should still be runnable (just unmodified
        # corrupted run)
        out = ct.output
        assert out is not None
        assert out.shape == ct.clean_output.shape


def test_patch_returns_self_for_chaining(interp_model):
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        ret = ct.patch("layers.5")
        assert ret is ct
        assert "layers.5" in ct.patches


def test_patch_actually_changes_output(interp_model):
    """A scheduled patch should produce a different corrupted output
    than an unpatched corrupted run."""
    # No patches: just baseline corrupted
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct_none:
        unpatched = ct_none.output

    # With a mid-network patch
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct_patched:
        ct_patched.patch("layers.5")
        patched = ct_patched.output

    # Outputs should differ since we swapped a layer's activation
    diff = float(mx.linalg.norm(unpatched - patched))
    assert diff > 0.0, (
        "Patching layers.5 didn't change the output. Either the patch "
        "didn't apply, or layers.5 makes no contribution at all (very "
        "unlikely). Either way, regression."
    )


def test_patch_after_output_raises(interp_model):
    """Calling patch() after .output has been read should error
    rather than silently dropping the patch."""
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        _ = ct.output  # triggers corrupted run
        with pytest.raises(RuntimeError, match="after \\.output"):
            ct.patch("layers.7")


def test_unknown_module_path_raises_keyerror(interp_model):
    """A patch on a nonexistent module should raise KeyError when the
    corrupted run tries to look up the clean activation."""
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        ct.patch("layers.999.nonexistent")
        with pytest.raises(KeyError, match="No clean activation"):
            _ = ct.output


def test_user_interventions_compose_with_patches(interp_model):
    """User-supplied interventions on the corrupted run apply alongside
    the scheduled patches, not instead of them."""
    from mlxterp import interventions as iv

    with interp_model.causal_trace(
        CLEAN, CORRUPTED, interventions={"layers.10": iv.scale(0.5)}
    ) as ct:
        ct.patch("layers.5")
        out = ct.output
    assert out is not None
    # We can't easily assert exact values, but the call shouldn't error
    # and the output should be a same-shaped array.
    assert out.ndim == 3


def test_output_is_idempotent(interp_model):
    """Reading .output twice must not re-run the corrupted forward
    (the lazy cache should hold)."""
    with interp_model.causal_trace(CLEAN, CORRUPTED) as ct:
        ct.patch("layers.5")
        first = ct.output
        second = ct.output
    diff = float(mx.linalg.norm(first - second))
    assert diff == 0.0
