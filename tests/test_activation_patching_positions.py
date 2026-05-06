"""Tests for the per-position activation patching extensions.

Covers:
- positions=None still returns Dict (backwards compat)
- positions='all' returns ndarray of shape (n_layers, seq_len)
- positions=[i, j, k] returns ndarray of shape (n_layers, 3)
- mismatched-length inputs raise ValueError under positions != None
- logit_diff metric requires clean_target/corrupted_target
- replace_at_positions intervention is wired through the public API
"""

from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
np = pytest.importorskip("numpy")
mlx_lm = pytest.importorskip("mlx_lm")

from mlxterp import InterpretableModel
from mlxterp import interventions as iv


MODEL_NAME = "lmstudio-community/Qwen3-4B-Instruct-2507-MLX-8bit"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    base, tok = mlx_lm.load(MODEL_NAME)
    return base, tok


@pytest.fixture(scope="module")
def interp_model(model_and_tokenizer):
    base, tok = model_and_tokenizer
    return InterpretableModel(base, tokenizer=tok)


CLEAN = "Paris is the capital of France"
CORRUPTED = "London is the capital of France"


def test_positions_none_returns_dict_backcompat(interp_model):
    out = interp_model.activation_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        component="output",
        layers=[0, 12, 30],
        metric="l2",
    )
    assert isinstance(out, dict)
    assert set(out.keys()) == {0, 12, 30}
    for v in out.values():
        assert isinstance(v, float)


def test_positions_all_returns_matrix(interp_model, model_and_tokenizer):
    _, tok = model_and_tokenizer
    seq_len = len(tok.encode(CLEAN))
    out = interp_model.activation_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        component="output",
        layers=[0, 12, 30],
        positions="all",
        metric="l2",
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, seq_len)
    assert out.dtype == np.float32


def test_positions_list_returns_matrix(interp_model):
    out = interp_model.activation_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        component="output",
        layers=[0, 12, 30],
        positions=[0, 2, 5],
        metric="l2",
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (3, 3)


def test_position_level_requires_same_length(interp_model):
    with pytest.raises(ValueError, match="same length"):
        interp_model.activation_patching(
            clean_text="Paris is the capital",
            corrupted_text="London is the capital of France",
            component="output",
            positions="all",
        )


def test_logit_diff_requires_targets(interp_model):
    with pytest.raises(ValueError, match="clean_target"):
        interp_model.activation_patching(
            clean_text=CLEAN,
            corrupted_text=CORRUPTED,
            component="output",
            metric="logit_diff",
        )


def test_logit_diff_metric_runs(interp_model):
    out = interp_model.activation_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        component="output",
        layers=[12, 30],
        positions=[0, 5],
        metric="logit_diff",
        clean_target=" Paris",
        corrupted_target=" London",
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == (2, 2)


def test_replace_at_positions_helper(model_and_tokenizer):
    """Direct unit test of the intervention helper itself."""
    seq_len, hidden = 5, 8
    activation = mx.ones((1, seq_len, hidden))
    value = mx.zeros((1, seq_len, hidden))

    # Replace position 2 only
    fn = iv.replace_at_positions(value, positions=2)
    out = fn(activation)
    out_np = np.asarray(out)
    assert out_np[0, 0, 0] == 1.0  # position 0 untouched
    assert out_np[0, 1, 0] == 1.0  # position 1 untouched
    assert out_np[0, 2, 0] == 0.0  # position 2 replaced with zero
    assert out_np[0, 3, 0] == 1.0  # position 3 untouched
    assert out_np[0, 4, 0] == 1.0  # position 4 untouched

    # Replace multiple positions
    fn2 = iv.replace_at_positions(value, positions=[1, 3])
    out2 = fn2(activation)
    out2_np = np.asarray(out2)
    assert out2_np[0, 0, 0] == 1.0
    assert out2_np[0, 1, 0] == 0.0  # replaced
    assert out2_np[0, 2, 0] == 1.0
    assert out2_np[0, 3, 0] == 0.0  # replaced
    assert out2_np[0, 4, 0] == 1.0


def test_kl_divergence_metric_runs(interp_model):
    out = interp_model.activation_patching(
        clean_text=CLEAN,
        corrupted_text=CORRUPTED,
        component="output",
        layers=[12, 30],
        metric="kl_divergence",
    )
    assert isinstance(out, dict)
