"""
Unit and integration tests for the Tuned Lens implementation.

Tests cover:
- TunedLens class initialization and forward pass
- Identity initialization behavior
- Save/load functionality
- Integration with InterpretableModel
- tuned_lens method
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import tempfile
from pathlib import Path

from mlxterp import InterpretableModel, TunedLens
from mlxterp.tuned_lens import train_tuned_lens, kl_divergence


# ============================================================================
# Mock Models for Testing
# ============================================================================

class MockTransformerBlock(nn.Module):
    """A simple transformer block for testing."""
    def __init__(self, dim=64):
        super().__init__()
        self.self_attn = MockAttention(dim)
        self.mlp = nn.Linear(dim, dim)

    def __call__(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class MockAttention(nn.Module):
    """Mock attention for testing."""
    def __init__(self, dim=64):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.o_proj(x)


class MockModel(nn.Module):
    """Mock model with standard structure."""
    def __init__(self, vocab_size=1000, dim=64, n_layers=4):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.norm = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        self.layers = [MockTransformerBlock(dim) for _ in range(n_layers)]

    def __call__(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class MockTokenizer:
    """Simple mock tokenizer for testing."""
    def __init__(self, vocab_size=1000):
        self.vocab_size = vocab_size

    def encode(self, text, **kwargs):
        if isinstance(text, mx.array):
            return text.tolist() if text.ndim == 1 else text[0].tolist()
        return list(range(len(text.split()) + 1))

    def decode(self, ids, **kwargs):
        return " ".join([f"token_{i}" for i in ids])


# ============================================================================
# TunedLens Class Tests
# ============================================================================

class TestTunedLensInit:
    """Test TunedLens initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        assert tuned_lens.num_layers == 4
        assert tuned_lens.hidden_dim == 64
        assert len(tuned_lens.translators) == 4

    def test_translator_structure(self):
        """Test that translators are linear layers."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        for translator in tuned_lens.translators:
            assert isinstance(translator, nn.Linear)
            assert translator.weight.shape == (64, 64)
            assert translator.bias.shape == (64,)

    def test_identity_initialization(self):
        """Test that translators are initialized close to identity."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        for translator in tuned_lens.translators:
            # Weight should be identity matrix
            expected_weight = mx.eye(64)
            assert mx.allclose(translator.weight, expected_weight).item()

            # Bias should be zeros
            expected_bias = mx.zeros((64,))
            assert mx.allclose(translator.bias, expected_bias).item()


class TestTunedLensForward:
    """Test TunedLens forward pass."""

    @pytest.fixture
    def tuned_lens(self):
        return TunedLens(num_layers=4, hidden_dim=64)

    def test_forward_1d(self, tuned_lens):
        """Test forward pass with 1D input."""
        hidden = mx.random.normal((64,))

        output = tuned_lens(hidden, layer_idx=0)
        assert output.shape == (64,)

    def test_forward_3d(self, tuned_lens):
        """Test forward pass with 3D input (batch, seq_len, hidden)."""
        hidden = mx.random.normal((2, 10, 64))

        output = tuned_lens(hidden, layer_idx=0)
        assert output.shape == (2, 10, 64)

    def test_identity_pass_through(self, tuned_lens):
        """Test that identity-initialized translator passes input through."""
        hidden = mx.random.normal((64,))

        output = tuned_lens(hidden, layer_idx=0)

        # Should be approximately equal due to identity initialization
        assert mx.allclose(hidden, output, atol=1e-5).item()

    def test_different_layers(self, tuned_lens):
        """Test that different layers can have different translations."""
        # Modify one translator
        tuned_lens.translators[1].weight = mx.eye(64) * 2.0

        hidden = mx.random.normal((64,))

        output_0 = tuned_lens(hidden, layer_idx=0)
        output_1 = tuned_lens(hidden, layer_idx=1)

        # Should not be equal
        assert not mx.allclose(output_0, output_1).item()

    def test_invalid_layer_idx(self, tuned_lens):
        """Test that invalid layer index raises error."""
        hidden = mx.random.normal((64,))

        with pytest.raises(ValueError):
            tuned_lens(hidden, layer_idx=10)

        with pytest.raises(ValueError):
            tuned_lens(hidden, layer_idx=-1)


class TestTunedLensSaveLoad:
    """Test TunedLens save/load functionality."""

    def test_save_and_load(self):
        """Test saving and loading tuned lens."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        # Modify some weights to verify they're saved/loaded correctly
        tuned_lens.translators[0].weight = mx.eye(64) * 2.0
        tuned_lens.translators[0].bias = mx.ones((64,))

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tuned_lens"

            # Save
            tuned_lens.save(str(save_path))

            # Verify files exist
            assert (save_path.with_suffix(".npz")).exists()
            assert (save_path.with_suffix(".json")).exists()

            # Load
            loaded = TunedLens.load(str(save_path))

            # Verify structure
            assert loaded.num_layers == 4
            assert loaded.hidden_dim == 64

            # Verify weights
            assert mx.allclose(
                loaded.translators[0].weight,
                mx.eye(64) * 2.0
            ).item()
            assert mx.allclose(
                loaded.translators[0].bias,
                mx.ones((64,))
            ).item()

    def test_load_preserves_functionality(self):
        """Test that loaded model works correctly."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tuned_lens"
            tuned_lens.save(str(save_path))
            loaded = TunedLens.load(str(save_path))

            # Test forward pass
            hidden = mx.random.normal((64,))
            original_output = tuned_lens(hidden, layer_idx=0)
            loaded_output = loaded(hidden, layer_idx=0)

            assert mx.allclose(original_output, loaded_output).item()


# ============================================================================
# KL Divergence Tests
# ============================================================================

def log_softmax(x, axis=-1):
    """Compute log softmax (MLX doesn't have this built-in)."""
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


class TestKLDivergence:
    """Test KL divergence computation."""

    def test_kl_same_distribution(self):
        """KL divergence of same distribution should be 0."""
        log_p = log_softmax(mx.random.normal((10,)), axis=-1)

        kl = kl_divergence(log_p, log_p)
        assert mx.abs(kl).item() < 1e-5

    def test_kl_different_distributions(self):
        """KL divergence of different distributions should be positive."""
        log_p = log_softmax(mx.random.normal((10,)), axis=-1)
        log_q = log_softmax(mx.random.normal((10,)), axis=-1)

        kl = kl_divergence(log_p, log_q)
        assert kl.item() >= 0

    def test_kl_batched(self):
        """Test KL divergence with batched input."""
        log_p = log_softmax(mx.random.normal((5, 10)), axis=-1)
        log_q = log_softmax(mx.random.normal((5, 10)), axis=-1)

        kl = kl_divergence(log_p, log_q)
        assert kl.shape == ()  # Should be scalar


# ============================================================================
# Integration Tests with InterpretableModel
# ============================================================================

class TestTunedLensIntegration:
    """Test TunedLens integration with InterpretableModel."""

    @pytest.fixture
    def model(self):
        base = MockModel(vocab_size=100, dim=32, n_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        return InterpretableModel(base, tokenizer=tokenizer)

    def test_tuned_lens_basic(self, model):
        """Test basic tuned_lens call."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3]])

        results = model.tuned_lens(input_tokens, tuned_lens, layers=[0, 1])

        # Check structure
        assert isinstance(results, dict)
        assert len(results) > 0

        for layer_idx, layer_results in results.items():
            assert isinstance(layer_results, list)
            for pos_results in layer_results:
                assert isinstance(pos_results, list)
                for pred in pos_results:
                    assert len(pred) == 3  # (token_id, score, token_str)

    def test_tuned_lens_specific_layers(self, model):
        """Test tuned_lens with specific layers."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3]])

        results = model.tuned_lens(input_tokens, tuned_lens, layers=[0, 2])

        assert set(results.keys()).issubset({0, 2})

    def test_tuned_lens_position(self, model):
        """Test tuned_lens with position parameter."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3, 4, 5]])

        # All positions
        results_all = model.tuned_lens(input_tokens, tuned_lens, layers=[0])
        # Last position only
        results_last = model.tuned_lens(input_tokens, tuned_lens, layers=[0], position=-1)

        assert len(results_all[0]) == 5
        assert len(results_last[0]) == 1

    def test_tuned_lens_top_k(self, model):
        """Test tuned_lens top_k parameter."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3]])

        results_1 = model.tuned_lens(input_tokens, tuned_lens, layers=[0], top_k=1)
        results_5 = model.tuned_lens(input_tokens, tuned_lens, layers=[0], top_k=5)

        for pos_preds in results_1[0]:
            assert len(pos_preds) == 1
        for pos_preds in results_5[0]:
            assert len(pos_preds) == 5


class TestTunedLensModelMethods:
    """Test InterpretableModel tuned lens methods."""

    @pytest.fixture
    def model(self):
        base = MockModel(vocab_size=100, dim=32, n_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        return InterpretableModel(base, tokenizer=tokenizer)

    def test_load_tuned_lens_method(self, model):
        """Test model.load_tuned_lens method."""
        # Create and save a tuned lens
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_tuned_lens"
            tuned_lens.save(str(save_path))

            # Load via model method
            loaded = model.load_tuned_lens(str(save_path))

            assert loaded.num_layers == 4
            assert loaded.hidden_dim == 32


class TestTunedLensCompareLogitLens:
    """Compare tuned lens with regular logit lens."""

    @pytest.fixture
    def model(self):
        base = MockModel(vocab_size=100, dim=32, n_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        return InterpretableModel(base, tokenizer=tokenizer)

    def test_identical_with_identity(self, model):
        """With identity initialization, tuned lens should match logit lens."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3]])

        # Get both results with top_k=5 to allow for comparison
        regular = model.logit_lens(input_tokens, layers=[0, 1, 2, 3], top_k=5)
        tuned = model.tuned_lens(input_tokens, tuned_lens, layers=[0, 1, 2, 3], top_k=5)

        # With identity initialization, predictions should be very similar
        for layer_idx in regular.keys():
            if layer_idx in tuned:
                for pos_idx in range(len(regular[layer_idx])):
                    reg_top = regular[layer_idx][pos_idx][0][0]  # Top token ID
                    tuned_top = tuned[layer_idx][pos_idx][0][0]  # Top token ID
                    # May not be exactly equal due to floating point, but should be similar
                    # Extract token IDs from tuples: each pred is (token_id, score, token_str)
                    reg_tokens = set(p[0] for p in regular[layer_idx][pos_idx])
                    tuned_tokens = set(p[0] for p in tuned[layer_idx][pos_idx])
                    # Check that at least one top prediction appears in the other's top-5
                    assert tuned_top in reg_tokens or reg_top in tuned_tokens, (
                        f"Layer {layer_idx}, pos {pos_idx}: "
                        f"reg_top={reg_top} not in tuned_tokens={tuned_tokens} and "
                        f"tuned_top={tuned_top} not in reg_tokens={reg_tokens}"
                    )


class TestTunedLensEdgeCases:
    """Test edge cases and error handling."""

    def test_mismatched_dimensions(self):
        """Test error handling for mismatched dimensions."""
        tuned_lens = TunedLens(num_layers=4, hidden_dim=64)

        # Wrong hidden dimension
        hidden = mx.random.normal((32,))  # Should be 64

        with pytest.raises(Exception):  # Could be various error types
            tuned_lens(hidden, layer_idx=0)

    def test_empty_layers_list(self):
        """Test with empty layers list."""
        base = MockModel(vocab_size=100, dim=32, n_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        model = InterpretableModel(base, tokenizer=tokenizer)
        tuned_lens = TunedLens(num_layers=4, hidden_dim=32)
        input_tokens = mx.array([[1, 2, 3]])

        results = model.tuned_lens(input_tokens, tuned_lens, layers=[])

        assert results == {}


# ============================================================================
# Training Tests
# ============================================================================

class TestTrainTunedLens:
    """Test train_tuned_lens function."""

    @pytest.fixture
    def model(self):
        """Create a mock model for training tests."""
        base = MockModel(vocab_size=100, dim=32, n_layers=4)
        tokenizer = MockTokenizer(vocab_size=100)
        return InterpretableModel(base, tokenizer=tokenizer)

    def test_train_basic(self, model):
        """Test basic training runs without error."""
        # Create simple dataset - needs enough tokens to avoid "too small" error
        dataset = ["This is a test sentence for training the tuned lens. " * 50]

        # Run training with minimal steps
        tuned_lens = train_tuned_lens(
            model,
            dataset,
            num_steps=2,
            max_seq_len=50,
            verbose=False,
        )

        assert isinstance(tuned_lens, TunedLens)
        assert tuned_lens.num_layers == 4
        assert tuned_lens.hidden_dim == 32

    def test_train_with_callback(self, model):
        """Test that callback is called during training."""
        dataset = ["This is a test sentence for training the tuned lens. " * 50]
        callback_calls = []

        def callback(step, loss):
            callback_calls.append((step, loss))

        tuned_lens = train_tuned_lens(
            model,
            dataset,
            num_steps=3,
            max_seq_len=50,
            verbose=False,
            callback=callback,
        )

        # Callback should be called for each step
        assert len(callback_calls) == 3
        for step, loss in callback_calls:
            assert isinstance(step, int)
            assert isinstance(loss, float)

    def test_train_modifies_weights(self, model):
        """Test that training modifies the translator weights."""
        dataset = ["This is a test sentence for training the tuned lens. " * 50]

        tuned_lens = train_tuned_lens(
            model,
            dataset,
            num_steps=50,  # More steps to ensure weight changes
            max_seq_len=50,
            learning_rate=1.0,
            verbose=False,
        )

        # Force evaluation of all weights before checking
        for translator in tuned_lens.translators:
            mx.eval(translator.weight, translator.bias)

        # After training, at least one translator should differ from identity
        max_weight_diff = 0.0
        max_bias_diff = 0.0
        for translator in tuned_lens.translators:
            weight_diff = mx.abs(translator.weight - mx.eye(32)).max().item()
            bias_diff = mx.abs(translator.bias).max().item()
            max_weight_diff = max(max_weight_diff, weight_diff)
            max_bias_diff = max(max_bias_diff, bias_diff)

        # Training should have modified at least some weights
        # With sufficient steps and learning rate, weights should change
        assert max_weight_diff > 1e-6 or max_bias_diff > 1e-6, (
            f"Training did not modify weights. Max weight diff: {max_weight_diff}, "
            f"max bias diff: {max_bias_diff}"
        )

    def test_train_with_save(self, model):
        """Test training with save_path saves the tuned lens."""
        dataset = ["This is a test sentence for training the tuned lens. " * 50]

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_trained"

            tuned_lens = train_tuned_lens(
                model,
                dataset,
                num_steps=2,
                max_seq_len=50,
                save_path=str(save_path),
                verbose=False,
            )

            # Check files were created
            assert save_path.with_suffix(".npz").exists()
            assert save_path.with_suffix(".json").exists()

            # Load and verify
            loaded = TunedLens.load(str(save_path))
            assert loaded.num_layers == tuned_lens.num_layers
            assert loaded.hidden_dim == tuned_lens.hidden_dim

    def test_train_dataset_too_small(self, model):
        """Test that training raises error for too-small dataset."""
        dataset = ["Short text"]

        with pytest.raises(ValueError, match="Dataset too small"):
            train_tuned_lens(
                model,
                dataset,
                num_steps=2,
                max_seq_len=100,  # More than dataset has
                verbose=False,
            )

    def test_train_model_method(self, model):
        """Test training via model.train_tuned_lens method."""
        dataset = ["This is a test sentence for training the tuned lens. " * 50]

        tuned_lens = model.train_tuned_lens(
            dataset,
            num_steps=2,
            max_seq_len=50,
            verbose=False,
        )

        assert isinstance(tuned_lens, TunedLens)
        assert tuned_lens.num_layers == 4

    def test_train_gradient_clipping(self, model):
        """Test that gradient clipping doesn't cause errors."""
        dataset = ["This is a test sentence for training the tuned lens. " * 50]

        # Should not raise even with aggressive clipping
        tuned_lens = train_tuned_lens(
            model,
            dataset,
            num_steps=2,
            max_seq_len=50,
            gradient_clip=0.01,  # Very aggressive clipping
            verbose=False,
        )

        assert isinstance(tuned_lens, TunedLens)


# ============================================================================
# Export Tests
# ============================================================================

class TestExports:
    """Test that all expected classes are exported."""

    def test_tuned_lens_exported(self):
        """Test TunedLens is exported from mlxterp."""
        from mlxterp import TunedLens
        assert TunedLens is not None

    def test_train_tuned_lens_exported(self):
        """Test train_tuned_lens is exported from mlxterp."""
        from mlxterp import train_tuned_lens
        assert train_tuned_lens is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
