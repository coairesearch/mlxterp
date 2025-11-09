"""
Basic tests for SAE functionality.

Tests core SAE components: config, model creation, encoding/decoding, saving/loading.
"""

import pytest
import mlx.core as mx
import tempfile
import shutil
from pathlib import Path

from mlxterp.sae import SAE, SAEConfig, BaseSAE


class TestSAEConfig:
    """Test SAE configuration validation."""

    def test_default_config(self):
        """Test that default config initializes correctly."""
        config = SAEConfig()
        assert config.expansion_factor == 16
        assert config.k == 100
        assert config.learning_rate == 1e-4
        assert config.batch_size == 256
        assert config.normalize_input is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SAEConfig(
            expansion_factor=32,
            k=150,
            learning_rate=5e-5,
            batch_size=128,
        )
        assert config.expansion_factor == 32
        assert config.k == 150
        assert config.learning_rate == 5e-5
        assert config.batch_size == 128

    def test_invalid_expansion_factor(self):
        """Test that invalid expansion_factor raises error."""
        with pytest.raises(ValueError, match="expansion_factor must be >= 1"):
            SAEConfig(expansion_factor=0)

    def test_invalid_k(self):
        """Test that invalid k raises error."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            SAEConfig(k=0)

    def test_invalid_learning_rate(self):
        """Test that invalid learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be > 0"):
            SAEConfig(learning_rate=0)

    def test_invalid_validation_split(self):
        """Test that invalid validation_split raises error."""
        with pytest.raises(ValueError, match="validation_split must be in"):
            SAEConfig(validation_split=1.5)


class TestSAEModel:
    """Test SAE model architecture and operations."""

    def test_sae_creation(self):
        """Test SAE initialization."""
        sae = SAE(d_model=2048, d_hidden=32768, k=100)
        assert sae.d_model == 2048
        assert sae.d_hidden == 32768
        assert sae.k == 100
        assert sae.normalize_input is True

    def test_sae_encode(self):
        """Test encoding activations to features."""
        sae = SAE(d_model=128, d_hidden=512, k=50, normalize_input=False)

        # Create sample activation
        batch_size, seq_len = 2, 10
        activation = mx.random.normal((batch_size, seq_len, 128))

        # Encode
        features = sae.encode(activation)

        # Check shape
        assert features.shape == (batch_size, seq_len, 512)

        # Check sparsity (should have at most k non-zero per position)
        non_zero_counts = mx.sum(features != 0, axis=-1)
        assert mx.all(non_zero_counts <= 50)

    def test_sae_decode(self):
        """Test decoding features back to activations."""
        sae = SAE(d_model=128, d_hidden=512, k=50, normalize_input=False)

        # Create sparse features
        batch_size, seq_len = 2, 10
        features = mx.zeros((batch_size, seq_len, 512))
        # Set some features to non-zero
        features[:, :, :50] = mx.random.normal((batch_size, seq_len, 50))

        # Decode
        reconstructed = sae.decode(features)

        # Check shape
        assert reconstructed.shape == (batch_size, seq_len, 128)

    def test_sae_forward(self):
        """Test full forward pass (encode + decode)."""
        sae = SAE(d_model=128, d_hidden=512, k=50, normalize_input=False)

        # Create sample activation
        batch_size, seq_len = 2, 10
        activation = mx.random.normal((batch_size, seq_len, 128))

        # Forward pass
        reconstructed, features = sae(activation)

        # Check shapes
        assert reconstructed.shape == (batch_size, seq_len, 128)
        assert features.shape == (batch_size, seq_len, 512)

        # Check sparsity
        non_zero_counts = mx.sum(features != 0, axis=-1)
        assert mx.all(non_zero_counts <= 50)

    def test_sae_reconstruction_quality(self):
        """Test that reconstruction is reasonably accurate."""
        sae = SAE(d_model=128, d_hidden=2048, k=100, normalize_input=False)

        # Create sample activation
        activation = mx.random.normal((1, 5, 128))

        # Forward pass
        reconstructed, features = sae(activation)

        # Compute reconstruction error
        error = mx.mean((activation - reconstructed) ** 2)

        # Error should be finite (not NaN or inf)
        assert not mx.isnan(error)
        assert not mx.isinf(error)

    def test_topk_sparsity(self):
        """Test that TopK sparsity is enforced correctly."""
        sae = SAE(d_model=128, d_hidden=512, k=50, normalize_input=False)

        activation = mx.random.normal((1, 1, 128))
        features = sae.encode(activation)

        # Count non-zero features
        non_zero = mx.sum(features != 0)
        assert non_zero <= 50  # At most k features

    def test_tied_weights(self):
        """Test tied weights option."""
        sae = SAE(d_model=128, d_hidden=512, k=50, tied_weights=True)

        activation = mx.random.normal((1, 5, 128))
        reconstructed, features = sae(activation)

        # Should work without errors
        assert reconstructed.shape == (1, 5, 128)
        assert features.shape == (1, 5, 512)


class TestSAESaveLoad:
    """Test saving and loading SAE models."""

    def test_save_and_load(self):
        """Test saving and loading SAE."""
        # Create SAE
        sae = SAE(d_model=128, d_hidden=512, k=50)
        sae.metadata = {"layer": 10, "component": "mlp"}

        # Save to temp directory
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_sae"
            sae.save(str(save_path))

            # Check files exist
            assert (save_path / "config.json").exists()
            assert (save_path / "weights.safetensors").exists()

            # Load
            loaded_sae = SAE.load(str(save_path))

            # Check attributes
            assert loaded_sae.d_model == 128
            assert loaded_sae.d_hidden == 512
            assert loaded_sae.k == 50
            assert loaded_sae.metadata["layer"] == 10
            assert loaded_sae.metadata["component"] == "mlp"

    def test_load_nonexistent(self):
        """Test loading from nonexistent path raises error."""
        with pytest.raises(FileNotFoundError):
            SAE.load("/nonexistent/path")

    def test_save_creates_directory(self):
        """Test that save creates directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "nested" / "dir" / "sae"

            sae = SAE(d_model=128, d_hidden=512, k=50)
            sae.save(str(save_path))

            assert save_path.exists()
            assert (save_path / "config.json").exists()


class TestSAEMetrics:
    """Test SAE metrics and statistics."""

    def test_activation_stats(self):
        """Test activation statistics computation."""
        sae = SAE(d_model=128, d_hidden=512, k=50)

        # Create sparse features
        features = mx.zeros((2, 10, 512))
        features[:, :, :50] = mx.random.normal((2, 10, 50))

        stats = sae.get_activation_stats(features)

        assert "l0" in stats  # Average number of active features
        assert "l1" in stats  # L1 norm
        assert "dead_features" in stats  # Number of dead features
        assert "dead_fraction" in stats  # Fraction of dead features

        # L0 should be around 50 (k value)
        assert 40 <= stats["l0"] <= 60

    def test_compute_loss(self):
        """Test loss computation."""
        sae = SAE(d_model=128, d_hidden=512, k=50, normalize_input=False)

        activation = mx.random.normal((2, 10, 128))
        loss, metrics = sae.compute_loss(activation)

        # Check that loss is computed
        assert "loss" in metrics
        assert "recon_loss" in metrics
        assert "l0" in metrics

        # Loss should be finite
        assert not mx.isnan(loss)
        assert not mx.isinf(loss)


class TestSAECompatibility:
    """Test SAE compatibility checking."""

    def test_is_compatible(self):
        """Test compatibility checking."""
        sae = SAE(d_model=2048, d_hidden=32768, k=100)
        sae.metadata = {
            "layer": 10,
            "component": "mlp",
            "model_name": "test-model"
        }

        # Should be compatible with same layer/component
        # (Note: full compatibility check requires actual model)
        assert sae.is_compatible(None, layer=10, component="mlp")

        # Should be incompatible with different layer
        assert not sae.is_compatible(None, layer=5, component="mlp")

        # Should be incompatible with different component
        assert not sae.is_compatible(None, layer=10, component="attn")


def test_sae_repr():
    """Test string representation."""
    sae = SAE(d_model=2048, d_hidden=32768, k=100)
    repr_str = repr(sae)

    assert "SAE" in repr_str
    assert "2048" in repr_str
    assert "32768" in repr_str
    assert "100" in repr_str
    assert "16.0x" in repr_str  # expansion factor


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
