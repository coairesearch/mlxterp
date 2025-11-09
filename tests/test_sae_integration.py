"""
Integration tests for SAE training with real models.

Tests the full workflow: model creation, activation collection, SAE training.
"""

import pytest
import mlx.core as mx
import mlx.nn as nn
import tempfile
from pathlib import Path

from mlxterp import InterpretableModel, SAEConfig


# Skip if mlx_lm not available
pytest.importorskip("mlx_lm")


class TestSAEIntegration:
    """Integration tests with real models."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load a small model for testing."""
        # Use the smallest available model for faster tests
        model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
        return model

    @pytest.fixture
    def sample_texts(self):
        """Generate sample texts for training."""
        return [
            "The capital of France is Paris",
            "Machine learning is a subset of artificial intelligence",
            "Python is a popular programming language",
            "The sun rises in the east",
            "Water freezes at zero degrees Celsius",
            "The Earth orbits around the Sun",
            "Photosynthesis occurs in plant leaves",
            "Shakespeare wrote many famous plays",
        ]

    def test_train_sae_simple(self, model, sample_texts):
        """Test basic SAE training with minimal config."""
        # Use small config for fast testing
        config = SAEConfig(
            expansion_factor=4,  # Small for testing
            k=20,
            num_epochs=1,  # Just 1 epoch for testing
            batch_size=4,
            warmup_steps=10,
        )

        # Train SAE
        sae = model.train_sae(
            layer=5,  # Middle layer
            dataset=sample_texts,
            config=config,
            verbose=False,  # Quiet for testing
        )

        # Check SAE was created
        assert sae is not None
        assert sae.d_model > 0
        assert sae.d_hidden == sae.d_model * 4

        # Check metadata
        assert sae.metadata["layer"] == 5
        assert sae.metadata["component"] == "mlp"

    def test_train_and_save(self, model, sample_texts):
        """Test training and saving SAE."""
        config = SAEConfig(
            expansion_factor=4,
            k=20,
            num_epochs=1,
            batch_size=4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_sae"

            # Train and save
            sae = model.train_sae(
                layer=5,
                dataset=sample_texts,
                config=config,
                save_path=str(save_path),
                verbose=False,
            )

            # Check saved
            assert save_path.exists()
            assert (save_path / "config.json").exists()
            assert (save_path / "weights.safetensors").exists()

            # Load and verify
            loaded_sae = model.load_sae(str(save_path))
            assert loaded_sae.d_model == sae.d_model
            assert loaded_sae.d_hidden == sae.d_hidden

    def test_sae_encode_real_activations(self, model, sample_texts):
        """Test encoding real model activations."""
        # Train SAE
        config = SAEConfig(
            expansion_factor=4,
            k=20,
            num_epochs=1,
            batch_size=4,
        )

        sae = model.train_sae(
            layer=5,
            dataset=sample_texts,
            config=config,
            verbose=False,
        )

        # Get activations from model
        with model.trace("Hello world") as trace:
            pass

        # Find the activation key for layer 5 MLP
        layer_keys = [k for k in trace.activations.keys() if "layers.5" in k and "mlp" in k]
        assert len(layer_keys) > 0

        activation = trace.activations[layer_keys[0]]

        # Encode
        features = sae.encode(activation)

        # Check features
        assert features.shape[0] == activation.shape[0]  # Same batch
        assert features.shape[1] == activation.shape[1]  # Same sequence length
        assert features.shape[2] == sae.d_hidden

        # Check sparsity
        non_zero_per_pos = mx.sum(features != 0, axis=-1)
        assert mx.all(non_zero_per_pos <= sae.k)

    def test_sae_different_components(self, model, sample_texts):
        """Test training SAE on different components."""
        config = SAEConfig(
            expansion_factor=4,
            k=20,
            num_epochs=1,
            batch_size=4,
        )

        # Test MLP component
        sae_mlp = model.train_sae(
            layer=5,
            component="mlp",
            dataset=sample_texts[:4],  # Fewer samples for speed
            config=config,
            verbose=False,
        )
        assert sae_mlp.metadata["component"] == "mlp"

    def test_sae_reconstruction_on_real_data(self, model, sample_texts):
        """Test that SAE can reconstruct real activations reasonably well."""
        config = SAEConfig(
            expansion_factor=8,  # Larger for better reconstruction
            k=50,
            num_epochs=2,  # More epochs
            batch_size=4,
        )

        sae = model.train_sae(
            layer=5,
            dataset=sample_texts,
            config=config,
            verbose=False,
        )

        # Get real activation
        with model.trace(sample_texts[0]) as trace:
            pass

        layer_keys = [k for k in trace.activations.keys() if "layers.5" in k and "mlp" in k]
        activation = trace.activations[layer_keys[0]]

        # Reconstruct
        reconstructed, features = sae(activation)

        # Compute reconstruction error
        error = mx.mean((activation - reconstructed) ** 2)

        # Error should be finite and reasonably small
        # (won't be perfect with limited training)
        assert not mx.isnan(error)
        assert not mx.isinf(error)
        assert float(error) < 100.0  # Reasonable upper bound

    def test_sae_compatible_with_layer(self, model, sample_texts):
        """Test SAE compatibility checking."""
        config = SAEConfig(
            expansion_factor=4,
            k=20,
            num_epochs=1,
            batch_size=4,
        )

        sae = model.train_sae(
            layer=5,
            dataset=sample_texts[:4],
            config=config,
            verbose=False,
        )

        # Should be compatible with layer 5, mlp
        assert sae.is_compatible(model, layer=5, component="mlp")

        # Should not be compatible with different layer
        assert not sae.is_compatible(model, layer=3, component="mlp")

        # Should not be compatible with different component
        assert not sae.is_compatible(model, layer=5, component="attn")


class TestSAEEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture(scope="class")
    def model(self):
        """Load model for testing."""
        model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
        return model

    def test_empty_dataset(self, model):
        """Test that empty dataset raises error."""
        config = SAEConfig(num_epochs=1)

        with pytest.raises(Exception):  # Should fail with empty dataset
            model.train_sae(
                layer=5,
                dataset=[],
                config=config,
                verbose=False,
            )

    def test_very_small_dataset(self, model):
        """Test training with very small dataset."""
        config = SAEConfig(
            expansion_factor=4,
            k=20,
            num_epochs=1,
            batch_size=2,
        )

        # Should work with just 2 samples
        sae = model.train_sae(
            layer=5,
            dataset=["Hello", "World"],
            config=config,
            verbose=False,
        )

        assert sae is not None

    def test_invalid_layer(self, model):
        """Test that invalid layer raises error."""
        config = SAEConfig(num_epochs=1)

        # Layer number too high should fail
        with pytest.raises(Exception):
            model.train_sae(
                layer=9999,
                dataset=["Hello"],
                config=config,
                verbose=False,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
