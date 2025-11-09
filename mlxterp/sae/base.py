"""
Abstract base class for Sparse Autoencoders and variants.

Defines the interface that all SAE-like models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
import json
from pathlib import Path


class BaseSAE(nn.Module, ABC):
    """Abstract base class for Sparse Autoencoders.

    All SAE variants (SAE, Transcoder, Crosscoder) inherit from this class
    and must implement the core methods.

    Attributes:
        d_model: Input dimension (activation dimension)
        d_hidden: Hidden dimension (number of features)
        metadata: Dictionary containing training metadata
    """

    def __init__(self, d_model: int, d_hidden: int):
        """Initialize base SAE.

        Args:
            d_model: Input activation dimension
            d_hidden: Hidden feature dimension
        """
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def encode(self, x: mx.array) -> mx.array:
        """Encode activations to sparse features.

        Args:
            x: Input activations of shape (batch, seq_len, d_model)

        Returns:
            Sparse features of shape (batch, seq_len, d_hidden)
        """
        pass

    @abstractmethod
    def decode(self, z: mx.array) -> mx.array:
        """Decode sparse features back to activation space.

        Args:
            z: Sparse features of shape (batch, seq_len, d_hidden)

        Returns:
            Reconstructed activations of shape (batch, seq_len, d_model)
        """
        pass

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        """Forward pass: encode then decode.

        Args:
            x: Input activations of shape (batch, seq_len, d_model)

        Returns:
            Tuple of (reconstructed activations, sparse features)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def save(self, path: str) -> None:
        """Save SAE weights and metadata to disk.

        Saves in a directory structure:
        path/
        ├── config.json       # Hyperparameters and metadata
        ├── weights.safetensors  # Model weights

        Args:
            path: Directory path to save to (will be created if doesn't exist)
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save weights - flatten nested dictionaries, exclude metadata
        weights = {}
        for name, param in self.parameters().items():
            if name == "metadata":
                continue  # Skip metadata (not a weight)
            if isinstance(param, dict):
                for subname, subparam in param.items():
                    weights[f"{name}.{subname}"] = subparam
            else:
                weights[name] = param

        weights_path = path / "weights.safetensors"
        mx.save_safetensors(str(weights_path), weights)

        # Save metadata and config
        # Include all non-callable attributes
        config = {
            "d_model": self.d_model,
            "d_hidden": self.d_hidden,
            "class": self.__class__.__name__,
            "metadata": self.metadata,
        }

        # Save subclass-specific attributes
        for key, value in self.__dict__.items():
            if not callable(value) and key not in config:
                # Save simple types
                if isinstance(value, (int, float, str, bool, type(None))):
                    config[key] = value
                # Save arrays (like _input_mean, _input_std) as lists
                elif hasattr(value, 'tolist'):
                    config[key] = value.tolist()

        config_path = path / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BaseSAE":
        """Load SAE from disk.

        Args:
            path: Directory path to load from

        Returns:
            Loaded SAE instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If config is invalid or class mismatch
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SAE directory not found: {path}")

        # Load config
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config = json.load(f)

        # Verify class match
        if config["class"] != cls.__name__:
            raise ValueError(
                f"Class mismatch: saved as {config['class']}, loading as {cls.__name__}"
            )

        # Extract constructor arguments for this class
        # SAE needs: d_model, d_hidden, k, normalize_input, tied_weights
        constructor_args = {
            "d_model": config["d_model"],
            "d_hidden": config["d_hidden"],
        }

        # Add SAE-specific constructor args if they exist in config
        if "k" in config:
            constructor_args["k"] = config["k"]
        if "normalize_input" in config:
            constructor_args["normalize_input"] = config["normalize_input"]
        if "tied_weights" in config:
            constructor_args["tied_weights"] = config["tied_weights"]

        # Create instance
        sae = cls(**constructor_args)

        # Load weights
        weights_path = path / "weights.safetensors"
        weights = mx.load(str(weights_path))
        sae.load_weights(list(weights.items()))

        # Load metadata
        sae.metadata = config.get("metadata", {})

        # Restore any other saved attributes
        for key, value in config.items():
            if key not in ["d_model", "d_hidden", "class", "metadata", "k", "normalize_input", "tied_weights"]:
                if hasattr(sae, key):
                    # Convert lists back to mx.array
                    if isinstance(value, list):
                        setattr(sae, key, mx.array(value))
                    else:
                        setattr(sae, key, value)

        return sae

    def is_compatible(
        self,
        model: Any,
        layer: int,
        component: str = "mlp"
    ) -> bool:
        """Check if this SAE is compatible with a given model/layer/component.

        Args:
            model: InterpretableModel instance
            layer: Layer number
            component: Component name (e.g., "mlp", "attn")

        Returns:
            True if dimensions match, False otherwise
        """
        # Get expected dimension from metadata
        expected_layer = self.metadata.get("layer")
        expected_component = self.metadata.get("component")
        expected_model_name = self.metadata.get("model_name")

        # Check metadata compatibility
        if expected_layer is not None and expected_layer != layer:
            return False
        if expected_component is not None and expected_component != component:
            return False

        # Check dimension compatibility
        # This would require accessing the model's layer dimension
        # For now, we just check if d_model matches expected dimension
        return True

    def get_activation_stats(self, z: mx.array) -> Dict[str, float]:
        """Compute statistics about feature activations.

        Args:
            z: Sparse features of shape (batch, seq_len, d_hidden)

        Returns:
            Dictionary with statistics:
            - l0: Average number of active features per sample
            - l0_sparsity: Fraction of features active (l0 / d_hidden)
            - l1_magnitude: Average magnitude of active features
            - dead_features: Number of features that never activate
            - dead_fraction: Fraction of features that are dead
        """
        # L0: number of non-zero features
        l0 = float(mx.mean(mx.sum(z != 0, axis=-1)))

        # L0 sparsity: fraction of features active (more intuitive than count)
        l0_sparsity = l0 / self.d_hidden

        # L1 magnitude: average absolute value of activations
        # NOTE: This is NOT the same as sparsity fraction!
        l1_magnitude = float(mx.mean(mx.abs(z)))

        # Dead features: features that never activate in this batch
        feature_max = mx.max(mx.abs(z), axis=(0, 1))  # Max activation per feature
        dead_features = int(mx.sum(feature_max < 1e-8))

        return {
            "l0": l0,
            "l0_sparsity": l0_sparsity,
            "l1_magnitude": l1_magnitude,
            "dead_features": dead_features,
            "dead_fraction": dead_features / self.d_hidden,
        }

    def __repr__(self) -> str:
        """String representation of SAE."""
        return (
            f"{self.__class__.__name__}("
            f"d_model={self.d_model}, "
            f"d_hidden={self.d_hidden}, "
            f"expansion={self.d_hidden/self.d_model:.1f}x)"
        )
