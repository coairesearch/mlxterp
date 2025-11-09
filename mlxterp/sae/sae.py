"""
Standard Sparse Autoencoder with TopK activation.

Implements a SAE that learns interpretable features from layer activations
using TopK sparsity constraint.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .base import BaseSAE


def topk_activation(x: mx.array, k: int) -> mx.array:
    """Apply TopK activation: keep only top k values, zero out the rest.

    Args:
        x: Input array of shape (..., d)
        k: Number of top values to keep

    Returns:
        Array with same shape as x, but only top-k values are non-zero
    """
    if k >= x.shape[-1]:
        return x  # If k >= dimension, no sparsity

    # Sort to find k-th largest value (threshold)
    # Sort in descending order
    sorted_vals = mx.sort(x, axis=-1)
    # Get k-th largest (which is at index -k from the end after sorting ascending)
    threshold = sorted_vals[..., -k:-k+1]  # Shape: (..., 1)

    # Keep only values >= threshold
    mask = x >= threshold
    output = mx.where(mask, x, mx.zeros_like(x))

    return output


class SAE(BaseSAE):
    """Sparse Autoencoder with TopK activation.

    Learns a sparse, overcomplete representation of layer activations using
    TopK sparsity constraint. This enforces exactly k features to be active
    per sample, leading to more interpretable features.

    Architecture:
        Input (d_model) → Encoder → ReLU → TopK(k) → Features (d_hidden) → Decoder → Output (d_model)

    Args:
        d_model: Input activation dimension
        d_hidden: Hidden feature dimension (typically d_model * expansion_factor)
        k: Number of top features to keep active per sample
        normalize_input: Whether to normalize input activations
        tied_weights: Whether to tie encoder and decoder weights (decoder = encoder.T)

    Example:
        >>> # Create SAE for layer with 2048-dim activations
        >>> sae = SAE(d_model=2048, d_hidden=32768, k=100)
        >>>
        >>> # Encode activations to features
        >>> features = sae.encode(activations)  # Shape: (batch, seq, 32768)
        >>>
        >>> # Decode back to activation space
        >>> reconstructed = sae.decode(features)  # Shape: (batch, seq, 2048)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        k: int = 100,
        normalize_input: bool = True,
        tied_weights: bool = False,
    ):
        """Initialize SAE.

        Args:
            d_model: Input dimension
            d_hidden: Hidden dimension (number of features)
            k: Number of top features to keep active
            normalize_input: Whether to normalize inputs before encoding
            tied_weights: Whether to tie encoder/decoder weights
        """
        super().__init__(d_model, d_hidden)

        self.k = k
        self.normalize_input = normalize_input
        self.tied_weights = tied_weights

        # Encoder: d_model → d_hidden
        self.encoder = nn.Linear(d_model, d_hidden, bias=True)

        # Decoder: d_hidden → d_model
        if tied_weights:
            # Decoder will use encoder weights transposed
            self.decoder_bias = mx.zeros((d_model,))
        else:
            self.decoder = nn.Linear(d_hidden, d_model, bias=True)

        # Statistics for normalization (frozen, not trainable parameters)
        # Freeze these by calling freeze() after assignment
        self.input_mean = mx.zeros((d_model,))
        self.input_std = mx.ones((d_model,))
        self.freeze(keys=["input_mean", "input_std"])

    def encode(self, x: mx.array) -> mx.array:
        """Encode activations to sparse features.

        Args:
            x: Input activations of shape (batch, seq_len, d_model)

        Returns:
            Sparse features of shape (batch, seq_len, d_hidden)
        """
        # Normalize input if enabled
        if self.normalize_input:
            x = (x - self.input_mean) / (self.input_std + 1e-8)

        # Encode: linear + ReLU
        h = self.encoder(x)
        h = mx.maximum(h, 0)  # ReLU

        # TopK sparsity
        z = topk_activation(h, k=self.k)

        return z

    def decode(self, z: mx.array) -> mx.array:
        """Decode sparse features back to activation space.

        Args:
            z: Sparse features of shape (batch, seq_len, d_hidden)

        Returns:
            Reconstructed activations of shape (batch, seq_len, d_model)
        """
        if self.tied_weights:
            # Use transposed encoder weights
            x_recon = mx.matmul(z, self.encoder.weight) + self.decoder_bias
        else:
            x_recon = self.decoder(z)

        # Denormalize if input was normalized
        if self.normalize_input:
            x_recon = x_recon * self.input_std + self.input_mean

        return x_recon

    def update_normalization_stats(self, activations: mx.array) -> None:
        """Update running statistics for input normalization.

        Should be called during training with batches of activations.

        Args:
            activations: Batch of activations of shape (batch, seq_len, d_model)
        """
        # Compute mean and std across batch and sequence dimensions
        flat = activations.reshape(-1, self.d_model)

        # Unfreeze, update, and refreeze
        self.unfreeze(keys=["input_mean", "input_std"])
        self.input_mean = mx.mean(flat, axis=0)
        self.input_std = mx.std(flat, axis=0)
        self.freeze(keys=["input_mean", "input_std"])

    def compute_loss(
        self,
        x: mx.array,
        lambda_sparse: float = 0.0
    ) -> tuple[mx.array, dict]:
        """Compute reconstruction loss and optional sparsity penalty.

        Args:
            x: Input activations of shape (batch, seq_len, d_model)
            lambda_sparse: L1 sparsity penalty coefficient (usually 0 for TopK)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Forward pass
        x_recon, z = self(x)

        # Reconstruction loss (MSE)
        recon_loss = mx.mean((x - x_recon) ** 2)

        # L1 sparsity (usually not needed with TopK, but available)
        if lambda_sparse > 0:
            l1_loss = mx.mean(mx.abs(z))
            total_loss = recon_loss + lambda_sparse * l1_loss
        else:
            l1_loss = mx.array(0.0)
            total_loss = recon_loss

        # Compute metrics
        metrics = {
            "loss": float(total_loss),
            "recon_loss": float(recon_loss),
            "l1_loss": float(l1_loss),
        }

        # Add activation statistics
        metrics.update(self.get_activation_stats(z))

        return total_loss, metrics

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"SAE(d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"k={self.k}, expansion={self.d_hidden/self.d_model:.1f}x, "
            f"normalize_input={self.normalize_input}, tied_weights={self.tied_weights})"
        )
