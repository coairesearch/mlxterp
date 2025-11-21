"""
BatchTopK Sparse Autoencoder (SAELens-style).

Modern variant of TopK that fixes mean L0 across batches rather than per-sample.
This provides more stable training and better feature utilization.
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional
from .base import BaseSAE


def batchtopk_activation(x: mx.array, k: int) -> mx.array:
    """Apply BatchTopK activation: keep top k values across batch, adjusting per-sample k.

    Unlike standard TopK which keeps exactly k values per sample, BatchTopK adjusts
    the per-sample k to achieve a target mean k across the batch. This leads to:
    - More stable training (batch-level statistics)
    - Better feature utilization
    - Fewer dead features

    Args:
        x: Input array of shape (batch, ..., d)
        k: Target mean number of active features across batch

    Returns:
        Array with same shape as x, sparsified to achieve mean k active features
    """
    batch_size = x.shape[0]
    d = x.shape[-1]

    # Flatten to (batch * ..., d)
    orig_shape = x.shape
    x_flat = x.reshape(-1, d)
    n_samples = x_flat.shape[0]

    # Sort all values across batch and feature dimensions
    # Target: keep k * n_samples total values
    target_active = k * n_samples

    # WORKAROUND for MLX int32 overflow bug (size > 2^31 - 1)
    # When flattening large arrays (n_samples * d > 2^31), MLX has an integer overflow
    # Instead of flattening, we collect values row by row
    total_elements = n_samples * d
    MAX_SAFE_SIZE = 2**30  # Stay well below 2^31

    if total_elements > MAX_SAFE_SIZE:
        # Process in chunks to avoid overflow
        # Collect absolute values without flattening
        abs_vals = mx.abs(x_flat)  # (n_samples, d)

        # Get top k*n_samples values across entire array
        # Strategy: get top values from each row, then sort globally
        k_per_sample = max(k * 2, 256)  # Get more per sample to ensure we have enough

        # Get top values from each sample
        row_topk_vals = []
        for i in range(n_samples):
            row_vals = mx.sort(abs_vals[i])[-k_per_sample:]
            row_topk_vals.append(row_vals)

        # Concatenate and get global threshold
        all_topk = mx.concatenate(row_topk_vals)
        sorted_vals = mx.sort(all_topk)
        threshold_idx = max(0, len(sorted_vals) - target_active)
        threshold = sorted_vals[threshold_idx]
    else:
        # Safe to flatten - array size fits in int32
        all_vals = mx.abs(x_flat).reshape((-1,))
        sorted_vals = mx.sort(all_vals)

        # Find threshold (k-th largest across entire batch)
        threshold_idx = max(0, len(sorted_vals) - target_active)
        threshold = sorted_vals[threshold_idx]

    # Keep only values >= threshold
    mask = mx.abs(x_flat) >= threshold
    output = mx.where(mask, x_flat, mx.zeros_like(x_flat))

    # Reshape back to original shape
    output = output.reshape(orig_shape)

    return output


class BatchTopKSAE(BaseSAE):
    """Sparse Autoencoder with BatchTopK activation (SAELens-style).

    Modern variant that fixes mean L0 across batches rather than per-sample.
    This provides:
    - More stable training (batch-level statistics instead of per-sample)
    - Better feature utilization (features can specialize to different samples)
    - Fewer dead features (batch-level sparsity is more forgiving)

    Architecture:
        Input (d_model) → Encoder → ReLU → BatchTopK(k) → Features (d_hidden) → Decoder → Output (d_model)

    Args:
        d_model: Input activation dimension
        d_hidden: Hidden feature dimension (typically d_model * expansion_factor)
        k: Target mean number of active features per sample across batch
        normalize_input: Whether to normalize input activations
        tied_weights: Whether to tie encoder and decoder weights

    Example:
        >>> # Create BatchTopK SAE (recommended over standard TopK)
        >>> sae = BatchTopKSAE(d_model=2048, d_hidden=32768, k=100)
        >>>
        >>> # Encode activations
        >>> features = sae.encode(activations)  # Shape: (batch, seq, 32768)
        >>>
        >>> # Decode back
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
        """Initialize BatchTopK SAE.

        Args:
            d_model: Input dimension
            d_hidden: Hidden dimension (number of features)
            k: Target mean number of active features per sample
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

        # Statistics for normalization (frozen, not trainable)
        self.input_mean = mx.zeros((d_model,))
        self.input_std = mx.ones((d_model,))
        self.freeze(keys=["input_mean", "input_std"])

    def encode(self, x: mx.array) -> mx.array:
        """Encode activations to sparse features using BatchTopK.

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

        # BatchTopK sparsity (batch-level instead of per-sample)
        z = batchtopk_activation(h, k=self.k)

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
            lambda_sparse: L1 sparsity penalty coefficient (usually 0 for BatchTopK)

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Forward pass
        x_recon, z = self(x)

        # Reconstruction loss (MSE)
        recon_loss = mx.mean((x - x_recon) ** 2)

        # L1 sparsity (usually not needed with BatchTopK, but available)
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
            f"BatchTopKSAE(d_model={self.d_model}, d_hidden={self.d_hidden}, "
            f"k={self.k}, expansion={self.d_hidden/self.d_model:.1f}x, "
            f"normalize_input={self.normalize_input}, tied_weights={self.tied_weights})"
        )
