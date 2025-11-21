"""
mlxterp.sae: Sparse Autoencoder module for interpretability

This module provides tools for training and analyzing Sparse Autoencoders (SAEs),
Transcoders, and Crosscoders on Apple Silicon using MLX.

Example:
    >>> from mlxterp import InterpretableModel
    >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
    >>>
    >>> # Train SAE with one line
    >>> sae = model.train_sae(layer=10, dataset=texts)
    >>>
    >>> # Analyze features
    >>> sae.visualize_feature(142, dataset=validation_texts)
"""

from .config import SAEConfig, TranscoderConfig, CrosscoderConfig
from .base import BaseSAE
from .sae import SAE
from .batchtopk import BatchTopKSAE
from .trainer import SAETrainer
from .visualization import (
    visualize_feature_activations,
    get_feature_activations_by_token,
    get_top_activating_tokens,
)

__all__ = [
    # Configuration
    "SAEConfig",
    "TranscoderConfig",
    "CrosscoderConfig",
    # Core classes
    "BaseSAE",
    "SAE",
    "BatchTopKSAE",
    # Training
    "SAETrainer",
    # Visualization
    "visualize_feature_activations",
    "get_feature_activations_by_token",
    "get_top_activating_tokens",
]
