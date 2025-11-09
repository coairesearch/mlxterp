"""
Configuration classes for SAE, Transcoder, and Crosscoder training.

Provides sensible defaults that work out-of-the-box, with options for customization.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SAEConfig:
    """Configuration for Sparse Autoencoder training.

    Provides sensible defaults optimized for most use cases. All parameters can be
    customized for advanced users.

    Args:
        expansion_factor: Hidden dimension = expansion_factor × input dimension
            Default: 32 (SAELens-validated, increased from 16)
        k: Number of top features to keep active (TopK sparsity)
            Default: 100
        learning_rate: Optimizer learning rate
            Default: 3e-4 (SAELens-validated, increased from 1e-4)
        batch_size: Number of activation samples per batch
            Default: 256
        num_epochs: Number of training epochs
            Default: 10
        lambda_sparse: L1 sparsity penalty coefficient (not used with TopK)
            Default: 0.0
        normalize_input: Whether to normalize input activations
            Default: True
        tied_weights: Whether to tie encoder and decoder weights
            Default: False
        dead_neuron_threshold: Threshold for considering a neuron dead
            Default: 1e-6
        use_ghost_grads: Apply ghost gradients to revive dead features (SAELens approach)
            Default: True
        feature_sampling_window: Window for tracking feature activity
            Default: 1000
        dead_feature_window: Steps before feature considered dead
            Default: 5000
        warmup_steps: Number of learning rate warmup steps
            Default: 1000
        lr_decay_steps: Steps for LR decay (None = total training steps)
            Default: None
        lr_scheduler: LR schedule type ("cosine", "linear", "constant")
            Default: "cosine"
        sparsity_warm_up_steps: Steps for sparsity warmup (None = no warmup)
            Default: None (will be set to total_steps if use_ghost_grads=True)
        gradient_clip: Maximum gradient norm (None = no clipping)
            Default: 1.0
        checkpoint_every: Save checkpoint every N steps
            Default: 5000
        validation_split: Fraction of data to use for validation
            Default: 0.05
        seed: Random seed for reproducibility
            Default: 42

    Example:
        >>> # Use defaults (SAELens-validated)
        >>> config = SAEConfig()
        >>>
        >>> # Customize specific parameters
        >>> config = SAEConfig(expansion_factor=64, k=150, learning_rate=5e-4)
    """

    # Architecture (SAELens-validated defaults)
    expansion_factor: int = 32  # Increased from 16
    k: int = 100
    sae_type: str = "topk"  # Options: "topk", "batchtopk"

    # Optimization (SAELens-validated defaults)
    learning_rate: float = 3e-4  # Increased from 1e-4
    batch_size: int = 256
    num_epochs: int = 10
    lambda_sparse: float = 0.0  # Not used with TopK

    # Preprocessing
    normalize_input: bool = True
    tied_weights: bool = False

    # Dead neuron handling (SAELens approach)
    dead_neuron_threshold: float = 1e-6
    use_ghost_grads: bool = True  # Apply ghost gradients to dead features
    feature_sampling_window: int = 1000  # Window for tracking feature activity
    dead_feature_window: int = 1000  # Steps before feature considered dead (reduced from 5000 for faster activation)

    # Training schedule (SAELens approach)
    warmup_steps: int = 1000
    lr_decay_steps: Optional[int] = None  # If None, defaults to total training steps
    lr_scheduler: str = "cosine"  # Options: "cosine", "linear", "constant"
    gradient_clip: Optional[float] = 1.0

    # Sparsity schedule (SAELens approach)
    sparsity_warm_up_steps: Optional[int] = None  # If None, no sparsity warmup

    # Checkpointing
    checkpoint_every: int = 5000
    validation_split: float = 0.05

    # Reproducibility
    seed: int = 42

    # Streaming optimization
    text_batch_size: int = 32  # Number of texts to process at once during streaming

    # Logging (optional Weights & Biases integration)
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    wandb_name: Optional[str] = None
    wandb_tags: Optional[list] = None

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.expansion_factor < 1:
            raise ValueError(f"expansion_factor must be >= 1, got {self.expansion_factor}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs must be >= 1, got {self.num_epochs}")
        if not 0 <= self.validation_split < 1:
            raise ValueError(f"validation_split must be in [0, 1), got {self.validation_split}")


@dataclass
class TranscoderConfig:
    """Configuration for Transcoder training.

    Transcoders learn transformations between layers (e.g., Layer N → Layer N+1).
    Default parameters are optimized for layer-to-layer feature learning.

    Args:
        expansion_factor: Hidden dimension multiplier (smaller than SAE, less reconstruction needed)
            Default: 8
        k: Number of top features to keep active
            Default: 50
        learning_rate: Optimizer learning rate (slower for stability)
            Default: 5e-5
        batch_size: Number of activation samples per batch
            Default: 256
        num_epochs: Number of training epochs (more epochs for harder task)
            Default: 15
        normalize_input: Whether to normalize input activations
            Default: True
        tied_weights: Whether to tie encoder and decoder weights
            Default: False
        dead_neuron_threshold: Threshold for considering a neuron dead
            Default: 1e-6
        resample_dead_every: Resample dead neurons every N steps
            Default: 10000
        warmup_steps: Number of learning rate warmup steps
            Default: 1000
        gradient_clip: Maximum gradient norm
            Default: 1.0
        checkpoint_every: Save checkpoint every N steps
            Default: 5000
        validation_split: Fraction of data to use for validation
            Default: 0.05
        seed: Random seed
            Default: 42

    Example:
        >>> config = TranscoderConfig(expansion_factor=16, k=100)
    """

    # Architecture (smaller expansion for transcoding)
    expansion_factor: int = 8
    k: int = 50

    # Optimization (slower learning for stability)
    learning_rate: float = 5e-5
    batch_size: int = 256
    num_epochs: int = 15

    # Preprocessing
    normalize_input: bool = True
    tied_weights: bool = False

    # Dead neuron handling
    dead_neuron_threshold: float = 1e-6
    resample_dead_every: int = 10000

    # Training schedule
    warmup_steps: int = 1000
    gradient_clip: Optional[float] = 1.0

    # Checkpointing
    checkpoint_every: int = 5000
    validation_split: float = 0.05

    # Reproducibility
    seed: int = 42


@dataclass
class CrosscoderConfig:
    """Configuration for Crosscoder training.

    Crosscoders learn shared features across multiple components or layers.
    Default parameters are optimized for multi-component feature learning.

    Args:
        expansion_factor: Hidden dimension multiplier (larger to capture multiple components)
            Default: 24
        k: Number of top features to keep active
            Default: 150
        learning_rate: Optimizer learning rate
            Default: 1e-4
        batch_size: Number of activation samples per batch (smaller due to multiple components)
            Default: 128
        num_epochs: Number of training epochs
            Default: 12
        normalize_input: Whether to normalize input activations
            Default: True
        component_weights: Optional weights for each component (None = equal weights)
            Default: None
        tied_weights: Whether to tie encoder and decoder weights
            Default: False
        dead_neuron_threshold: Threshold for considering a neuron dead
            Default: 1e-6
        resample_dead_every: Resample dead neurons every N steps
            Default: 10000
        warmup_steps: Number of learning rate warmup steps
            Default: 1000
        gradient_clip: Maximum gradient norm
            Default: 1.0
        checkpoint_every: Save checkpoint every N steps
            Default: 5000
        validation_split: Fraction of data to use for validation
            Default: 0.05
        seed: Random seed
            Default: 42

    Example:
        >>> # Equal weights for all components
        >>> config = CrosscoderConfig()
        >>>
        >>> # Custom weights (e.g., weight MLP more than attention)
        >>> config = CrosscoderConfig(component_weights=[0.6, 0.4])
    """

    # Architecture (larger expansion for multiple components)
    expansion_factor: int = 24
    k: int = 150

    # Optimization
    learning_rate: float = 1e-4
    batch_size: int = 128  # Smaller due to multiple components
    num_epochs: int = 12

    # Preprocessing
    normalize_input: bool = True
    component_weights: Optional[list] = None  # Equal weight by default
    tied_weights: bool = False

    # Dead neuron handling
    dead_neuron_threshold: float = 1e-6
    resample_dead_every: int = 10000

    # Training schedule
    warmup_steps: int = 1000
    gradient_clip: Optional[float] = 1.0

    # Checkpointing
    checkpoint_every: int = 5000
    validation_split: float = 0.05

    # Reproducibility
    seed: int = 42
