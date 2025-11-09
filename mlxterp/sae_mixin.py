"""
SAE mixin for InterpretableModel.

Provides methods for training and using Sparse Autoencoders with the model.
"""

from typing import List, Optional
from pathlib import Path

from .sae import SAE, SAEConfig, SAETrainer


class SAEMixin:
    """Mixin class providing SAE-related methods.

    This mixin adds SAE training and loading capabilities to InterpretableModel.
    """

    def train_sae(
        self,
        layer: int,
        dataset: List[str],
        component: str = "mlp",
        save_path: Optional[str] = None,
        config: Optional[SAEConfig] = None,
        verbose: bool = True,
    ) -> SAE:
        """Train a Sparse Autoencoder on activations from a specific layer.

        This is the simple, one-line API for training SAEs. It automatically:
        - Collects activations from the model
        - Initializes and trains the SAE
        - Handles normalization and optimization
        - Optionally saves the trained SAE

        Args:
            layer: Layer number to train on (e.g., 10 for layer 10)
            dataset: List of text samples to collect activations from
            component: Component name - options:
                - "mlp": MLP/FFN output
                - "attn": Attention output
                - "residual": Residual stream
            save_path: Optional path to save trained SAE (e.g., "sae_layer10.mlx")
            config: Optional SAE configuration (uses sensible defaults if None)
            verbose: Whether to show progress and metrics (default: True)

        Returns:
            Trained SAE instance

        Example:
            >>> # Simple usage with defaults
            >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
            >>> texts = ["Paris is the capital of France", ...]  # 10k+ samples
            >>>
            >>> sae = model.train_sae(
            ...     layer=10,
            ...     dataset=texts,
            ...     save_path="sae_layer10.mlx"
            ... )
            >>>
            >>> # Custom configuration
            >>> from mlxterp.sae import SAEConfig
            >>> config = SAEConfig(expansion_factor=32, k=150, learning_rate=5e-5)
            >>> sae = model.train_sae(layer=10, dataset=texts, config=config)
        """
        # Create trainer with config
        trainer = SAETrainer(config=config)

        # Train SAE
        sae = trainer.train(
            model=self,
            layer=layer,
            component=component,
            dataset=dataset,
            save_path=save_path,
            verbose=verbose,
        )

        return sae

    def load_sae(self, path: str) -> SAE:
        """Load a trained SAE from disk.

        Args:
            path: Path to saved SAE directory

        Returns:
            Loaded SAE instance

        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If SAE file is invalid

        Example:
            >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
            >>> sae = model.load_sae("sae_layer10.mlx")
            >>>
            >>> # Check compatibility
            >>> if sae.is_compatible(model, layer=10, component="mlp"):
            ...     print("SAE is compatible!")
            >>>
            >>> # Use SAE to encode activations
            >>> with model.trace("Hello world") as trace:
            ...     activation = trace.activations["model.model.layers.10.mlp"]
            ...     features = sae.encode(activation)
        """
        sae = SAE.load(path)
        return sae

    def train_transcoder(
        self,
        from_layer: int,
        to_layer: int,
        dataset: List[str],
        component: str = "mlp",
        from_component: Optional[str] = None,
        to_component: Optional[str] = None,
        save_path: Optional[str] = None,
        config: Optional['TranscoderConfig'] = None,
        verbose: bool = True,
    ) -> 'Transcoder':
        """Train a Transcoder to learn layer-to-layer transformations.

        Transcoders learn how information is transformed between layers,
        mapping features from layer N to layer N+1.

        Args:
            from_layer: Source layer number
            to_layer: Target layer number
            dataset: List of text samples
            component: Component name if same for both layers
            from_component: Source component (overrides component)
            to_component: Target component (overrides component)
            save_path: Optional path to save trained transcoder
            config: Optional transcoder configuration
            verbose: Show progress

        Returns:
            Trained Transcoder instance

        Example:
            >>> # Learn MLP layer 10 → 11 transformation
            >>> transcoder = model.train_transcoder(
            ...     from_layer=10,
            ...     to_layer=11,
            ...     dataset=texts
            ... )
            >>>
            >>> # Cross-component: MLP → Attention
            >>> transcoder = model.train_transcoder(
            ...     from_layer=10,
            ...     from_component="mlp",
            ...     to_layer=11,
            ...     to_component="attn",
            ...     dataset=texts
            ... )

        Note:
            This feature is planned for Phase 3. Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "Transcoder training is planned for Phase 3. "
            "See SAE_INTEGRATION_PLAN.md for details."
        )

    def train_crosscoder(
        self,
        layer: Optional[int] = None,
        layers: Optional[List[int]] = None,
        components: Optional[List[str]] = None,
        dataset: List[str] = None,
        save_path: Optional[str] = None,
        config: Optional['CrosscoderConfig'] = None,
        verbose: bool = True,
    ) -> 'Crosscoder':
        """Train a Crosscoder to find shared features across components/layers.

        Crosscoders learn features that appear in multiple components
        simultaneously, useful for circuit discovery.

        Args:
            layer: Single layer number (if learning across components)
            layers: Multiple layer numbers (if learning across layers)
            components: List of components (e.g., ["mlp", "attn"])
            dataset: List of text samples
            save_path: Optional path to save trained crosscoder
            config: Optional crosscoder configuration
            verbose: Show progress

        Returns:
            Trained Crosscoder instance

        Example:
            >>> # Find features shared between MLP and Attention
            >>> crosscoder = model.train_crosscoder(
            ...     layer=10,
            ...     components=["mlp", "attn"],
            ...     dataset=texts
            ... )
            >>>
            >>> # Find features across multiple layers
            >>> crosscoder = model.train_crosscoder(
            ...     layers=[8, 10, 12],
            ...     components=["mlp"],
            ...     dataset=texts
            ... )

        Note:
            This feature is planned for Phase 4. Currently raises NotImplementedError.
        """
        raise NotImplementedError(
            "Crosscoder training is planned for Phase 4. "
            "See SAE_INTEGRATION_PLAN.md for details."
        )
