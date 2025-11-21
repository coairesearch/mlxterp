"""
SAE mixin for InterpretableModel.

Provides methods for training and using Sparse Autoencoders with the model.
"""

from typing import List, Optional, Tuple, Union
from pathlib import Path
import mlx.core as mx

from .sae import SAE, SAEConfig, SAETrainer, BatchTopKSAE


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

    def load_sae(self, path: str) -> Union[SAE, BatchTopKSAE]:
        """Load a trained SAE from disk.

        Args:
            path: Path to saved SAE directory

        Returns:
            Loaded SAE instance (SAE or BatchTopKSAE)

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
        # Try BatchTopKSAE first, then fall back to regular SAE
        try:
            sae = BatchTopKSAE.load(path)
        except ValueError:
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

    # =============================================================================
    # Phase 2: SAE Feature Analysis
    # =============================================================================

    def get_top_features_for_text(
        self,
        text: str,
        sae: Union[SAE, BatchTopKSAE, str],
        layer: int,
        component: str = "mlp",
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """Find which SAE features activate most strongly for a given text.

        This helps you understand what high-level concepts the SAE has learned
        by showing which features "light up" when processing specific inputs.

        Args:
            text: Input text to analyze
            sae: Either an SAE instance or path to saved SAE
            layer: Layer number where SAE was trained
            component: Component name ("mlp", "attn", etc.)
            top_k: Number of top features to return

        Returns:
            List of (feature_id, activation_value) tuples, sorted by activation

        Example:
            >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
            >>> sae = model.load_sae("sae_layer10.mlx")
            >>>
            >>> # What features activate for this text?
            >>> top_features = model.get_top_features_for_text(
            ...     text="Paris is the capital of France",
            ...     sae=sae,
            ...     layer=10,
            ...     component="mlp",
            ...     top_k=10
            ... )
            >>>
            >>> for feature_id, activation in top_features:
            ...     print(f"Feature {feature_id}: {activation:.3f}")
        """
        # Load SAE if path provided
        if isinstance(sae, str):
            sae = self.load_sae(sae)

        # Get activations from model
        with self.trace(text) as trace:
            pass

        # Find activation key for this layer and component
        activation_key = None
        for key in trace.activations.keys():
            if f"layers.{layer}" in key and key.endswith(f".{component}"):
                activation_key = key
                break

        if activation_key is None:
            raise ValueError(
                f"Could not find activations for layer {layer} component {component}. "
                f"Available keys: {list(trace.activations.keys())}"
            )

        activations = trace.activations[activation_key]  # (seq_len, d_model)

        # Add batch dimension for SAE: (seq_len, 1, d_model)
        activations_3d = activations[:, None, :]

        # Run through SAE
        _, features = sae(activations_3d)  # features: (seq_len, 1, d_hidden)

        # Get max activation per feature across sequence
        feature_max = mx.max(mx.abs(features), axis=(0, 1))  # (d_hidden,)

        # Get top-k features
        top_indices = mx.argsort(feature_max)[-top_k:]

        # Convert to Python lists
        # Note: feature_max is 1D array (d_hidden,)
        # top_indices is 1D array of indices
        import numpy as np
        feature_max_np = np.array(feature_max).flatten()
        top_indices_np = np.array(top_indices).flatten()

        # Reverse to get highest first
        top_indices_np = top_indices_np[::-1]

        results = []
        for idx in top_indices_np:
            idx_int = int(idx)
            activation_value = float(feature_max_np[idx_int])
            if activation_value > 0:  # Only include if feature activated
                results.append((idx_int, activation_value))

        return results

    def get_top_texts_for_feature(
        self,
        feature_id: int,
        sae: Union[SAE, BatchTopKSAE, str],
        texts: List[str],
        layer: int,
        component: str = "mlp",
        top_k: int = 10
    ) -> List[Tuple[str, float, int]]:
        """Find texts where a specific SAE feature activates most strongly.

        This helps you understand what concept a feature represents by finding
        examples where it activates with high strength.

        Args:
            feature_id: The feature index to analyze
            sae: Either an SAE instance or path to saved SAE
            texts: Dataset of texts to search through
            layer: Layer number where SAE was trained
            component: Component name ("mlp", "attn", etc.)
            top_k: Number of top examples to return

        Returns:
            List of (text, activation_value, token_position) tuples

        Example:
            >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
            >>> sae = model.load_sae("sae_layer10.mlx")
            >>>
            >>> # Load dataset
            >>> from datasets import load_dataset
            >>> dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            >>> texts = [item["text"] for item in dataset if len(item["text"]) > 50][:1000]
            >>>
            >>> # What texts activate feature 1234?
            >>> examples = model.get_top_texts_for_feature(
            ...     feature_id=1234,
            ...     sae=sae,
            ...     texts=texts,
            ...     layer=10,
            ...     component="mlp",
            ...     top_k=20
            ... )
            >>>
            >>> print(f"Feature {feature_id} activates most on:")
            >>> for text, activation, pos in examples[:5]:
            ...     print(f"  [{activation:.3f}] {text[:100]}...")
        """
        # Load SAE if path provided
        if isinstance(sae, str):
            sae = self.load_sae(sae)

        # Collect activations and track which text they came from
        all_feature_activations = []

        for text_idx, text in enumerate(texts):
            # Get activations
            with self.trace(text) as trace:
                pass

            # Find activation key
            activation_key = None
            for key in trace.activations.keys():
                if f"layers.{layer}" in key and key.endswith(f".{component}"):
                    activation_key = key
                    break

            if activation_key is None:
                continue

            activations = trace.activations[activation_key]  # (seq_len, d_model)

            # Add batch dimension: (seq_len, 1, d_model)
            activations_3d = activations[:, None, :]

            # Run through SAE
            _, features = sae(activations_3d)  # (seq_len, 1, d_hidden)

            # Get activations for this specific feature
            feature_acts = features[:, 0, feature_id]  # (seq_len,)

            # Find max activation and position
            max_activation = float(mx.max(mx.abs(feature_acts)))
            max_pos = int(mx.argmax(mx.abs(feature_acts)))

            if max_activation > 0:
                all_feature_activations.append((text_idx, max_activation, max_pos))

        # Sort by activation strength
        all_feature_activations.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with text content
        results = []
        for text_idx, activation, pos in all_feature_activations[:top_k]:
            results.append((texts[text_idx], activation, pos))

        return results
