"""
Analysis and interpretability utilities for InterpretableModel.

This module provides analysis-related methods as a mixin class, including:
- get_token_predictions: Decode hidden states to token predictions
- logit_lens: See what each layer predicts at each position
- activation_patching: Identify important layers for a task
"""

import mlx.core as mx
from typing import Optional, Dict, List, Union, Any


class AnalysisMixin:
    """
    Mixin class providing analysis and interpretability methods.

    This mixin assumes the class has:
    - self.model: The wrapped MLX model
    - self.tokenizer: Tokenizer for text/token conversion
    - self.layers: Access to model layers
    - self.vocab_size: Vocabulary size property
    - self.trace(): Tracing context manager
    - self.output: Output property
    - self.encode(), token_to_str(): Tokenization methods
    """

    def get_token_predictions(
        self,
        hidden_state: mx.array,
        top_k: int = 10,
        return_scores: bool = False,
        embedding_layer: Optional[Any] = None,
        lm_head: Optional[Any] = None,
    ) -> Union[List[int], List[tuple], List[List]]:
        """
        Decode hidden states to token predictions using the model's output projection.

        For models with weight tying (like Llama), uses the embedding layer's weights
        transposed as the output projection. Handles quantized embeddings automatically.

        Args:
            hidden_state: Hidden state tensor, shape (hidden_dim,) or (batch, hidden_dim)
            top_k: Number of top predictions to return
            return_scores: If True, return (token_id, score) tuples instead of just token_ids
            embedding_layer: Override embedding layer for weight-tied projection.
                            If provided, uses this layer's weights for output projection.
            lm_head: Override lm_head layer. If provided, uses this layer directly.
                    Takes precedence over embedding_layer.

        Returns:
            For 1D input: List of token IDs or (token_id, score) tuples
            For 2D input: List of lists, one per batch element

        Example:
            >>> with model.trace("Hello") as trace:
            >>>     layer_6 = trace.activations["model.model.layers.6"]
            >>>
            >>> # Get predictions from layer 6's last token
            >>> hidden = layer_6[0, -1, :]
            >>> predictions = model.get_token_predictions(hidden, top_k=5)
            >>>
            >>> # Decode to words
            >>> for token_id in predictions:
            >>>     print(model.token_to_str(token_id))
            >>>
            >>> # Batched predictions
            >>> hidden_batch = layer_6[:, -1, :]  # (batch, hidden_dim)
            >>> batch_preds = model.get_token_predictions(hidden_batch, top_k=5)
            >>> # batch_preds is a list of lists, one per batch element
            >>>
            >>> # Custom model with non-standard paths
            >>> predictions = model.get_token_predictions(
            >>>     hidden, top_k=5, embedding_layer=model.model.my_embed
            >>> )
        """
        # Handle batched input
        is_batched = hidden_state.ndim == 2
        if is_batched:
            # Process each batch element separately
            batch_results = []
            for i in range(hidden_state.shape[0]):
                single_result = self.get_token_predictions(
                    hidden_state[i], top_k=top_k, return_scores=return_scores,
                    embedding_layer=embedding_layer, lm_head=lm_head
                )
                batch_results.append(single_result)
            return batch_results

        # Resolve output projection using the module resolver
        # Priority: lm_head override > embedding_layer override > resolver
        if lm_head is not None:
            # Direct lm_head override
            logits = lm_head(hidden_state)
        elif embedding_layer is not None:
            # Embedding layer override - use weight-tied projection
            logits = self._compute_weight_tied_logits(hidden_state, embedding_layer)
        else:
            # Use module resolver
            proj_module, proj_path, is_weight_tied = self._module_resolver.get_output_projection()

            if proj_module is None:
                raise AttributeError(
                    "Cannot find output projection layer. Model must have either "
                    "'lm_head' or weight-tied embeddings. Tried paths:\n"
                    f"  lm_head: {self._module_resolver.LM_HEAD_PATHS}\n"
                    f"  embedding: {self._module_resolver.EMBEDDING_PATHS}\n"
                    "Use embedding_path or lm_head_path constructor args to specify custom paths."
                )

            if is_weight_tied:
                logits = self._compute_weight_tied_logits(hidden_state, proj_module)
            else:
                logits = proj_module(hidden_state)

        # Get top-k predictions
        top_k_indices = mx.argpartition(-logits, kth=min(top_k, logits.shape[-1] - 1))[:top_k]

        # Force evaluation
        mx.eval(top_k_indices)

        if return_scores:
            # Return (token_id, score) tuples
            results = []
            for idx in top_k_indices:
                token_id = int(idx)
                score = float(logits[idx])
                results.append((token_id, score))
            # Sort by score descending
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        else:
            # Return just token IDs, sorted by score
            token_ids = [int(idx) for idx in top_k_indices]
            scores = [float(logits[idx]) for idx in top_k_indices]
            # Sort by score
            sorted_pairs = sorted(zip(token_ids, scores), key=lambda x: x[1], reverse=True)
            return [token_id for token_id, _ in sorted_pairs]

    def _compute_weight_tied_logits(
        self,
        hidden_state: mx.array,
        embed_layer: Any,
    ) -> mx.array:
        """
        Compute logits using weight-tied embedding layer.

        For weight-tied models, the embedding weights are used (transposed) as
        the output projection. This handles both standard and quantized embeddings.

        Args:
            hidden_state: Hidden state tensor, shape (hidden_dim,)
            embed_layer: Embedding layer to use for projection

        Returns:
            Logits tensor, shape (vocab_size,)
        """
        # Check if it's a quantized embedding by looking for quantization attributes
        is_quantized = hasattr(embed_layer, 'scales') or hasattr(embed_layer, 'biases') or \
                      (hasattr(embed_layer, 'weight') and embed_layer.weight.shape[0] != self.vocab_size)

        if is_quantized:
            # Quantized embedding - compute similarities in batches
            vocab_size = self.vocab_size if self.vocab_size else 128256

            # Batch process to avoid OOM
            batch_size = 1024
            all_logits = []

            for start_idx in range(0, vocab_size, batch_size):
                end_idx = min(start_idx + batch_size, vocab_size)
                batch_indices = mx.arange(start_idx, end_idx)

                # Get embeddings for this batch
                batch_embeds = embed_layer(batch_indices)  # Shape: (batch_size, hidden_dim)

                # Compute similarities
                batch_logits = batch_embeds @ hidden_state  # Shape: (batch_size,)
                all_logits.append(batch_logits)

            return mx.concatenate(all_logits, axis=0)
        else:
            # Standard embedding with weight attribute
            embed_weights = embed_layer.weight  # Shape: (vocab_size, hidden_dim)
            return hidden_state @ embed_weights.T

    def logit_lens(
        self,
        text: str,
        top_k: int = 1,
        layers: Optional[List[int]] = None,
        position: Optional[int] = None,
        plot: bool = False,
        max_display_tokens: int = 15,
        figsize: tuple = (16, 10),
        cmap: str = 'viridis',
        font_family: Optional[str] = None,
        final_norm: Optional[Any] = None,
        skip_norm: bool = False,
    ) -> Dict[int, List[List[tuple]]]:
        """
        Apply logit lens to see what each layer predicts at each token position.

        The logit lens technique projects each layer's hidden states through the
        final layer norm and embedding matrix to see what tokens each layer predicts
        at each position in the sequence.

        Args:
            text: Input text to analyze
            top_k: Number of top predictions to return per position (default: 1)
            layers: Specific layer indices to analyze (None = all layers)
            position: Specific position to analyze (None = all positions).
                     Supports negative indexing (-1 = last position).
            plot: If True, display a heatmap visualization showing top predictions
            max_display_tokens: Maximum number of tokens to show in visualization (from the end)
            figsize: Figure size for plot (width, height)
            cmap: Colormap for heatmap (default: 'viridis')
            font_family: Font to use for plot (for CJK support use 'Arial Unicode MS' or None for auto-detect)
            final_norm: Override final layer normalization module. If provided,
                       uses this module instead of auto-detected norm layer.
            skip_norm: If True, skip final layer normalization entirely.
                      Useful for models without a final norm layer.

        Returns:
            Dict mapping layer_idx -> list of positions -> list of (token_id, score, token_str) tuples
            Structure: {layer_idx: [[pos_0_predictions], [pos_1_predictions], ...]}
            If position is specified, each layer will have a single list of predictions.

        Example:
            >>> # Get top prediction per position per layer
            >>> results = model.logit_lens("The capital of France is")
            >>>
            >>> # Print predictions for layer 10
            >>> for pos_idx, predictions in enumerate(results[10]):
            >>>     top_token = predictions[0][2]  # Get top token string
            >>>     print(f"Position {pos_idx}: {top_token}")
            >>>
            >>> # Analyze only the last position
            >>> results = model.logit_lens("The capital of France is", position=-1)
            >>>
            >>> # Visualize with heatmap
            >>> results = model.logit_lens("The Eiffel Tower is located in the city of", plot=True)
            >>>
            >>> # Model without final norm
            >>> results = model.logit_lens("Hello", skip_norm=True)
            >>>
            >>> # Custom final norm
            >>> results = model.logit_lens("Hello", final_norm=model.model.my_norm)
        """
        import warnings
        from .core.module_resolver import find_layer_key_pattern

        # Run trace to get all layer outputs
        with self.trace(text) as trace:
            pass

        # Get tokens for displaying input
        tokens = self.encode(text)

        # Resolve final layer norm
        # Priority: skip_norm > final_norm override > resolver
        if skip_norm:
            final_norm_layer = None
        elif final_norm is not None:
            final_norm_layer = final_norm
        else:
            final_norm_layer = self._module_resolver.get_final_norm()
            if final_norm_layer is None:
                warnings.warn(
                    "Cannot find final layer norm. Tried paths:\n"
                    f"  {self._module_resolver.NORM_PATHS}\n"
                    "Proceeding without normalization. Use final_norm parameter "
                    "to specify a custom norm layer, or skip_norm=True to suppress this warning."
                )

        # Determine which layers to analyze
        if layers is None:
            layers = list(range(len(self.layers)))

        results = {}

        for layer_idx in layers:
            # Find the correct layer key pattern for this model
            layer_key = find_layer_key_pattern(trace.activations, layer_idx)
            if layer_key is None:
                continue

            layer_output = trace.activations[layer_key]  # Shape: (batch, seq_len, hidden_dim)
            if layer_output.ndim != 3:
                # Skip if not proper shape (might be a different type of activation)
                continue
            batch_size, seq_len, hidden_dim = layer_output.shape

            layer_predictions = []

            # Determine positions to analyze
            if position is not None:
                # Handle negative indexing
                actual_pos = position if position >= 0 else seq_len + position
                positions_to_analyze = [actual_pos]
            else:
                positions_to_analyze = range(seq_len)

            # For each position in the sequence
            for pos in positions_to_analyze:
                hidden = layer_output[0, pos, :]  # Shape: (hidden_dim,)

                # Apply final layer norm if available
                if final_norm_layer is not None:
                    normalized = final_norm_layer(hidden)
                else:
                    normalized = hidden

                # Get token predictions
                predictions = self.get_token_predictions(normalized, top_k=top_k, return_scores=True)

                # Add token strings
                predictions_with_str = [
                    (token_id, score, self.token_to_str(token_id))
                    for token_id, score in predictions
                ]
                layer_predictions.append(predictions_with_str)

            results[layer_idx] = layer_predictions

        # Generate visualization if requested
        if plot:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                import warnings
            except ImportError:
                print("Warning: matplotlib not available. Install with: pip install matplotlib")
                return results

            # Configure font for CJK support
            if font_family is None:
                # Auto-detect: try common CJK fonts
                import platform
                system = platform.system()
                if system == 'Darwin':  # macOS
                    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'PingFang SC', 'DejaVu Sans']
                elif system == 'Windows':
                    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
                else:  # Linux
                    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Micro Hei', 'DejaVu Sans']
            else:
                plt.rcParams['font.sans-serif'] = [font_family, 'DejaVu Sans']

            plt.rcParams['axes.unicode_minus'] = False

            # Suppress font warnings for missing glyphs
            warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*missing from font.*')

            # Prepare data for heatmap
            layer_indices = sorted(results.keys())
            seq_len = len(results[layer_indices[0]])

            # Limit displayed tokens if sequence is too long
            start_pos = max(0, seq_len - max_display_tokens)
            display_seq_len = seq_len - start_pos

            # Build matrix of top predictions (layers × positions)
            predictions_matrix = []
            input_token_labels = []

            for layer_idx in layer_indices:
                layer_row = []
                for pos in range(start_pos, seq_len):
                    # Get top prediction for this position
                    top_pred = results[layer_idx][pos][0][2]  # Top token string
                    layer_row.append(top_pred)
                predictions_matrix.append(layer_row)

            # Get input tokens for x-axis labels
            for pos in range(start_pos, seq_len):
                token_str = self.token_to_str(tokens[pos])
                input_token_labels.append(token_str)

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create a categorical colormap - we need unique colors for unique tokens
            # First collect all unique tokens
            all_tokens = set()
            for row in predictions_matrix:
                all_tokens.update(row)
            all_tokens = sorted(list(all_tokens))

            # Create a mapping from token to integer
            token_to_idx = {token: idx for idx, token in enumerate(all_tokens)}

            # Convert predictions matrix to integer indices
            numeric_matrix = np.array([
                [token_to_idx[token] for token in row]
                for row in predictions_matrix
            ])

            # Create heatmap
            im = ax.imshow(numeric_matrix, cmap=cmap, aspect='auto', interpolation='nearest')

            # Set ticks and labels
            ax.set_xticks(np.arange(display_seq_len))
            ax.set_yticks(np.arange(len(layer_indices)))
            ax.set_xticklabels(input_token_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels([f"Layer {i}" for i in layer_indices])

            # Add text annotations showing the predicted tokens
            for i in range(len(layer_indices)):
                for j in range(display_seq_len):
                    pred_token = predictions_matrix[i][j]
                    # Truncate long tokens for display
                    display_token = pred_token if len(pred_token) <= 10 else pred_token[:8] + ".."
                    ax.text(j, i, display_token,
                           ha="center", va="center", color="white",
                           fontsize=8, weight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.3))

            # Labels and title
            ax.set_xlabel("Input Token Position", fontsize=12, weight='bold')
            ax.set_ylabel("Layer", fontsize=12, weight='bold')
            title_text = text if len(text) <= 60 else f"{text[:60]}..."
            ax.set_title(f'Token Predictions Across Layers (Logit Lens)\nInput: "{title_text}"',
                        fontsize=14, pad=20, weight='bold')

            # Add a note about the color legend
            colorbar_labels = all_tokens[:20] if len(all_tokens) > 20 else all_tokens
            legend_text = "Color represents predicted token\nShowing unique tokens across all predictions"
            ax.text(0.02, -0.15, legend_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            plt.tight_layout()
            plt.show()

        return results

    def tuned_lens(
        self,
        text: str,
        tuned_lens: Any,
        top_k: int = 1,
        layers: Optional[List[int]] = None,
        position: Optional[int] = None,
        plot: bool = False,
        max_display_tokens: int = 15,
        figsize: tuple = (16, 10),
        cmap: str = 'viridis',
        font_family: Optional[str] = None,
        final_norm: Any = None,
        skip_norm: bool = False,
    ) -> Dict[int, List[List[tuple]]]:
        """
        Apply tuned lens for improved layer-wise predictions.

        The tuned lens technique (Belrose et al., 2023) uses learned affine
        transformations for each layer to correct for coordinate system mismatches,
        producing more accurate intermediate predictions than the standard logit lens.

        Args:
            text: Input text to analyze
            tuned_lens: Trained TunedLens instance with layer translators
            top_k: Number of top predictions to return per position (default: 1)
            layers: Specific layer indices to analyze (None = all layers)
            position: Specific position to analyze (None = all positions).
                     Supports negative indexing (-1 = last position).
            plot: If True, display a heatmap visualization showing top predictions
            max_display_tokens: Maximum number of tokens to show in visualization (from the end)
            figsize: Figure size for plot (width, height)
            cmap: Colormap for heatmap (default: 'viridis')
            font_family: Font to use for plot (for CJK support use 'Arial Unicode MS' or None for auto-detect)
            final_norm: Optional override for the final layer norm. Pass a callable to use a custom norm.
            skip_norm: If True, skip final layer normalization (for models without it)

        Returns:
            Dict mapping layer_idx -> list of positions -> list of (token_id, score, token_str) tuples
            Structure: {layer_idx: [[pos_0_predictions], [pos_1_predictions], ...]}

        Example:
            >>> # Train or load tuned lens
            >>> tuned_lens = model.train_tuned_lens(texts, num_steps=250)
            >>> # Or load pre-trained:
            >>> # tuned_lens = TunedLens.load("tuned_lens")  # Loads .npz and .json files
            >>>
            >>> # Apply tuned lens
            >>> results = model.tuned_lens(
            ...     "The capital of France is",
            ...     tuned_lens,
            ...     layers=[0, 5, 10, 15],
            ...     plot=True
            ... )
            >>>
            >>> # Compare with regular logit lens
            >>> regular = model.logit_lens("The capital of France is", layers=[0, 5, 10, 15])

        Reference:
            Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens"
            https://arxiv.org/abs/2303.08112
        """
        import warnings
        from .core.module_resolver import find_layer_key_pattern

        # Run trace to get all layer outputs
        with self.trace(text) as trace:
            pass

        # Get tokens for displaying input
        tokens = self.encode(text)

        # Get final layer norm (handle overrides)
        if skip_norm:
            final_norm_layer = None
        elif final_norm is not None:
            final_norm_layer = final_norm
        else:
            final_norm_layer = self._module_resolver.get_final_norm()
            if final_norm_layer is None:
                warnings.warn(
                    "Cannot find final layer norm for tuned lens. Tried paths:\n"
                    f"  {self._module_resolver.NORM_PATHS}\n"
                    "Proceeding without normalization. This may affect prediction quality.\n"
                    "Use skip_norm=True to suppress this warning, or final_norm=<norm> to provide one."
                )

        # Determine which layers to analyze
        if layers is None:
            layers = list(range(len(self.layers)))

        results = {}

        for layer_idx in layers:
            # Find the correct layer key pattern for this model
            layer_key = find_layer_key_pattern(trace.activations, layer_idx)
            if layer_key is None:
                continue

            layer_output = trace.activations[layer_key]  # Shape: (batch, seq_len, hidden_dim)
            if layer_output.ndim != 3:
                continue
            batch_size, seq_len, hidden_dim = layer_output.shape

            layer_predictions = []

            # Determine positions to analyze with bounds checking
            if position is not None:
                actual_pos = position if position >= 0 else seq_len + position
                # Validate position is within bounds
                if actual_pos < 0 or actual_pos >= seq_len:
                    raise ValueError(
                        f"Position {position} is out of bounds for sequence length {seq_len}. "
                        f"Valid range: [{-seq_len}, {seq_len - 1}]"
                    )
                positions_to_analyze = [actual_pos]
            else:
                positions_to_analyze = range(seq_len)

            for pos in positions_to_analyze:
                hidden = layer_output[0, pos, :]  # Shape: (hidden_dim,)

                # Apply tuned lens translator for this layer
                translated = tuned_lens(hidden, layer_idx)

                # Apply final layer norm if available
                if final_norm_layer is not None:
                    normalized = final_norm_layer(translated)
                else:
                    normalized = translated

                # Get token predictions
                predictions = self.get_token_predictions(normalized, top_k=top_k, return_scores=True)

                # Add token strings
                predictions_with_str = [
                    (token_id, score, self.token_to_str(token_id))
                    for token_id, score in predictions
                ]
                layer_predictions.append(predictions_with_str)

            results[layer_idx] = layer_predictions

        # Generate visualization if requested
        if plot:
            self._plot_logit_lens(
                results, text, tokens, max_display_tokens, figsize, cmap, font_family,
                title_prefix="Tuned Lens"
            )

        return results

    def _plot_logit_lens(
        self,
        results: Dict[int, List[List[tuple]]],
        text: str,
        tokens: List[int],
        max_display_tokens: int = 15,
        figsize: tuple = (16, 10),
        cmap: str = 'viridis',
        font_family: Optional[str] = None,
        title_prefix: str = "Logit Lens"
    ) -> None:
        """
        Internal method to plot logit lens or tuned lens results.

        Args:
            results: Dict mapping layer_idx -> predictions
            text: Original input text
            tokens: Token IDs
            max_display_tokens: Max tokens to display
            figsize: Figure size
            cmap: Colormap
            font_family: Font family for rendering
            title_prefix: Prefix for the title ("Logit Lens" or "Tuned Lens")
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib import rcParams
        except ImportError:
            raise ImportError("Plotting requires matplotlib. Install with: pip install matplotlib")

        # Set up font for potential CJK characters
        if font_family:
            rcParams['font.family'] = font_family
        else:
            import platform
            if platform.system() == 'Darwin':
                rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']
            else:
                rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

        # Get input tokens for display
        input_tokens_str = [self.token_to_str(t) for t in tokens]

        # Limit to last N tokens for display
        if len(input_tokens_str) > max_display_tokens:
            input_tokens_str = input_tokens_str[-max_display_tokens:]
            # Adjust results to match
            offset = len(tokens) - max_display_tokens
            trimmed_results = {}
            for layer_idx, preds in results.items():
                trimmed_results[layer_idx] = preds[offset:]
            results = trimmed_results

        # Build prediction matrix for heatmap
        layer_indices = sorted(results.keys())
        num_layers = len(layer_indices)
        num_positions = len(input_tokens_str)

        if num_layers == 0 or num_positions == 0:
            print("No data to plot")
            return

        # Create matrix of top predictions (token strings)
        predictions_matrix = []
        for layer_idx in layer_indices:
            layer_preds = results[layer_idx]
            row = []
            for pos_preds in layer_preds:
                if pos_preds:
                    top_token_str = pos_preds[0][2]
                    row.append(top_token_str)
                else:
                    row.append("")
            predictions_matrix.append(row)

        # Create color mapping for unique tokens
        all_tokens = set()
        for row in predictions_matrix:
            all_tokens.update(row)
        all_tokens = sorted(list(all_tokens))
        token_to_color = {t: i for i, t in enumerate(all_tokens)}

        # Create numeric matrix for coloring
        color_matrix = []
        for row in predictions_matrix:
            color_row = [token_to_color.get(t, 0) for t in row]
            color_matrix.append(color_row)

        # Plot
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        import numpy as np
        color_array = np.array(color_matrix)
        im = ax.imshow(color_array, cmap=cmap, aspect='auto')

        # Set ticks
        ax.set_xticks(range(num_positions))
        ax.set_yticks(range(num_layers))

        # Format x-tick labels
        x_labels = []
        for i, t in enumerate(input_tokens_str):
            display_t = repr(t)[1:-1] if len(t) <= 10 else repr(t[:10])[1:-1] + "..."
            x_labels.append(f"{display_t}")
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels([f"L{i}" for i in layer_indices], fontsize=9)

        # Add text annotations
        for i in range(num_layers):
            for j in range(num_positions):
                pred_token = predictions_matrix[i][j]
                display_pred = pred_token[:8] if len(pred_token) > 8 else pred_token
                display_pred = repr(display_pred)[1:-1]
                text_color = 'white' if color_array[i, j] > len(all_tokens) / 2 else 'black'
                ax.text(j, i, display_pred, ha='center', va='center',
                       fontsize=7, color=text_color, weight='bold')

        ax.set_xlabel("Input Token Position", fontsize=12, weight='bold')
        ax.set_ylabel("Layer", fontsize=12, weight='bold')
        title_text = text if len(text) <= 60 else f"{text[:60]}..."
        ax.set_title(f'Token Predictions Across Layers ({title_prefix})\nInput: "{title_text}"',
                    fontsize=14, pad=20, weight='bold')

        legend_text = "Color represents predicted token\nShowing unique tokens across all predictions"
        ax.text(0.02, -0.15, legend_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.show()

    def activation_patching(
        self,
        clean_text: str,
        corrupted_text: str,
        component: str = "mlp",
        layers: Optional[List[int]] = None,
        metric: str = "l2",
        plot: bool = False,
        figsize: tuple = (12, 8),
        cmap: str = "RdBu_r"
    ) -> Dict[int, float]:
        """
        Automated activation patching to find important layers for a task.

        Patches clean activations into corrupted runs at each layer and measures
        how much this recovers the clean output. High recovery indicates the
        layer is important for the task.

        Args:
            clean_text: Clean/correct input text
            corrupted_text: Corrupted/incorrect input text
            component: Component to patch. Valid options:
                - "mlp": The MLP/feed-forward component
                - "self_attn": The self-attention component
                - Full paths like "mlp.gate_proj", "self_attn.q_proj"
                Note: Use the component names from your model architecture.
                For mlx-lm models, use "self_attn" (not "attn").
            layers: Specific layers to test (None = all layers)
            metric: Distance metric. Options:
                - "l2": Euclidean distance (default, with overflow protection)
                - "cosine": Cosine distance (recommended for large vocabularies)
                - "mse": Mean squared error (most stable for huge models)
            plot: If True, display heatmap of results
            figsize: Figure size for plot
            cmap: Colormap for heatmap (default: "RdBu_r" - blue=positive, red=negative)

        Returns:
            Dict mapping layer_idx -> recovery percentage
            Positive % = layer is important (patching helps)
            Negative % = layer encodes corruption (patching hurts)
            ~0% = layer not relevant for this task

        Example:
            >>> # Find which MLPs are important for factual knowledge
            >>> results = model.activation_patching(
            >>>     clean_text="Paris is the capital of France",
            >>>     corrupted_text="London is the capital of France",
            >>>     component="mlp",
            >>>     plot=True
            >>> )
            >>>
            >>> # Get most important layers
            >>> sorted_layers = sorted(results.items(), key=lambda x: x[1], reverse=True)
            >>> print(f"Most important: Layer {sorted_layers[0][0]} ({sorted_layers[0][1]:.1f}%)")
        """
        from . import interventions as iv

        # Get baseline outputs
        print(f"Getting clean output...")
        with self.trace(clean_text):
            clean_output = self.output.save()

        print(f"Getting corrupted output...")
        with self.trace(corrupted_text):
            corrupted_output = self.output.save()

        mx.eval(clean_output, corrupted_output)

        # Distance function with numerical stability
        if metric == "l2":
            def distance(a, b):
                """L2 distance with numerical stability for large vocabularies"""
                diff = a - b
                # Use float32 for accumulation to prevent overflow
                diff_f32 = diff.astype(mx.float32)
                squared_sum = mx.sum(diff_f32 * diff_f32)
                # Check for overflow
                if mx.isinf(squared_sum) or mx.isnan(squared_sum):
                    # Fallback: use mean squared error instead of sum
                    mse = mx.mean(diff_f32 * diff_f32)
                    return float(mx.sqrt(mse) * mx.sqrt(float(diff.size)))
                return float(mx.sqrt(squared_sum))
        elif metric == "cosine":
            def distance(a, b):
                """Cosine distance with numerical stability"""
                # Use float32 for better precision
                a_f32 = a.astype(mx.float32)
                b_f32 = b.astype(mx.float32)
                a_norm = mx.sqrt(mx.sum(a_f32 * a_f32))
                b_norm = mx.sqrt(mx.sum(b_f32 * b_f32))
                if mx.isinf(a_norm) or mx.isinf(b_norm) or a_norm < 1e-10 or b_norm < 1e-10:
                    # Fallback: use normalized mean
                    a_normalized = a_f32 / (mx.sqrt(mx.mean(a_f32 * a_f32)) + 1e-10)
                    b_normalized = b_f32 / (mx.sqrt(mx.mean(b_f32 * b_f32)) + 1e-10)
                    return float(1.0 - mx.mean(a_normalized * b_normalized))
                a_normalized = a_f32 / a_norm
                b_normalized = b_f32 / b_norm
                return float(1.0 - mx.sum(a_normalized * b_normalized))
        elif metric == "mse":
            def distance(a, b):
                """Mean squared error - stable for large vocabularies"""
                diff = a.astype(mx.float32) - b.astype(mx.float32)
                return float(mx.mean(diff * diff))
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'l2', 'cosine', or 'mse'")

        baseline = distance(corrupted_output[0, -1], clean_output[0, -1])
        print(f"Baseline {metric} distance: {baseline:.4f}\n")

        # Determine layers to test
        if layers is None:
            layers = list(range(len(self.layers)))

        results = {}

        print(f"Patching {component} at each layer...")
        for layer_idx in layers:
            print(f"  Layer {layer_idx:2d}...", end="\r")

            # Build component key - try multiple path patterns for different model types
            # Supports: Llama/Mistral (layers.X), GPT-2 (h.X), and others

            # Map component names for different architectures
            # GPT-2 uses "attn" instead of "self_attn"
            component_variants = [component]
            if component == "self_attn":
                component_variants.append("attn")  # GPT-2 style
            elif component == "attn":
                component_variants.append("self_attn")  # Llama style

            # Special handling for "output" component - refers to the layer itself
            if component == "output":
                # "output" means the layer's output, which is stored under the layer name
                path_patterns = [
                    f"model.model.layers.{layer_idx}",   # mlx-lm Llama models
                    f"model.layers.{layer_idx}",         # models with .model wrapper
                    f"layers.{layer_idx}",               # direct layers
                    f"model.model.h.{layer_idx}",        # GPT-2 style
                    f"model.h.{layer_idx}",              # GPT-2 without double model
                    f"h.{layer_idx}",                    # Direct GPT-2
                ]
            elif "." in component:
                # Full path provided (e.g., "mlp.gate_proj")
                path_patterns = []
                for comp in component_variants:
                    path_patterns.extend([
                        f"model.model.layers.{layer_idx}.{comp}",  # mlx-lm Llama models
                        f"model.layers.{layer_idx}.{comp}",        # models with .model wrapper
                        f"layers.{layer_idx}.{comp}",              # direct layers
                        f"model.model.h.{layer_idx}.{comp}",       # GPT-2 style
                        f"model.h.{layer_idx}.{comp}",             # GPT-2 without double model
                        f"h.{layer_idx}.{comp}",                   # Direct GPT-2
                    ])
            else:
                # Simple component name (e.g., "mlp", "self_attn", "attn")
                path_patterns = []
                for comp in component_variants:
                    path_patterns.extend([
                        f"model.model.layers.{layer_idx}.{comp}",  # mlx-lm Llama models
                        f"model.layers.{layer_idx}.{comp}",        # models with .model wrapper
                        f"layers.{layer_idx}.{comp}",              # direct layers
                        f"model.model.h.{layer_idx}.{comp}",       # GPT-2 style
                        f"model.h.{layer_idx}.{comp}",             # GPT-2 without double model
                        f"h.{layer_idx}.{comp}",                   # Direct GPT-2
                    ])

            # Get clean activation - find which path works
            activation_key = None
            with self.trace(clean_text) as trace:
                for path in path_patterns:
                    if path in trace.activations:
                        activation_key = path
                        clean_activation = trace.activations[path]
                        break

                if activation_key is None:
                    print(f"\nWarning: No activation found for layer {layer_idx}.{component}, skipping")
                    print(f"  Tried: {path_patterns[:3]}")
                    continue

            mx.eval(clean_activation)

            # Derive intervention key from activation key
            # Remove "model." or "model.model." prefix for interventions
            if activation_key.startswith("model.model."):
                intervention_key = activation_key[12:]  # Remove "model.model."
            elif activation_key.startswith("model."):
                intervention_key = activation_key[6:]   # Remove "model."
            else:
                intervention_key = activation_key

            # Patch into corrupted
            with self.trace(corrupted_text,
                           interventions={intervention_key: iv.replace_with(clean_activation)}):
                patched_output = self.output.save()

            mx.eval(patched_output)

            # Calculate recovery
            dist = distance(patched_output[0, -1], clean_output[0, -1])
            if baseline > 1e-10:
                recovery = (baseline - dist) / baseline * 100
            else:
                # Baseline is zero (clean == corrupted), can't compute recovery
                recovery = 0.0
            results[layer_idx] = recovery

        print(f"\nCompleted patching {len(results)} layers")

        # Generate visualization if requested
        if plot:
            try:
                import matplotlib.pyplot as plt
                import numpy as np
            except ImportError:
                print("Warning: matplotlib not available. Install with: pip install matplotlib")
                return results

            # Prepare data
            layer_indices = sorted(results.keys())
            recoveries = [results[layer_idx] for layer_idx in layer_indices]

            # Create figure
            fig, ax = plt.subplots(figsize=figsize)

            # Create bar plot
            colors = ['#2166ac' if r > 0 else '#b2182b' for r in recoveries]
            bars = ax.bar(layer_indices, recoveries, color=colors, alpha=0.7, edgecolor='black')

            # Add horizontal line at 0
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

            # Labels and title
            ax.set_xlabel("Layer", fontsize=12, weight='bold')
            ax.set_ylabel("Recovery (%)", fontsize=12, weight='bold')
            clean_short = clean_text if len(clean_text) <= 40 else f"{clean_text[:40]}..."
            corrupted_short = corrupted_text if len(corrupted_text) <= 40 else f"{corrupted_text[:40]}..."
            ax.set_title(
                f'Activation Patching: {component.upper()}\n'
                f'Clean: "{clean_short}"\n'
                f'Corrupted: "{corrupted_short}"',
                fontsize=12, pad=20, weight='bold'
            )

            # Add value labels on bars
            for i, (layer_idx, recovery) in enumerate(zip(layer_indices, recoveries)):
                height = recovery
                ax.text(layer_idx, height + (3 if height > 0 else -3),
                       f'{recovery:.1f}%',
                       ha='center', va='bottom' if height > 0 else 'top',
                       fontsize=8)

            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2166ac', alpha=0.7, label='Positive (important)'),
                Patch(facecolor='#b2182b', alpha=0.7, label='Negative (encodes corruption)')
            ]
            ax.legend(handles=legend_elements, loc='best')

            # Grid
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

            plt.tight_layout()
            plt.show()

        return results
