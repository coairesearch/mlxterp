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
        return_scores: bool = False
    ) -> Union[List[int], List[tuple]]:
        """
        Decode hidden states to token predictions using the model's output projection.

        For models with weight tying (like Llama), uses the embedding layer's weights
        transposed as the output projection. Handles quantized embeddings automatically.

        Args:
            hidden_state: Hidden state tensor, shape (hidden_dim,) or (batch, hidden_dim)
            top_k: Number of top predictions to return
            return_scores: If True, return (token_id, score) tuples instead of just token_ids

        Returns:
            List of token IDs or (token_id, score) tuples

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
        """
        # Get embedding weights for output projection
        # For weight-tied models, embedding.T is used as lm_head
        if hasattr(self.model, 'lm_head'):
            # Direct lm_head (not weight-tied)
            logits = self.model.lm_head(hidden_state)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            # Weight-tied: use embedding weights transposed
            embed_layer = self.model.model.embed_tokens

            # For quantized embeddings, we need to use the layer as a function
            # to properly dequantize. Create a dummy input to get output shape.
            # Then we can use it as the lm_head by transposing the operation.

            # The simplest approach: treat the embedding layer as the lm_head
            # by using it in "reverse" mode through matrix multiplication
            # But for quantized layers, we need the dequantized weights

            # Check if it's a quantized embedding by looking for quantization attributes
            is_quantized = hasattr(embed_layer, 'scales') or hasattr(embed_layer, 'biases') or \
                          (hasattr(embed_layer, 'weight') and embed_layer.weight.shape[0] != self.vocab_size)

            if is_quantized:
                # Quantized embedding - we need to use a different approach
                # For now, create a workaround by computing similarities
                # This is less efficient but works with quantized embeddings

                # Get vocab size
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

                logits = mx.concatenate(all_logits, axis=0)
            else:
                # Standard embedding with weight attribute
                embed_weights = embed_layer.weight  # Shape: (vocab_size, hidden_dim)
                logits = hidden_state @ embed_weights.T
        else:
            raise AttributeError(
                "Cannot find output projection layer. Model must have either "
                "'lm_head' or weight-tied embeddings ('model.embed_tokens')"
            )

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

    def logit_lens(
        self,
        text: str,
        top_k: int = 1,
        layers: Optional[List[int]] = None,
        plot: bool = False,
        max_display_tokens: int = 15,
        figsize: tuple = (16, 10),
        cmap: str = 'viridis',
        font_family: Optional[str] = None
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
            plot: If True, display a heatmap visualization showing top predictions
            max_display_tokens: Maximum number of tokens to show in visualization (from the end)
            figsize: Figure size for plot (width, height)
            cmap: Colormap for heatmap (default: 'viridis')
            font_family: Font to use for plot (for CJK support use 'Arial Unicode MS' or None for auto-detect)

        Returns:
            Dict mapping layer_idx -> list of positions -> list of (token_id, score, token_str) tuples
            Structure: {layer_idx: [[pos_0_predictions], [pos_1_predictions], ...]}

        Example:
            >>> # Get top prediction per position per layer
            >>> results = model.logit_lens("The capital of France is")
            >>>
            >>> # Print predictions for layer 10
            >>> for pos_idx, predictions in enumerate(results[10]):
            >>>     top_token = predictions[0][2]  # Get top token string
            >>>     print(f"Position {pos_idx}: {top_token}")
            >>>
            >>> # Visualize with heatmap
            >>> results = model.logit_lens("The Eiffel Tower is located in the city of", plot=True)
        """
        # Run trace to get all layer outputs
        with self.trace(text) as trace:
            pass

        # Get tokens for displaying input
        tokens = self.encode(text)

        # Get final layer norm for projection
        if "model.model.norm" in trace.activations:
            final_norm_layer = self.model.model.norm
        else:
            raise ValueError("Cannot find final layer norm (model.model.norm)")

        # Determine which layers to analyze
        if layers is None:
            layers = list(range(len(self.layers)))

        # Get embedding layer for projection
        embed_layer = self.model.model.embed_tokens
        vocab_size = self.vocab_size

        results = {}

        for layer_idx in layers:
            # Get layer output
            layer_key = f"model.model.layers.{layer_idx}"
            if layer_key not in trace.activations:
                continue

            layer_output = trace.activations[layer_key]  # Shape: (batch, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = layer_output.shape

            layer_predictions = []

            # For each position in the sequence
            for pos in range(seq_len):
                hidden = layer_output[0, pos, :]  # Shape: (hidden_dim,)

                # Apply final layer norm
                normalized = final_norm_layer(hidden)

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
            component: Component to patch ("mlp", "self_attn", "output", or full path like "mlp.gate_proj")
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

            # Build component key
            if "." in component:
                # Full path provided (e.g., "mlp.gate_proj")
                activation_key = f"model.model.layers.{layer_idx}.{component}"
                intervention_key = f"layers.{layer_idx}.{component}"
            else:
                # Simple component name
                activation_key = f"model.model.layers.{layer_idx}.{component}"
                intervention_key = f"layers.{layer_idx}.{component}"

            # Get clean activation
            with self.trace(clean_text) as trace:
                if activation_key not in trace.activations:
                    print(f"\nWarning: {activation_key} not found, skipping")
                    continue
                clean_activation = trace.activations[activation_key]

            mx.eval(clean_activation)

            # Patch into corrupted
            with self.trace(corrupted_text,
                           interventions={intervention_key: iv.replace_with(clean_activation)}):
                patched_output = self.output.save()

            mx.eval(patched_output)

            # Calculate recovery
            dist = distance(patched_output[0, -1], clean_output[0, -1])
            recovery = (baseline - dist) / baseline * 100
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
