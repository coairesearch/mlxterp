"""
Tuned Lens implementation for improved layer-wise predictions.

The tuned lens technique (Belrose et al., 2023) trains small affine transformations
for each layer to correct for coordinate system mismatches between layers, producing
more accurate intermediate predictions than the standard logit lens.

Reference:
    Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens"
    https://arxiv.org/abs/2303.08112
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
from typing import Optional, List, Dict, Any, Callable
from pathlib import Path
import json


def log_softmax(x: mx.array, axis: int = -1) -> mx.array:
    """
    Compute log softmax.

    MLX doesn't have a built-in log_softmax, so we compute it as:
    log_softmax(x) = x - logsumexp(x)

    Args:
        x: Input array
        axis: Axis along which to compute log softmax

    Returns:
        Log softmax of x
    """
    return x - mx.logsumexp(x, axis=axis, keepdims=True)


class TunedLens(nn.Module):
    """
    Learned affine translators for each layer.

    The tuned lens uses layer-specific affine transformations (Wx + b) to map
    hidden states from each layer into a space where the final output projection
    can make accurate predictions. This corrects for coordinate system mismatches
    between layers.

    Attributes:
        num_layers: Number of transformer layers
        hidden_dim: Dimension of hidden states
        translators: List of linear layers, one per transformer layer

    Example:
        >>> # Create tuned lens for a model
        >>> tuned_lens = TunedLens(num_layers=32, hidden_dim=4096)
        >>>
        >>> # Apply to a hidden state from layer 10
        >>> translated = tuned_lens(hidden_state, layer_idx=10)
        >>>
        >>> # Save and load (creates .npz and .json files)
        >>> tuned_lens.save("tuned_lens_llama")  # Creates tuned_lens_llama.npz and tuned_lens_llama.json
        >>> loaded = TunedLens.load("tuned_lens_llama")
    """

    def __init__(self, num_layers: int, hidden_dim: int):
        """
        Initialize tuned lens with identity-initialized translators.

        Args:
            num_layers: Number of transformer layers in the model
            hidden_dim: Dimension of hidden states
        """
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # Create translators for each layer, initialized close to identity
        self.translators = []
        for i in range(num_layers):
            translator = nn.Linear(hidden_dim, hidden_dim)
            self.translators.append(translator)

        # Initialize close to identity for stability
        self._init_identity()

    def _init_identity(self):
        """Initialize translators close to identity transformation."""
        for translator in self.translators:
            # Set weight to identity matrix
            translator.weight = mx.eye(self.hidden_dim)
            # Set bias to zeros
            translator.bias = mx.zeros((self.hidden_dim,))

    def __call__(self, h: mx.array, layer_idx: int) -> mx.array:
        """
        Apply the translator for a specific layer.

        Args:
            h: Hidden state tensor, shape (hidden_dim,) or (batch, seq_len, hidden_dim)
            layer_idx: Index of the layer this hidden state came from

        Returns:
            Translated hidden state with same shape as input
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(
                f"layer_idx {layer_idx} out of range [0, {self.num_layers})"
            )
        return self.translators[layer_idx](h)

    def save(self, path: str) -> None:
        """
        Save tuned lens weights and config to a file.

        Args:
            path: Path to save the weights (will create .npz and .json files)
        """
        path = Path(path)

        # Save config
        config = {
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
        }
        config_path = path.with_suffix(".json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Save weights using mx.savez
        weights_path = path.with_suffix(".npz")
        weights = {}
        for i, translator in enumerate(self.translators):
            weights[f"translators_{i}_weight"] = translator.weight
            weights[f"translators_{i}_bias"] = translator.bias
        mx.savez(str(weights_path), **weights)

    @classmethod
    def load(cls, path: str) -> "TunedLens":
        """
        Load tuned lens from saved files.

        Args:
            path: Path to the saved weights (expects .npz and .json files)

        Returns:
            Loaded TunedLens instance
        """
        path = Path(path)

        # Load config
        config_path = path.with_suffix(".json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            num_layers=config["num_layers"],
            hidden_dim=config["hidden_dim"],
        )

        # Load weights
        weights_path = path.with_suffix(".npz")
        weights = mx.load(str(weights_path))

        for i in range(instance.num_layers):
            instance.translators[i].weight = weights[f"translators_{i}_weight"]
            instance.translators[i].bias = weights[f"translators_{i}_bias"]

        return instance


def kl_divergence(log_p: mx.array, log_q: mx.array) -> mx.array:
    """
    Compute KL divergence KL(p || q) from log probabilities.

    Args:
        log_p: Log probabilities of target distribution (from final layer)
        log_q: Log probabilities of predicted distribution (from tuned lens)

    Returns:
        KL divergence (scalar)
    """
    p = mx.exp(log_p)
    return mx.sum(p * (log_p - log_q), axis=-1).mean()


def train_tuned_lens(
    model: Any,
    dataset: List[str],
    num_steps: int = 250,
    learning_rate: float = 1.0,
    momentum: float = 0.9,
    max_seq_len: int = 2048,
    gradient_clip: float = 1.0,
    save_path: Optional[str] = None,
    verbose: bool = True,
    callback: Optional[Callable[[int, float], None]] = None,
) -> TunedLens:
    """
    Train tuned lens translators using KL divergence loss.

    The training minimizes KL divergence between:
    - Target: Model's final output distribution
    - Prediction: Tuned lens prediction from each intermediate layer

    Training follows the paper's recommendations:
    - SGD with Nesterov momentum (0.9)
    - Learning rate 1.0 with linear decay over training steps
    - Gradient clipping with norm 1.0

    Note:
        Training processes one sequence chunk at a time (no batching).
        This follows the paper's approach where each training step uses
        a single sequence of up to max_seq_len tokens.

    Args:
        model: InterpretableModel instance with loaded model
        dataset: List of text strings for training
        num_steps: Number of training steps (default: 250)
        learning_rate: Initial learning rate (default: 1.0)
        momentum: Nesterov momentum coefficient (default: 0.9)
        max_seq_len: Maximum sequence length for training chunks (default: 2048)
        gradient_clip: Gradient clipping norm (default: 1.0)
        save_path: Optional path to save trained weights (creates .npz and .json files)
        verbose: If True, print training progress
        callback: Optional callback function called with (step, loss) after each step

    Returns:
        Trained TunedLens instance

    Example:
        >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
        >>> texts = ["Sample text 1", "Sample text 2", ...]
        >>> tuned_lens = train_tuned_lens(
        ...     model, texts,
        ...     num_steps=250,
        ...     save_path="tuned_lens"  # Creates tuned_lens.npz and tuned_lens.json
        ... )
    """
    import mlx.optimizers as optim
    from .core.module_resolver import find_layer_key_pattern

    # Get model dimensions
    num_layers = len(model.layers)

    # Get hidden dimension from model
    # Try to infer from first layer's weight matrix or embedding
    try:
        # Try to get from embedding layer
        embed_layer = model._module_resolver.get_embedding_layer()
        if hasattr(embed_layer, "weight"):
            hidden_dim = embed_layer.weight.shape[1]
        elif hasattr(embed_layer, "scales"):
            # Quantized embedding
            hidden_dim = embed_layer.scales.shape[0]
        else:
            # Fallback: try to get from a trace
            sample_tokens = model.encode(dataset[0][:100])
            with model.trace(mx.array([sample_tokens[:10]])) as trace:
                pass
            # Get dimension from first layer output
            layer_key = find_layer_key_pattern(trace.activations, 0)
            if layer_key:
                hidden_dim = trace.activations[layer_key].shape[-1]
            else:
                raise ValueError("Could not determine hidden dimension from model")
    except Exception as e:
        raise ValueError(f"Could not determine model hidden dimension: {e}")

    # Create tuned lens
    tuned_lens = TunedLens(num_layers=num_layers, hidden_dim=hidden_dim)

    # Get final layer norm and output projection
    final_norm = model._module_resolver.get_final_norm()
    proj_module, proj_path, is_weight_tied = model._module_resolver.get_output_projection()

    if proj_module is None:
        raise ValueError("Could not find output projection (lm_head or embedding) in model")

    # Create optimizer with Nesterov momentum
    optimizer = optim.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)

    # Build a single large text from dataset for chunking
    all_text = " ".join(dataset)
    all_tokens = model.encode(all_text)

    if len(all_tokens) < max_seq_len:
        raise ValueError(
            f"Dataset too small: {len(all_tokens)} tokens, need at least {max_seq_len}. "
            "Provide more text data."
        )

    # Training loop
    step = 0
    text_position = 0

    if verbose:
        print(f"Training tuned lens: {num_layers} layers, {hidden_dim} hidden dim")
        print(f"Dataset: {len(all_tokens)} tokens, {num_steps} steps")

    # Define loss function
    def compute_loss(tuned_lens_params, input_tokens, layer_activations, target_logits):
        """Compute KL divergence loss for all layers."""
        # Update tuned lens with current params
        tuned_lens.update(tuned_lens_params)

        total_loss = mx.array(0.0)
        seq_len = input_tokens.shape[1]

        for layer_idx in range(num_layers):
            layer_key = find_layer_key_pattern(layer_activations, layer_idx)
            if layer_key is None:
                continue

            layer_output = layer_activations[layer_key]  # (batch, seq_len, hidden_dim)

            # Apply tuned lens translator
            translated = tuned_lens(layer_output, layer_idx)

            # Apply final norm if available
            if final_norm is not None:
                translated = final_norm(translated)

            # Compute logits through output projection
            if is_weight_tied:
                # Weight-tied: use embedding weights transposed
                if hasattr(proj_module, "scales"):
                    # Quantized
                    embed_weights = mx.dequantize(
                        proj_module.weight,
                        proj_module.scales,
                        proj_module.biases,
                        proj_module.group_size,
                        proj_module.bits,
                    )
                else:
                    embed_weights = proj_module.weight
                pred_logits = translated @ embed_weights.T
            else:
                pred_logits = proj_module(translated)

            # Compute log probabilities
            pred_log_probs = log_softmax(pred_logits, axis=-1)
            target_log_probs = log_softmax(target_logits, axis=-1)

            # KL divergence loss
            layer_loss = kl_divergence(target_log_probs, pred_log_probs)
            total_loss = total_loss + layer_loss

        return total_loss / num_layers

    # Training iterations
    while step < num_steps:
        # Get next chunk of tokens
        end_pos = min(text_position + max_seq_len, len(all_tokens))
        chunk_tokens = all_tokens[text_position:end_pos]

        if len(chunk_tokens) < 10:
            # Wrap around if we've reached the end
            text_position = 0
            continue

        input_tokens = mx.array([chunk_tokens])

        # Run forward pass to get activations and target logits
        with model.trace(input_tokens) as trace:
            pass

        # Get target logits from final output
        # We need to get the model's actual output logits
        try:
            # Run model directly to get final logits
            if hasattr(model.model, '__call__'):
                target_logits = model.model(input_tokens)
                if isinstance(target_logits, tuple):
                    target_logits = target_logits[0]
            else:
                # Fallback: compute from last layer
                last_layer_key = find_layer_key_pattern(trace.activations, num_layers - 1)
                if last_layer_key:
                    last_hidden = trace.activations[last_layer_key]
                    if final_norm is not None:
                        last_hidden = final_norm(last_hidden)
                    if is_weight_tied:
                        if hasattr(proj_module, "scales"):
                            embed_weights = mx.dequantize(
                                proj_module.weight,
                                proj_module.scales,
                                proj_module.biases,
                                proj_module.group_size,
                                proj_module.bits,
                            )
                        else:
                            embed_weights = proj_module.weight
                        target_logits = last_hidden @ embed_weights.T
                    else:
                        target_logits = proj_module(last_hidden)
                else:
                    raise ValueError("Could not get target logits")
        except Exception as e:
            raise ValueError(f"Error computing target logits: {e}")

        # Compute loss and gradients
        loss_fn = lambda params: compute_loss(
            params, input_tokens, trace.activations, target_logits
        )
        loss, grads = mx.value_and_grad(loss_fn)(tuned_lens.parameters())

        # Gradient clipping
        flat_grads = mlx.utils.tree_flatten(grads)
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in flat_grads))
        if grad_norm > gradient_clip:
            scale = gradient_clip / (grad_norm + 1e-6)
            grads = mlx.utils.tree_map(lambda g: g * scale, grads)

        # Linear learning rate decay
        current_lr = learning_rate * (1 - step / num_steps)
        optimizer.learning_rate = current_lr

        # Update parameters
        optimizer.update(tuned_lens, grads)
        mx.eval(tuned_lens.parameters())

        loss_val = float(loss)

        if verbose and step % 10 == 0:
            print(f"Step {step}/{num_steps}, Loss: {loss_val:.4f}, LR: {current_lr:.4f}")

        if callback is not None:
            callback(step, loss_val)

        step += 1
        text_position = end_pos

        # Wrap around if needed
        if text_position >= len(all_tokens) - max_seq_len:
            text_position = 0

    if verbose:
        print(f"Training complete. Final loss: {loss_val:.4f}")

    # Save if path provided
    if save_path:
        tuned_lens.save(save_path)
        if verbose:
            print(f"Saved tuned lens to {save_path}")

    return tuned_lens
