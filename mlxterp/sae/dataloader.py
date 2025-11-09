"""
Memory-efficient data loader for SAE training.

Streams activations in batches instead of loading everything into memory.
"""

import mlx.core as mx
from typing import Any, List, Iterator, Tuple
import numpy as np


class ActivationDataLoader:
    """Memory-efficient loader for activation data.

    Instead of loading all activations into memory, this:
    1. Stores text samples and model reference
    2. Yields batches of activations on-demand
    3. Minimizes memory usage

    Example:
        >>> loader = ActivationDataLoader(model, layer=10, component="mlp",
        ...                               texts=dataset, batch_size=64)
        >>> for batch in loader:
        ...     # batch is a small mx.array of activations
        ...     pass
    """

    def __init__(
        self,
        model: Any,
        layer: int,
        component: str,
        texts: List[str],
        batch_size: int = 64,
        shuffle: bool = False,
        seed: int = 42,
    ):
        """Initialize data loader.

        Args:
            model: InterpretableModel instance
            layer: Layer number to extract from
            component: Component name (e.g., "mlp")
            texts: List of text samples
            batch_size: Number of activation samples per batch
            shuffle: Whether to shuffle data
            seed: Random seed for shuffling
        """
        self.model = model
        self.layer = layer
        self.component = component
        self.texts = texts
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # We'll collect activations in chunks and yield batches
        # This keeps memory usage low
        self._current_epoch = 0

    def __len__(self) -> int:
        """Estimate number of batches (approximate)."""
        # This is approximate - actual count depends on token counts
        avg_tokens_per_text = 50  # Conservative estimate
        total_samples = len(self.texts) * avg_tokens_per_text
        return total_samples // self.batch_size

    def _get_activation_key(self) -> str:
        """Find the activation key for this layer/component."""
        # Use a sample to find the key
        with self.model.trace(self.texts[0]) as trace:
            pass

        for key in trace.activations.keys():
            if f"layers.{self.layer}" in key and self.component in key:
                if (key.endswith(f".{self.component}") or
                    f".{self.component}." in key or
                    key.endswith(self.component)):
                    return key

        raise ValueError(
            f"Could not find activations for layer {self.layer}, "
            f"component '{self.component}'. "
            f"Available keys: {list(trace.activations.keys())[:5]}..."
        )

    def __iter__(self) -> Iterator[mx.array]:
        """Iterate over batches of activations.

        Yields:
            Batches of activations, shape (batch_size, d_model)
        """
        # Get activation key
        target_key = self._get_activation_key()

        # Shuffle texts if requested
        if self.shuffle:
            rng = np.random.default_rng(self.seed + self._current_epoch)
            indices = rng.permutation(len(self.texts))
            texts = [self.texts[i] for i in indices]
        else:
            texts = self.texts

        # Process texts in chunks to collect activation batches
        text_chunk_size = 8  # Process 8 texts at a time
        activation_buffer = []

        for i in range(0, len(texts), text_chunk_size):
            chunk = texts[i:i + text_chunk_size]

            # Trace this chunk
            with self.model.trace(chunk) as trace:
                pass

            # Get activations
            acts = trace.activations[target_key]  # (text_batch, seq_len, d_model)

            # Flatten to (text_batch * seq_len, d_model)
            flat = acts.reshape(-1, acts.shape[-1])

            # Add to buffer
            activation_buffer.append(flat)

            # Concatenate buffer
            if len(activation_buffer) > 0:
                combined = mx.concatenate(activation_buffer, axis=0)

                # Yield batches while we have enough samples
                while len(combined) >= self.batch_size:
                    # Yield one batch
                    batch = combined[:self.batch_size]
                    yield batch

                    # Keep remainder
                    combined = combined[self.batch_size:]

                # Update buffer with remainder
                if len(combined) > 0:
                    activation_buffer = [combined]
                else:
                    activation_buffer = []

        # Yield final partial batch if any
        if len(activation_buffer) > 0:
            final = mx.concatenate(activation_buffer, axis=0)
            if len(final) > 0:
                yield final

        self._current_epoch += 1

    def estimate_total_samples(self) -> int:
        """Estimate total number of activation samples.

        This processes a few texts to estimate average tokens per text,
        then extrapolates.

        Returns:
            Estimated total activation samples
        """
        # Sample first few texts
        sample_size = min(10, len(self.texts))
        target_key = self._get_activation_key()

        total_tokens = 0
        for text in self.texts[:sample_size]:
            with self.model.trace(text) as trace:
                pass
            acts = trace.activations[target_key]
            total_tokens += acts.shape[1]  # seq_len

        avg_tokens = total_tokens / sample_size
        estimated_total = int(len(self.texts) * avg_tokens)

        return estimated_total
