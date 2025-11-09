"""
SAE training with progress tracking and automatic optimization.

Provides a clean API for training SAEs with sensible defaults and automatic
handling of common issues like dead neurons.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Optional, Dict, Any, Callable
from tqdm import tqdm
import time
from mlx.utils import tree_map

from .config import SAEConfig
from .sae import SAE
from .batchtopk import BatchTopKSAE

# Optional Weights & Biases support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class SAETrainer:
    """Trainer for Sparse Autoencoders with automatic optimization.

    Handles:
    - Training loop with progress tracking
    - Learning rate warmup and scheduling
    - Dead neuron detection and resampling
    - Checkpointing
    - Validation
    - Metrics tracking

    Example:
        >>> # Simple training
        >>> trainer = SAETrainer(config=SAEConfig())
        >>> sae = trainer.train(model=model, layer=10, component="mlp", dataset=texts)
        >>>
        >>> # Advanced: custom config
        >>> config = SAEConfig(expansion_factor=32, k=150, learning_rate=5e-5)
        >>> trainer = SAETrainer(config=config)
        >>> sae = trainer.train(model=model, layer=10, dataset=texts)
    """

    def __init__(self, config: Optional[SAEConfig] = None):
        """Initialize trainer.

        Args:
            config: SAE configuration (uses defaults if None)
        """
        self.config = config or SAEConfig()
        self.wandb_run = None

        # Ghost gradient tracking
        self.feature_act_history = {}  # Track feature activation history {feature_id: [step1, step2, ...]}

        # Initialize W&B if configured
        if self.config.use_wandb:
            if not WANDB_AVAILABLE:
                raise ImportError(
                    "Weights & Biases is not installed. Install with: pip install wandb"
                )
            self._init_wandb()

    def train(
        self,
        model: Any,
        layer: int,
        component: str,
        dataset: List[str],
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> SAE:
        """Train SAE on activations from a specific layer/component.

        Args:
            model: InterpretableModel instance
            layer: Layer number to train on
            component: Component name (e.g., "mlp", "attn", "residual")
            dataset: List of text samples to collect activations from
            save_path: Optional path to save trained SAE
            verbose: Whether to show progress bar and metrics

        Returns:
            Trained SAE instance

        Example:
            >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct")
            >>> texts = ["Example text 1", "Example text 2", ...]
            >>>
            >>> trainer = SAETrainer()
            >>> sae = trainer.train(
            ...     model=model,
            ...     layer=10,
            ...     component="mlp",
            ...     dataset=texts,
            ...     save_path="sae_layer10.mlx"
            ... )
        """
        if verbose:
            print(f"🎯 Training SAE on Layer {layer} {component}")
            print(f"📊 Config: {self.config.expansion_factor}x expansion, k={self.config.k}")

        # Step 1: Determine d_model from a sample
        if verbose:
            print(f"\n📥 Determining model dimensions from sample...")

        d_model = self._get_activation_dimension(
            model=model,
            layer=layer,
            component=component,
            sample_text=dataset[0]
        )

        if verbose:
            print(f"✓ Activation dimension: {d_model}")

        # Step 2: Initialize SAE
        d_hidden = d_model * self.config.expansion_factor

        # Create SAE based on type
        if self.config.sae_type == "batchtopk":
            sae = BatchTopKSAE(
                d_model=d_model,
                d_hidden=d_hidden,
                k=self.config.k,
                normalize_input=self.config.normalize_input,
                tied_weights=self.config.tied_weights,
            )
        else:  # "topk" (default)
            sae = SAE(
                d_model=d_model,
                d_hidden=d_hidden,
                k=self.config.k,
                normalize_input=self.config.normalize_input,
                tied_weights=self.config.tied_weights,
            )

        # Store metadata
        sae.metadata = {
            "layer": layer,
            "component": component,
            "model_name": getattr(model, "model_path", "unknown"),
            "config": self.config.__dict__,
            "training_samples": len(dataset),
        }

        # Log SAE metadata to W&B
        if self.wandb_run:
            wandb.config.update({
                "d_model": d_model,
                "d_hidden": d_hidden,
                "layer": layer,
                "component": component,
                "model_name": sae.metadata["model_name"],
                "training_samples": len(dataset),
            })

        if verbose:
            print(f"\n🏗️  Created {sae}")

        # Step 3: Update normalization statistics (if needed)
        if self.config.normalize_input:
            if verbose:
                print(f"\n📊 Computing normalization statistics from sample...")
            self._update_normalization_from_dataset(
                sae=sae,
                model=model,
                layer=layer,
                component=component,
                dataset=dataset[:min(100, len(dataset))],  # Use sample for stats
            )
            if verbose:
                print(f"✓ Normalization stats computed")

        # Step 4: Split dataset into train/val
        val_size = int(len(dataset) * self.config.validation_split)
        if val_size > 0:
            train_dataset = dataset[:-val_size]
            val_dataset = dataset[-val_size:]
        else:
            train_dataset = dataset
            val_dataset = None

        if verbose:
            print(f"✓ Split: {len(train_dataset)} train texts, " +
                  (f"{len(val_dataset)} val texts" if val_dataset else "0 val"))

        # Step 5: Train with streaming
        if verbose:
            print(f"\n🚀 Training for {self.config.num_epochs} epochs (streaming mode)...")

        trained_sae = self._training_loop_streaming(
            sae=sae,
            model=model,
            layer=layer,
            component=component,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            save_path=save_path,
            verbose=verbose
        )

        # Step 6: Save if requested
        if save_path:
            trained_sae.save(save_path)
            if verbose:
                print(f"\n💾 Saved SAE to {save_path}")

        if verbose:
            print("\n✅ Training complete!")

        return trained_sae

    def _get_activation_dimension(
        self,
        model: Any,
        layer: int,
        component: str,
        sample_text: str
    ) -> int:
        """Get activation dimension from a sample.

        Args:
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            sample_text: Sample text to trace

        Returns:
            Activation dimension (d_model)
        """
        with model.trace(sample_text) as trace:
            pass

        # Find exact component match (not sub-components)
        for key in trace.activations.keys():
            if f"layers.{layer}" in key:
                # Match only if key ends with ".{component}" (exact match)
                if key.endswith(f".{component}"):
                    acts = trace.activations[key]
                    return acts.shape[-1]  # d_model

        raise ValueError(
            f"Could not find activations for layer {layer}, component '{component}'"
        )

    def _update_normalization_from_dataset(
        self,
        sae: SAE,
        model: Any,
        layer: int,
        component: str,
        dataset: List[str]
    ) -> None:
        """Update SAE normalization statistics from dataset sample.

        Args:
            sae: SAE instance
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            dataset: Sample of texts
        """
        # Collect a sample of activations
        sample_acts = []

        for text in dataset:
            with model.trace(text) as trace:
                pass

            # Find exact component match (not sub-components)
            for key in trace.activations.keys():
                if f"layers.{layer}" in key:
                    if key.endswith(f".{component}"):
                        acts = trace.activations[key]
                        flat = acts.reshape(-1, acts.shape[-1])
                        sample_acts.append(flat)
                        break

        # Concatenate and update stats
        if sample_acts:
            combined = mx.concatenate(sample_acts, axis=0)
            sae.update_normalization_stats(combined)

    def _training_loop_streaming(
        self,
        sae: SAE,
        model: Any,
        layer: int,
        component: str,
        train_dataset: List[str],
        val_dataset: Optional[List[str]],
        save_path: Optional[str],
        verbose: bool
    ) -> SAE:
        """Streaming training loop that generates batches on-demand.

        Args:
            sae: SAE to train
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            train_dataset: Training text samples
            val_dataset: Optional validation text samples
            save_path: Optional path to save checkpoints
            verbose: Show progress

        Returns:
            Trained SAE
        """
        # Initialize optimizer
        optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            betas=[0.9, 0.999],
            eps=1e-8,
        )

        # Find the activation key once
        activation_key = self._find_activation_key(model, layer, component, train_dataset[0])

        # Estimate total steps
        # Process a few samples to estimate tokens per text
        sample_tokens = []
        for text in train_dataset[:min(10, len(train_dataset))]:
            with model.trace(text) as trace:
                pass
            acts = trace.activations[activation_key]
            sample_tokens.append(acts.shape[1])  # seq_len

        avg_tokens = sum(sample_tokens) / len(sample_tokens)
        estimated_activations = int(len(train_dataset) * avg_tokens)
        steps_per_epoch = estimated_activations // self.config.batch_size
        total_steps = steps_per_epoch * self.config.num_epochs

        if verbose:
            print(f"Estimated ~{estimated_activations:,} activation samples")
            print(f"~{steps_per_epoch:,} steps per epoch, {total_steps:,} total steps")

        # Training state
        step = 0
        best_val_loss = float('inf')

        # Create LR schedule
        lr_schedule = self._get_lr_schedule(total_steps)

        # Create progress bar
        if verbose:
            pbar = tqdm(total=total_steps, desc="Training")

        # Training loop - STREAMING
        for epoch in range(self.config.num_epochs):
            # Shuffle dataset for this epoch
            import random
            epoch_dataset = train_dataset.copy()
            random.Random(self.config.seed + epoch).shuffle(epoch_dataset)

            # Stream batches from dataset
            batch_buffer = []

            # Process texts in larger chunks for efficiency
            # Larger chunks = fewer model traces = better GPU utilization
            text_chunk_size = self.config.text_batch_size

            for text_idx in range(0, len(epoch_dataset), text_chunk_size):
                text_batch = epoch_dataset[text_idx:text_idx + text_chunk_size]

                # Trace this batch of texts
                with model.trace(text_batch) as trace:
                    pass

                # Get activations
                acts = trace.activations[activation_key]  # (text_batch, seq_len, d_model)

                # Flatten to individual activation samples
                flat = acts.reshape(-1, acts.shape[-1])  # (text_batch*seq_len, d_model)

                # CRITICAL FIX: Force evaluation to prevent lazy computation buildup
                mx.eval(flat)

                # Add to buffer
                batch_buffer.append(flat)

                # CRITICAL FIX: Concatenate and immediately clear buffer to prevent memory leak
                if len(batch_buffer) >= 3:  # Buffer a few chunks for efficiency
                    combined = mx.concatenate(batch_buffer, axis=0)
                    batch_buffer = []  # Clear buffer immediately after concatenation

                    # Yield training batches while we have enough samples
                    while combined.shape[0] >= self.config.batch_size:
                        # Get one batch
                        batch = combined[:self.config.batch_size]

                        # Update learning rate
                        current_lr = lr_schedule(step)
                        optimizer.learning_rate = current_lr

                        # Training step (with sparsity warmup)
                        loss, metrics = self._train_step(sae, batch, optimizer, step, total_steps)
                        step += 1

                        # Clear cache frequently to prevent memory buildup
                        if step % 10 == 0:
                            mx.clear_cache()

                        # Log to W&B
                        if self.wandb_run:
                            train_metrics = {**metrics, "learning_rate": current_lr}
                            self._log_to_wandb(train_metrics, step, prefix="train")

                        # Update progress bar
                        if verbose:
                            pbar.update(1)
                            pbar.set_postfix({
                                "loss": f"{metrics['loss']:.4f}",
                                "l0": f"{metrics['l0']:.1f}",
                                "dead": f"{metrics['dead_fraction']:.2%}"
                            })

                        # Checkpoint and cleanup
                        if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
                            if verbose:
                                print(f"\n📸 Checkpoint at step {step}")

                            # Save checkpoint if save_path provided
                            if save_path:
                                import os
                                # Create checkpoint filename: base_step{N}.mlx
                                base_name = os.path.splitext(save_path)[0]
                                checkpoint_path = f"{base_name}_step{step}.mlx"
                                sae.save(checkpoint_path)
                                if verbose:
                                    print(f"   💾 Saved checkpoint to {checkpoint_path}")

                            import gc
                            gc.collect()
                            mx.clear_cache()

                        # Keep remainder
                        combined = combined[self.config.batch_size:]

                    # CRITICAL FIX: Only keep remainder, don't accumulate old buffers
                    if combined.shape[0] > 0:
                        batch_buffer = [combined]

            # Validation
            if val_dataset is not None:
                val_metrics = self._validate_streaming(
                    sae, model, layer, component, activation_key, val_dataset
                )

                # Log validation metrics
                if self.wandb_run:
                    self._log_to_wandb(val_metrics, step, prefix="val")

                if verbose:
                    print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Val loss: {val_metrics['loss']:.4f}")

                # Track best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']

        if verbose:
            pbar.close()

        # Finish W&B run
        self._finish_wandb()

        return sae

    def _find_activation_key(
        self,
        model: Any,
        layer: int,
        component: str,
        sample_text: str
    ) -> str:
        """Find the activation key for a layer/component.

        Args:
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            sample_text: Sample text to trace

        Returns:
            Activation key string
        """
        with model.trace(sample_text) as trace:
            pass

        # Find exact component match (not sub-components)
        # For component="mlp", match "layers.23.mlp" but NOT "layers.23.mlp.gate_proj"
        for key in trace.activations.keys():
            if f"layers.{layer}" in key:
                # Match only if key ends with ".{component}" (exact match)
                # This prevents matching sub-components like mlp.gate_proj
                if key.endswith(f".{component}"):
                    return key

        raise ValueError(
            f"Could not find activations for layer {layer}, component '{component}'"
        )

    def _validate_streaming(
        self,
        sae: SAE,
        model: Any,
        layer: int,
        component: str,
        activation_key: str,
        val_dataset: List[str]
    ) -> dict:
        """Run validation in streaming mode.

        Args:
            sae: SAE model
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            activation_key: Key for activations
            val_dataset: Validation texts

        Returns:
            Validation metrics
        """
        all_metrics = []

        # Process validation set in chunks
        for i in range(0, len(val_dataset), 4):
            text_batch = val_dataset[i:i + 4]

            with model.trace(text_batch) as trace:
                pass

            acts = trace.activations[activation_key]
            flat = acts.reshape(-1, acts.shape[-1])

            # Add sequence dimension if needed
            if len(flat.shape) == 2:
                flat = flat[:, None, :]

            # Compute loss
            _, metrics = sae.compute_loss(flat)
            all_metrics.append(metrics)

        # Average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        return avg_metrics

    def _collect_activations(
        self,
        model: Any,
        layer: int,
        component: str,
        dataset: List[str],
        verbose: bool = True
    ) -> mx.array:
        """Collect activations from model on dataset.

        Args:
            model: InterpretableModel instance
            layer: Layer number
            component: Component name
            dataset: List of text samples
            verbose: Show progress

        Returns:
            Array of activations, shape (num_samples, d_model)
        """
        all_activations = []

        # Process in batches for memory efficiency
        batch_size = min(32, len(dataset))

        for i in range(0, len(dataset), batch_size):
            batch_prompts = dataset[i:i + batch_size]

            # Trace the batch
            with model.trace(batch_prompts) as trace:
                pass

            # Find the activation key for this layer and component
            # Keys look like: "model.model.layers.10.mlp"
            target_key = None
            for key in trace.activations.keys():
                if f"layers.{layer}" in key and component in key:
                    # Prefer exact component match (e.g., "mlp" not "mlp.gate")
                    if key.endswith(f".{component}") or f".{component}." in key or key.endswith(component):
                        target_key = key
                        break

            if target_key is None:
                raise ValueError(
                    f"Could not find activations for layer {layer}, component '{component}'. "
                    f"Available keys: {list(trace.activations.keys())[:5]}..."
                )

            # Get activations
            acts = trace.activations[target_key]  # Shape: (batch, seq_len, d_model)

            # Flatten sequence dimension - each token position is a training sample
            # This gives us more diverse training data
            batch_flat = acts.reshape(-1, acts.shape[-1])  # Shape: (batch*seq, d_model)
            all_activations.append(batch_flat)

        # Concatenate all batches
        all_acts = mx.concatenate(all_activations, axis=0)

        return all_acts

    def _training_loop(
        self,
        sae: SAE,
        train_activations: mx.array,
        val_activations: Optional[mx.array],
        verbose: bool
    ) -> SAE:
        """Main training loop.

        Args:
            sae: SAE to train
            train_activations: Training data
            val_activations: Optional validation data
            verbose: Show progress

        Returns:
            Trained SAE
        """
        # Initialize optimizer
        # Note: MLX optimizers expect a scalar learning rate, not a schedule function
        # We'll manually update the learning rate during training
        optimizer = optim.AdamW(
            learning_rate=self.config.learning_rate,
            betas=[0.9, 0.999],
            eps=1e-8,
        )

        # Training state
        step = 0
        best_val_loss = float('inf')
        num_samples = len(train_activations)
        steps_per_epoch = num_samples // self.config.batch_size
        total_steps = self.config.num_epochs * steps_per_epoch

        # Create LR schedule
        lr_schedule = self._get_lr_schedule(total_steps)

        # Create progress bar
        if verbose:
            pbar = tqdm(total=total_steps, desc="Training")

        # Training loop
        for epoch in range(self.config.num_epochs):
            # Shuffle training data each epoch
            indices = mx.random.permutation(num_samples)
            train_shuffled = train_activations[indices]

            # Batch loop
            for batch_idx in range(0, num_samples, self.config.batch_size):
                # Get batch
                batch = train_shuffled[batch_idx:batch_idx + self.config.batch_size]

                # Update learning rate with warmup and decay schedule
                current_lr = lr_schedule(step)
                optimizer.learning_rate = current_lr

                # Forward and backward (with sparsity warmup)
                loss, metrics = self._train_step(sae, batch, optimizer, step, total_steps)

                step += 1

                # Log to W&B
                if self.wandb_run:
                    train_metrics = {**metrics, "learning_rate": current_lr}
                    self._log_to_wandb(train_metrics, step, prefix="train")

                # Update progress bar
                if verbose:
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "l0": f"{metrics['l0']:.1f}",
                        "dead": f"{metrics['dead_fraction']:.2%}"
                    })

                # Checkpoint and cleanup
                if self.config.checkpoint_every and step % self.config.checkpoint_every == 0:
                    if verbose:
                        print(f"\n📸 Checkpoint at step {step}")
                    # Force garbage collection and memory cleanup
                    import gc
                    gc.collect()
                    mx.clear_cache()

            # Validation
            if val_activations is not None:
                val_metrics = self._validate(sae, val_activations)

                # Log validation metrics to W&B
                if self.wandb_run:
                    self._log_to_wandb(val_metrics, step, prefix="val")

                if verbose:
                    print(f"\nEpoch {epoch+1}/{self.config.num_epochs} - Val loss: {val_metrics['loss']:.4f}")

                # Track best model
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    # Could save best checkpoint here

        if verbose:
            pbar.close()

        # Finish W&B run
        self._finish_wandb()

        return sae

    def _train_step(
        self,
        sae: SAE,
        batch: mx.array,
        optimizer: optim.Optimizer,
        step: int = 0,
        total_steps: int = 1
    ) -> tuple[mx.array, dict]:
        """Single training step.

        Args:
            sae: SAE model
            batch: Batch of activations (2D: batch_size, d_model)
            optimizer: Optimizer
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Tuple of (loss, metrics)
        """
        # Add sequence dimension if needed (SAE expects 3D input)
        if len(batch.shape) == 2:
            batch = batch[:, None, :]  # (batch, 1, d_model)

        # Get trainable parameters only (excludes frozen params)
        trainable_params = sae.trainable_parameters()

        # Get sparsity coefficient (with warmup if configured)
        sparsity_coef = self._get_sparsity_coefficient(step, total_steps)
        lambda_sparse = self.config.lambda_sparse * sparsity_coef

        # Define loss function that takes trainable parameters
        def loss_fn(params):
            # Update model with new parameters
            sae.update(params)
            # Compute loss with sparsity warmup
            loss_val, _ = sae.compute_loss(batch, lambda_sparse=lambda_sparse)
            return loss_val

        # Compute loss and gradients
        try:
            loss_and_grad_fn = mx.value_and_grad(loss_fn)
            loss, grads = loss_and_grad_fn(trainable_params)
        except Exception as e:
            print(f"\n❌ Gradient computation failed!")
            print(f"   Error: {e}")
            print(f"   Batch shape: {batch.shape}")
            print(f"   Trainable params: {list(trainable_params.keys())}")
            raise

        # Apply ghost gradients if enabled
        if self.config.use_ghost_grads:
            # Forward pass to get features for tracking
            x_recon, features = sae(batch)

            # Update feature activation history
            self._update_feature_history(features, step)

            # Get dead features
            dead_features = self._get_dead_features(sae.d_hidden, step)

            # Apply ghost grads if we have dead features
            if dead_features and step >= self.config.dead_feature_window:
                ghost_loss, ghost_grads = self._apply_ghost_grads(
                    sae, batch, dead_features, trainable_params
                )

                # Combine gradients (weighted sum)
                ghost_grad_scale = 0.1  # Scale down ghost grads
                grads = tree_map(
                    lambda g, gg: g + ghost_grad_scale * gg,
                    grads,
                    ghost_grads
                )

        # Clip gradients if configured
        if self.config.gradient_clip:
            grads = self._clip_gradients(grads, self.config.gradient_clip)

        # Update parameters
        optimizer.update(sae, grads)

        # Evaluate to execute the graph and clear lazy computations
        mx.eval(sae.parameters(), optimizer.state)

        # Get metrics (recompute after update)
        _, metrics = sae.compute_loss(batch)

        # Force evaluation of metrics to clear computation graph
        mx.eval(metrics)

        return loss, metrics

    def _validate(self, sae: SAE, val_activations: mx.array) -> dict:
        """Run validation.

        Args:
            sae: SAE model
            val_activations: Validation data (2D: batch_size, d_model)

        Returns:
            Dictionary of validation metrics
        """
        # Add sequence dimension if needed
        if len(val_activations.shape) == 2:
            val_activations = val_activations[:, None, :]  # (batch, 1, d_model)

        # Compute loss on validation set
        _, metrics = sae.compute_loss(val_activations)
        return metrics

    def _get_lr_schedule(self, total_steps: int) -> Callable:
        """Create learning rate schedule with warmup and decay.

        Args:
            total_steps: Total number of training steps

        Returns:
            Learning rate schedule function
        """
        import math

        base_lr = self.config.learning_rate
        warmup_steps = self.config.warmup_steps
        decay_steps = self.config.lr_decay_steps if self.config.lr_decay_steps else total_steps
        scheduler_type = self.config.lr_scheduler

        def schedule(step):
            if step < warmup_steps:
                # Linear warmup
                return base_lr * (step / warmup_steps)
            else:
                if scheduler_type == "cosine":
                    # Cosine decay after warmup
                    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
                    progress = min(progress, 1.0)  # Clamp to [0, 1]
                    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
                elif scheduler_type == "linear":
                    # Linear decay after warmup
                    progress = (step - warmup_steps) / (decay_steps - warmup_steps)
                    progress = min(progress, 1.0)
                    return base_lr * (1 - progress)
                else:  # "constant"
                    # Constant after warmup
                    return base_lr

        return schedule

    def _get_sparsity_coefficient(self, step: int, total_steps: int) -> float:
        """Get sparsity coefficient with optional warmup.

        Args:
            step: Current training step
            total_steps: Total number of training steps

        Returns:
            Sparsity coefficient (0.0 to 1.0)
        """
        if self.config.sparsity_warm_up_steps is None:
            # No warmup, always use full sparsity penalty
            return 1.0

        warmup_steps = self.config.sparsity_warm_up_steps
        if step < warmup_steps:
            # Linear warmup from 0 to 1
            return float(step) / warmup_steps
        else:
            # Full sparsity after warmup
            return 1.0

    def _update_feature_history(self, features: mx.array, step: int):
        """Track which features activated in this batch.

        Args:
            features: Sparse features array (batch, seq, d_hidden)
            step: Current training step
        """
        # Find which features are active
        # Sum across batch and sequence dimensions to get per-feature activity
        feature_activity = mx.sum(mx.abs(features), axis=(0, 1))  # Shape: (d_hidden,)

        # Convert to list and find active features (MLX doesn't have argwhere/nonzero)
        activity_list = feature_activity.tolist()
        active_features = set(i for i, val in enumerate(activity_list) if val > 0)

        # Update history for active features
        for feat_id in active_features:
            if feat_id not in self.feature_act_history:
                self.feature_act_history[feat_id] = []
            self.feature_act_history[feat_id].append(step)

            # Keep only recent history (within sampling window)
            self.feature_act_history[feat_id] = [
                s for s in self.feature_act_history[feat_id]
                if step - s < self.config.feature_sampling_window
            ]

    def _get_dead_features(self, d_hidden: int, step: int) -> set:
        """Get list of features that are considered dead.

        Args:
            d_hidden: Number of hidden features
            step: Current training step

        Returns:
            Set of dead feature indices
        """
        if not self.config.use_ghost_grads:
            return set()

        if step < self.config.dead_feature_window:
            # Not enough history yet
            return set()

        dead_features = set()
        for feat_id in range(d_hidden):
            if feat_id not in self.feature_act_history:
                # Never activated
                dead_features.add(feat_id)
            else:
                # Check if activated recently
                last_steps = self.feature_act_history[feat_id]
                if not last_steps or (step - last_steps[-1]) > self.config.dead_feature_window:
                    # Not activated within dead feature window
                    dead_features.add(feat_id)

        return dead_features

    def _apply_ghost_grads(
        self,
        sae: Any,
        batch: mx.array,
        dead_features: set,
        trainable_params: dict
    ) -> tuple[mx.array, dict]:
        """Apply ghost gradients to dead features.

        Ghost gradients are auxiliary gradients that encourage dead features to activate
        on random samples from the batch. This helps revive dead neurons.

        Args:
            sae: SAE model
            batch: Current batch (batch, 1, d_model)
            dead_features: Set of dead feature indices
            trainable_params: Trainable parameters

        Returns:
            Tuple of (ghost_loss, ghost_grads)
        """
        if not dead_features or len(dead_features) == 0:
            # No dead features, return zero loss and empty grads
            return mx.array(0.0), tree_map(lambda x: mx.zeros_like(x), trainable_params)

        # Sample random examples from batch for ghost grads
        batch_size = batch.shape[0]
        num_samples = min(batch_size, 8)  # Sample a few examples
        sample_indices = mx.random.randint(0, batch_size, (num_samples,))
        ghost_batch = batch[sample_indices]

        # Define ghost loss function
        def ghost_loss_fn(params):
            sae.update(params)

            # Encode
            x = ghost_batch
            if sae.normalize_input:
                x = (x - sae.input_mean) / (sae.input_std + 1e-8)

            # Get hidden pre-activations (before ReLU and TopK)
            h = sae.encoder(x)

            # Compute L2 norm of dead feature activations
            # Encourage dead features to have non-zero pre-activations
            dead_feat_list = list(dead_features)
            if len(dead_feat_list) > 100:  # Subsample if too many dead
                import random
                dead_feat_list = random.sample(dead_feat_list, 100)

            dead_feat_activations = h[:, :, dead_feat_list]
            ghost_loss = -mx.mean(mx.abs(dead_feat_activations))  # Negative to maximize

            return ghost_loss

        # Compute ghost gradients
        ghost_loss_and_grad = mx.value_and_grad(ghost_loss_fn)
        ghost_loss, ghost_grads = ghost_loss_and_grad(trainable_params)

        return ghost_loss, ghost_grads

    def _clip_gradients(self, grads: dict, max_norm: float) -> dict:
        """Clip gradients by global norm.

        Args:
            grads: Dictionary of gradients
            max_norm: Maximum gradient norm

        Returns:
            Clipped gradients
        """
        # Compute global norm by recursively processing nested dicts
        def compute_norm(g):
            if isinstance(g, dict):
                norm = 0.0
                for v in g.values():
                    norm += compute_norm(v)
                return norm
            elif isinstance(g, (list, tuple)):
                # Skip lists/tuples (frozen parameters have no gradients)
                return 0.0
            elif g is None:
                # Skip None values
                return 0.0
            else:
                return float(mx.sum(g * g))

        total_norm_sq = compute_norm(grads)
        total_norm = mx.sqrt(mx.array(total_norm_sq))

        # Clip if necessary
        if float(total_norm) > max_norm:
            scale = max_norm / (float(total_norm) + 1e-6)

            # Recursively scale gradients
            def scale_grads(g, s):
                if isinstance(g, dict):
                    return {k: scale_grads(v, s) for k, v in g.items()}
                elif isinstance(g, (list, tuple)):
                    # Return as-is (frozen parameters, no scaling needed)
                    return g
                elif g is None:
                    # Return None as-is
                    return g
                else:
                    return g * s

            grads = scale_grads(grads, scale)

        return grads

    def _init_wandb(self) -> None:
        """Initialize Weights & Biases logging."""
        if not WANDB_AVAILABLE:
            return

        # Prepare config dict
        wandb_config = {
            "expansion_factor": self.config.expansion_factor,
            "k": self.config.k,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "warmup_steps": self.config.warmup_steps,
            "normalize_input": self.config.normalize_input,
            "tied_weights": self.config.tied_weights,
            "gradient_clip": self.config.gradient_clip,
            "validation_split": self.config.validation_split,
        }

        # Initialize W&B run
        self.wandb_run = wandb.init(
            project=self.config.wandb_project or "mlxterp-sae",
            entity=self.config.wandb_entity,
            name=self.config.wandb_name,
            tags=self.config.wandb_tags,
            config=wandb_config,
        )

    def _log_to_wandb(self, metrics: dict, step: int, prefix: str = "train") -> None:
        """Log metrics to Weights & Biases.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            prefix: Metric prefix (e.g., "train", "val")
        """
        if not self.wandb_run:
            return

        # Prepare metrics with prefix
        wandb_metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        wandb_metrics["step"] = step

        # Log to W&B
        wandb.log(wandb_metrics)

    def _finish_wandb(self) -> None:
        """Finish Weights & Biases run."""
        if self.wandb_run:
            wandb.finish()
            self.wandb_run = None
