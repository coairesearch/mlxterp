"""
Realistic SAE Training with HuggingFace Dataset

This example demonstrates training an SAE with a substantial dataset from HuggingFace,
which is necessary for learning meaningful sparse features.

Requirements:
    pip install datasets

Recommended datasets:
    - "wikitext" (wikitext-103-raw-v1): 28k articles, ~100M tokens
    - "openwebtext": 8M documents
    - "the_pile_openwebtext2": Subset of The Pile
    - "allenai/c4": Colossal Clean Crawled Corpus
"""

from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig
from datasets import load_dataset
from typing import List
import mlx.core as mx

def load_training_texts(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-103-raw-v1",
    split: str = "train",
    num_samples: int = 20000,
    min_length: int = 50,
    max_length: int = 512,
) -> List[str]:
    """Load and prepare text samples from HuggingFace dataset.

    Args:
        dataset_name: HuggingFace dataset name
        dataset_config: Dataset configuration/subset
        split: Dataset split (train/validation/test)
        num_samples: Number of samples to use for training
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters (for memory efficiency)

    Returns:
        List of text strings for SAE training
    """
    print(f"📦 Loading {dataset_name} dataset...")

    # Load dataset
    dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)

    print(f"   Total samples in dataset: {len(dataset)}")

    # Filter and prepare texts
    texts = []
    for item in dataset:
        # Get text field (may vary by dataset)
        text = item.get('text', item.get('content', ''))

        # Skip empty or very short texts
        if len(text) < min_length:
            continue

        # Truncate very long texts
        if len(text) > max_length:
            text = text[:max_length]

        texts.append(text.strip())

        # Stop when we have enough samples
        if len(texts) >= num_samples:
            break

    print(f"   ✓ Prepared {len(texts)} text samples")
    print(f"   Average length: {sum(len(t) for t in texts) / len(texts):.0f} chars")

    return texts


def main():
    print("=" * 70)
    print("Realistic SAE Training with HuggingFace Dataset")
    print("=" * 70)

    # Configuration
    MODEL_NAME = "arogister/Qwen3-8B-ShiningValiant3-mlx-4Bit"
    #MODEL_NAME = "mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit"
    #MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
    LAYER = 23
    COMPONENT = "mlp"
    NUM_SAMPLES = 10000  # 20k samples for realistic training

    # Step 1: Load model
    print(f"\n1. Loading model: {MODEL_NAME}")
    base_model, tokenizer = load(MODEL_NAME)
    model = InterpretableModel(base_model, tokenizer=tokenizer)
    print(f"   ✓ Model loaded")

    # Step 2: Load training dataset
    print(f"\n2. Loading training dataset...")
    try:
        # Option A: WikiText (high-quality Wikipedia text)
        texts = load_training_texts(
            dataset_name="wikitext",
            dataset_config="wikitext-103-raw-v1",
            split="train",
            num_samples=NUM_SAMPLES,
            min_length=50,
            max_length=512,
        )
    except Exception as e:
        print(f"   ⚠️  Could not load wikitext: {e}")
        print(f"   Falling back to alternative dataset...")

        # Option B: Fallback to smaller dataset
        try:
            texts = load_training_texts(
                dataset_name="roneneldan/TinyStories",
                dataset_config=None,
                split="train",
                num_samples=NUM_SAMPLES,
                min_length=50,
                max_length=512,
            )
        except Exception as e2:
            print(f"   ❌ Could not load fallback dataset: {e2}")
            print(f"   Please install datasets: pip install datasets")
            return

    # Step 3: Configure SAE training
    print(f"\n3. Configuring SAE training...")

    config = SAEConfig(
        sae_type="batchtopk",
        expansion_factor=16,    # 4x expansion → 16,384 features (4096 × 4)
        k=128,                  # Top-64 sparsity (was 32 - causing 99.91% dead features)
        learning_rate=3e-4,     # Standard learning rate
        lr_scheduler="cosine",
        sparsity_warm_up_steps=None,
        use_ghost_grads=True,
        num_epochs=3,           # Fewer epochs with large dataset
        batch_size=64,          # Training batch size
        text_batch_size=32,     # Number of texts to process at once (streaming optimization)
        warmup_steps=2000,       # Warmup for stability
        validation_split=0.05,  # 5% validation
        normalize_input=True,
        gradient_clip=1,
        use_wandb=True,
        checkpoint_every=5000,  # More frequent checkpoints (was 25000)
        wandb_project="mlxterp-sae-experiments",
        wandb_name="V13_Qwen3-8B-ShiningValiant3-mlx-4Bit_sae_layer23_mlp_16x_k128",
        wandb_tags=["sae", "layer23", "mlp"],
    )

    print(f"   Config:")
    print(f"   - Expansion: {config.expansion_factor}x")
    print(f"   - Sparsity: k={config.k}")
    print(f"   - Learning rate: {config.learning_rate}")
    print(f"   - Epochs: {config.num_epochs}")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Training samples: {len(texts)}")

    # Estimate training details
    print(f"\n4. Training estimates...")

    # Rough estimate of activation samples
    avg_tokens_per_sample = 50  # Conservative estimate
    total_activation_samples = len(texts) * avg_tokens_per_sample
    steps_per_epoch = total_activation_samples // config.batch_size
    total_steps = steps_per_epoch * config.num_epochs

    print(f"   Estimated activation samples: ~{total_activation_samples:,}")
    print(f"   Steps per epoch: ~{steps_per_epoch:,}")
    print(f"   Total training steps: ~{total_steps:,}")
    print(f"   ⏱️  Estimated time: ~{total_steps / 40:.0f}-{total_steps / 20:.0f} minutes")

    # Step 5: Train SAE
    print(f"\n5. Training SAE on Layer {LAYER} {COMPONENT}...")
    print(f"   (This will take several minutes with {NUM_SAMPLES:,} samples)")
    print()

    sae = model.train_sae(
        layer=LAYER,
        component=COMPONENT,
        dataset=texts,
        config=config,
        save_path=f"sae_layer{LAYER}_{COMPONENT}_{NUM_SAMPLES}samples.mlx",
        verbose=True
    )

    # Step 6: Evaluate trained SAE
    print(f"\n6. Evaluating trained SAE...")

    # Test on sample text
    test_text = "The study of artificial intelligence explores how machines can learn and reason."

    with model.trace(test_text) as trace:
        pass

    # Get activation
    target_keys = [k for k in trace.activations.keys()
                   if f"layers.{LAYER}" in k and COMPONENT in k]

    if target_keys:
        activation = trace.activations[target_keys[0]]

        # Encode to features
        features = sae.encode(activation)

        # Get active features
        active_mask = features != 0
        num_active = int(mx.sum(active_mask))
        active_feature_indices = mx.where(active_mask[0, -1])  # Last token

        print(f"   Test text: '{test_text}'")
        print(f"   Activation shape: {activation.shape}")
        print(f"   Feature shape: {features.shape}")
        print(f"   Active features: {num_active}/{sae.d_hidden}")
        print(f"   Active feature IDs (last token): {active_feature_indices[0][:10].tolist()}...")

        # Reconstruction quality
        reconstructed = sae.decode(features)
        mse = float(mx.mean((activation - reconstructed) ** 2))
        print(f"   Reconstruction MSE: {mse:.6f}")

    print(f"\n" + "=" * 70)
    print(f"✅ Realistic SAE training complete!")
    print(f"=" * 70)
    print(f"\nSaved to: sae_layer{LAYER}_{COMPONENT}_{NUM_SAMPLES}samples.mlx")
    print(f"\nNext steps:")
    print(f"  1. Analyze features: sae.get_activation_stats()")
    print(f"  2. Find top features for text (Phase 2)")
    print(f"  3. Visualize feature activations (Phase 2)")
    print(f"  4. Use features for model steering (Phase 2)")


if __name__ == "__main__":
    main()
