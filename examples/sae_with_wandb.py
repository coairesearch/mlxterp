"""
SAE Training with Weights & Biases Logging

This example demonstrates how to train SAEs with automatic logging to Weights & Biases
for experiment tracking, visualization, and collaboration.

Requirements:
    pip install wandb datasets

Before running:
    wandb login
"""

from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig
from datasets import load_dataset
from typing import List


def load_training_texts(num_samples: int = 10000) -> List[str]:
    """Load training dataset from HuggingFace."""
    print(f"📦 Loading dataset...")

    # Load WikiText dataset
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", trust_remote_code=True)

    # Filter and prepare texts
    texts = []
    for item in dataset:
        text = item.get('text', '').strip()
        if len(text) >= 50:
            texts.append(text[:512])  # Truncate long texts
        if len(texts) >= num_samples:
            break

    print(f"   ✓ Loaded {len(texts)} text samples")
    return texts


def main():
    print("=" * 70)
    print("SAE Training with Weights & Biases")
    print("=" * 70)

    # Configuration
    MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    LAYER = 10
    COMPONENT = "mlp"
    NUM_SAMPLES = 10000

    # Step 1: Load model
    print(f"\n1. Loading model: {MODEL_NAME}")
    base_model, tokenizer = load(MODEL_NAME)
    model = InterpretableModel(base_model, tokenizer=tokenizer)
    print(f"   ✓ Model loaded")

    # Step 2: Load dataset
    print(f"\n2. Loading training dataset...")
    texts = load_training_texts(num_samples=NUM_SAMPLES)

    # Step 3: Configure SAE with W&B logging
    print(f"\n3. Configuring SAE with W&B logging...")

    config = SAEConfig(
        # SAE architecture
        expansion_factor=16,
        k=100,

        # Training
        learning_rate=3e-4,
        num_epochs=5,
        batch_size=64,
        warmup_steps=500,
        validation_split=0.05,

        # Optimization
        normalize_input=True,
        gradient_clip=1.0,

        # Weights & Biases configuration
        use_wandb=True,                          # Enable W&B logging
        wandb_project="mlxterp-sae-experiments", # W&B project name
        wandb_entity=None,                       # Your W&B username/team (optional)
        wandb_name=f"sae_layer{LAYER}_{COMPONENT}",  # Run name
        wandb_tags=["sae", f"layer{LAYER}", COMPONENT],  # Tags for organization
    )

    print(f"   ✓ W&B logging enabled")
    print(f"   Project: {config.wandb_project}")
    print(f"   Run name: {config.wandb_name}")
    print(f"   Tags: {config.wandb_tags}")

    # Step 4: Train SAE
    print(f"\n4. Training SAE (metrics will be logged to W&B)...")
    print(f"   Visit: https://wandb.ai/{config.wandb_entity or 'your-username'}/{config.wandb_project}")
    print()

    sae = model.train_sae(
        layer=LAYER,
        component=COMPONENT,
        dataset=texts,
        config=config,
        save_path=f"sae_layer{LAYER}_{COMPONENT}_wandb.mlx",
        verbose=True
    )

    print(f"\n" + "=" * 70)
    print(f"✅ Training complete!")
    print(f"=" * 70)
    print(f"\nView results at: https://wandb.ai")
    print(f"\nLogged metrics:")
    print(f"  - train/loss - Training reconstruction loss")
    print(f"  - train/l0 - Average number of active features")
    print(f"  - train/dead_fraction - Fraction of dead neurons")
    print(f"  - train/learning_rate - Current learning rate")
    print(f"  - val/loss - Validation loss (per epoch)")
    print(f"  - val/l0 - Validation feature activation")
    print(f"  - val/dead_fraction - Validation dead neurons")


if __name__ == "__main__":
    # Check if wandb is installed
    try:
        import wandb
    except ImportError:
        print("❌ Weights & Biases is not installed!")
        print("   Install with: pip install wandb")
        print("   Then run: wandb login")
        exit(1)

    main()
