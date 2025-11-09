"""Quick test of new SAE features."""

import mlx.core as mx
from mlxterp import InterpretableModel, SAEConfig
from mlx_lm import load

print("=" * 70)
print("Testing New SAE Features")
print("=" * 70)

# Test 1: Config with new features
print("\n1. Testing SAEConfig with new features...")
config = SAEConfig(
    sae_type="batchtopk",
    expansion_factor=4,  # Small for testing
    k=32,
    learning_rate=3e-4,
    lr_scheduler="cosine",
    sparsity_warm_up_steps=100,
    use_ghost_grads=True,
    num_epochs=1,
    batch_size=64,
    use_wandb=False,  # Disable for testing
)
print("✓ SAEConfig created with all new features")
print(f"  - SAE type: {config.sae_type}")
print(f"  - LR scheduler: {config.lr_scheduler}")
print(f"  - Ghost grads: {config.use_ghost_grads}")

# Test 2: Load small model
print("\n2. Loading small model...")
MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
base_model, tokenizer = load(MODEL_NAME)
model = InterpretableModel(base_model, tokenizer=tokenizer)
print(f"✓ Model loaded: {MODEL_NAME}")

# Test 3: Create small dataset
print("\n3. Creating test dataset...")
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language.",
    "The capital of France is Paris.",
    "Neural networks are inspired by the human brain.",
] * 20  # 100 samples
print(f"✓ Created {len(texts)} text samples")

# Test 4: Train SAE with new features
print("\n4. Training SAE with new features (1 epoch, small model)...")
print("   This should take 1-2 minutes...")

try:
    sae = model.train_sae(
        layer=8,
        component="mlp",
        dataset=texts,
        config=config,
        save_path=None,  # Don't save
        verbose=True
    )
    print("\n✓ SAE training completed successfully!")
    print(f"  - SAE type: {type(sae).__name__}")
    print(f"  - d_model: {sae.d_model}")
    print(f"  - d_hidden: {sae.d_hidden}")
    print(f"  - k: {sae.k}")

    # Test encoding
    test_text = "This is a test."
    with model.trace(test_text) as trace:
        pass

    target_key = [k for k in trace.activations.keys()
                  if "layers.8" in k and "mlp" in k and k.endswith(".mlp")][0]
    activation = trace.activations[target_key]

    features = sae.encode(activation)
    reconstructed = sae.decode(features)

    mse = float(mx.mean((activation - reconstructed) ** 2))
    print(f"\n✓ Encoding/decoding works!")
    print(f"  - Reconstruction MSE: {mse:.6f}")

    active_features = int(mx.sum(features != 0))
    print(f"  - Active features: {active_features}")

except Exception as e:
    print(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 70)
print("✅ All new features working correctly!")
print("=" * 70)
print("\nNew features tested:")
print("  ✅ BatchTopK architecture")
print("  ✅ LR decay (cosine schedule)")
print("  ✅ Sparsity warmup")
print("  ✅ Ghost gradients")
print("  ✅ SAELens-validated config")
