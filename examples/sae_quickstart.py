"""
SAE Quickstart: Train and use Sparse Autoencoders

This example demonstrates the basic SAE workflow:
1. Load a model
2. Train an SAE on a layer
3. Use the SAE to encode activations
4. Save and load SAEs
"""
from mlx_lm import load
from mlxterp import InterpretableModel, SAEConfig

# Step 1: Load model
print("Loading model...")
base_model, tokenizer = load("mlx-community/Qwen3-30B-A3B-Thinking-2507-4bit")
model = InterpretableModel(base_model, tokenizer=tokenizer)

# Step 2: Prepare dataset
# Use diverse text samples for better feature learning
dataset = [
    "The capital of France is Paris",
    "Machine learning is revolutionizing artificial intelligence",
    "Python is a popular programming language for data science",
    "The sun rises in the east and sets in the west",
    "Water freezes at zero degrees Celsius",
    "The Earth orbits around the Sun",
    "Photosynthesis occurs in plant leaves using sunlight",
    "Shakespeare wrote many famous plays including Hamlet",
    "DNA contains the genetic information of living organisms",
    "Gravity is the force that attracts objects toward each other",
    # Add more diverse samples for better training...
]

# Step 3: Train SAE (one line!)
print("\n🎯 Training SAE on Layer 10...")

# # Option A: Simple (use defaults)
# sae = model.train_sae(
#     layer=10,
#     dataset=dataset,
#     save_path="my_sae.mlx"
# )

# Option B: Custom configuration
config = SAEConfig(
    expansion_factor=16,  # 16x more features than input dimension
    k=100,                # Keep top 100 features active
    learning_rate=1e-4,
    num_epochs=10,
)

sae_custom = model.train_sae(
    layer=10,
    dataset=dataset,
    config=config,
    save_path="my_sae_custom.mlx"
)

# Step 4: Use the trained SAE
print("\n📊 Using trained SAE...")

# Get activations from model
with model.trace("The capital of Germany is") as trace:
    pass

# Find the MLP activation for layer 10
layer_keys = [k for k in trace.activations.keys() if "layers.10" in k and "mlp" in k]
activation = trace.activations[layer_keys[0]]

# Encode to sparse features
features = sae_custom.encode(activation)
print(f"Feature shape: {features.shape}")
print(f"Active features: {sum(features[0, -1] != 0)}")  # Should be ≤ k

# Decode back
reconstructed = sae_custom.decode(features)
print(f"Reconstruction shape: {reconstructed.shape}")

# Step 5: Load saved SAE
print("\n💾 Loading saved SAE...")
loaded_sae = model.load_sae("my_sae_custom.mlx")

# Verify compatibility
if loaded_sae.is_compatible(model, layer=10, component="mlp"):
    print("✓ SAE is compatible!")

# Step 6: Analyze features (future functionality)
print("\n🔬 Future: Feature analysis")
print("Coming in Phase 2:")
print("  - sae.analyze_text() - Find top features for text")
print("  - sae.visualize_feature() - See what features represent")
print("  - sae.steer() - Control model behavior with features")

print("\n✅ SAE Quickstart complete!")
