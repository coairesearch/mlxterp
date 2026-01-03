# Tutorial 6: Sparse Autoencoders

**Paper**: [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/) by Anthropic (2023)

**Difficulty**: Advanced | **Time**: 5-6 hours

**Status**: Coming Soon

---

## Overview

This tutorial will demonstrate how to train and analyze Sparse Autoencoders (SAEs) to decompose polysemantic neurons into interpretable features.

## Planned Content

1. The superposition hypothesis and polysemanticity
2. Sparse autoencoder architecture and training objective
3. Training an SAE on model activations using mlxterp's built-in SAE support
4. Identifying and interpreting learned features
5. Feature steering: using SAE features for targeted interventions
6. Limitations and future directions

## Implementation Status

This tutorial is tracked in [GitHub Issue #6](https://github.com/coairesearch/mlxterp/issues/6). Contributions welcome!

mlxterp already has SAE training and analysis built-in:
- `model.train_sae()` for training
- `get_top_features_for_text()` for feature analysis

---

## References

1. Anthropic (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/).

2. Cunningham et al. (2023). [Sparse Autoencoders Find Highly Interpretable Features in Language Models](https://arxiv.org/abs/2309.08600).
