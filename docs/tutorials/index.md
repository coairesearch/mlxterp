# Paper Reproduction Tutorials

This section contains tutorials that reproduce key mechanistic interpretability research papers using mlxterp. Each tutorial demonstrates the library's capabilities through concrete, educational examples that match the original papers' findings.

## Why Paper Reproductions?

1. **Validation**: Verify that mlxterp produces the same results as established research
2. **Education**: Learn mechanistic interpretability concepts through hands-on implementation
3. **Reference**: Use these as starting points for your own research

## Tutorials

| # | Paper | Difficulty | Time | Key Concept |
|---|-------|------------|------|-------------|
| 1 | [Logit Lens](01_logit_lens.md) | Beginner | 1-2h | Intermediate layer predictions |
| 2 | [Tuned Lens](02_tuned_lens.md) | Beginner-Int | 3-4h | Learned prediction probes |
| 3 | [Causal Tracing (ROME)](03_causal_tracing.md) | Intermediate | 3-4h | Knowledge localization |
| 4 | [Steering Vectors (CAA)](04_steering_vectors.md) | Intermediate | 2-3h | Behavior control |
| 5 | [Induction Heads](05_induction_heads.md) | Int-Advanced | 4-5h | Pattern completion circuits |
| 6 | [Sparse Autoencoders](06_sparse_autoencoders.md) | Advanced | 5-6h | Feature decomposition |

## Prerequisites

Before starting these tutorials, ensure you have:

- mlxterp installed with all extras: `pip install mlxterp[dev,docs,viz]`
- A Mac with Apple Silicon (M1/M2/M3/M4) for optimal performance
- Basic familiarity with transformers and neural networks
- Python knowledge (intermediate level)

## Getting Started

We recommend following the tutorials in order, as concepts build upon each other:

1. **Start with Logit Lens** - Introduces the core concept of examining intermediate predictions
2. **Move to Tuned Lens** - Shows how to improve upon the basic logit lens
3. **Then Causal Tracing** - Learn to localize where information is stored
4. **Continue with Steering** - Apply what you've learned to control model behavior
5. **Explore Induction Heads** - Understand a fundamental transformer circuit
6. **Finish with SAEs** - The frontier of interpretability research

## Running the Examples

Each tutorial has an accompanying Python script in `examples/tutorials/`:

```bash
# Run the Logit Lens tutorial
python examples/tutorials/01_logit_lens/logit_lens_tutorial.py

# Run the Causal Tracing tutorial
python examples/tutorials/02_causal_tracing/causal_tracing_tutorial.py
```

## References

These tutorials are based on the following papers:

1. **Logit Lens**: nostalgebraist (2020). [interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)

2. **Tuned Lens**: Belrose et al. (2023). [Eliciting Latent Predictions from Transformers with the Tuned Lens](https://arxiv.org/abs/2303.08112). NeurIPS 2023.

3. **ROME / Causal Tracing**: Meng et al. (2022). [Locating and Editing Factual Associations in GPT](https://arxiv.org/abs/2202.05262). NeurIPS 2022.

4. **Contrastive Activation Addition**: Rimsky et al. (2024). [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681). ACL 2024.

5. **Induction Heads**: Olsson et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.

6. **Sparse Autoencoders**: Anthropic (2023). [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/).
