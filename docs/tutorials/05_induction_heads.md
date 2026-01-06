# Tutorial 5: Induction Heads

**Paper**: [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/) by Olsson et al. (Anthropic, 2022)

**Difficulty**: Intermediate-Advanced | **Time**: 4-5 hours

**Status**: Coming Soon

---

## Overview

This tutorial will demonstrate how to identify and analyze induction heads - attention heads that implement a pattern completion algorithm: given `[A][B]...[A]`, predict `[B]`.

## Planned Content

1. Understanding the induction head algorithm
2. The two-step composition (previous token head + induction head)
3. Measuring induction behavior in language models
4. Finding induction heads through attention pattern analysis
5. Ablation studies to confirm causal role
6. Connection to in-context learning

## Implementation Status

This tutorial is tracked in [GitHub Issue #5](https://github.com/coairesearch/mlxterp/issues/5). Contributions welcome!

---

## References

1. Olsson, C., et al. (2022). [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/). Anthropic.

2. Elhage et al. (2021). [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html). Anthropic.
