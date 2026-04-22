# Example Notebooks

This directory contains Jupyter notebooks demonstrating mlxterp features.

## Tutorial Notebooks

Work through these in order to learn mlxterp's causal interpretability tools:

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [01_causal_patching.ipynb](01_causal_patching.ipynb) | Activation patching (layer/position/head), metrics (logit_diff, l2, kl, cosine), CausalTrace, PatchingResult |
| 2 | [02_circuit_discovery.ipynb](02_circuit_discovery.ipynb) | Residual stream analysis, Direct Logit Attribution, attribution patching, path patching, ACDC |
| 3 | [03_generation.ipynb](03_generation.ipynb) | Text generation (greedy, temperature, top-k, top-p), interventions during generation, steering vectors, callbacks |
| 4 | [04_conversation_analysis.ipynb](04_conversation_analysis.ipynb) | Turn detection, TurnList filtering, ConversationTrace, per-turn activations, cross-turn attention |
| 5 | [05_agentic_interp.ipynb](05_agentic_interp.ipynb) | Research workflows, AutoInterp ratchet loop, automated feature labeling, report generation |

## Other Notebooks

- **Test Library.ipynb** - Interactive testing and demonstration of core mlxterp features

## Usage

Start Jupyter:
```bash
jupyter notebook
```

Or use Jupyter Lab:
```bash
jupyter lab
```

Then open any notebook in this directory.

## Creating New Notebooks

When creating example notebooks:
1. Use clear section headers and markdown explanations
2. Include imports and setup in the first cell
3. Add comments explaining each step
4. Clear output before committing (unless the output is important to show)
5. Keep notebooks focused on one feature or workflow

## Note

Notebooks are great for interactive exploration but scripts (in `examples/*.py`) are better for:
- Reproducible examples
- CI/CD testing
- Quick reference of API usage
