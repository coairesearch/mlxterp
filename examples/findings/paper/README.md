# Sycophancy in pythia-410m — Mini-Paper

Academic-style paper documenting the causal interpretability investigation of sycophancy in pythia-410m, conducted end-to-end using mlxterp.

## Files

| File | Description |
|------|-------------|
| `sycophancy_pythia.tex` | LaTeX source (~400 lines, 6 pages) |
| `sycophancy_pythia.pdf` | Compiled PDF |
| `*.png` | Figures (copied from parent directory) |

## Compiling

```bash
cd examples/findings/paper
pdflatex sycophancy_pythia.tex
pdflatex sycophancy_pythia.tex  # Second pass for references
```

Requires: `pdflatex` with standard packages (`graphicx`, `amsmath`, `booktabs`, `hyperref`, `tikz`). No external bibliography manager needed (bibliography is embedded).

## Structure

1. **Abstract** — one paragraph summary
2. **Introduction** — research questions Q1-Q3
3. **Methodology** — probes, causal measures, implementation
4. **Results** — behavioral check + patching + DLA + aggregation
5. **Discussion** — distributed circuit, MLP L0 anomaly, early attention role, null result on Python probe
6. **Limitations** — explicit
7. **Conclusion** + reproducibility note + future work
8. **References** — 8 mechanistic interpretability citations

## Reproduction

The results in this paper come directly from running:
```bash
python examples/sycophancy_investigation.py
```
in the mlxterp repository. Runtime: ~3 minutes on Apple Silicon.
