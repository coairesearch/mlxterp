# mlxterp docs/code rescan report

## Scope
Full rescan of `docs/` and verification of the current codebase after the listed fixes. Confirmation tests (synthetic + real-model) executed.

## Verified Fixes (code)
- `Trace.saved_values` now copies in `__exit__`, while activations remain available inside the context. (`mlxterp/core/trace.py`)
- List-of-int inputs are wrapped into a batch dimension. (`mlxterp/core/trace.py`)
- `collect_activations` now filters by layer and normalizes keys to short form. (`mlxterp/core/cache.py`)
- `get_token_predictions` handles batched hidden states. (`mlxterp/analysis.py`)
- `logit_lens` supports `position`. (`mlxterp/analysis.py`)
- `activation_patching` supports multiple path patterns, `component="output"`, and division-by-zero protection. (`mlxterp/analysis.py`)

## Tests Run
- `pytest tests/doubleReview/confirmationtests -q --no-cov`
- Result: 16 collected, 16 passed

## Remaining Doc Mismatches (found in 100% rescan)
- `docs/API.md:84` and `docs/API.md:631` still reference `.attn` and `layers.3.attn`. For mlx-lm Llama, this should be `self_attn`, and `Trace.get_activation(...)` requires the full key (e.g., `model.model.layers.3.self_attn`) because `Trace.get_activation` does not normalize keys.
- `docs/architecture.md:274-286` includes a context-manager pattern snippet that executes the forward pass in `__exit__`. This contradicts the current implementation, which executes in `__enter__`.
- `docs/examples.md:423-427` uses `results[15][0]` from `logit_lens(...)`; this indexes position 0, not the last position. It should use `results[15][-1][0]` or pass `position=-1`.

## Residual Codebase Note (non-doc)
- `logit_lens` still assumes `model.model.norm` and `model.model.embed_tokens`. Custom/non-mlx-lm models without these attributes will fail; if full model-agnostic behavior is intended, this is a remaining limitation.

## Conclusion
All previously reported bugs are fixed and verified by tests. A few doc mismatches remain (noted above) and should be updated to fully align with current behavior.
