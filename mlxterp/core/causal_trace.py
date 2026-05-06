"""
Multi-input "causal" trace: clean/corrupted pair tracing.

The standard activation-patching pattern requires capturing activations
on a clean run, then injecting them into a corrupted run via the
`replace_with` intervention. This works but is verbose and easy to get
wrong (missing context manager, wrong key prefix, forgetting to save
output, etc.). ``CausalTrace`` is the ergonomic wrapper:

    with model.causal_trace(clean_input, corrupted_input) as ct:
        ct.patch("layers.5.mlp")
        ct.patch("layers.10.self_attn")
        patched = ct.output             # corrupted run, with clean
                                        # activations swapped in at the
                                        # patched modules
        clean   = ct.clean_output       # captured during __enter__

Resolves CAUSAL_INTERP_ROADMAP.md Tier 1 #2 (Multi-Input Trace).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import mlx.core as mx


class CausalTrace:
    """Context manager for clean/corrupted comparative tracing.

    On ``__enter__`` runs a single forward pass over ``clean_input``,
    capturing every activation. ``patch(module_name)`` schedules a
    swap-in for the corrupted run. On the first access of ``output``
    we run the corrupted forward pass with all scheduled patches as
    ``replace_with`` interventions on top of any user-supplied ones.

    Patches and the corrupted run are deferred so users can call
    ``patch()`` multiple times in any order before reading the result,
    and the corrupted forward only runs once even if ``output`` is
    read repeatedly.

    Args:
        model: The InterpretableModel being traced.
        clean_input: Text or token sequence for the clean run.
        corrupted_input: Text or token sequence for the corrupted run.
        interventions: Optional dict of additional, user-supplied
            interventions to apply on the corrupted run alongside the
            scheduled patches. Same format as
            ``model.trace(..., interventions=...)``.
    """

    def __init__(
        self,
        model,
        clean_input,
        corrupted_input,
        interventions: Optional[Dict[str, Any]] = None,
    ):
        self._model = model
        self._clean_input = clean_input
        self._corrupted_input = corrupted_input
        self._user_interventions: Dict[str, Any] = dict(interventions or {})
        self._patches: List[str] = []
        self._clean_activations: Optional[Dict[str, mx.array]] = None
        self._clean_output: Optional[mx.array] = None
        self._corrupted_output: Optional[mx.array] = None
        self._corrupted_run: bool = False

    def __enter__(self) -> "CausalTrace":
        with self._model.trace(self._clean_input) as t:
            pass
        self._clean_activations = dict(t.activations)
        self._clean_output = self._clean_activations.get("__model_output__")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # The corrupted forward (if requested) ran inside the with block on
        # first .output access; nothing further to clean up. If patches were
        # scheduled but .output was never accessed, we silently do not run
        # the corrupted forward — same behaviour as a torch trainer where
        # you only pay for what you ask for.
        return False

    def patch(self, module_name: str) -> "CausalTrace":
        """Schedule a clean→corrupted activation swap for one module.

        Multiple ``patch`` calls compose: each accumulates one more
        replace_with intervention. Call before reading ``output``.
        Returns ``self`` to support chaining.
        """
        if self._corrupted_run:
            raise RuntimeError(
                "Cannot patch after .output has been accessed. The corrupted "
                "forward already ran with the patches scheduled at that "
                "point. Schedule all patches before reading .output."
            )
        self._patches.append(module_name)
        return self

    @property
    def clean_output(self) -> Optional[mx.array]:
        """The model's output on the clean input, captured during __enter__."""
        return self._clean_output

    @property
    def output(self) -> mx.array:
        """The model's output on the corrupted input with all scheduled
        patches applied. Lazily triggers the corrupted forward on first
        access; cached for subsequent reads.
        """
        if not self._corrupted_run:
            self._run_corrupted()
        return self._corrupted_output

    @property
    def patches(self) -> List[str]:
        """The list of scheduled patches, in the order they were added."""
        return list(self._patches)

    def _run_corrupted(self) -> None:
        from . import intervention as iv

        interventions = dict(self._user_interventions)
        for name in self._patches:
            clean_act = self._lookup_clean_activation(name)
            interventions[name] = iv.replace_with(clean_act)
        with self._model.trace(
            self._corrupted_input, interventions=interventions
        ) as t:
            pass
        self._corrupted_output = t.activations.get("__model_output__")
        self._corrupted_run = True

    def _lookup_clean_activation(self, name: str) -> mx.array:
        """Find the clean activation matching ``name``, accommodating the
        path prefixes that ``trace.activations`` keys use ("model.",
        "model.model.") relative to what the user types.
        """
        if self._clean_activations is None:
            raise RuntimeError("CausalTrace was not entered before use.")
        for prefix in ("", "model.", "model.model."):
            full = f"{prefix}{name}"
            if full in self._clean_activations:
                return self._clean_activations[full]
        sample = list(self._clean_activations.keys())
        raise KeyError(
            f"No clean activation captured for module {name!r}. "
            f"Tried prefixes '', 'model.', 'model.model.'. "
            f"Sample keys actually captured: {sample[:5]}..."
        )
