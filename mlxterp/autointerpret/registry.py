"""
MetricRegistry for AutoInterp experiments.

Allows the human to define metrics in setup.py that the agent can use
in experiments without needing to know implementation details.
"""

from typing import Any, Callable, Dict, List, Optional


class MetricRegistry:
    """
    Registry of metric functions for AutoInterp experiments.

    The human defines metrics in setup.py. The agent reads available
    metrics and uses them in experiments without modifying setup.py.

    Example in setup.py:
        metrics = MetricRegistry()
        metrics.register("logit_diff", lambda clean, patched, target, foil:
            patched[0, -1, target] - patched[0, -1, foil])
        metrics.register("prob_correct", lambda clean, patched, target, **kw:
            float(mx.softmax(patched[0, -1])[target]))
    """

    def __init__(self):
        self._metrics: Dict[str, Callable] = {}
        self._descriptions: Dict[str, str] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        description: str = "",
    ):
        """Register a metric function.

        Args:
            name: Metric name (used by agent to reference it)
            fn: Metric function
            description: Human-readable description of what this measures
        """
        self._metrics[name] = fn
        self._descriptions[name] = description

    def get(self, name: str) -> Callable:
        """Get a metric by name."""
        if name not in self._metrics:
            raise KeyError(
                f"Unknown metric: {name}. "
                f"Available: {list(self._metrics.keys())}"
            )
        return self._metrics[name]

    def list(self) -> List[Dict[str, str]]:
        """List all registered metrics with descriptions."""
        return [
            {"name": name, "description": self._descriptions.get(name, "")}
            for name in self._metrics
        ]

    @property
    def names(self) -> List[str]:
        """List of registered metric names."""
        return list(self._metrics.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._metrics

    def __len__(self) -> int:
        return len(self._metrics)
