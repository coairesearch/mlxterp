"""
AutoInterp: Karpathy AutoResearch-style ratchet loop for interpretability.

Provides the three-file contract (setup.py, experiment.py, program.md),
scaffold generation, experiment logging, and the ratchet loop.

Designed for two modes:
1. Zero orchestration — open Claude Code in the directory, say "read program.md and start"
2. Programmatic — use AutoInterpret class for automation

See: https://github.com/karpathy/autoresearch
"""

from .runner import AutoInterpret
from .registry import MetricRegistry
from .scaffold import init_autointerpret
from .logging import ExperimentLog, ExperimentEntry

__all__ = [
    "AutoInterpret",
    "MetricRegistry",
    "init_autointerpret",
    "ExperimentLog",
    "ExperimentEntry",
]
