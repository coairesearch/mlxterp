"""
Visualization module for mlxterp.

Provides attention pattern visualization and pattern detection utilities
for mechanistic interpretability analysis.

Example:
    >>> from mlxterp import InterpretableModel
    >>> from mlxterp.visualization import attention_heatmap, attention_from_trace
    >>>
    >>> model = InterpretableModel("mlx-community/Llama-3.2-1B-Instruct-4bit")
    >>>
    >>> with model.trace("The capital of France is Paris") as trace:
    >>>     pass
    >>>
    >>> # Visualize attention patterns
    >>> attention_from_trace(trace, model.to_str_tokens("The capital of France is Paris"))
"""

from .attention import (
    attention_heatmap,
    attention_from_trace,
    get_attention_patterns,
    AttentionVisualizationConfig,
)

from .patterns import (
    AttentionPatternDetector,
    detect_head_types,
    detect_induction_heads,
    induction_score,
    previous_token_score,
    first_token_score,
    copying_score,
)

from .interactive import (
    interactive_attention,
    display_interactive_attention,
    save_interactive_attention,
    interactive_attention_from_trace,
    InteractiveAttentionConfig,
)

__all__ = [
    # Attention visualization
    "attention_heatmap",
    "attention_from_trace",
    "get_attention_patterns",
    "AttentionVisualizationConfig",
    # Interactive visualization
    "interactive_attention",
    "display_interactive_attention",
    "save_interactive_attention",
    "interactive_attention_from_trace",
    "InteractiveAttentionConfig",
    # Pattern detection
    "AttentionPatternDetector",
    "detect_head_types",
    "detect_induction_heads",
    "induction_score",
    "previous_token_score",
    "first_token_score",
    "copying_score",
]
