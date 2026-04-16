"""
AutoInterpret runner: programmatic ratchet loop for automated experiments.

For zero-orchestration mode, use the scaffold directly with Claude Code.
For programmatic automation, use the AutoInterpret class.
"""

import time
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .logging import ExperimentLog, ExperimentEntry
from .registry import MetricRegistry
from ..results import AnalysisResult


class AutoInterpret:
    """
    Programmatic AutoInterp runner.

    Runs a ratchet loop of interpretability experiments, logging results
    and accumulating findings.

    Two modes of use:

    **Mode 1: Zero orchestration (recommended)**
    Just run `init_autointerpret()`, open the directory in Claude Code,
    and say "read program.md and start." No Python orchestration needed.

    **Mode 2: Programmatic (this class)**
    For automation, testing, or integration into larger pipelines.

    Example:
        from mlxterp.autointerpret import AutoInterpret

        runner = AutoInterpret(
            model=model,
            clean="The Eiffel Tower is in",
            corrupted="The Colosseum is in",
            output_dir="my_experiment",
        )

        # Run a single experiment
        entry = runner.run_experiment(
            name="mlp_patching",
            fn=lambda: activation_patching(model, clean, corrupted, component="mlp"),
            hypothesis="MLPs at mid-layers are important for factual recall",
        )

        # Get experiment history
        print(runner.log.summary())
    """

    def __init__(
        self,
        model,
        clean: Union[str, Any] = "",
        corrupted: Union[str, Any] = "",
        output_dir: str = "autointerpret",
        max_experiments: int = 100,
        time_per_experiment: int = 120,
    ):
        self.model = model
        self.clean = clean
        self.corrupted = corrupted
        self.output_dir = Path(output_dir)
        self.max_experiments = max_experiments
        self.time_per_experiment = time_per_experiment

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "findings").mkdir(exist_ok=True)

        # Initialize log
        self.log = ExperimentLog(str(self.output_dir / "results.jsonl"))

    def run_experiment(
        self,
        name: str,
        fn: Callable[[], Any],
        hypothesis: str = "",
        informative_check: Optional[Callable[[Any], bool]] = None,
    ) -> ExperimentEntry:
        """
        Run a single experiment and log the result.

        Args:
            name: Experiment name / method description
            fn: Function that runs the experiment and returns a result
            hypothesis: What you expect to find
            informative_check: Optional function that takes the result and
                returns True if informative. Default: always True.

        Returns:
            ExperimentEntry with results
        """
        start = time.time()

        try:
            result = fn()
            duration = time.time() - start

            # Convert result to serializable form
            if isinstance(result, AnalysisResult):
                result_data = json.loads(result.to_json())
                conclusion = result.summary()
            elif isinstance(result, dict):
                result_data = result
                conclusion = str(result)[:200]
            else:
                result_data = {"raw": str(result)[:500]}
                conclusion = str(result)[:200]

            # Check if informative
            informative = True
            if informative_check is not None:
                informative = informative_check(result)

        except Exception as e:
            duration = time.time() - start
            result_data = {"error": str(e)}
            conclusion = f"FAILED: {str(e)}"
            informative = False

        entry = ExperimentEntry(
            hypothesis=hypothesis,
            method=name,
            result=result_data,
            informative=informative,
            conclusion=conclusion,
            duration_seconds=duration,
        )

        self.log.append(entry)

        # Save finding if informative
        if informative:
            finding_path = self.output_dir / "findings" / f"{self.log.count:03d}_{name}.json"
            finding_path.write_text(entry.to_json())

        return entry

    def history(self) -> List[ExperimentEntry]:
        """Get all experiment entries."""
        return self.log.entries()

    def findings(self) -> List[ExperimentEntry]:
        """Get only informative experiments."""
        return self.log.informative_entries()

    def summary(self) -> str:
        """Generate a summary of all experiments."""
        return self.log.summary()

    @property
    def n_experiments(self) -> int:
        """Number of experiments run so far."""
        return self.log.count

    @property
    def is_done(self) -> bool:
        """Whether the maximum number of experiments has been reached."""
        return self.n_experiments >= self.max_experiments
