"""
Experiment logging for AutoInterp.

Structured JSONL logging of experiments with metadata, results,
and informative/uninformative classification.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExperimentEntry:
    """A single experiment entry in the log.

    Attributes:
        experiment_id: Unique ID (auto-generated from timestamp)
        hypothesis: What the agent expected to find
        method: What analysis was run
        result: Structured result data
        informative: Whether the finding was informative (kept)
        conclusion: What was learned
        timestamp: When the experiment was run
        duration_seconds: How long it took
        metadata: Additional context
    """

    experiment_id: str = ""
    hypothesis: str = ""
    method: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    informative: bool = True
    conclusion: str = ""
    timestamp: float = 0.0
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = f"exp_{int(time.time())}_{id(self) % 10000:04d}"
        if self.timestamp == 0.0:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


class ExperimentLog:
    """
    Append-only experiment log backed by a JSONL file.

    Each line is a JSON object representing one experiment.

    Usage:
        log = ExperimentLog("results.jsonl")
        log.append(ExperimentEntry(
            hypothesis="Layer 5 MLP is important for factual recall",
            method="activation_patching",
            result={"layer_5_effect": 0.82},
            informative=True,
            conclusion="Confirmed: layer 5 MLP has 82% recovery",
        ))

        # Read back
        for entry in log.entries():
            print(f"{entry.experiment_id}: {entry.conclusion}")
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def append(self, entry: ExperimentEntry):
        """Append an experiment entry to the log."""
        with open(self.path, "a") as f:
            f.write(entry.to_json() + "\n")

    def entries(self) -> List[ExperimentEntry]:
        """Read all entries from the log."""
        if not self.path.exists():
            return []

        entries = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    entries.append(ExperimentEntry(**data))
                except (json.JSONDecodeError, TypeError):
                    continue
        return entries

    def informative_entries(self) -> List[ExperimentEntry]:
        """Get only informative (kept) experiments."""
        return [e for e in self.entries() if e.informative]

    def summary(self) -> str:
        """Generate a summary of all experiments."""
        entries = self.entries()
        if not entries:
            return "No experiments logged yet."

        informative = sum(1 for e in entries if e.informative)
        total_time = sum(e.duration_seconds for e in entries)

        lines = [
            f"Experiments: {len(entries)} total, {informative} informative",
            f"Total time: {total_time:.1f}s",
            "",
            "Findings:",
        ]
        for entry in entries:
            marker = "+" if entry.informative else "-"
            lines.append(f"  [{marker}] {entry.conclusion}")

        return "\n".join(lines)

    @property
    def count(self) -> int:
        """Number of entries in the log."""
        return len(self.entries())
