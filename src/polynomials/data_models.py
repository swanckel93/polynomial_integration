from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
from pathlib import Path
import re


@dataclass
class PartialResult:
    """Represents a partial result during adaptive integration refinement."""

    result: float
    error: float
    samples: int
    time_elapsed: float


@dataclass
class SolverResult:
    """Represents the complete result for a single solver."""

    name: str
    description: str
    partial_results: List[PartialResult]
    final_result: float
    final_error: float
    total_time: float
    final_samples: int
    within_tolerance: bool
    success: bool
    error_message: Optional[str] = None


@dataclass
class IntegrationRun:
    """Represents a complete integration run with all solver results."""

    polynomial_coefficients: List[float]
    polynomial_string: str
    interval_start: float
    interval_end: float
    analytic_solution: float
    tolerance: float
    timeout: int
    start_n: int
    seed: int
    batch_size: int
    timestamp: str
    solver_results: List[SolverResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntegrationRun":
        """Create instance from dictionary (JSON deserialization)."""
        # Convert nested dictionaries back to dataclasses
        solver_results = []
        for solver_data in data["solver_results"]:
            partial_results = [
                PartialResult(**partial) for partial in solver_data["partial_results"]
            ]
            solver_results.append(
                SolverResult(**{**solver_data, "partial_results": partial_results})
            )

        return cls(**{**data, "solver_results": solver_results})

    def save_to_json(self, filepath: Path) -> None:
        """Save the integration run to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: Path) -> "IntegrationRun":
        """Load integration run from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


def generate_run_filename(polynomial_string: str, interval: tuple) -> str:
    """Generate a descriptive filename for the integration run."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract polynomial order (highest power)
    # Find all x^ patterns and extract the highest power
    powers = re.findall(r"x\^(\d+)", polynomial_string)
    if powers:
        order = max(int(p) for p in powers)
    else:
        # Check for x without explicit power (implies x^1)
        if "x" in polynomial_string.lower():
            order = 1
        else:
            order = 0  # constant polynomial

    # Format interval with from_to_ structure using integer parts
    interval_str = f"from_{int(interval[0])}_to_{int(interval[1])}"

    return f"{timestamp}_order{order}_{interval_str}.json"
