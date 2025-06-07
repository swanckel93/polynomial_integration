import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class PartialResult:
    result: float
    error: float
    samples: float
    time: float


@dataclass
class SolverResult:
    name: str
    description: str
    partial_results: List[PartialResult]
    final_result: float
    final_error: float
    total_time: float
    final_samples: float
    within_tolerance: bool


@dataclass
class IntegrationData:
    polynomial: str
    poly_order: int
    interval: tuple
    tolerance: float
    solver_results: List[SolverResult]


def parse_integration_output(filepath: Path) -> IntegrationData:
    """Parse the integration output file and return structured data."""

    with open(filepath, "r") as f:
        content = f.read()

    # Extract polynomial and interval from the first line
    poly_line = content.split("\n")[1]
    poly_match = re.search(
        r"Integrating Polynomial (.+?) in interval \[(.+?), (.+?)\]", poly_line
    )
    polynomial = poly_match.group(1) if poly_match else "Unknown"

    # Extract polynomial order (highest power)
    poly_order = 0
    if poly_match:
        poly_str = poly_match.group(1)
        order_match = re.search(r"x\^(\d+)(?![^x]*x\^)", poly_str)
        poly_order = int(order_match.group(1)) if order_match else 0

    interval_start = float(poly_match.group(2)) if poly_match else 0.0
    interval_end = float(poly_match.group(3)) if poly_match else 1.0
    interval = (interval_start, interval_end)

    # Extract tolerance from command line (assuming 1e-5 from the command)
    tol_match = re.search(r"--tol (\S+)", content)
    tolerance = float(tol_match.group(1)) if tol_match else 1e-5

    # Split content by solver sections
    solver_sections = re.split(r"Executing \[([^\]]+)\]", content)[
        1:
    ]  # Skip first empty part

    solver_results = []

    for i in range(0, len(solver_sections), 2):
        solver_name = solver_sections[i]
        solver_content = solver_sections[i + 1] if i + 1 < len(solver_sections) else ""

        # Skip if solver not implemented
        if "not implemented" in solver_content:
            continue

        # Extract description
        desc_match = re.search(
            r"Description:\s*(.+?)(?=\n\n|\n  Final|\n    Partial)",
            solver_content,
            re.DOTALL,
        )
        description = desc_match.group(1).strip() if desc_match else ""

        # Extract partial results
        partial_results = []
        partial_matches = re.findall(
            r"Partial Result\s*=\s*([\d.]+)\s*\n\s*Partial Error\s*=\s*([\d.e+-]+)%\s*\n\s*Number Of Samples\s*=\s*([\d.e+-]+)\s*\n\s*Time Elapsed\s*=\s*([\d.]+)s",
            solver_content,
        )

        for match in partial_matches:
            partial_results.append(
                PartialResult(
                    result=float(match[0]),
                    error=float(match[1]),
                    samples=float(match[2]),
                    time=float(match[3]),
                )
            )

        # Extract final results
        final_result_match = re.search(r"Final Result\s*=\s*([\d.]+)", solver_content)
        final_error_match = re.search(r"Final Error\s*=\s*([\d.e+-]+)%", solver_content)
        total_time_match = re.search(
            r"Total Time Elapsed\s*=\s*([\d.]+)s", solver_content
        )
        final_samples_match = re.search(
            r"Number Of Samples\s*=\s*([\d.e+-]+)(?=\s*\n\s*Error within)",
            solver_content,
        )
        tolerance_match = re.search(
            r"Error within Tolerance\s*=\s*(True|False)", solver_content
        )

        final_result = float(final_result_match.group(1)) if final_result_match else 0.0
        final_error = float(final_error_match.group(1)) if final_error_match else 0.0
        total_time = float(total_time_match.group(1)) if total_time_match else 0.0
        final_samples = (
            float(final_samples_match.group(1)) if final_samples_match else 0.0
        )
        within_tolerance = (
            tolerance_match.group(1) == "True" if tolerance_match else False
        )

        solver_results.append(
            SolverResult(
                name=solver_name,
                description=description,
                partial_results=partial_results,
                final_result=final_result,
                final_error=final_error,
                total_time=total_time,
                final_samples=final_samples,
                within_tolerance=within_tolerance,
            )
        )

    return IntegrationData(
        polynomial=polynomial,
        poly_order=poly_order,
        interval=interval,
        tolerance=tolerance,
        solver_results=solver_results,
    )


def create_integration_plot(data: IntegrationData, save_path: Path) -> None:
    """Create and save the integration methods performance plot."""

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(data.solver_results)))

    # Plot each method
    for i, solver in enumerate(data.solver_results):
        # Collect all time and error points (partial + final)
        times = []
        errors = []
        samples = []

        # Add partial results
        for partial in solver.partial_results:
            times.append(partial.time)
            errors.append(partial.error)
            samples.append(partial.samples)

        # Add final result if it's different from the last partial
        if (
            not solver.partial_results
            or solver.total_time != solver.partial_results[-1].time
        ):
            times.append(solver.total_time)
            errors.append(solver.final_error)
            samples.append(solver.final_samples)

        # Handle zero errors for logarithmic scale
        plot_errors = [max(err, 1e-10) if err == 0 else err for err in errors]

        # Plot the line
        ax.plot(
            times,
            plot_errors,
            "o-",
            label=solver.name,
            linewidth=2,
            markersize=6,
            color=colors[i],
        )

        # Add annotations for sample counts
        for j, (time, error, sample) in enumerate(zip(times, plot_errors, samples)):
            # Only annotate every other point to avoid crowding, or all points if few
            if len(times) <= 3 or j % 2 == 0 or j == len(times) - 1:
                ax.annotate(
                    f"{sample:.0e}",
                    (time, error),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                    alpha=0.8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=colors[i], alpha=0.3),
                )

    # Add tolerance line
    ax.axhline(
        y=data.tolerance * 100,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Tolerance ({data.tolerance:.0e})",
        alpha=0.8,
    )

    # Set logarithmic scale for y-axis
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Labels and title
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Error [%]", fontsize=12)
    ax.set_title(
        f"Integration Methods Performance\nPolynomial of order {data.poly_order}, Interval: {data.interval}",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, alpha=0.3, which="both")

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Set axis limits for better visualization
    all_times = [solver.total_time for solver in data.solver_results] + [
        partial.time
        for solver in data.solver_results
        for partial in solver.partial_results
    ]
    max_time = max(all_times) if all_times else 1
    ax.set_xlim(-0.1, max_time * 1.1)
    ax.set_ylim(1e-10, 1e2)

    # Save the plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # Close the figure to free memory


def main():
    """Main function to process the integration output and create plot."""

    filename = "polynomial_10_interval_10000.txt"
    fpath = Path(__file__).parents[1] / "sample_runs"
    dstpath = Path(__file__).parents[1] / "plots"

    # Parse the input file
    input_file = fpath / filename
    data = parse_integration_output(input_file)

    # Create and save the plot
    plot_filename = filename.replace(".txt", "_plot.png")
    output_file = dstpath / plot_filename
    create_integration_plot(data, output_file)

    print(f"Plot saved to: {output_file}")
    print(f"Parsed {len(data.solver_results)} solver results")


if __name__ == "__main__":
    main()
