import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from polynomials.data_models import IntegrationRun
import numpy as np


def create_integration_plot(integration_run: IntegrationRun, save_path: Path) -> None:
    """Create and save the integration methods performance plot."""

    # Set style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Colors for different methods
    colors = plt.cm.tab10(np.linspace(0, 1, len(integration_run.solver_results)))

    # Collect all times and errors for proper axis limits
    all_times = []
    all_errors = []

    # Plot each successful solver
    color_idx = 0
    for solver in integration_run.solver_results:
        # if not solver.success or not solver.partial_results:
        #     continue

        # Collect all time and error points from partial results
        times = [partial.time_elapsed for partial in solver.partial_results]
        errors = [partial.error for partial in solver.partial_results]
        samples = [partial.samples for partial in solver.partial_results]

        # Add to global lists for axis limits
        all_times.extend(times)
        all_errors.extend(errors)

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
            color=colors[color_idx],
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
                    bbox=dict(
                        boxstyle="round,pad=0.2", facecolor=colors[color_idx], alpha=0.3
                    ),
                )

        color_idx += 1

    # Add tolerance line
    ax.axhline(
        y=integration_run.tolerance,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Tolerance ({integration_run.tolerance:.0e}%)",
        alpha=0.8,
    )

    # Set logarithmic scale for y-axis
    ax.set_yscale("log")
    ax.set_xscale("log")

    # Labels and title
    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("Error [%]", fontsize=12)

    # Determine polynomial order for title
    poly_order = len(integration_run.polynomial_coefficients) - 1
    ax.set_title(
        f"Integration Methods Performance\n"
        f"Polynomial of order {poly_order}, "
        f"Interval: [{integration_run.interval_start}, {integration_run.interval_end}]",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(True, alpha=0.3, which="both")

    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Set axis limits to show all points with some margin
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        # Add margin for log scale
        time_margin = (
            0.1 * (max_time - min_time) if max_time > min_time else max_time * 0.1
        )
        ax.set_xlim(max(min_time - time_margin, min_time * 0.5), max_time + time_margin)

    if all_errors:
        # Filter out zero errors for limit calculation
        non_zero_errors = [err for err in all_errors if err > 0]
        if non_zero_errors:
            min_error = min(non_zero_errors)
            max_error = max(all_errors)
            # Add margin for log scale
            ax.set_ylim(min_error * 0.1, max_error * 10)
        else:
            ax.set_ylim(1e-10, 1e2)
    else:
        ax.set_ylim(1e-10, 1e2)

    # Save the plot
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()  # Close the figure to free memory
