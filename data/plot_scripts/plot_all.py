import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Config
tolerance = 1e-11
filename = "sample1.txt"
fpath = Path(__file__).parents[1] / "sample_runs"
dstpath = Path(__file__).parents[1] / "plots"

# Read and parse
with open(fpath / filename) as f:
    lines = f.readlines()

# Regular expressions
poly_re = re.compile(r"^Integrating Polynomial (.+) in interval \[(.+)\]")
solver_re = re.compile(r"^Executing \[([A-Z0-9\-]+)\]")
partial_error_re = re.compile(r"Partial Error\s+=\s+([0-9.+\-eE%]+)")
time_elapsed_re = re.compile(r"Time Elapsed\s+=\s+([0-9.+\-eE]+)s")
samples_re = re.compile(r"Number Of Samples\s+=\s+([0-9.+\-eE]+)")

poly_title = ""
poly_order = 0
current_solver = None
solvers_data = {}

for line in lines:
    line = line.strip()

    poly_match = poly_re.search(line)

    if poly_match:
        poly_str = poly_match.group(1)
        interval = poly_match.group(2)
        poly_title = f"Integration of f(x) = {poly_str} on [{interval}]"

        # Extract polynomial order (highest power)
        order_match = re.search(r"x\^(\d+)(?![^x]*x\^)", poly_str)
        poly_order = int(order_match.group(1)) if order_match else 0

        continue

    solver_match = solver_re.search(line)
    if solver_match:
        current_solver = solver_match.group(1)
        solvers_data[current_solver] = {
            "times": [],
            "errors": [],
            "samples": [],
        }
        continue

    if current_solver:
        time_match = time_elapsed_re.search(line)
        error_match = partial_error_re.search(line)
        samples_match = samples_re.search(line)

        if all([time_match, error_match, samples_match]):
            solvers_data[current_solver]["times"].append(float(time_match.group(1)))
            solvers_data[current_solver]["errors"].append(
                float(error_match.group(1).replace("%", ""))
            )
            solvers_data[current_solver]["samples"].append(samples_match.group(1))

# Collect all data for axis limit calculation
all_times = []
all_errors = []

# Plot
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(12, 8))

for solver, data in solvers_data.items():
    times = np.array(data["times"])
    errors = np.array(data["errors"])
    samples = data["samples"]

    if not len(times):  # skip if no data
        continue

    # Collect data for axis limits
    all_times.extend(times)
    all_errors.extend(errors)

    plt.plot(times, errors, label=solver, marker="o")

    # Annotate with sample count
    for x, y, n in zip(times, errors, samples):
        plt.annotate(
            f"{n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8
        )

# Calculate extended axis limits
if all_times and all_errors:
    # Include tolerance in error calculations
    all_errors_with_tol = all_errors + [
        tolerance * 100
    ]  # Convert tolerance to percentage

    min_time, max_time = min(all_times), max(all_times)
    min_error, max_error = min(all_errors_with_tol), max(all_errors_with_tol)

    # Calculate log bounds with one order of magnitude extension
    time_log_min = np.floor(np.log10(min_time)) - 1
    time_log_max = np.ceil(np.log10(max_time)) + 1
    error_log_min = np.floor(np.log10(min_error)) - 1
    error_log_max = np.ceil(np.log10(max_error)) + 1

    # Set extended limits
    plt.xlim(10**time_log_min, 10**time_log_max)
    plt.ylim(10**error_log_min, 10**error_log_max)

# Plot formatting
plt.axhline(
    y=tolerance * 100, color="gray", linestyle="--", label="Tolerance"
)  # convert tol to percent
plt.yscale("log")
plt.xscale("log")  # Also set x-axis to log scale for consistency
plt.xlabel("Time Elapsed [s]")
plt.ylabel("Relative Error [%]")
plt.title(poly_title)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(dstpath / f"{filename.split('.')[0]}.png", dpi=600)  # Fixed filename bug
