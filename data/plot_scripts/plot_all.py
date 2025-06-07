import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Config
tolerance = 1e-5
filename = "sample1.txt"
fpath = Path(__file__).parents[1] / "sample_runs"
dstpath = Path(__file__).parents[1] / "plots"
# Initialize plot data structure


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

# Plot
sns.set(style="whitegrid", palette="muted")
plt.figure(figsize=(12, 8))

for solver, data in solvers_data.items():
    times = np.array(data["times"])
    errors = np.array(data["errors"])
    samples = data["samples"]

    if not len(times):  # skip if no data
        continue

    plt.plot(times, errors, label=solver, marker="o")

    # Annotate with sample count
    for x, y, n in zip(times, errors, samples):
        plt.annotate(
            f"{n}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8
        )

# Plot formatting
plt.axhline(
    y=tolerance * 100, color="gray", linestyle="--", label="Tolerance"
)  # convert tol to percent
plt.yscale("log")
plt.xlabel("Time Elapsed [s]")
plt.ylabel("Relative Error [%]")
plt.title(poly_title)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(dstpath / f"{filename.split('.')[0]} + .png", dpi=600)
