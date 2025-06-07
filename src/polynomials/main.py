import sys
import typer

from pathlib import Path
from typing import List, Optional
from rich.panel import Panel
from rich.table import Table
from datetime import datetime
from pathlib import Path

from polynomials.solvers import SolverName
from polynomials.polynom import Polynom
from polynomials.solvers import AnalyticSolver, SOLVER_MAP, SolverName
from helpers.adaptive_integration import AdaptiveIntegration
from polynomials.data_models import IntegrationRun
from helpers.rich_console import get_console
from polynomials.data_models import (
    IntegrationRun,
    SolverResult,
    generate_run_filename,
)
from helpers.cli import (
    parse_polynomial_coefficients,
    format_polynomial_display,
    create_aligned_polynomial_operation,
)
from helpers.plotting import create_integration_plot

# Rich console for beautiful output
console = get_console()
app = typer.Typer(
    name="poly",
    help="🧮 Polynomial operations CLI tool with beautiful output",
    rich_markup_mode="rich",
)


@app.command("display")
def show_polynom(
    coefficients: str = typer.Argument(
        ...,
        help="Polynomial coefficients in ascending order, space-separated (e.g., '3 0 -1' for 3 + 0x - x²)",
    )
):
    """
    📊 Display a polynomial from coefficients.

    Renders a polynomial in a beautiful format from space-separated coefficients.
    """
    coeffs = parse_polynomial_coefficients(coefficients)
    poly = Polynom(coeffs)

    console.print(format_polynomial_display(poly, "Your Polynomial", "cyan"))


@app.command("add")
def add_polynom(
    p1: str = typer.Argument(
        ..., help="First polynomial coefficients (space-separated)"
    ),
    p2: str = typer.Argument(
        ..., help="Second polynomial coefficients (space-separated)"
    ),
):
    """
    ➕ Add two polynomials: P₁ + P₂

    Displays the polynomials and their sum in a beautifully aligned format.
    """
    coeffs1 = parse_polynomial_coefficients(p1)
    coeffs2 = parse_polynomial_coefficients(p2)

    poly1 = Polynom(coeffs1)
    poly2 = Polynom(coeffs2)
    result = poly1 + poly2

    console.print(create_aligned_polynomial_operation(poly1, poly2, result, "add"))


@app.command("subtract")
def subtract_polynom(
    p1: str = typer.Argument(
        ..., help="First polynomial coefficients (space-separated)"
    ),
    p2: str = typer.Argument(
        ..., help="Second polynomial coefficients (space-separated)"
    ),
):
    """
    ➖ Subtract two polynomials: P₁ - P₂

    Displays the polynomials and their difference in a beautifully aligned format.
    """
    coeffs1 = parse_polynomial_coefficients(p1)
    coeffs2 = parse_polynomial_coefficients(p2)

    poly1 = Polynom(coeffs1)
    poly2 = Polynom(coeffs2)
    result = poly1 - poly2

    console.print(create_aligned_polynomial_operation(poly1, poly2, result, "subtract"))


@app.command("multiply")
def multiply_polynom(
    p1: str = typer.Argument(
        ..., help="First polynomial coefficients (space-separated)"
    ),
    p2: str = typer.Argument(
        ..., help="Second polynomial coefficients (space-separated)"
    ),
):
    """
    ✖️ Multiply two polynomials: P₁ × P₂

    Displays the polynomials and their product in a beautifully aligned format.
    """
    coeffs1 = parse_polynomial_coefficients(p1)
    coeffs2 = parse_polynomial_coefficients(p2)

    poly1 = Polynom(coeffs1)
    poly2 = Polynom(coeffs2)
    result = poly1 * poly2

    console.print(create_aligned_polynomial_operation(poly1, poly2, result, "multiply"))


@app.command("integrate")
def integrate(
    coefficients: str = typer.Argument(
        ..., help="Polynomial coefficients in ascending order, space-separated"
    ),
    interval_a: float = typer.Argument(..., help="Integration interval start"),
    interval_b: float = typer.Argument(..., help="Integration interval end"),
    solvers: Optional[List[str]] = typer.Option(
        None,
        "--solver",
        "-s",
        help="Solver names to use (can specify multiple). If not specified, all solvers will be used",
    ),
    tol: float = typer.Option(1e-6, "--tolerance", help="Integration tolerance"),
    timeout: int = typer.Option(30, "--timeout", help="Timeout per solver (seconds)"),
    start_n: int = typer.Option(10, "--start-n", help="Initial subintervals/samples"),
    seed: int = typer.Option(42, "--seed", help="Random seed for Monte Carlo"),
    batch_size: int = typer.Option(
        1024, "--batch-size", help="Batch size for multiprocess solvers"
    ),
    save_data: bool = typer.Option(
        False, "--save-data", help="Save integration results to JSON file in ./data/"
    ),
):
    """
    🧮 Integrate polynomials using numerical methods.

    Integrates a polynomial over a specified interval using various numerical
    integration methods with beautiful progress tracking and results display.

    Usage: poly integrate "1 2 3" 0 1 --solver SIMPSON --save-data

    Available solvers: MIDPOINT, TRAPEZ, SIMPSON, SIMPSON38, BOOLE,
    MIDPOINT_MP, TRAPEZ_MP, SIMPSON_MP, SIMPSON38_MP, BOOLE_MP, MONTECARLO
    """

    # Parse polynomial coefficients
    coeffs = parse_polynomial_coefficients(coefficients)
    poly = Polynom(coeffs)
    interval_tuple = (interval_a, interval_b)

    integration_run = IntegrationRun(
        polynomial_coefficients=coeffs,
        polynomial_string=str(poly),
        interval_start=interval_a,
        interval_end=interval_b,
        analytic_solution=0.0,  # for now, dont wanna make this None
        tolerance=tol,
        timeout=timeout,
        start_n=start_n,
        seed=seed,
        batch_size=batch_size,
        timestamp=datetime.now().isoformat(),
        solver_results=[],
    )

    # Better to check twice for solvers, ran into source of truth def in different places.
    if not solvers:
        solvers_to_run = list(SolverName)
    else:
        valid_solver_names = {s.value: s for s in SolverName}
        solvers_to_run = []
        for solver_name in solvers:
            if solver_name.upper() in valid_solver_names:
                solvers_to_run.append(valid_solver_names[solver_name.upper()])
            else:
                console.print(f"[red]❌ Unknown solver: {solver_name}[/red]")
                console.print(
                    f"[yellow]Available solvers: {', '.join(valid_solver_names.keys())}[/yellow]"
                )
                raise typer.Exit(1)
    # Really dislike the syntax here, but I guess its fine for the main function
    header_content = f"[bold blue]Polynomial:[/bold blue] [yellow]{poly}[/yellow]\n"
    header_content += f"[bold blue]Interval:[/bold blue] [[cyan]{interval_tuple[0]}[/cyan], [cyan]{interval_tuple[1]}[/cyan]]"

    console.print(
        Panel(
            header_content,
            title="🧮 Integration Task",
            border_style="blue",
            padding=(1, 2),
        )
    )
    # Get the ref value from the analytic solution
    analytic_solution = AnalyticSolver.integrate(poly, interval_tuple)
    integration_run.analytic_solution = analytic_solution
    console.print(
        f"[bold green]📐 Analytic Solution: {analytic_solution:.8f}[/bold green]\n"
    )

    # TODO: Styling could be externalized for better mantainability.
    results_table = Table(title="[bold not italic]📊[/] Integration Results Summary")
    results_table.add_column("Solver", style="cyan", no_wrap=True)
    results_table.add_column("Result", style="green", justify="right")
    results_table.add_column("Error %", style="yellow", justify="right")
    results_table.add_column("Time (s)", style="blue", justify="right")
    results_table.add_column("Samples", style="magenta", justify="right")
    results_table.add_column("✓", style="bold", justify="center", width=3)

    total_solvers = 0
    successful_solvers = 0

    # Main Loop for so sovlers.
    for solver_name in solvers_to_run:
        total_solvers += 1
        solver_class = SOLVER_MAP.get(solver_name)
        if solver_class is None:
            console.print(
                f"[red]❌ Solver {solver_name} not implemented. Skipping.[/red]"
            )
            # Add failed solver result to data
            integration_run.solver_results.append(
                SolverResult(
                    name=solver_name.value,
                    description="Not implemented",
                    partial_results=[],
                    final_result=0.0,
                    final_error=0.0,
                    total_time=0.0,
                    final_samples=0,
                    within_tolerance=False,
                    success=False,
                    error_message="Solver not implemented",
                )
            )
            continue

        # Display current solver info
        solver_panel = Panel(
            f"[bold]{solver_class.description.strip()}[/bold]",
            title=f"⚙️ {solver_name.value.upper()}",
            border_style="green",
            padding=(0, 1),
        )
        console.print(solver_panel)

        # TODO: this is terrible.
        # Injecting solvers with their respective kwargs.
        # Needed Cause sibling classes expect different params.
        # I guess that happens when you cut corners.
        solver_kwargs = {}
        if "n_samples" in solver_class.expected_kwargs:
            solver_kwargs["n_samples"] = start_n
        if "n_subintervals" in solver_class.expected_kwargs:
            solver_kwargs["n_subintervals"] = start_n
        if "_seed" in solver_class.expected_kwargs:
            solver_kwargs["_seed"] = seed
        if "batch_size" in solver_class.expected_kwargs:
            solver_kwargs["batch_size"] = batch_size

        try:
            with console.status(f"[bold green]Running {solver_name.value}...") as _:
                partial_results, is_success = AdaptiveIntegration.refine(
                    polynom=poly,
                    solver=solver_class,
                    interval=interval_tuple,
                    analytic_solution=analytic_solution,
                    tolerance=tol,
                    timeout=timeout,
                    solver_kwargs=solver_kwargs,
                )

            if is_success:
                successful_solvers += 1

            if partial_results:
                final_partial = partial_results[-1]
                final_result = final_partial.result
                final_error = final_partial.error
                total_time = final_partial.time_elapsed
                final_samples = final_partial.samples
            else:
                final_result = 0.0
                final_error = 0.0
                total_time = 0.0
                final_samples = 0

            solver_result = SolverResult(
                name=solver_name.value,
                description=solver_class.description.strip(),
                partial_results=partial_results,
                final_result=final_result,
                final_error=final_error,
                total_time=total_time,
                final_samples=final_samples,
                within_tolerance=final_error <= tol,
                success=is_success,
                error_message=None,
            )
            integration_run.solver_results.append(solver_result)

            success_icon = "✅" if is_success else "❌"

            results_table.add_row(
                solver_name.value,
                f"{final_result:.6f}",
                f"{final_error:.2e}",
                f"{total_time:.3f}",
                f"{final_samples:.1e}",
                success_icon,
            )

        except Exception as e:
            console.print(f"[red]❌ Error running {solver_name}: {str(e)}[/red]")

            solver_result = SolverResult(
                name=solver_name.value,
                description=(
                    solver_class.description.strip() if solver_class else "Unknown"
                ),
                partial_results=[],
                final_result=0.0,
                final_error=0.0,
                total_time=0.0,
                final_samples=0,
                within_tolerance=False,
                success=False,
                error_message=str(e),
            )
            integration_run.solver_results.append(solver_result)

            results_table.add_row(
                solver_name.value,
                "[red]ERROR[/red]",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
                "[red]N/A[/red]",
                "❌",
            )
            continue

    console.print("\n")
    console.print(results_table)

    summary_text = (
        f"[green]✅ {successful_solvers} solvers completed successfully[/green]"
    )
    if successful_solvers < total_solvers:
        failed = total_solvers - successful_solvers
        summary_text += f"\n[red]❌ {failed} solvers failed or timed out[/red]"

    console.print(Panel(summary_text, title="📈 Summary", border_style="blue"))

    # Dumping the json
    if save_data:
        data_dir = Path("./data")
        filename = generate_run_filename(str(poly), interval_tuple)
        filepath = data_dir / filename

        try:
            integration_run.save_to_json(filepath)
            console.print(f"\n[green]💾 Data saved to: {filepath}[/green]")
        except Exception as e:
            console.print(f"\n[red]❌ Failed to save data: {str(e)}[/red]")


@app.command("plot")
def plot(
    json_filename: str = typer.Argument(
        ..., help="JSON filename from ./data/ directory"
    ),
    output_path: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output plot path (default: ./plots/)"
    ),
):
    """
    📊 Create performance plots from integration run data.

    Loads integration results from JSON file and creates a performance plot
    showing error vs time for all solvers.

    Usage: poly plot run_20241215_123456_polynomial_0to1.json
    """

    # Setup paths
    data_dir = Path("./data")
    plots_dir = Path("./plots")

    # Load data
    json_path = data_dir / json_filename
    if not json_path.exists():
        console.print(f"[red]❌ JSON file not found: {json_path}[/red]")
        # List available files
        if data_dir.exists():
            available_files = list(data_dir.glob("*.json"))
            if available_files:
                console.print("\n[yellow]Available JSON files:[/yellow]")
                for file in available_files:
                    console.print(f"  • {file.name}")
        raise typer.Exit(1)

    try:
        integration_run = IntegrationRun.load_from_json(json_path)
        console.print(f"[green]✅ Loaded data from: {json_path}[/green]")
    except Exception as e:
        console.print(f"[red]❌ Failed to load JSON: {str(e)}[/red]")
        raise typer.Exit(1)

    final_output_path: Path
    if output_path is None:
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_filename = json_filename.replace(".json", "_plot.png")
        final_output_path = plots_dir / output_filename
    else:
        final_output_path = Path(output_path)
        final_output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create plot
    try:
        create_integration_plot(integration_run, final_output_path)
        console.print(f"[green]📊 Plot saved to: {final_output_path}[/green]")

        # Display summary
        successful_solvers = sum(
            1 for solver in integration_run.solver_results if solver.success
        )
        total_solvers = len(integration_run.solver_results)

        summary_text = (
            f"[bold blue]Polynomial:[/bold blue] {integration_run.polynomial_string}\n"
        )
        summary_text += f"[bold blue]Interval:[/bold blue] [{integration_run.interval_start}, {integration_run.interval_end}]\n"
        summary_text += (
            f"[bold blue]Tolerance:[/bold blue] {integration_run.tolerance:.2e}\n"
        )
        summary_text += f"[bold blue]Solvers:[/bold blue] {successful_solvers}/{total_solvers} successful"

        console.print(Panel(summary_text, title="📈 Plot Summary", border_style="blue"))

    except Exception as e:
        console.print(f"[red]❌ Failed to create plot: {str(e)}[/red]")
        raise typer.Exit(1)


@app.command("list-runs")
def list_runs():
    """
    📋 List available integration run JSON files.

    Shows all available JSON files in the ./data/ directory.
    """
    data_dir = Path("./data")

    if not data_dir.exists():
        console.print(
            "[yellow]📁 No data directory found. Run some integrations with --save-data first.[/yellow]"
        )
        return

    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        console.print("[yellow]📁 No JSON files found in ./data/ directory.[/yellow]")
        return

    console.print(f"[green]📋 Found {len(json_files)} integration run(s):[/green]\n")

    for json_file in sorted(json_files):
        try:
            # Load basic info without full parsing
            integration_run = IntegrationRun.load_from_json(json_file)
            successful_solvers = sum(
                1 for solver in integration_run.solver_results if solver.success
            )
            total_solvers = len(integration_run.solver_results)

            console.print(f"[cyan]📊 {json_file.name}[/cyan]")
            console.print(f"   Polynomial: {integration_run.polynomial_string}")
            console.print(
                f"   Interval: [{integration_run.interval_start}, {integration_run.interval_end}]"
            )
            console.print(f"   Tolerance: {integration_run.tolerance:.2e}")
            console.print(
                f"   Solvers: {successful_solvers}/{total_solvers} successful"
            )
            console.print(f"   Timestamp: {integration_run.timestamp}")
            console.print()

        except Exception as e:
            console.print(f"[red]❌ Error reading {json_file.name}: {str(e)}[/red]")


def main():
    """Main entry point with error handling."""
    sys.setrecursionlimit(1000)
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️ Operation cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]💥 Unexpected error: {str(e)}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    main()
