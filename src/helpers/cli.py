from polynomials.polynom import Polynom
from rich.panel import Panel
import typer
from .rich_console import get_console
from rich.text import Text
from rich.align import Align
from rich.table import Table

console = get_console()


def parse_polynomial_coefficients(coeffs_str: str) -> list[float]:
    """Parse polynomial coefficients from string format."""
    try:
        return [float(x) for x in coeffs_str.split()]
    except ValueError:
        console.print(
            "[red]❌ Error: Invalid coefficient format. Use space-separated numbers.[/red]"
        )
        raise typer.Exit(1)


def format_polynomial_display(
    poly: Polynom, title: str = "Polynomial", color: str = "blue"
) -> Panel:
    """Create a beautiful polynomial display panel."""
    poly_text = Text(str(poly), style=f"bold {color}")
    return Panel(
        Align.center(poly_text), title=f"📊 {title}", border_style=color, padding=(1, 2)
    )


def create_aligned_polynomial_operation(
    p1: Polynom, p2: Polynom, result: Polynom, operation: str
) -> Panel:
    """Create beautifully aligned polynomial arithmetic display."""

    # Get coefficient lists (pad to same length)
    max_degree = max(len(p1), len(p2), len(result))

    p1_coeffs = list(p1) + [0] * (max_degree - len(p1))
    p2_coeffs = list(p2) + [0] * (max_degree - len(p2))
    result_coeffs = list(result) + [0] * (max_degree - len(result))

    # Create table for aligned display
    table = Table(show_header=False, show_lines=True, box=None, padding=(0, 1))
    table.add_column("", style="cyan", width=12)

    # Add columns for each degree (reverse order for display)
    for i in range(max_degree - 1, -1, -1):
        if i == 0:
            table.add_column(f"x⁰", style="bold", justify="center", width=8)
        else:
            table.add_column(f"x^{i}", style="bold", justify="center", width=8)

    # Add polynomial rows
    def format_coeff(coeff):
        if coeff == 0:
            return "[dim]0[/dim]"
        elif coeff == 1:
            return "[green]1[/green]"
        elif coeff == -1:
            return "[red]-1[/red]"
        elif coeff > 0:
            return f"[green]{coeff}[/green]"
        else:
            return f"[red]{coeff}[/red]"

    # First polynomial
    row1 = ["[blue]P₁ =[/blue]"] + [
        format_coeff(p1_coeffs[i]) for i in range(max_degree - 1, -1, -1)
    ]
    table.add_row(*row1)

    # Operation symbol
    op_symbol = {"add": "+", "subtract": "-", "multiply": "×"}[operation]
    row2 = [f"[yellow]{op_symbol}[/yellow]"] + [
        format_coeff(p2_coeffs[i]) for i in range(max_degree - 1, -1, -1)
    ]
    table.add_row(*row2)

    # Separator line
    table.add_row(*["[dim]─[/dim]"] * (max_degree + 1))

    # Result
    row3 = ["[magenta]Result =[/magenta]"] + [
        format_coeff(result_coeffs[i]) for i in range(max_degree - 1, -1, -1)
    ]
    table.add_row(*row3)

    operation_names = {
        "add": "Addition",
        "subtract": "Subtraction",
        "multiply": "Multiplication",
    }
    return Panel(
        table,
        title=f"🧮 Polynomial {operation_names[operation]}",
        border_style="green",
        padding=(1, 1),
    )
