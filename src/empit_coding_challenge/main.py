import argparse
import sys
from .polynom import Polynom
from .routines import Routines

# from .solvers import ...

POLYN_HELP_MSG = "List of coefficients in ascending order, separated by spaces (e.g., '3 0 -1' for 3 + 0x - x²)."


def main():
    sys.setrecursionlimit(1000)  # Set recursion limit to 100

    parser = argparse.ArgumentParser(
        prog="poly", description="Polynomial operations CLI tool."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    display_parser = subparsers.add_parser(
        name="display", help="Render and display a polynomial from coefficients."
    )
    display_parser.add_argument("p1", help=POLYN_HELP_MSG)
    display_parser.set_defaults(func=Routines.show_polynom)

    addition_parser = subparsers.add_parser(
        name="add", help="Add two polynomials: p1 + p2"
    )
    addition_parser.add_argument("p1", help=POLYN_HELP_MSG)
    addition_parser.add_argument("p2", help=POLYN_HELP_MSG)
    addition_parser.set_defaults(func=Routines.add_polynom)

    subraction_parser = subparsers.add_parser(
        name="subtract", help="Subtract two polynomials: p1 - p2"
    )
    subraction_parser.add_argument("p1", help=POLYN_HELP_MSG)
    subraction_parser.add_argument("p2", help=POLYN_HELP_MSG)
    subraction_parser.set_defaults(func=Routines.subtract_polynom)

    multiplication_parser = subparsers.add_parser(
        name="multiply", help="Multiply two polynomials: p1 * p2"
    )
    multiplication_parser.add_argument("p1", help=POLYN_HELP_MSG)
    multiplication_parser.add_argument("p2", help=POLYN_HELP_MSG)
    multiplication_parser.set_defaults(func=Routines.multiply_polynom)

    integration_parser = subparsers.add_parser(name="integration")
    integration_parser.add_argument("p1")

    integrate_all_parser = subparsers.add_parser("integrate_all")
    integrate_all_parser.add_argument(
        "--p1", nargs="+", type=float, required=True, help="Polynomial coefficients"
    )
    integrate_all_parser.add_argument(
        "--interval", nargs=2, required=True, help="Integration interval (a b)"
    )
    integrate_all_parser.add_argument(
        "--tol", type=float, default=1e-6, help="Tolerance"
    )
    integrate_all_parser.add_argument(
        "--timeout", type=int, default=3, help="Timeout per solver in seconds"
    )
    integrate_all_parser.set_defaults(func=Routines.integrate_all)

    # create a Polynom from a list of cofficients given in the cl
    # create an interval for integrating from arguments given in the cl

    # solve the integral of the polynom with your different solvers integrate method
    # compare the results and run times

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
