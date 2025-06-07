from .polynom import Polynom
from .solvers import AnalyticSolver, SOLVER_MAP, SolverName
from .utils import AdaptiveIntegration


class Routines:
    @staticmethod
    def show_polynom(args):
        # coeffs = Polynom.parse_from_string(p1)
        p = Polynom(args.p1)
        print(p)
        return p

    @staticmethod
    def add_polynom(args):
        result = Polynom(args.p1) + Polynom(args.p2)
        print(f"{result}")

    @staticmethod
    def subtract_polynom(args):
        result = Polynom(args.p1) - Polynom(args.p2)
        print(f"{result}")

    @staticmethod
    def multiply_polynom(args):
        p1 = Polynom(args.p1)
        p2 = Polynom(args.p2)
        result = p1 * p2

        print(f"{result}")

    @staticmethod
    def integrate(args):
        poly = Polynom(args.p1)
        interval = tuple(args.interval)
        solver = SOLVER_MAP.get(args.solver)
        tolerance = args.tol
        if solver is None:
            raise ValueError("Solver of that name not found. Aborting.")
        kwargs = {
            # "start_n_intervals": args.start_n,
            "_seed": args.seed,
            "n_samples": args.start_n,
            "n_subintervals": args.start_n,
            "batch_size": args.batch_size,
        }
        solver_kwargs = {k: kwargs[k] for k in solver.expected_kwargs}
        timeout = args.timeout
        analytic_solution = AnalyticSolver.integrate(poly, interval)
        a, b = interval
        print(f"Integrating Polynomial {poly} in interval [{a}, {b}]")
        print()
        print(f"Executing [{args.solver.upper()}]")
        print(f"Description:")
        print(" " * 2 + f"{solver.description}")
        print()
        result, error, elapsed, n_samples, is_success = AdaptiveIntegration.refine(
            polynom=poly,
            solver=solver,
            interval=interval,
            analytic_solution=analytic_solution,
            tolerance=tolerance,
            timeout=timeout,
            solver_kwargs=solver_kwargs,
        )
        print(" " * 2 + f"Final Result           = {result:.6f}")
        print(" " * 2 + f"Final Error            = {error:.2e}%")
        print(" " * 2 + f"Total Time Elapsed     = {elapsed:.3f}s")
        print(" " * 2 + f"Number Of Samples      = {n_samples:.1e}")
        print(" " * 2 + f"Error within Tolerance = {is_success}")
        print()
        print(" " * 2 + "-" * 50)
        print()

    @staticmethod
    def integrate_all(args):
        poly = Polynom(args.p1)
        interval = tuple(args.interval)
        tol = args.tol
        timeout = args.timeout
        a, b = interval
        print(f"Integrating Polynomial {poly} in interval [{a}, {b}]")
        print()
        for solver_name in SolverName:
            solver = SOLVER_MAP.get(solver_name)
            if solver is None:
                print(
                    f"Solver {solver_name} not implemented in SOLVER_MAP. Skipping.\n"
                )
                continue

            kwargs = {
                "_seed": args.seed,
                "n_samples": args.start_n,
                "n_subintervals": args.start_n,
                "batch_size": args.batch_size,
            }
            solver_kwargs = {
                k: kwargs[k] for k in solver.expected_kwargs if k in kwargs
            }
            analytic_solution = AnalyticSolver.integrate(poly, interval)

            print(f"Executing [{solver_name}]")
            print("Description:")
            print(" " * 2 + f"{solver.description.strip()}")
            print()

            result, error, elapsed, n_samples, is_success = AdaptiveIntegration.refine(
                polynom=poly,
                solver=solver,
                interval=interval,
                analytic_solution=analytic_solution,
                tolerance=tol,
                timeout=timeout,
                solver_kwargs=solver_kwargs,
            )

            print(" " * 2 + f"Final Result           = {result:.6f}")
            print(" " * 2 + f"Final Error            = {error:.2e}%")
            print(" " * 2 + f"Total Time Elapsed     = {elapsed:.3f}s")
            print(" " * 2 + f"Number Of Samples      = {n_samples:.1e}")
            print(" " * 2 + f"Error within Tolerance = {is_success}")
            print()
            print(" " * 2 + "-" * 50)
            print()
