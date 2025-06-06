from .polynom import Polynom
from .solvers import (
    AnalyticSolver,
    NumericV1,
    NumericMP,
    NumericRecursive,
    MonteCarlo,
    NumericV1Meta,
    NumericMPMeta,
    NumericRecursiveMeta,
    MonteCarloMeta,
    SolverMeta,
)
import signal
import time


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException()


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
    def integrate_polynom(args):
        assert len(args.interval) == 2
        for val in args.interval:
            assert isinstance(val, float) | isinstance(val, int)
        interval = tuple([float(val) for val in args.interval])
        assert len(interval) == 2

        result = AnalyticSolver.integrate(Polynom(args.p1), interval)
        print(result)

    @staticmethod
    def integrate_all(args):
        poly = Polynom(args.p1)
        interval = tuple(map(float, args.interval))
        tol = float(args.tol)
        timeout = int(args.timeout)

        print(f"Running integration for {poly} over {interval}")
        print(f"Tolerance: {tol}, Timeout per solver: {timeout}s\n")

        # Compute analytical reference solution
        analytic_meta = SolverMeta()
        analytic_result = AnalyticSolver.integrate(poly, interval, meta=analytic_meta)
        print(f"[Analytic] Result = {analytic_result:.10f}\n")

        solvers = [
            (
                "Simple Numeric",
                NumericV1,
                NumericV1Meta(),
                {"n_samples": 1e4},
            ),
            (
                "Numeric Multi-Process",
                NumericMP,
                NumericMPMeta(),
                {
                    "n_samples": 1e4,
                    "batch_size": 2**13,
                },
            ),
            (
                "Recursive Numeric",
                NumericRecursive,
                NumericRecursiveMeta(),
                {
                    "tol": tol,
                    "max_depth": 999,
                },
            ),
            (
                "MonteCarlo",
                MonteCarlo,
                MonteCarloMeta(),
                {
                    "n_samples": 1e4,
                },
            ),
        ]

        for name, Solver, meta, kwargs in solvers:
            print(f"[{name}]")
            try:
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(timeout)

                start = time.time()
                result = Solver.integrate(poly, interval, meta=meta, **kwargs)
                signal.alarm(0)
                meta.execution_time = time.time() - start

                abs_error = abs(result - analytic_result)
                print(f"  Result       = {result:.10f}")
                print(f"  Error        = {abs_error:.2e}")
                print(f"  Time         = {meta.execution_time:.3f}s")
                if "n_samples" in kwargs:
                    print(f"  N Samples    = {kwargs['n_samples']:.2e}")
                if hasattr(meta, "depth"):
                    print(f"  Depth        = {meta.depth}")
                if abs_error <= tol:
                    continue

                # Retry loop
                while abs_error > tol:
                    time_left = timeout - (time.time() - start)
                    if time_left <= 0:
                        raise TimeoutException

                    for key in ["n_samples"]:
                        if key in kwargs:
                            kwargs[key] = int(kwargs[key] * 10)

                    signal.alarm(int(time_left))
                    retry_start = time.time()
                    result = Solver.integrate(poly, interval, meta=meta, **kwargs)
                    signal.alarm(0)
                    meta.execution_time += time.time() - retry_start

                    abs_error = abs(result - analytic_result)

                    print(f"  Retried:")
                    print(f"    Result     = {result:.10f}")
                    print(f"    Error      = {abs_error:.2e}")
                    print(f"    Time       = {meta.execution_time:.3f}s")
                    if "n_samples" in kwargs:
                        print(f"  N Samples    = {kwargs['n_samples']:.2e}")
                    if hasattr(meta, "depth"):
                        print(f"  Depth        = {meta.depth}")

                    if abs_error <= tol:
                        break

            except TimeoutException:
                print(f"  Timeout after {timeout}s")
                if meta.result is not None:
                    abs_error = abs(meta.result - analytic_result)
                    print(f"  Partial Result = {meta.result:.10f}")
                    print(f"  Error          = {abs_error:.2e}")
                    if hasattr(meta, "depth"):
                        print(f"    Depth        = {meta.depth}")
                else:
                    print("  No result available")
            print()
