import timeit
from polynomials.polynom import Polynom, IntegralSolver
from polynomials.data_models import PartialResult
from .errors_calc import ErrorCalculations


class AdaptiveIntegration:
    @staticmethod
    def refine(
        polynom: Polynom,
        solver: IntegralSolver,
        interval: tuple[float, float],
        analytic_solution: float,
        tolerance: float,
        timeout: float,
        solver_kwargs: dict,
    ) -> tuple[list[PartialResult], bool]:
        start_time = timeit.default_timer()
        elapsed = 0.0
        curr_n = solver_kwargs.get("n_subintervals") or solver_kwargs.get("n_samples")

        assert curr_n is not None

        result = None
        is_success = False
        error = None
        partial_results = []

        while elapsed < timeout:
            result = polynom.integrate(interval, solver, **solver_kwargs)
            error = ErrorCalculations.get_absolute_error(analytic_solution, result)
            elapsed = timeit.default_timer() - start_time

            # Capture partial result
            partial_result = PartialResult(
                result=result, error=error, samples=int(curr_n), time_elapsed=elapsed
            )
            partial_results.append(partial_result)

            # Display progress (keep existing output for user feedback)
            print(" " * 4 + f"Partial Result    = {result:.6f}")
            print(" " * 4 + f"Partial Error     = {error:.2e}%")
            print(" " * 4 + f"Number Of Samples = {curr_n:.1e}")
            print(" " * 4 + f"Time Elapsed      = {elapsed:.3f}s")
            print()

            if error < tolerance:
                is_success = True
                break

            curr_n *= 10

            if "n_samples" in solver.expected_kwargs:
                solver_kwargs["n_samples"] = curr_n
            if "n_subintervals" in solver.expected_kwargs:
                solver_kwargs["n_subintervals"] = curr_n

        return partial_results, is_success
