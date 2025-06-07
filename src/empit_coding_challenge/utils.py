import timeit
import time
from inspect import signature
import numpy as np
from .solvers import Polynom, IntegralSolver


class Stats:
    @staticmethod
    def compute_mean_M2(data: np.ndarray) -> tuple[int, float, float]:
        n = data.shape[0]
        mean = float(np.mean(data))
        M2 = float(np.sum((data - mean) ** 2))
        return n, mean, M2

    @staticmethod
    def combine_stats(
        stats_list: list[tuple[int, float, float]],
    ) -> tuple[int, float, float]:
        total_n = 0
        total_mean = 0
        total_M2 = 0

        for n_i, mean_i, M2_i in stats_list:
            if total_n == 0:
                total_mean = mean_i
                total_M2 = M2_i
                total_n = n_i
            else:
                delta = mean_i - total_mean
                new_n = total_n + n_i
                total_mean += delta * (n_i / new_n)
                total_M2 += M2_i + delta**2 * total_n * n_i / new_n
                total_n = new_n

        variance = total_M2 / total_n
        std_dev = variance**0.5
        return total_n, total_mean, std_dev

    @staticmethod
    def combine_integrals(
        stats_list: list[tuple[int, float, float]],
    ) -> tuple[int, float, float]:
        """
        Combine batch results into total integral, mean, and stddev.
        Each element in stats_list is (n, mean, M2) for one batch of areas.
        """
        total_n = 0
        weighted_sum = 0
        M2_sum = 0

        for n_i, mean_i, M2_i in stats_list:
            total_n += n_i
            weighted_sum += n_i * mean_i
            M2_sum += M2_i  # This may need a more accurate combining, but good for now

        std_dev = (M2_sum / total_n) ** 0.5
        return total_n, weighted_sum, std_dev


class Timer:
    @staticmethod
    def __extract_meta_arg__(func, args, kwargs):
        """
        Attempts to retrieve the `meta` argument (positional or keyword) from the provided args and kwargs.
        """
        if "meta" in kwargs:
            return kwargs["meta"]

        sig = signature(func)
        params = list(sig.parameters)
        if "meta" in params:
            meta_index = params.index("meta")
            if meta_index < len(args):
                return args[meta_index]
        return None

    @staticmethod
    def timeit(func):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            meta = Timer.__extract_meta_arg__(func, args, kwargs)
            if meta is not None and hasattr(meta, "execution_time"):
                meta.execution_time = end - start
            print(f"{func.__name__} took {end - start:.6f} seconds")
            return result

        return wrapper


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
    ) -> tuple[float, float, float, float, bool]:
        start_time = timeit.default_timer()
        elapsed = 0.0
        curr_n = solver_kwargs["n_subintervals"]
        result = None
        is_success = False
        error = None
        while elapsed < timeout:
            result = polynom.integrate(interval, solver, **solver_kwargs)
            error = ErrorCalculations.get_relative_error(analytic_solution, result)
            elapsed = timeit.default_timer() - start_time
            if error < tolerance:
                is_success = True
                break
            print(" " * 4 + f"Partial Result    = {result:.6f}")
            print(" " * 4 + f"Partial Error     = {error:.2e}%")
            print(" " * 4 + f"Number Of Samples = {curr_n:.1e}")
            print(" " * 4 + f"Time Elapsed      = {elapsed:.3f}s")
            print()
            curr_n *= 10
            if "n_samples" in solver.expected_kwargs:
                solver_kwargs["n_samples"] = curr_n
            if "n_subintervals" in solver.expected_kwargs:
                solver_kwargs["n_subintervals"] = curr_n

        assert result is not None
        assert error is not None

        return result, error, elapsed, curr_n, is_success


class ErrorCalculations:
    @staticmethod
    def get_relative_error(target_value: float, current_value: float) -> float:
        assert target_value is not None
        return abs(current_value - target_value) / target_value * 100
