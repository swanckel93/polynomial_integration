from .polynom import Polynom, IntegralSolver
import random
import statistics
import math
import multiprocessing as mp
import os
import numpy as np
from dataclasses import dataclass


@dataclass
class SolverMeta:
    result: float | None = None
    execution_time: float | None = None


class AnalyticSolver(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: SolverMeta | None,
    ) -> float:
        "Analytic integration, assuming Offset == 0"
        F_0 = Polynom([0] + [val / (index + 1) for index, val in enumerate(polynom)])
        result = F_0(interval[1]) - F_0(interval[0])
        if meta is not None:
            meta.result = result
        return result


@dataclass
class NumericFixedStepMeta(SolverMeta):
    n_subintervals: int | None = None
    step_size: float | None = None


class NumericFixedStep(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: NumericFixedStepMeta | None,
        n=100,
    ) -> float:
        """
        Fixed step size.
        Numerical integration using the midpoint (rectangle) rule.

        Approximates the definite integral of a polynomial over [a, b] by dividing
        the interval into N subintervals and evaluating the function at the midpoint
        of each subinterval.

        Parameters:
        - polynom: The polynomial to integrate
        - interval: A tuple (a, b) specifying the bounds of integration
        - N: Number of subintervals (default is 100)

        Returns:
        - Approximate value of the integral

        Editor's Notes:
        Time Complexity: O(N)
        Limitations:
            - Decrease in accuracy for big interval and small N
            - Unnecessary effort in 'smooth' regions.
        """
        a, b = interval
        step_size = (b - a) / n
        mid_points = [a + step_size * (i + 0.5) for i in range(n)]
        Y = [polynom(x_mid) for x_mid in mid_points]
        result = sum(Y) * step_size
        if meta is not None:
            meta.result = result
            meta.n_subintervals = len(Y)
            meta.step_size
        return result


@dataclass
class NumericAdaptiveStepMeta(SolverMeta):
    tolerance_achieved: float | None = None
    depth: int | None = None
    is_within_tolerance: bool | None = None


class NumericAdaptiveStep(IntegralSolver):
    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        meta: NumericAdaptiveStepMeta | None,
        tol=1e-6,
        max_depth=90,
        depth=0,
    ) -> float:
        """
        Adaptive step size.

        Numerical integration using the midpoint (rectangle) rule with adaptive refinement.

        Approximates the definite integral of a polynomial over [a, b] by recursively subdividing
        the interval until the estimated error is below a given tolerance. Evaluates the function
        at midpoints of subintervals and adapts the step size to improve accuracy where needed.

        Parameters:
        - polynom: The polynomial to integrate
        - interval: A tuple (a, b) specifying the bounds of integration
        - tol: Desired error tolerance (default is 1e-6)
        - max_depth: Maximum recursion depth to prevent infinite subdivision

        Returns:
        - Approximate value of the integral

        Editor's Notes:
        Time Complexity: O(2^max_depth) worst, better in practice,
        Advantages:
            - Higher accuracy with fewer function evaluations
            - Efficient handling of intervals with varying function behavior
        Limitations:
            - Overhead due to recursion and error estimation
            - Requires careful choice of tolerance and max_depth

        For more info, see https://personal.math.ubc.ca/~CLP/CLP2/clp_2_ic/ap_adaptive.html#:~:text=%E2%80%9CAdaptive%20quadrature%E2%80%9D%20refers%20to%20a,easy%20to%20get%20good%20accuracy.
        """

        a, b = interval
        mid = (a + b) / 2
        h = b - a

        # Midpoint rule estimates
        full = polynom(mid) * h
        left = polynom((a + mid) / 2) * (h / 2)
        right = polynom((mid + b) / 2) * (h / 2)
        refined = left + right

        if abs(refined - full) < tol or depth >= max_depth:
            if meta is not None:
                meta.result = refined
                meta.tolerance_achieved = refined - full
                meta.depth = depth
                meta.is_within_tolerance = abs(refined - full) <= tol
            return refined
        else:
            return cls.integrate(
                polynom,
                (a, mid),
                meta=meta,
                tol=tol / 2,
                max_depth=max_depth,
                depth=depth + 1,
            ) + cls.integrate(
                polynom,
                (mid, b),
                meta=meta,
                tol=tol / 2,
                max_depth=max_depth,
                depth=depth + 1,
            )


@dataclass
class AdaptiveMonteCarloMeta(SolverMeta):
    n_samples: int | None = None
    standard_error: float | None = None
    is_within_standard_error: bool | None = None


class AdaptiveMonteCarlo(IntegralSolver):

    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: AdaptiveMonteCarloMeta | None,
        start_n=1e4,
        _seed=42,
        tol=1e-6,
        max_n=1e7,
    ) -> float:
        """
        Adaptive Monte Carlo integration.

        Numerical integration using a Monte Carlo method with adaptive sample size control.

        Approximates the definite integral of a polynomial over [a, b] by drawing uniformly distributed
        random samples, evaluating the polynomial at these points, and computing the mean value.
        The number of samples is increased until the estimated standard error of the result falls
        below a specified tolerance or a maximum number of samples is reached.

        Parameters:
        - polynom: The polynomial to integrate.
        - interval: A tuple (a, b) specifying the bounds of integration.
        - start_n: Initial number of random samples (default is 10,000).
        - _seed: Random seed for reproducibility (default is 42).
        - tol: Desired standard error tolerance for the result (default is 1e-6).
        - max_n: Maximum total number of samples allowed (default is 10 million).

        Returns:
        - An AdaptiveMonteCarloResult object containing:
            - result: Estimated value of the integral.
            - standard_err: Estimated standard error of the result.
            - samples_used: Total number of samples used when stopping.
            - tolerance_reached: Whether the desired tolerance was achieved.
            - execution_time: Optional field to hold time taken for integration.

        Editor's Notes:
        Time Complexity: O(n), where n is the number of samples used.
        Advantages:
            - Simple and robust method, even for noisy or non-smooth functions.
            - Naturally parallelizable (e.g., sample evaluation).
        Limitations:
            - Convergence is slow compared to deterministic methods.
            - Not ideal for low-tolerance precision unless many samples are used.
        """
        random.seed(_seed)
        a, b = interval
        result = 1e6
        standard_err = 1e6
        curr_n = start_n

        while standard_err > tol and curr_n < max_n:
            samples = [polynom(random.uniform(a, b)) for _ in range(int(curr_n))]
            _mean = sum(samples) / curr_n
            result = (b - a) * _mean
            standard_err = (b - a) * statistics.stdev(samples) / math.sqrt(curr_n)
            curr_n *= 10
        if meta is not None:
            meta.result = result
            meta.n_samples = len(samples)
            meta.standard_error = float(standard_err)
            meta.is_within_standard_error = bool(standard_err <= tol)

        return result


@dataclass
class AdaptiveMonteCarloMPMeta(SolverMeta):
    cpu_count: int | None = None
    n_workers: int | None = None
    n_samples: int | None = None
    standard_error: float | None = None
    is_within_standard_error: bool | None = None


class AdaptiveMonteCarloMP(IntegralSolver):
    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        meta: AdaptiveMonteCarloMPMeta | None,
        start_n=1e4,
        _seed=42,
        tol=1e-6,
        max_n=1e7,
        n_workers: int | None = None,
    ) -> float:
        """
        https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
        https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap
        """
        cpu_count = os.cpu_count()
        assert cpu_count is not None
        if n_workers is None:

            n_workers = int(
                cpu_count * 0.75
            )  # dont want to use all cpus, else WSL may crash..
        else:
            n_workers = min(n_workers, int(cpu_count * 0.75))
        intervals = cls.partition_interval(interval, n_workers)
        standard_err = 1e6
        result = 1e6
        curr_n = start_n
        while standard_err > tol and curr_n < max_n:
            f_args = [
                (polynom, _interval, int(curr_n // n_workers))
                for _interval in intervals
            ]
            with mp.Pool(processes=n_workers) as pool:
                samples = pool.starmap(cls.generate_samples, f_args)

            samples = np.array(samples).flatten()
            a, b = interval
            result = (b - a) * np.mean(samples)
            standard_err = (b - a) * np.std(samples) / np.sqrt(curr_n)
            curr_n *= 10
        if meta is not None:
            meta.result = float(result)
            meta.cpu_count = cpu_count
            meta.n_workers = n_workers
            meta.n_samples = samples.shape[0]
            meta.standard_error = float(standard_err)
            meta.is_within_standard_error = bool(standard_err <= tol)

        return float(result)

    @staticmethod
    def generate_samples(
        polynom: Polynom, interval: tuple[float, float], n: int
    ) -> np.ndarray:
        """
        Reason for the use of vectorize can be found here.
        https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
        """
        a, b = interval
        x = np.random.uniform(a, b, size=n)
        v_polynom = np.vectorize(polynom, otypes=[float])
        return v_polynom(x)

    @staticmethod
    def partition_interval(
        interval: tuple[float, float], n: int
    ) -> list[tuple[float, float]]:
        """
        retrieved from https://math.stackexchange.com/questions/3812011/dividing-an-interval-into-equal-length-segments-formula
        """
        a, b = interval
        delta = (b - a) / n
        print(delta)
        intervals = []
        for i in range(n - 1):
            intervals.append((a + i * delta, a + (i + 1) * delta))
        intervals.append((a + (n - 1) * delta, b))
        return intervals
