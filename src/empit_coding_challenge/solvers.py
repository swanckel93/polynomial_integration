from .polynom import Polynom, IntegralSolver
from .utils import Stats
import random
import statistics
import math
import multiprocessing as mp
import os
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class SolverMeta:
    result: Optional[float] = None
    execution_time: Optional[float] = None


class AnalyticSolver(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: Optional[SolverMeta],
    ) -> float:
        "Analytic integration, assuming Offset == 0"
        F_0 = Polynom([0] + [val / (index + 1) for index, val in enumerate(polynom)])
        result = F_0(interval[1]) - F_0(interval[0])
        if meta is not None:
            meta.result = result
        return result


@dataclass
class NumericV1Meta(SolverMeta):
    n_subintervals: Optional[int] = None
    step_size: Optional[float] = None


class NumericV1(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: Optional[NumericV1Meta],
        n_samples: int,
    ) -> float:
        a, b = interval
        step_size = (b - a) / n_samples
        mid_points = [a + step_size * (i + 0.5) for i in range(int(n_samples))]
        Y = [polynom(x_mid) for x_mid in mid_points]
        result = sum(Y) * step_size
        if meta is not None:
            meta.result = result
            meta.n_subintervals = n_samples
            meta.step_size = step_size
        return result


@dataclass
class NumericRecursiveMeta(SolverMeta):
    tolerance_achieved: float | None = None
    depth: int | None = None
    is_within_tolerance: bool | None = None


class NumericRecursive(IntegralSolver):
    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        meta: Optional[NumericRecursiveMeta],
        tol=1e-6,
        max_depth=90,
        depth=0,
    ) -> float:
        a, b = interval
        mid = (a + b) / 2
        h = b - a

        # Midpoint rule estimates
        full = polynom(mid) * h
        left = polynom((a + mid) / 2) * (h / 2)
        right = polynom((mid + b) / 2) * (h / 2)
        refined = left + right

        # Stop condition
        if abs(refined - full) < tol or depth >= max_depth:
            return refined

        # Recursive subdivision
        left_result = cls.integrate(
            polynom,
            (a, mid),
            meta,
            tol=tol / 2,
            max_depth=max_depth,
            depth=depth + 1,
        )
        right_result = cls.integrate(
            polynom,
            (mid, b),
            meta,
            tol=tol / 2,
            max_depth=max_depth,
            depth=depth + 1,
        )

        result = left_result + right_result

        # Only update meta once, at root, this cost me 2 hours of my life...
        if meta is not None and depth == 0:
            meta.result = result
            meta.tolerance_achieved = abs(refined - full)
            meta.is_within_tolerance = abs(refined - full) <= tol
        if meta is not None:
            if meta.result is None:
                meta.result = 0.0
            meta.result += refined
            if meta.depth is None:
                meta.depth = 0
                meta.depth = max(meta.depth, depth)

        return result


@dataclass
class NumericMPMeta(SolverMeta):
    n_subintervals: Optional[int] = None
    n_samples: Optional[int] = None
    step_size: Optional[float] = None
    standard_error: Optional[float] = None


class NumericMP(IntegralSolver):
    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        meta: Optional[NumericMPMeta],
        n_samples: int,
        batch_size: int = 2**10,
    ) -> float:
        n_pools = os.cpu_count()
        if n_pools is None:
            n_pools = 4
        n_intervals = int(max(n_samples // batch_size, 1))
        vpoly = np.vectorize(polynom, otypes=[float])
        intervals = cls.partition_interval(interval, n_intervals)
        func_args = [(interval, batch_size, vpoly) for interval in intervals]
        with mp.Pool(n_pools) as pool:
            aggregations = pool.starmap(cls.aggregate_interval, func_args)
        total_n, total_mean, total_stdev = Stats.combine_integrals(aggregations)

        if meta is not None:
            meta.result = total_mean
            meta.n_subintervals = n_intervals
            meta.n_samples = total_n
            meta.step_size = (interval[1] - interval[0]) / total_n
            meta.standard_error = total_stdev / math.sqrt(total_n)

        return float(total_mean)

    @staticmethod
    def partition_interval(
        interval: tuple[float, float], n: int
    ) -> list[tuple[float, float]]:
        """
        retrieved from https://math.stackexchange.com/questions/3812011/dividing-an-interval-into-equal-length-segments-formula
        """
        a, b = interval
        delta = (b - a) / n
        intervals = []
        for i in range(n - 1):
            intervals.append((a + i * delta, a + (i + 1) * delta))
        intervals.append((a + (n - 1) * delta, b))
        return intervals

    @staticmethod
    def aggregate_interval(
        interval: tuple[float, float],
        batch_size: int,
        vfunc: Callable,
    ):
        a, b = interval
        step_size = (b - a) / batch_size
        mid_points = a + (np.arange(batch_size) + 0.5) * step_size
        Y = vfunc(mid_points) * step_size  # Multiply with step_size to get area
        result = Stats.compute_mean_M2(Y)

        return result


@dataclass
class MonteCarloMeta(SolverMeta):
    n_samples: Optional[int] = None
    standard_error: Optional[float] = None


class MonteCarlo(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        meta: Optional[MonteCarloMeta],
        n_samples: int,
        _seed: int = 42,
    ) -> float:
        random.seed(_seed)
        a, b = interval
        samples = [polynom(random.uniform(a, b)) for _ in range(int(n_samples))]
        _mean = sum(samples) / n_samples
        result = (b - a) * _mean
        standard_err = (b - a) * statistics.stdev(samples) / math.sqrt(n_samples)

        if meta is not None:
            meta.result = result
            meta.n_samples = n_samples
            meta.standard_error = float(standard_err)
        return result
