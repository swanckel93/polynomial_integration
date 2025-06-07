from .polynom import Polynom, IntegralSolver
import random
import statistics
import math
import multiprocessing as mp
import os
import numpy as np
from typing import Callable
from strenum import StrEnum
import enum


class AnalyticSolver(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
    ) -> float:
        "Analytic integration, assuming Offset == 0"
        F_0 = Polynom([0] + [val / (index + 1) for index, val in enumerate(polynom)])
        result = F_0(interval[1]) - F_0(interval[0])
        return result


class NewtonCoteSolver(IntegralSolver):
    expected_kwargs = {"n_subintervals"}

    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        n_subintervals: int,
    ) -> float:
        a, b = interval
        h = (b - a) / n_subintervals
        total = 0.0

        for i in range(n_subintervals):
            sub_a = a + i * h
            sub_b = sub_a + h
            total += cls._apply_rule(polynom, sub_a, sub_b)

        return total

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        raise NotImplementedError("Subclasses must implement _apply_rule()")


class MidpointSolver(NewtonCoteSolver):
    description = "∫[a,b] f(x) dx ≈ (b - a) * f((a + b) / 2)"

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        mid = (a + b) / 2
        return (b - a) * polynom(mid)


class TrapezoidalSolver(NewtonCoteSolver):
    description = "∫[a,b] f(x) dx ≈ (b - a) / 2 * (f(a) + f(b))"

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        return (b - a) * (polynom(a) + polynom(b)) / 2


class SimpsonSolver(NewtonCoteSolver):
    description = "∫[a,b] f(x) dx ≈ (b - a) / 6 * (f(a) + 4f((a + b)/2) + f(b))"

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        mid = (a + b) / 2
        return (b - a) / 6 * (polynom(a) + 4 * polynom(mid) + polynom(b))


class Simpson38Solver(NewtonCoteSolver):
    description = """
    Let h = (b - a) / 3,
    ∫[a,b] f(x) dx ≈ (b - a) / 8 * [f(a) + 3f(a + h) + 3f(a + 2h) + f(b)]
    """

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        h = (b - a) / 3
        x1 = a + h
        x2 = a + 2 * h
        return (
            (b - a) * (polynom(a) + 3 * polynom(x1) + 3 * polynom(x2) + polynom(b)) / 8
        )


class BooleSolver(NewtonCoteSolver):
    description = """
    Let h = (b - a) / 4,
    ∫[a,b] f(x) dx ≈ (b - a) / 90 * [7f(a) + 32f(a + h) + 12f(a + 2h) + 32f(a + 3h) + 7f(b)]
    """

    @staticmethod
    def _apply_rule(polynom: Polynom, a: float, b: float) -> float:
        h = (b - a) / 4
        x1 = a + h
        x2 = a + 2 * h
        x3 = a + 3 * h
        return (
            (b - a)
            / 90
            * (
                7 * polynom(a)
                + 32 * polynom(x1)
                + 12 * polynom(x2)
                + 32 * polynom(x3)
                + 7 * polynom(b)
            )
        )


# Recursive function works. But not included in the CLI. properties are too different and would make the cli hacky.
class MidpointRecursive(IntegralSolver):
    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        tol=1e-6,
        max_depth=90,
        depth=0,
        max_reached_depth: list[int] | None = None,
    ) -> float:
        if max_reached_depth is None:
            max_reached_depth = [0]

        max_reached_depth[0] = max(max_reached_depth[0], depth)
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
            tol=tol / 2,
            max_depth=max_depth,
            depth=depth + 1,
        )
        right_result = cls.integrate(
            polynom,
            (mid, b),
            tol=tol / 2,
            max_depth=max_depth,
            depth=depth + 1,
        )

        result = left_result + right_result

        return result


class NewtonCoteMP(IntegralSolver):
    expected_kwargs = {"n_subintervals", "batch_size"}

    @classmethod
    def integrate(
        cls,
        polynom: Polynom,
        interval: tuple[float, float],
        n_subintervals: int,
        batch_size: int = 64,  # default batch size
    ) -> float:
        intervals = cls.partition_interval(interval, n_subintervals)

        batches = [
            intervals[i : i + batch_size] for i in range(0, len(intervals), batch_size)
        ]

        vpoly = np.vectorize(polynom, otypes=[float])
        args = [(batch, vpoly) for batch in batches]
        if os.cpu_count() is None:
            n_pools = 4
        else:
            cpu_count = os.cpu_count()
            n_pools = int(cpu_count * 0.75)  # type: ignore

        with mp.Pool(n_pools) as pool:
            results = pool.starmap(cls.integrate_batch, args)

        return sum(results)

    @staticmethod
    def partition_interval(
        interval: tuple[float, float], n: int
    ) -> list[tuple[float, float]]:
        a, b = interval
        delta = (b - a) / n
        return [(a + i * delta, a + (i + 1) * delta) for i in range(n)]

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        raise NotImplementedError("Subclasses must override integrate_interval.")

    @classmethod
    def integrate_batch(
        cls,
        intervals: list[tuple[float, float]],
        vpoly: Callable[[float], float],
    ) -> float:
        return sum(cls.integrate_interval(interval, vpoly) for interval in intervals)


class MidpointMP(NewtonCoteMP):
    description = """ 
    apply (b - a) * f((a + b)/2) 
    over subintervals
    """

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        a, b = interval
        mid = (a + b) / 2
        return (b - a) * vpoly(mid)


class TrapezoidalMP(NewtonCoteMP):
    description = """ 
    apply (b - a) * f((a + b)/2) 
    over subintervals
    """

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        a, b = interval
        return (b - a) * (vpoly(a) + vpoly(b)) / 2


class SimpsonMP(NewtonCoteMP):
    description = """ 
    apply (b - a) * f((a + b)/2) 
    over subintervals
    """

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        a, b = interval
        m = (a + b) / 2
        return (b - a) * (vpoly(a) + 4 * vpoly(m) + vpoly(b)) / 6


class Simpson38MP(NewtonCoteMP):
    description = """ 
    apply (b - a) * f((a + b)/2) 
    over subintervals
    """

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        a, b = interval
        h = (b - a) / 3
        x1, x2 = a + h, a + 2 * h
        return (b - a) * (vpoly(a) + 3 * vpoly(x1) + 3 * vpoly(x2) + vpoly(b)) / 8


class BooleMP(NewtonCoteMP):
    description = """ 
    apply (b - a) * f((a + b)/2) 
    over subintervals
    """

    @staticmethod
    def integrate_interval(
        interval: tuple[float, float], vpoly: Callable[[float], float]
    ) -> float:
        a, b = interval
        h = (b - a) / 4
        x1 = a + h
        x2 = a + 2 * h
        x3 = a + 3 * h
        return (
            (b - a)
            * (
                7 * vpoly(a)
                + 32 * vpoly(x1)
                + 12 * vpoly(x2)
                + 32 * vpoly(x3)
                + 7 * vpoly(b)
            )
            / 90
        )


class MonteCarlo(IntegralSolver):
    expected_kwargs = {"n_samples", "_seed"}
    description = """ 
    Estimates ∫[a,b] f(x) dx by uniformly sampling points in [a, b].
    Computes the average of f(x) over n_samples random x-values, scaled by (b - a).
    """

    @staticmethod
    def integrate(
        polynom: Polynom,
        interval: tuple[float, float],
        n_samples: int,
        _seed: int = 42,
    ) -> float:
        random.seed(_seed)
        a, b = interval
        samples = [polynom(random.uniform(a, b)) for _ in range(int(n_samples))]
        _mean = sum(samples) / n_samples
        result = (b - a) * _mean
        standard_err = (b - a) * statistics.stdev(samples) / math.sqrt(n_samples)
        return result


# There is a way of doing this with inheritance properties, but I want it to be explicitly written for better readability.
SOLVER_GROUP_MAP = {
    NewtonCoteSolver: [
        MidpointSolver,
        TrapezoidalSolver,
        SimpsonSolver,
        Simpson38Solver,
        BooleSolver,
    ],
    NewtonCoteMP: [
        MidpointMP,
        TrapezoidalMP,
        SimpsonMP,
        Simpson38MP,
        BooleMP,
    ],
    MonteCarlo: [
        MonteCarlo,
    ],
}


class SolverName(StrEnum):
    MIDPOINT = enum.auto()
    TRAPEZ = enum.auto()
    SIMPSON = enum.auto()
    SIMPSON38 = enum.auto()
    BOOLE = enum.auto()
    MIDPOINT_MP = "MIDPOINT-MP"
    TRAPEZ_MP = "TRAPEZ-MP"
    SIMPSON_MP = "SIMPSON-MP"
    SIMPSON38_MP = "SIMPSON38-MP"
    BOOLE_MP = "BOOLE-MP"
    MONTECARLO = "MONTECARLO"


SOLVER_MAP = {
    SolverName.MIDPOINT: MidpointSolver,
    SolverName.TRAPEZ: TrapezoidalSolver,
    SolverName.SIMPSON: SimpsonSolver,
    SolverName.SIMPSON38: Simpson38Solver,
    SolverName.BOOLE: BooleSolver,
    SolverName.MIDPOINT_MP: MidpointMP,
    SolverName.TRAPEZ_MP: TrapezoidalMP,
    SolverName.SIMPSON_MP: SimpsonMP,
    SolverName.SIMPSON38_MP: Simpson38MP,
    SolverName.BOOLE_MP: BooleMP,
    SolverName.MONTECARLO: MonteCarlo,
}
