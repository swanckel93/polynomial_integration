import sys
from src.empit_coding_challenge.solvers import (
    NumericFixedStep,
    NumericAdaptiveStep,
    AnalyticSolver,
    AdaptiveMonteCarlo,
    AdaptiveMonteCarloMP,
    AdaptiveMonteCarloMPMeta,
)
from src.empit_coding_challenge.polynom import Polynom
from tests.utils import timeit
import random

sys.setrecursionlimit(100)  # Set recursion limit to 100


@timeit
def adaptive_monte_carlo():
    p = Polynom([0, 0, 1])
    interval = (0.0, 1.0)
    start_n = int(1e5)
    max_n = int(1e6)
    meta = AdaptiveMonteCarloMPMeta()
    res = AdaptiveMonteCarloMP.integrate(
        p,
        interval,
        meta,
        start_n=start_n,
        max_n=max_n,
    )

    return res


@timeit
def adaptive_monte_carlo_mp(meta):
    p = Polynom([0, 0, 1])
    interval = (0.0, 1.0)
    tolerance = 1e-6
    start_n = int(1e4)
    max_n = int(1e9)
    n_workers = 1000
    res = AdaptiveMonteCarloMP.integrate(
        p,
        interval,
        meta,
        tol=tolerance,
        start_n=start_n,
        max_n=max_n,
        n_workers=n_workers,
    )

    return res


# analytic_result = AnalyticSolver.integrate(p, interval)
# numeric_result_fixed_step = NumericFixedStep.integrate(p, interval, n=1000)
# numeric_result_adaptive_step = NumericAdaptiveStep.integrate(p, interval, max_depth=90)
# monte_carlo_result = AdaptiveMonteCarlo.integrate(p, interval)
# err_fixed = abs(analytic_result - numeric_result_fixed_step)
# err_adaptive = abs(analytic_result - numeric_result_adaptive_step)
# err_monte_carlo = abs(analytic_result - monte_carlo_result)
# print(f"analytic_result: {analytic_result}")
# print(f"numeric_result_fixed_step: {numeric_result_fixed_step}")
# print(f"monte_carlo_result: {monte_carlo_result}")
# print(f"Error fixed: {err_fixed}")
# print(f"Error adaptive: {err_adaptive}")
# print(f"Error montecarlo: {err_monte_carlo}")
if __name__ == "__main__":
    import os

    p = Polynom([0, 0, 1])
    interval = (0.0, 1.0)
    start_n = int(1e5)
    meta = AdaptiveMonteCarloMPMeta()
    result = adaptive_monte_carlo_mp(meta)
    print(result)
    print(meta)
