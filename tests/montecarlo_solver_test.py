import pytest
from src.empit_coding_challenge.polynom import Polynom
from src.empit_coding_challenge.solvers import (
    AdaptiveMonteCarlo,
    AdaptiveMonteCarloResult,
    AnalyticSolver,
)
from tests.shared.bounded_integrals import TO_INTEGRATE


@pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
def test_adaptive_montecarlo_accuracy(coeffs, interval):
    p = Polynom(coeffs)
    expected = AnalyticSolver.integrate(p, interval)

    result: AdaptiveMonteCarloResult = AdaptiveMonteCarlo.integrate(
        p, interval, tol=1e-1, start_n=1e4, max_n=1e7
    )
    confidence = 1  # how many std deviations we deem acceptable.

    assert (
        result.tolerance_reached
    ), "Monte Carlo method did not reach desired tolerance"
    assert abs(result.result - expected) <= confidence * result.standard_err, (
        f"Monte Carlo result {result.result} deviates too far from expected {expected} "
        f"with standard error {result.standard_err}"
    )
