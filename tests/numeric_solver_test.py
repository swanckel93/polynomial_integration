import pytest
from src.empit_coding_challenge.polynom import Polynom
from src.empit_coding_challenge.solvers import (
    AnalyticSolver,
    NumericFixedStep,
    NumericAdaptiveStep,
)
from tests.shared.bounded_integrals import TO_INTEGRATE


@pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
def test_numeric_fixed_vs_analytic(coeffs, interval):
    p = Polynom(coeffs)
    expected = AnalyticSolver.integrate(p, interval)
    result = NumericFixedStep.integrate(p, interval, n=1000)
    assert result == pytest.approx(
        expected, rel=1e-6
    ), f"Fixed step inaccurate: {result} vs {expected}"


@pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
def test_numeric_adaptive_vs_analytic(coeffs, interval):
    p = Polynom(coeffs)
    tol = 1e-6
    expected = AnalyticSolver.integrate(p, interval)
    result = NumericAdaptiveStep.integrate(p, interval, tol=tol)
    assert (
        abs(result - expected) < tol
    ), f"Adaptive did not respect tol={tol}: {result} vs {expected}"
