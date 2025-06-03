import pytest
from src.empit_coding_challenge.polynom import Polynom
from src.empit_coding_challenge.solvers import (
    AnalyticSolver,
    NumericFixedStep,
    NumericAdaptiveStep,
)

TO_INTEGRATE = [
    ([1], (0, 1)),  # ∫1 dx = 1
    ([0, 1], (0, 2)),  # ∫x dx = 2
    ([1, 0, 1], (0, 1)),  # ∫(1 + x^2) dx = 4/3
    ([2, -3, 1], (-1, 1)),  # ∫(2 - 3x + x^2) dx
    ([5, 4, 3, 2, 1], (0, 1)),  # Higher-degree test
]


@pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
def test_numeric_fixed_vs_analytic(coeffs, interval):
    p = Polynom(coeffs)
    expected = AnalyticSolver.integrate(p, interval)
    result = NumericFixedStep.integrate(p, interval, N=1000)
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
