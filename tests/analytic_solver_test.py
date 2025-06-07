import pytest
from src.polynomials.polynom import Polynom
from src.polynomials.solvers import AnalyticSolver


@pytest.mark.parametrize(
    "coeffs, interval, expected",
    [
        ([0], (0, 1), 0),  # Zero polynomial, integral should be zero
        ([1], (0, 1), 1),  # Constant polynomial f(x)=1, integral over [0,1] is 1
        ([0, 1], (0, 1), 0.5),  # f(x) = x, integral over [0,1] = 0.5
        ([1, 1], (0, 1), 1.5),  # f(x) = 1 + x, integral over [0,1] = 1 + 0.5 = 1.5
        ([3, 0, 2], (1, 2), 23.0 / 3),  # f(x) = 3 + 0*x + 2x^2, integral over [1,2]
    ],
)
def test_integrate_analyticsolver(coeffs, interval, expected):
    poly = Polynom(coeffs)
    result = AnalyticSolver.integrate(poly, interval)
    assert pytest.approx(result, rel=1e-9) == expected
