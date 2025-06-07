import pytest
from src.polynomials.polynom import Polynom


@pytest.mark.unit
@pytest.mark.parametrize(
    "p1_str, p2_str, expected",
    [
        ("1 2", "1 2", [1, 4, 4]),  # (1 + 2x) * (1 + 2x)
        ("2", "3", [6]),  # constant * constant
        ("1 0 1", "1", [1, 0, 1]),  # poly * 1
        ("0", "5 6 7", [0, 0, 0]),  # zero * poly
        ("1 1", "1 -1", [1, 0, -1]),  # (1 + x)(1 - x) = 1 - x^2
    ],
)
def test_polynom_multiplication(p1_str, p2_str, expected):
    p1 = Polynom(p1_str)
    p2 = Polynom(p2_str)
    result = p1 * p2
    assert result == expected
