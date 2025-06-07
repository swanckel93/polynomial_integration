import pytest
from src.polynomials.polynom import Polynom


@pytest.mark.unit
@pytest.mark.parametrize(
    "p1_str, p2_str, expected",
    [
        ("1 2 3", "4 5 6", [5, 7, 9]),  # same length
        ("1 2", "3", [4, 2]),  # different lengths
        ("0 0 0", "1 1 1", [1, 1, 1]),  # adding to zeros
        ("0", "1 2 3", [1, 2, 3]),  # empty string = error or zero
        ("1.5 2.5", "2.5 3.5", [4.0, 6.0]),  # floats
    ],
)
def test_polynom_addition(p1_str, p2_str, expected):
    p1 = Polynom(p1_str)
    p2 = Polynom(p2_str)
    result = p1 + p2
    assert result == expected
