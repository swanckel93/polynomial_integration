import pytest
from src.empit_coding_challenge.polynom import Polynom


@pytest.mark.parametrize(
    "p1_str, p2_str, expected",
    [
        ("5 4 3", "1 2 3", [4, 2, 0]),  # same length
        ("1 2", "3", [-2, 2]),  # different lengths
        ("0 0 0", "1 1 1", [-1, -1, -1]),  # subtracting from zeros
        ("4 5 6", "4 5 6", [0, 0, 0]),  # subtraction to zero
        ("1.5 2.5", "0.5 1.5", [1.0, 1.0]),  # floats
    ],
)
def test_polynom_subtraction(p1_str, p2_str, expected):
    p1 = Polynom(p1_str)
    p2 = Polynom(p2_str)
    result = p1 - p2
    assert result == expected
