import pytest
from src.polynomials.polynom import Polynom


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("1 2 3", [1.0, 2.0, 3.0]),  # normal case
        ("  4.5   -2.1  0  ", [4.5, -2.1, 0.0]),  # extra spacies and floats
        ("0", [0.0]),  # single zero
        ("1e2 3.14", [100.0, 3.14]),  # scientific notation
    ],
)
def test_parse_from_string_valid(input_str, expected):
    result = Polynom.parse_from_string(input_str)
    assert result == expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_str",
    [
        "",  # empty string
        "   ",  # just whitespace
        "abc def",  # non-numeric input
        "1 2 three",  # mixed bad input
        None,  # completely invalid type
    ],
)
def test_parse_from_string_invalid(input_str):
    with pytest.raises((ValueError, AttributeError)):
        Polynom.parse_from_string(input_str)
