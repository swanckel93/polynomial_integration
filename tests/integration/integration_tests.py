import pytest
from src.polynomials.polynom import Polynom


@pytest.mark.integration
@pytest.mark.skip(reason="Not implemented yet")
@pytest.mark.parametrize(
    "got, expected",
    [("dogs", "cats"), ("apple" "bananas")],
)
def test_cli(got, expected):
    assert got == expected
