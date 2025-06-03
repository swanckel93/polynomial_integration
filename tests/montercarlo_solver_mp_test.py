import pytest
from src.empit_coding_challenge.solvers import AdaptiveMonteCarloMP
from src.empit_coding_challenge.polynom import Polynom


@pytest.mark.parametrize(
    "coeffs, interval, n",
    [
        ([1], (0, 1), 100),
        ([0, 1], (-1, 1), 500),
        ([2, -3, 1], (0, 2), 1000),
    ],
)
def test_generate_samples(coeffs, interval, n):
    p = Polynom(coeffs)
    samples = AdaptiveMonteCarloMP.generate_samples(p, interval, n)

    assert len(samples) == n
    for val in samples:
        assert isinstance(val, float)


@pytest.mark.parametrize(
    "interval, n, expected_intervals",
    [
        ((0, 1), 2, [(0.0, 0.5), (0.5, 1.0)]),
        ((-1, 1), 4, [(-1.0, -0.5), (-0.5, 0.0), (0.0, 0.5), (0.5, 1.0)]),
        ((0, 10), 5, [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0)]),
    ],
)
def test_partition_interval(interval, n, expected_intervals):
    intervals = AdaptiveMonteCarloMP.partition_interval(interval, n)

    assert intervals == expected_intervals
    assert len(intervals) == n
    for start, end in intervals:
        assert end > start
