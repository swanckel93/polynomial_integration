import pytest
import numpy as np
from src.empit_coding_challenge.utils import (
    Stats,
)


def compute_mean_M2(data):
    n = len(data)
    mean = np.mean(data)
    M2 = np.sum((data - mean) ** 2)
    return n, mean, M2


@pytest.mark.parametrize("num_groups", [2, 3])
def test_combine_stats(num_groups):
    np.random.seed(42)
    data_sets = [
        np.random.normal(loc=i * 2, scale=1.0, size=1000) for i in range(num_groups)
    ]

    # Individual stats
    stats_list = [compute_mean_M2(data) for data in data_sets]

    # Combined via Stats.combine_stats
    combined_mean, combined_std = Stats.combine_stats(stats_list)

    # Ground truth by joining all data
    full_data = np.concatenate(data_sets)
    expected_mean = np.mean(full_data)
    expected_std = np.std(full_data, ddof=0)

    # Assert close within tolerance
    assert np.isclose(combined_mean, expected_mean, atol=1e-10)
    assert np.isclose(combined_std, expected_std, atol=1e-10)
