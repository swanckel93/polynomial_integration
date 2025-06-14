import numpy as np


class Stats:
    @staticmethod
    def compute_mean_M2(data: np.ndarray) -> tuple[int, float, float]:
        n = data.shape[0]
        mean = float(np.mean(data))
        M2 = float(np.sum((data - mean) ** 2))
        return n, mean, M2

    @staticmethod
    def combine_stats(
        stats_list: list[tuple[int, float, float]],
    ) -> tuple[int, float, float]:
        total_n = 0
        total_mean = 0
        total_M2 = 0

        for n_i, mean_i, M2_i in stats_list:
            if total_n == 0:
                total_mean = mean_i
                total_M2 = M2_i
                total_n = n_i
            else:
                delta = mean_i - total_mean
                new_n = total_n + n_i
                total_mean += delta * (n_i / new_n)
                total_M2 += M2_i + delta**2 * total_n * n_i / new_n
                total_n = new_n

        variance = total_M2 / total_n
        std_dev = variance**0.5
        return total_n, total_mean, std_dev

    @staticmethod
    def combine_integrals(
        stats_list: list[tuple[int, float, float]],
    ) -> tuple[int, float, float]:
        """
        Combine batch results into total integral, mean, and stddev.
        Each element in stats_list is (n, mean, M2) for one batch of areas.
        """
        total_n = 0
        weighted_sum = 0
        M2_sum = 0

        for n_i, mean_i, M2_i in stats_list:
            total_n += n_i
            weighted_sum += n_i * mean_i
            M2_sum += M2_i  # This may need a more accurate combining, but good for now

        std_dev = (M2_sum / total_n) ** 0.5
        return total_n, weighted_sum, std_dev
