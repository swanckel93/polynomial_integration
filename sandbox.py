import sys
from src.empit_coding_challenge.solvers import (
    NumericV1,
    NumericRecursive,
    AnalyticSolver,
    MonteCarlo,
    NumericMPMeta,
    NumericMP,
)
from src.empit_coding_challenge.polynom import Polynom
from src.empit_coding_challenge.utils import Timer
import random

sys.setrecursionlimit(100)  # Set recursion limit to 100


@Timer.timeit
def numeric_fixed_mp(meta):
    p = Polynom([0, 0, 1])
    interval = (0.0, 1.0)
    start_n = int(1e5)
    max_n = int(1e10)
    batch_size = 2**10
    tolerance = 1e-6
    meta = meta
    res = NumericMP.integrate(
        p,
        interval,
        meta,
        start_n=start_n,
        max_n=max_n,
        batch_size=batch_size,
    )

    return res


if __name__ == "__main__":
    import os

    meta = NumericMPMeta()
    result = numeric_fixed_mp(meta)
    print(result)
    print(meta)
