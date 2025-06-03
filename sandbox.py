import sys
from src.empit_coding_challenge.solvers import (
    NumericFixedStep,
    NumericAdaptiveStep,
    AnalyticSolver,
)
from src.empit_coding_challenge.polynom import Polynom

sys.setrecursionlimit(100)  # Set recursion limit to 100

p = Polynom([0, 0, 1])
interval = (0.0, 1.0)
analytic_result = AnalyticSolver.integrate(p, interval)
numeric_resul_fixed_step = NumericFixedStep.integrate(p, interval, N=1000)
numeric_result_adaptive_step = NumericAdaptiveStep.integrate(p, interval, max_depth=90)
err_fixed = abs(analytic_result - numeric_resul_fixed_step)
err_adaptive = abs(analytic_result - numeric_result_adaptive_step)
print(f"analytic_result: {analytic_result}")
print(f"numeric_result_fixed_step: {numeric_resul_fixed_step}")
print(f"Error fixed: {err_fixed}")
print(f"Error adaptive: {err_adaptive}")
