import pytest
from src.polynomials.solvers import *
from ..shared.bounded_integrals import TO_INTEGRATE

# Test tolerance for numerical methods
TOLERANCE = 1e-3
TIGHT_TOLERANCE = 1e-6
MONTE_CARLO_TOLERANCE = 1e-1  # Monte Carlo needs looser tolerance


class TestNewtonCoteSolvers:
    """Test all Newton-Cotes solvers against analytic solutions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_midpoint_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = MidpointSolver.integrate(
            polynom, interval, n_subintervals=1000
        )

        assert abs(numerical_result - analytic_result) < TOLERANCE, (
            f"Midpoint solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_trapezoidal_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = TrapezoidalSolver.integrate(
            polynom, interval, n_subintervals=1000
        )

        assert abs(numerical_result - analytic_result) < TOLERANCE, (
            f"Trapezoidal solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_simpson_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = SimpsonSolver.integrate(
            polynom, interval, n_subintervals=100
        )

        # Simpson's rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"Simpson solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_simpson38_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = Simpson38Solver.integrate(
            polynom, interval, n_subintervals=99
        )

        # Simpson 3/8 rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"Simpson 3/8 solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_boole_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = BooleSolver.integrate(polynom, interval, n_subintervals=100)

        # Boole's rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"Boole solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )


class TestNewtonCoteMPSolvers:
    """Test all multiprocessing Newton-Cotes solvers against analytic solutions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_midpoint_mp_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = MidpointMP.integrate(
            polynom, interval, n_subintervals=1000, batch_size=64
        )

        assert abs(numerical_result - analytic_result) < TOLERANCE, (
            f"MidpointMP solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_trapezoidal_mp_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = TrapezoidalMP.integrate(
            polynom, interval, n_subintervals=1000, batch_size=64
        )

        assert abs(numerical_result - analytic_result) < TOLERANCE, (
            f"TrapezoidalMP solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_simpson_mp_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = SimpsonMP.integrate(
            polynom, interval, n_subintervals=100, batch_size=32
        )

        # Simpson's rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"SimpsonMP solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_simpson38_mp_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = Simpson38MP.integrate(
            polynom, interval, n_subintervals=99, batch_size=33
        )

        # Simpson 3/8 rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"Simpson38MP solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_boole_mp_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = BooleMP.integrate(
            polynom, interval, n_subintervals=100, batch_size=25
        )

        # Boole's rule should be very accurate for polynomials
        assert abs(numerical_result - analytic_result) < TIGHT_TOLERANCE, (
            f"BooleMP solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )


class TestMonteCarloSolver:
    """Test Monte Carlo solver against analytic solutions."""

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_monte_carlo_solver(self, coeffs, interval):
        polynom = Polynom(coeffs)
        analytic_result = AnalyticSolver.integrate(polynom, interval)
        numerical_result = MonteCarlo.integrate(
            polynom, interval, n_samples=100000, _seed=42
        )

        # Monte Carlo needs looser tolerance due to stochastic nature
        assert abs(numerical_result - analytic_result) < MONTE_CARLO_TOLERANCE, (
            f"Monte Carlo solver failed for {coeffs} on {interval}: "
            f"got {numerical_result}, expected {analytic_result}"
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_monte_carlo_reproducibility(self, coeffs, interval):
        """Test that Monte Carlo produces the same result with the same seed."""
        polynom = Polynom(coeffs)

        result1 = MonteCarlo.integrate(polynom, interval, n_samples=10000, _seed=123)
        result2 = MonteCarlo.integrate(polynom, interval, n_samples=10000, _seed=123)

        assert result1 == result2, (
            f"Monte Carlo solver not reproducible for {coeffs} on {interval}: "
            f"got {result1} and {result2}"
        )


class TestSolverComparison:
    """Test that different solvers produce similar results for the same polynomial."""

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_solver_consistency(self, coeffs, interval):
        """Test that Newton-Cotes methods agree with each other."""
        polynom = Polynom(coeffs)

        # Get results from different solvers
        midpoint = MidpointSolver.integrate(polynom, interval, n_subintervals=1000)
        trapezoidal = TrapezoidalSolver.integrate(
            polynom, interval, n_subintervals=1000
        )
        simpson = SimpsonSolver.integrate(polynom, interval, n_subintervals=100)

        # They should all be close to each other (and to the analytic solution)
        analytic = AnalyticSolver.integrate(polynom, interval)

        assert abs(midpoint - analytic) < TOLERANCE
        assert abs(trapezoidal - analytic) < TOLERANCE
        assert abs(simpson - analytic) < TIGHT_TOLERANCE

    @pytest.mark.unit
    @pytest.mark.parametrize("coeffs, interval", TO_INTEGRATE)
    def test_mp_vs_single_thread(self, coeffs, interval):
        """Test that MP versions produce the same results as single-threaded versions."""
        polynom = Polynom(coeffs)
        n_intervals = 1000

        # Compare single-threaded vs multiprocessing versions
        midpoint_st = MidpointSolver.integrate(
            polynom, interval, n_subintervals=n_intervals
        )
        midpoint_mp = MidpointMP.integrate(
            polynom, interval, n_subintervals=n_intervals, batch_size=64
        )

        trapezoidal_st = TrapezoidalSolver.integrate(
            polynom, interval, n_subintervals=n_intervals
        )
        trapezoidal_mp = TrapezoidalMP.integrate(
            polynom, interval, n_subintervals=n_intervals, batch_size=64
        )

        simpson_st = SimpsonSolver.integrate(polynom, interval, n_subintervals=100)
        simpson_mp = SimpsonMP.integrate(
            polynom, interval, n_subintervals=100, batch_size=32
        )

        # Single-threaded and MP versions should give identical results
        assert (
            abs(midpoint_st - midpoint_mp) < 1e-10
        ), f"Midpoint ST vs MP mismatch for {coeffs}: {midpoint_st} vs {midpoint_mp}"

        assert (
            abs(trapezoidal_st - trapezoidal_mp) < 1e-10
        ), f"Trapezoidal ST vs MP mismatch for {coeffs}: {trapezoidal_st} vs {trapezoidal_mp}"

        assert (
            abs(simpson_st - simpson_mp) < 1e-10
        ), f"Simpson ST vs MP mismatch for {coeffs}: {simpson_st} vs {simpson_mp}"


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.unit
    def test_constant_polynomial(self):
        """Test integration of constant polynomials."""
        polynom = Polynom([5])  # f(x) = 5
        interval = (0, 3)
        expected = 15  # 5 * 3

        midpoint = MidpointSolver.integrate(polynom, interval, n_subintervals=1)
        trapezoidal = TrapezoidalSolver.integrate(polynom, interval, n_subintervals=1)
        simpson = SimpsonSolver.integrate(polynom, interval, n_subintervals=1)

        assert abs(midpoint - expected) < 1e-10
        assert abs(trapezoidal - expected) < 1e-10
        assert abs(simpson - expected) < 1e-10

    @pytest.mark.unit
    def test_linear_polynomial(self):
        """Test integration of linear polynomials."""
        polynom = Polynom([2, 3])  # f(x) = 2 + 3x
        interval = (0, 2)
        expected = 2 * 2 + 3 * 2**2 / 2  # 4 + 6 = 10

        # Trapezoidal and Simpson should be exact for linear polynomials
        trapezoidal = TrapezoidalSolver.integrate(polynom, interval, n_subintervals=1)
        simpson = SimpsonSolver.integrate(polynom, interval, n_subintervals=1)

        assert abs(trapezoidal - expected) < 1e-10
        assert abs(simpson - expected) < 1e-10

    @pytest.mark.unit
    def test_zero_interval(self):
        """Test integration over zero-length interval."""
        polynom = Polynom([1, 2, 3])
        interval = (1, 1)  # Same start and end
        expected = 0

        for solver in [MidpointSolver, TrapezoidalSolver, SimpsonSolver]:
            result = solver.integrate(polynom, interval, n_subintervals=10)
            assert abs(result - expected) < 1e-10

    @pytest.mark.unit
    def test_negative_interval(self):
        """Test integration over negative intervals."""
        polynom = Polynom([1, 1])  # f(x) = 1 + x
        interval = (2, 0)  # Backwards interval
        forward_interval = (0, 2)

        # Integration over backwards interval should be negative of forward
        backward_result = TrapezoidalSolver.integrate(
            polynom, interval, n_subintervals=100
        )
        forward_result = TrapezoidalSolver.integrate(
            polynom, forward_interval, n_subintervals=100
        )

        assert abs(backward_result + forward_result) < TOLERANCE
