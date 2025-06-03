from .polynom import Polynom, IntegralSolver


# TODO: create your solver classes here using the IntegralSolver as a base class
class AnalyticSolver(IntegralSolver):
    @staticmethod
    def integrate(polynom: Polynom, interval: tuple[float, float]):
        "Analytical integration, assuming Offset == 0"
        # the Integral F_0 is the special solution of the integral with constant = 0.
        # For the purpose of calculating value of interval, it is irrelevant which solution is chosen.
        F_0 = Polynom([0] + [val / (index + 1) for index, val in enumerate(polynom)])

        return F_0(interval[1]) - F_0(interval[0])


class NumericFixedStep(IntegralSolver):
    @staticmethod
    def integrate(polynom: Polynom, interval: tuple[float, float], N=100) -> float:
        """
        Fixed step size.
        Numerical integration using the midpoint (rectangle) rule.

        Approximates the definite integral of a polynomial over [a, b] by dividing
        the interval into N subintervals and evaluating the function at the midpoint
        of each subinterval.

        Parameters:
        - polynom: The polynomial to integrate
        - interval: A tuple (a, b) specifying the bounds of integration
        - N: Number of subintervals (default is 100)

        Returns:
        - Approximate value of the integral

        Editor's Notes:
        Time Complexity: O(N)
        Limitations:
            - Decrease in accuracy for big interval and small N
            - Unnecessary effort in 'smooth' regions.
        """
        a, b = interval
        step_size = (b - a) / N
        mid_points = [a + step_size * (i + 0.5) for i in range(N)]
        Y = [polynom(x_mid) for x_mid in mid_points]
        return sum(Y) * step_size


class NumericAdaptiveStep(IntegralSolver):
    @staticmethod
    def integrate(
        polynom: Polynom, interval: tuple[float, float], tol=1e-6, max_depth=10, depth=0
    ) -> float:
        """
        Adaptive step size.

        Numerical integration using the midpoint (rectangle) rule with adaptive refinement.

        Approximates the definite integral of a polynomial over [a, b] by recursively subdividing
        the interval until the estimated error is below a given tolerance. Evaluates the function
        at midpoints of subintervals and adapts the step size to improve accuracy where needed.

        Parameters:
        - polynom: The polynomial to integrate
        - interval: A tuple (a, b) specifying the bounds of integration
        - tol: Desired error tolerance (default is 1e-6)
        - max_depth: Maximum recursion depth to prevent infinite subdivision (optional)

        Returns:
        - Approximate value of the integral

        Editor's Notes:
        Time Complexity: O(2^max_depth) worst, better in practice,
        Advantages:
            - Higher accuracy with fewer function evaluations
            - Efficient handling of intervals with varying function behavior
        Limitations:
            - Overhead due to recursion and error estimation
            - Requires careful choice of tolerance and max_depth

        For more info, see https://personal.math.ubc.ca/~CLP/CLP2/clp_2_ic/ap_adaptive.html#:~:text=%E2%80%9CAdaptive%20quadrature%E2%80%9D%20refers%20to%20a,easy%20to%20get%20good%20accuracy.
        """

        a, b = interval
        mid = (a + b) / 2
        h = b - a

        # Midpoint rule estimates
        full = polynom(mid) * h
        left = polynom((a + mid) / 2) * (h / 2)
        right = polynom((mid + b) / 2) * (h / 2)
        refined = left + right

        if abs(refined - full) < tol or depth >= max_depth:
            return refined
        else:
            return NumericAdaptiveStep.integrate(
                polynom, (a, mid), tol / 2, max_depth, depth + 1
            ) + NumericAdaptiveStep.integrate(
                polynom, (mid, b), tol / 2, max_depth, depth + 1
            )
