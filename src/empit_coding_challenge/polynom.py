from typing import Self
from dataclasses import dataclass


class Polynom(list):
    def __init__(self, coeffs):
        # TODO This could be better. Try to allow list and str only, rest raises Value Error.
        if isinstance(coeffs, str):
            coeffs = Polynom.parse_from_string(coeffs)
        super().__init__(coeffs)

    def __repr__(self):
        return "Polynom({})".format(super().__repr__())

    def __add__(self, other):
        max_len = max(len(self), len(other))
        result = []
        for i in range(max_len):
            a = self[i] if i < len(self) else 0
            b = other[i] if i < len(other) else 0
            result.append(a + b)
        return Polynom(result)

    def __sub__(self, other):
        """Similar to add, just with a '-'"""
        max_len = max(len(self), len(other))
        result = []
        for i in range(max_len):
            a = self[i] if i < len(self) else 0
            b = other[i] if i < len(other) else 0
            result.append(a - b)
        return Polynom(result)

    def __mul__(self, other):
        """
        Multiplying coeffs is equivalent to convolution operation.
        See https://en.wikipedia.org/wiki/Convolution and
        https://math.stackexchange.com/questions/1937630/convolution-and-multiplication-of-polynomials-is-the-same
        """
        result_len = len(self) + len(other) - 1
        result = [0] * result_len
        for i, a in enumerate(self):
            for j, b in enumerate(other):
                result[i + j] += a * b
        return Polynom(result)

    def __call__(self, x):
        result = 0
        for i in range(len(self)):
            result += self[i] * x**i
        return result

    def __str__(self):
        terms = []
        for i, coeff in enumerate(self):
            if coeff == 0:
                continue
            term = str(abs(round(coeff, 3)))
            if i > 0:
                term += "*x"
            if i > 1:
                term += "^{}".format(i)
            if coeff < 0:
                term = "-" + term
            elif terms:
                term = "+" + term
            terms.append(term)
        if not terms:
            return "0"
        return "".join((terms))

    def integrate(self, interval: tuple[float, float], solver: "IntegralSolver"):
        return solver.integrate(self, interval)

    @staticmethod
    def parse_from_string(input_str: str) -> list[float]:
        try:
            coeffs = [float(token) for token in input_str.strip().split()]
        except ValueError:
            raise ValueError(
                f"Invalid input: all the coefficients must be numbers. But got: {input_str!r}"
            )
        if not coeffs:
            raise ValueError("No coefficients provided.")
        return coeffs


class IntegralSolver:
    @staticmethod
    def integrate(polynom: Polynom, interval: tuple[float, float]) -> float: ...


@dataclass
class SolverMetadata:
    result: float
    tolerance: float | None = None
    tolerance_reached: bool | None = None
    n_samples_used: int | None = None
    execution_time: float | None = None
