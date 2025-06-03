from .polynom import Polynom
from .solvers import AnalyticSolver


class Routines:
    @staticmethod
    def show_polynom(args):
        # coeffs = Polynom.parse_from_string(p1)
        p = Polynom(args.p1)
        print(p)
        return p

    @staticmethod
    def add_polynom(args):
        result = Polynom(args.p1) + Polynom(args.p2)
        print(f"{result}")

    @staticmethod
    def subtract_polynom(args):
        result = Polynom(args.p1) - Polynom(args.p2)
        print(f"{result}")

    @staticmethod
    def multiply_polynom(args):
        p1 = Polynom(args.p1)
        p2 = Polynom(args.p2)
        result = p1 * p2

        print(f"{result}")

    @staticmethod
    def integrate_polynom(args):
        assert len(args.interval) == 2
        for val in args.interval:
            assert isinstance(val, float) | isinstance(val, int)
        interval = tuple([float(val) for val in args.interval])
        assert len(interval) == 2
        result = AnalyticSolver.integrate(Polynom(args.p1), interval)
        print(result)
