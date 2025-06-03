import time
from src.empit_coding_challenge.solvers import SolverMeta
from inspect import signature


def extract_meta_arg(func, args, kwargs):
    """
    Attempts to retrieve the `meta` argument (positional or keyword) from the provided args and kwargs.
    """
    if "meta" in kwargs:
        return kwargs["meta"]

    sig = signature(func)
    params = list(sig.parameters)
    if "meta" in params:
        meta_index = params.index("meta")
        if meta_index < len(args):
            return args[meta_index]
    return None


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        # TODO: This is ok, but it would be nicer to have a Typed Result structure for all Solvers. Beyond the scope for now.
        meta = extract_meta_arg(func, args, kwargs)
        if meta is not None and hasattr(meta, "execution_time"):
            meta.execution_time = end - start
        print(f"{func.__name__} took {end - start:.6f} seconds")
        return result

    return wrapper
