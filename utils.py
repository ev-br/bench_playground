import functools
import importlib

from array_api_compat import (
    is_jax_namespace as is_jax,
    is_torch_namespace as is_torch,
    is_cupy_namespace as is_cupy,
    is_numpy_namespace as is_numpy,
)


def try_import(name: str):
    """Import a module if available."""
    try:
        module = importlib.import_module(name)
    except ImportError:
        module = None
    return module

# import available Array API backends
jax = try_import("jax")
torch = try_import("torch")
cupy = try_import("cupy")


HAVE_JAX = jax is not None
HAVE_TORCH = torch is not None
HAVE_CUPY = cupy is not None


def configure_backend(xp, device: str):
    """Set the defaults for a backend.
    """
    if is_jax(xp):
        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_default_device", jax.devices(device)[0])
    elif is_torch(xp):
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float64)
    elif is_numpy(xp):
        if device != "cpu":
            raise ValueError(f"{device=} is invalid for NumPy.")
    elif is_cupy(xp):
        if device != "cuda":
            raise ValueError(f"{device=} is invalid for CuPy.")


# Early bind the synchronization symbols to avoid the attribute lookup overhead

if HAVE_CUPY:
    # In [8]: %timeit cupy.cuda.stream.get_current_stream().synchronize()
    # 571 ns ± 2.04 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    # f = cupy.cuda.stream.get_current_stream().synchronize
    # In [9]: %timeit f()
    # 355 ns ± 2.59 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    cupy_synchronize = cupy.cuda.stream.get_current_stream().synchronize

if HAVE_TORCH and torch.backends.cuda.is_built() and torch.cuda.is_available():
    # In [15]: %timeit torch.cuda.synchronize()
    # 4.27 μs ± 16.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    torch_synchronize = torch.cuda.synchronize



# XXX: Ideally, this should happen in the framework instead
def wrap_function(func, xp):
    """ Wrap the callable to synchronize CUDA and other async dispatch.
    """
    if HAVE_JAX and is_jax(xp):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            result = func(*args, **kwds)
            jax.block_until_ready(result)
            return result
        return wrapped

    elif HAVE_TORCH and is_torch(xp):
        # https://github.com/pytorch/pytorch/blob/v2.8.0/torch/utils/benchmark/utils/timer.py#L16
        # NB: torch does in the timer, we do it here instead to avoid having to patch
        # both `function_to_benchmark` and the timer.
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            @functools.wraps(func)
            def wrapped(*args, **kwds):
                result = func(*args, **kwds)
                torch_synchronize()
                return result
            return wrapped
        else:
            return func

    elif HAVE_CUPY and is_cupy(xp):
        # https://github.com/cupy/cupy/issues/1317#issuecomment-394558903
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            result = func(*args, **kwds)
            cupy_synchronize()
            return result
        return wrapped

    else:
        return func


##########################

def get_timer(base_timer, xp):
    if HAVE_JAX and is_jax(xp):
        def timer():
            jax.block_until_ready()      # XXX: does not work, needs a jax object. Ooops
            return base_timer()
    else:
        timer = base_timer

    return timer


###############################

# Helpers for loading, slicing and dicing benchmark results
import json
import re
from collections import defaultdict


def read_json(fname: str, machine=None) -> list[dict]:
    """Read the json, return a list of BenchmarkResult dicts.
    """
    with open(fname, "r") as f:
        raw_json = json.load(f)

    if machine is not None:
        _machine = raw_json["machine_info"]["node"]
        if _machine != machine:
            raise ValueError(
                f"{machine = } requested by {fname} has the runs on node = {_machine}"
            )

    return raw_json["benchmarks"]


def group_by(list_of_results, pred):
    """ Group the list of BenchmarkResult dicts by the predicate value.

    Parameters
    ----------
    list_of_results: list[dict]
        List of BenchmarkResult dicts
    pred: callable
        The signature is pred(BenchmarkResult) -> value
        list_of_result entries will be grouped by the return value

    Returns
    -------
    dict
        The keys are values as returned by `pred`, the values are lists of BenchmarkResults
    """
    grouped = defaultdict(list)
    for bmark in list_of_results:
        value = pred(bmark)
        grouped[value].append(bmark)

    return dict(grouped)



def normalize_module_name(s: str):
    """Normalize the module repr to module.__name__

    Usage:

    >>> lst = read_json("path/to.json")
    >>> for elem in lst:
    ...     elem["params"]["xp"] = normalize_module_name(elem["params"]["xp"])
    """
    match = re.search(r"UNSERIALIZABLE\[<module '([\w\.]+)'", s)
    if match:
        module_name = match.group(1)
        return module_name
    else:
        return s 
