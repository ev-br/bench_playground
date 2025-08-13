import functools

try:
    import jax
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

try:
    import cupy
    HAVE_CUPY = True
except ImportError:
    HAVE_CUPY = False

from array_api_compat import (
    is_jax_namespace as is_jax,
    is_torch_namespace as is_torch,
    is_cupy_namespace as is_cupy
)


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

