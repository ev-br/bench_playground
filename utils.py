import functools

try:
    import jax
    HAVE_JAX = True
except ImportError:
    HAVE_JAX = False

from array_api_compat import is_jax_namespace as is_jax


def get_timer(base_timer, xp):
    if HAVE_JAX and is_jax(xp):
        def timer():
            jax.block_until_ready()      # XXX: does not work, needs a jax object. Ooops
            return base_timer()
    else:
        timer = base_timer

    return timer


##########################

def wrap_function(func, xp):
    if HAVE_JAX and is_jax(xp):
        @functools.wraps(func)
        def wrapped(*args, **kwds):
            result = func(*args, **kwds)
            jax.block_until_ready(result)
            return result

        return wrapped
    else:
        return func


