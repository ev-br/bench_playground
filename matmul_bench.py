import pytest
import numpy as np

import utils

# import the available array backends
jnp = utils.try_import("jax.numpy")
torch = utils.try_import("torch")
cupy = utils.try_import("cupy")

AVAILABLE_MODULES = [x for x in [np, jnp, torch, cupy] if x is not None]

N = 100


def run_matmul(a, xp):
    res = xp.matmul(a, a)
    return res


#@pytest.mark.benchmark(warmup=True, warmup_iterations=100, disable_gc=True) -- moved to pytest.ini
@pytest.mark.parametrize('xp', AVAILABLE_MODULES)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_matmul(benchmark, xp, device):

    utils.configure_backend(xp, device=device)    # use f64 etc

    # generate the same data for all backends
    rng = np.random.default_rng(123)
    a = rng.uniform(size=(N, N))
    aa = xp.asarray(a)     # NB: rely on the default device from configure_backend

    # use xp-specific synchronization
    func = utils.wrap_function(run_matmul, xp)

    # run the benchmark
    result = benchmark(func, aa, xp)

