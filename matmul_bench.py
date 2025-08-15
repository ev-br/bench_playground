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


@pytest.mark.parametrize('N', [10, 20, 50, 100, 200, 500])
@pytest.mark.parametrize('xp', AVAILABLE_MODULES)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_matmul(benchmark, N, xp, device):

    try:
        utils.configure_backend(xp, device=device)    # use f64 etc
        xp.asarray([1, 2, 3])   # pytorch errors out if CUDA is not available
    except:
        pytest.xfail(f"{xp.__name__} & device {device.upper()} do not play ball.")

    # generate the same data for all backends
    rng = np.random.default_rng(123)
    a = rng.uniform(size=(N, N))
    aa = xp.asarray(a)     # NB: rely on the default device from configure_backend

    # use xp-specific synchronization
    func = utils.wrap_function(run_matmul, xp)

    # run the benchmark
    result = benchmark(func, aa, xp)

