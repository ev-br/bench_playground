import os
import socket
import pytest
import numpy as np

import utils

from scipy.interpolate import RBFInterpolator
from scipy.stats.qmc import Halton

# import the available array backends
jnp = utils.try_import("jax.numpy")
torch = utils.try_import("torch")
cupy = utils.try_import("cupy")

AVAILABLE_MODULES = [x for x in [np, jnp, torch, cupy] if x is not None]

# JIT or eager
jit = "jit" if os.environ.get("SCIPY_JIT", "0") == "1" else "eager"

# XXX this should happen in the suite-specific config instead?
machine = socket.gethostname()


Nobs = 100
Ns = [50, 100, 200, 500, 1000, 2000]

# set up the data to interpolate
rng = np.random.default_rng(123)
xobs_np = 2*Halton(2, seed=rng).random(Nobs) - 1
yobs_np = np.sum(xobs_np, axis=1)*np.exp(-6*np.sum(xobs_np**2, axis=1))


@pytest.mark.parametrize('xp', AVAILABLE_MODULES)
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('N', Ns)
def test_rbf(benchmark, xp, device, N):

    # configure the backend
    try:
        utils.configure_backend(xp, device=device)    # use f64 etc
        xp.asarray([1, 2, 3])   # pytorch errors out if CUDA is not available
    except:
        pytest.skip(f"{xp.__name__} & device {device.upper()} are incompatible.")

    # https://pytest-benchmark.readthedocs.io/en/latest/usage.html#extra-info
    benchmark.extra_info["jit"] = jit
    benchmark.extra_info["machine"] = machine
    benchmark.extra_info["env_vars"] = utils.get_env_vars()
    benchmark.extra_info["scipy_commit"] = "413afb7bc9e922a0b6778ae407cc95ab9bac56c1"  # XXX

    # construct the interpolator
    xobs, yobs = map(xp.asarray, (xobs_np, yobs_np))
    rbf = RBFInterpolator(xobs, yobs)

    # problem size dependent data
    x1 = xp.linspace(-1, 1, N)
    xgrid = xp.stack(xp.meshgrid(x1, x1, indexing='ij'))
    xflat = xgrid.reshape(2, -1).T     # make it a 2-D array

    # pre-compile
    rbf(xflat)

    # use xp-specific synchronization
    func = utils.wrap_function(rbf, xp)

    # benchmark
    benchmark(func, xflat)
