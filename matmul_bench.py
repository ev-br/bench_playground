import pytest
import numpy as np
import jax.numpy as jnp
import torch

import utils

def run_matmul(a, xp):
    res = xp.matmul(a, a)
    return res


#@pytest.mark.benchmark(warmup=True, warmup_iterations=100, disable_gc=True) -- moved to pytest.ini
@pytest.mark.parametrize('xp', [jnp, np, torch])
def test_matmul(benchmark, xp):

    rng = np.random.default_rng(123)
    a = rng.uniform(size=(100, 100))
    aa = xp.asarray(a)

    # benchmark._timer = utils.get_timer(benchmark._timer, xp)
    func = utils.wrap_function(run_matmul, xp)

    result = benchmark(func, aa, xp)

