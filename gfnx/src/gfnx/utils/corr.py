from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax.scipy.stats import rankdata


@jax.jit
def pearson_corr(a: chex.Array, b: chex.Array):
    """Computation of Pearson correlation.

    Assumes that at least two components for both a and b are different
    """
    chex.assert_equal_shape([a, b])
    return jnp.corrcoef(a, b)[0, 1]


@partial(jax.jit, static_argnames=["method", "dtype"])
def spearman_corr(a: chex.Array, b: chex.Array, method="average", dtype=jnp.float32):
    """
    JIT-compatible Spearman correlation that propagates NaNs. Handles integer and float inputs.
    """
    a = jnp.asarray(a, dtype=dtype)
    b = jnp.asarray(b, dtype=dtype)

    has_nan = jnp.isnan(a).any() | jnp.isnan(b).any()
    is_too_small = a.size < 2

    def compute_corr(x, y):
        x_ranked = rankdata(x, method=method)
        y_ranked = rankdata(y, method=method)
        return jnp.corrcoef(x_ranked, y_ranked)[0, 1]

    return jax.lax.cond(has_nan | is_too_small, lambda: jnp.nan, lambda: compute_corr(a, b))
