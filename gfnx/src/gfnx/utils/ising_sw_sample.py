"""JAX Swendsen-Wang sampler for the 2D Ising model.

Target:
    p(s) ∝ exp(beta * J * sum_<i,j> s_i s_j + beta * h * sum_i s_i)

with s_i ∈ {-1, +1} and periodic boundary conditions.

For the critical Ising experiment:
    L = 16
    beta = 0.4407
    J = 1.0
    h = 0.0

The implementation follows the Swendsen-Wang construction used in the
offpolicy-discrete-diffusion-samplers-and-bridges-public repository.
"""

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax


def _sw_step(
    key: jax.Array,
    spins: jnp.ndarray,
    beta: float,
    J: float,
) -> tuple[jnp.ndarray, jax.Array]:
    """One Swendsen-Wang update.

    Args:
        key: JAX PRNG key.
        spins: (B, L, L), values in {-1, +1}.
        beta: inverse temperature.
        J: ferromagnetic coupling.

    Returns:
        new_spins: (B, L, L)
        new_key: updated PRNG key
    """
    B, L, _ = spins.shape
    n_sites = L * L

    # Same bond probability as the reference Ising2D implementation:
    #
    #     p = 1 - exp(-2 * beta * J)
    #
    p_bond = 1.0 - jnp.exp(-2.0 * beta * J)

    # ------------------------------------------------------------------
    # 1. Sample activated bonds between equal neighbouring spins.
    # ------------------------------------------------------------------

    key, key_h, key_v, key_color = jax.random.split(key, 4)

    # Horizontal bond (i,j) -- (i,j+1)
    right_spins = jnp.roll(spins, -1, axis=2)
    equal_h = spins == right_spins
    bond_h = (
        equal_h
        & (jax.random.uniform(key_h, (B, L, L)) < p_bond)
    )

    # Vertical bond (i,j) -- (i+1,j)
    down_spins = jnp.roll(spins, -1, axis=1)
    equal_v = spins == down_spins
    bond_v = (
        equal_v
        & (jax.random.uniform(key_v, (B, L, L)) < p_bond)
    )

    # ------------------------------------------------------------------
    # 2. Find connected components.
    #
    # We use iterative label propagation instead of a Python/Numba
    # union-find so the sampler remains JAX compatible.
    # ------------------------------------------------------------------

    # Each site initially has its own component label.
    labels = jnp.broadcast_to(
        jnp.arange(n_sites, dtype=jnp.int32)[None, :],
        (B, n_sites),
    )

    labels = labels.reshape(B, L, L)

    def propagate(_, labels):
        # Neighbour labels.
        right = jnp.roll(labels, -1, axis=2)
        left = jnp.roll(labels, +1, axis=2)
        down = jnp.roll(labels, -1, axis=1)
        up = jnp.roll(labels, +1, axis=1)

        # Bonds are stored at the source site.
        right_bond = bond_h
        left_bond = jnp.roll(bond_h, +1, axis=2)

        down_bond = bond_v
        up_bond = jnp.roll(bond_v, +1, axis=1)

        # If an edge is active, propagate the neighbour's label.
        candidates = jnp.stack(
            [
                labels,
                jnp.where(right_bond, right, labels),
                jnp.where(left_bond, left, labels),
                jnp.where(down_bond, down, labels),
                jnp.where(up_bond, up, labels),
            ],
            axis=0,
        )

        return jnp.min(candidates, axis=0)

    # L*L iterations are more than enough for a LxL torus.
    labels = lax.fori_loop(
        0,
        n_sites,
        propagate,
        labels,
    )

    # ------------------------------------------------------------------
    # 3. Give each connected cluster an independent Ising colour.
    # ------------------------------------------------------------------

    cluster_colours = jax.random.randint(
        key_color,
        (B, n_sites),
        minval=0,
        maxval=2,
    ).astype(jnp.int8)

    cluster_colours = cluster_colours.reshape(B, L, L)

    new_spins = 2 * jnp.take_along_axis(
        cluster_colours.reshape(B, -1),
        labels.reshape(B, -1),
        axis=1,
    ).reshape(B, L, L) - 1

    return new_spins.astype(jnp.int8), key


@partial(
    jax.jit,
    static_argnames=(
        "L",
        "num_samples",
        "batch_size",
        "burn_in",
        "collect_every",
    ),
)
def swendsen_wang_sampler(
    key: jax.Array,
    L: int,
    beta: float,
    J: float,
    num_samples: int,
    batch_size: int = 256,
    burn_in: int = 1000,
    collect_every: int = 100,
) -> tuple[jnp.ndarray, jnp.ndarray, jax.Array]:
    """Generate Swendsen-Wang Ising samples.

    This mirrors the reference sampler's structure:

        B = 256 parallel chains
        burn_in = 1000
        collect_every = 100

    Args:
        key: JAX PRNG key.
        L: lattice side length.
        beta: inverse temperature.
        J: coupling.
        num_samples: number of configurations to return.
        batch_size: number of parallel chains.
        burn_in: SW updates discarded initially.
        collect_every: SW updates between saved samples.

    Returns:
        samples:
            (num_samples, L, L), spins in {-1, +1}
        final_spins:
            final state of all parallel chains
        key:
            updated PRNG key
    """

    # Reference sampler uses ceil(n / B) collection rounds.
    n_collect = (num_samples + batch_size - 1) // batch_size

    # Random initial Ising configurations.
    key, init_key = jax.random.split(key)

    spins = (
        2
        * jax.random.randint(
            init_key,
            (batch_size, L, L),
            minval=0,
            maxval=2,
            dtype=jnp.int8,
        )
        - 1
    )

    # --------------------------------------------------------------
    # Burn-in.
    # --------------------------------------------------------------

    def burn_body(carry, _):
        key, spins = carry
        spins, key = _sw_step(key, spins, beta, J)
        return (key, spins), None

    (key, spins), _ = lax.scan(
        burn_body,
        (key, spins),
        xs=None,
        length=burn_in,
    )

    # --------------------------------------------------------------
    # Collect.
    #
    # Reference:
    #
    # for _ in range(num_collect):
    #     for _ in range(collect_every):
    #         SW_step()
    #     samples.append(S)
    # --------------------------------------------------------------

    def collect_body(carry, _):
        key, spins = carry

        def thin_body(carry, _):
            key, spins = carry
            spins, key = _sw_step(key, spins, beta, J)
            return (key, spins), None

        (key, spins), _ = lax.scan(
            thin_body,
            (key, spins),
            xs=None,
            length=collect_every,
        )

        return (key, spins), spins

    (key, spins), collected = lax.scan(
        collect_body,
        (key, spins),
        xs=None,
        length=n_collect,
    )

    # collected:
    #     (n_collect, batch_size, L, L)
    #
    # Reference concatenates all chains and then randomly selects n if
    # the requested size is smaller than the generated amount.
    samples = collected.reshape(-1, L, L)[:num_samples]

    return samples, spins, key