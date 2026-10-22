"""Evaluation metrics for the GFNx Ising-Bitseq experiment.

The target-side evaluation pool is fixed outside these metric modules.  Each
metric module samples a fresh subset from that pool on every evaluation round,
so one can use a large fixed SW pool (typically 16384 samples) while evaluating
on smaller batches (typically 2048 total, or smaller per round).

The physics metrics follow the reference implementation used by the authors:
  - magnetization error
  - two-point correlation error
  - Sinkhorn distance with Hamming cost

The EUBO module performs the GFNx backward-rollout estimator on a sampled
subset of the fixed reference-state pool.
"""

from typing import Any, Dict

import chex
import jax
import jax.numpy as jnp

from gfnx.base import TEnvParams
from gfnx.metrics.base import (
    BaseMetricsModule,
    BaseProcessArgs,
    EmptyInitArgs,
    EmptyUpdateArgs,
    MetricsState,
)
from gfnx.utils.bitseq import detokenize
from gfnx.utils.rollout import (
    TPolicyFn,
    TPolicyParams,
    backward_rollout,
    backward_trajectory_log_probs,
    forward_rollout,
)


# -----------------------------------------------------------------------------
# Ising physics helpers
# -----------------------------------------------------------------------------


def _corr_curve(spins: chex.Array, axis: int) -> chex.Array:
    """Connected two-point correlation C(r) for a batch of Ising spins.

    This mirrors Ising2D.two_point_correlation() in the reference repository:
    for every distance r, compute

        <S_i S_{i+r}> - <S_i><S_{i+r}>

    averaged over all sites. Periodic boundary conditions are implemented by
    ``jnp.roll``.

    Args:
        spins: Array of shape (batch, L, L), values in {-1, +1}.
        axis: 1 for vertical/row direction, 2 for horizontal/column direction.

    Returns:
        Array of shape (L,) containing the correlation for r = 0, ..., L-1.
    """
    L = spins.shape[axis]
    mean_spins = jnp.mean(spins, axis=0)  # (L, L)
    mean_axis = axis - 1  # remove batch dimension

    def at_shift(r: int) -> chex.Array:
        shifted = jnp.roll(spins, -r, axis=axis)
        shifted_mean = jnp.roll(mean_spins, -r, axis=mean_axis)
        pair_mean = jnp.mean(spins * shifted, axis=0)
        return jnp.mean(pair_mean - mean_spins * shifted_mean)

    return jnp.stack([at_shift(r) for r in range(L)])


def _hamming_block(x_blk: chex.Array, y: chex.Array) -> chex.Array:
    """Pairwise Hamming distances between rows of binary arrays.

    For x, y in {0, 1}^d:  |x - y|_1 = sum(x) + sum(y) - 2 <x, y>.
    """
    inner = jnp.matmul(x_blk, y.T, precision=jax.lax.Precision.HIGHEST)
    return jnp.sum(x_blk, axis=-1)[:, None] + jnp.sum(y, axis=-1)[None, :] - 2.0 * inner


def sinkhorn_distance(
    x: chex.Array,
    y: chex.Array,
    reg: float = 1e-3,
    n_iters: int = 100,
    chunk_size: int | None = None,
) -> chex.Array:
    """Entropy-regularised OT cost with Hamming ground cost.

    Mirrors the reference ``eval_metrics.sinkhorn_distance`` (PyTorch): uniform
    empirical weights on both sample sets, log-domain Sinkhorn iterations with
    ``n_iters`` (f, g) updates, and the returned value is the *transport cost*
    ``<P, C>`` of the final plan (not the regularised objective).

    Args:
        x: (n, d) binary array (values in {0, 1}), e.g. model samples.
        y: (m, d) binary array (values in {0, 1}), e.g. target samples.
            NOTE: inputs must be bits, not {-1, +1} spins; the Hamming cost
            is computed as sum(x) + sum(y) - 2 <x, y>.
        reg: Entropic regularisation (epsilon). The paper uses 1e-3.
        n_iters: Number of Sinkhorn iterations. The reference uses 100.
        chunk_size: If None, the full (n, m) cost matrix is materialised
            (fine for the 2048-sized per-round subsets). If set, the cost is
            recomputed block by block and never stored, so memory is
            O(chunk_size * max(n, m)); use this for 16384 x 16384. Must divide
            both n and m.

    Returns:
        Scalar array with the transport cost.
    """
    x = x.astype(jnp.float32)
    y = y.astype(jnp.float32)
    n, m = x.shape[0], y.shape[0]
    log_mu = -jnp.log(jnp.asarray(n, dtype=jnp.float32))
    log_nu = -jnp.log(jnp.asarray(m, dtype=jnp.float32))

    if chunk_size is None:
        cost = _hamming_block(x, y)

        def rows_lse(g):  # (n,)  logsumexp_j (g_j - C_ij) / reg
            return jax.scipy.special.logsumexp((g[None, :] - cost) / reg, axis=1)

        def cols_lse(f):  # (m,)  logsumexp_i (f_i - C_ij) / reg
            return jax.scipy.special.logsumexp((f[:, None] - cost) / reg, axis=0)

        def final_cost(f, g):
            plan = jnp.exp((f[:, None] + g[None, :] - cost) / reg)
            return jnp.sum(plan * cost)

    else:
        if n % chunk_size != 0 or m % chunk_size != 0:
            raise ValueError(
                f"chunk_size={chunk_size} must divide both n={n} and m={m}"
            )
        d = x.shape[1]
        x_ch = x.reshape(n // chunk_size, chunk_size, d)
        y_ch = y.reshape(m // chunk_size, chunk_size, d)

        def rows_lse(g):
            def one(x_blk):
                c = _hamming_block(x_blk, y)  # (chunk, m)
                return jax.scipy.special.logsumexp((g[None, :] - c) / reg, axis=1)

            return jax.lax.map(one, x_ch).reshape(-1)

        def cols_lse(f):
            def one(y_blk):
                c = _hamming_block(y_blk, x)  # (chunk, n) = C[:, block].T
                return jax.scipy.special.logsumexp((f[None, :] - c) / reg, axis=1)

            return jax.lax.map(one, y_ch).reshape(-1)

        def final_cost(f, g):
            f_ch = f.reshape(n // chunk_size, chunk_size)

            def one(args):
                x_blk, f_blk = args
                c = _hamming_block(x_blk, y)
                plan = jnp.exp((f_blk[:, None] + g[None, :] - c) / reg)
                return jnp.sum(plan * c)

            return jnp.sum(jax.lax.map(one, (x_ch, f_ch)))

    def body(_, carry):
        _, g = carry
        f = reg * (log_mu - rows_lse(g))
        g = reg * (log_nu - cols_lse(f))
        return f, g

    f, g = jax.lax.fori_loop(
        0,
        n_iters,
        body,
        (jnp.zeros(n, dtype=jnp.float32), jnp.zeros(m, dtype=jnp.float32)),
    )
    return final_cost(f, g)


# -----------------------------------------------------------------------------
# Physics metrics
# -----------------------------------------------------------------------------


@chex.dataclass
class IsingPhysicsMetricState(MetricsState):
    mag_error: jnp.ndarray
    corr_error: jnp.ndarray
    sinkhorn: jnp.ndarray


class IsingPhysicsMetricsModule(BaseMetricsModule):
    """Magnetization error, two-point correlation error, and Sinkhorn.

    ``gt_spins`` is a fixed reference pool, normally 16384 SW samples.  On
    every evaluation round a fresh subset of size ``batch_size`` is drawn from
    that pool and used for *all three* metrics in that round. The round values
    are then averaged over ``n_rounds``.

    This matches the reference metric definitions. In particular, for the
    Ising setup used here (h=0), the reference magnetization target is exactly
    zero; it is NOT estimated from the sampled GT batch.
    """

    def __init__(
        self,
        env,
        L: int,
        k: int,
        fwd_policy_fn: TPolicyFn,
        n_rounds: int,
        batch_size: int,
        sinkhorn_reg: float = 1e-3,
        sinkhorn_iters: int = 100,
        compute_sinkhorn: bool = True,
        sinkhorn_chunk_size: int | None = None,
    ):
        self.env = env
        self.L = L
        self.k = k
        self.fwd_policy_fn = fwd_policy_fn

        self.n_rounds = n_rounds
        self.batch_size = batch_size

        self.sinkhorn_reg = sinkhorn_reg
        self.sinkhorn_iters = sinkhorn_iters
        self.compute_sinkhorn = compute_sinkhorn
        self.sinkhorn_chunk_size = sinkhorn_chunk_size

    InitArgs = EmptyInitArgs

    def init(
        self,
        rng_key: chex.PRNGKey,
        args: InitArgs,
    ) -> IsingPhysicsMetricState:
        del rng_key, args
        return IsingPhysicsMetricState(
            mag_error=jnp.array(jnp.inf, dtype=jnp.float32),
            corr_error=jnp.array(jnp.inf, dtype=jnp.float32),
            sinkhorn=jnp.array(jnp.inf, dtype=jnp.float32),
        )

    UpdateArgs = EmptyUpdateArgs

    def update(self, metrics_state, rng_key, args=None):
        del rng_key, args
        return metrics_state

    def get(self, metrics_state: IsingPhysicsMetricState) -> Dict[str, Any]:
        return {
            "mag_error": metrics_state.mag_error,
            "corr_error": metrics_state.corr_error,
            "sinkhorn": metrics_state.sinkhorn,
        }

    @chex.dataclass
    class ProcessArgs(BaseProcessArgs):
        policy_params: TPolicyParams
        env_params: TEnvParams

    def process(
        self,
        metrics_state: IsingPhysicsMetricState,
        rng_key: chex.PRNGKey,
        args: ProcessArgs,
    ) -> IsingPhysicsMetricState:

        def process_round(carry_rng_key, _):
            rng_key, gt_key, model_key = jax.random.split(
                carry_rng_key,
                3,
            )

            # ------------------------------------------------------------
            # Fresh GT batch from the fixed .npz pool.
            # get_ground_truth_sampling() returns BitseqEnvState(tokens=...).
            # ------------------------------------------------------------
            gt_state = self.env.get_ground_truth_sampling(
                rng_key=gt_key,
                batch_size=self.batch_size,
                env_params=args.env_params,
            )

            # tokens -> bits {0,1}
            gt_bits = jax.vmap(
                lambda t: detokenize(t, self.k)
            )(gt_state.tokens)

            # bits {0,1} -> spins {-1,+1}
            gt_spins = (
                2.0 * gt_bits.astype(jnp.float32) - 1.0
            ).reshape(
                self.batch_size,
                self.L,
                self.L,
            )

            # ------------------------------------------------------------
            # Fresh model samples.
            # ------------------------------------------------------------
            _, aux_info = forward_rollout(
                rng_key=model_key,
                num_envs=self.batch_size,
                policy_fn=self.fwd_policy_fn,
                policy_params=args.policy_params,
                env=self.env,
                env_params=args.env_params,
            )

            final_state = aux_info["final_env_state"]

            model_bits = jax.vmap(
                lambda t: detokenize(t, self.k)
            )(final_state.tokens)

            model_spins = (
                2.0 * model_bits.astype(jnp.float32) - 1.0
            ).reshape(
                self.batch_size,
                self.L,
                self.L,
            )

            # ------------------------------------------------------------
            # Magnetization error.
            # ------------------------------------------------------------
            model_mean = jnp.mean(model_spins, axis=0)

            row_model = jnp.mean(model_mean, axis=1)
            col_model = jnp.mean(model_mean, axis=0)

            # h = 0 -> target magnetization is zero.
            row_gt = jnp.zeros_like(row_model)
            col_gt = jnp.zeros_like(col_model)

            mag_error = (
                jnp.sum(jnp.abs(row_model - row_gt))
                + jnp.sum(jnp.abs(col_model - col_gt))
            ) / (2.0 * self.L)

            # ------------------------------------------------------------
            # Two-point correlation error.
            # ------------------------------------------------------------
            row_model_corr = _corr_curve(
                model_spins,
                axis=1,
            )
            col_model_corr = _corr_curve(
                model_spins,
                axis=2,
            )

            row_gt_corr = _corr_curve(
                gt_spins,
                axis=1,
            )
            col_gt_corr = _corr_curve(
                gt_spins,
                axis=2,
            )

            # Average over L distances in each of two directions (2L terms),
            # matching Carter et al., arXiv:2602.05961, Appendix C.1.3.
            corr_error = (
                jnp.sum(
                    jnp.abs(row_model_corr - row_gt_corr)
                )
                + jnp.sum(
                    jnp.abs(col_model_corr - col_gt_corr)
                )
            ) / (2.0 * self.L)

            # ------------------------------------------------------------
            # Sinkhorn distance (Hamming cost, uniform weights) between the
            # model batch and the GT batch, on bits {0, 1}.
            # Cost is O(n_iters * batch_size^2 * L^2); pass
            # sinkhorn_chunk_size for large batches (e.g. 16384) to avoid
            # materialising the full cost matrix, or compute_sinkhorn=False
            # to skip it (the key is kept with a zero placeholder).
            # ------------------------------------------------------------
            if self.compute_sinkhorn:
                sinkhorn = sinkhorn_distance(
                    model_bits.astype(jnp.float32),
                    gt_bits.astype(jnp.float32),
                    reg=self.sinkhorn_reg,
                    n_iters=self.sinkhorn_iters,
                    chunk_size=self.sinkhorn_chunk_size,
                )
            else:
                sinkhorn = jnp.zeros((), dtype=jnp.float32)

            return rng_key, (
                mag_error,
                corr_error,
                sinkhorn,
            )

        _, (
            mag_rounds,
            corr_rounds,
            sinkhorn_rounds,
        ) = jax.lax.scan(
            process_round,
            rng_key,
            None,
            length=self.n_rounds,
        )

        return metrics_state.replace(
            mag_error=jnp.mean(mag_rounds),
            corr_error=jnp.mean(corr_rounds),
            sinkhorn=jnp.mean(sinkhorn_rounds),
        )
