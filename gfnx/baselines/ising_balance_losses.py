"""Trajectory losses matching the repository's TB/DB/SubTB/VarGrad conventions."""

import jax.numpy as jnp


def balance_loss(objective, log_pf, log_pb, log_flow, log_reward, done, pad, log_z, lmbd=1.9):
    valid = ~pad
    increments = jnp.where(valid, log_pf - log_pb, 0.0)
    reward = jnp.where(valid, log_reward, 0.0)
    residual = increments.sum(axis=1) - reward.sum(axis=1)
    if objective == "tb":
        return jnp.mean((log_z + residual) ** 2)
    if objective == "vargrad":
        return 0.5 * jnp.sum((residual - residual.mean()) ** 2) / (residual.shape[0] - 1)
    flows = log_flow.at[:, 1:].set(jnp.where(done & valid, reward, log_flow[:, 1:]))
    if objective == "db":
        errors = flows[:, :-1] + increments - flows[:, 1:]
        return 0.5 * jnp.where(valid, errors**2, 0.0).sum() / valid.sum()
    if objective == "subtb":
        # Prefix sums compute all subtrajectory residuals without a cubic tensor.
        prefix = jnp.pad(jnp.cumsum(increments, axis=1), ((0, 0), (1, 0)))
        i, j = jnp.triu_indices(log_pf.shape[1] + 1, k=1)
        errors = flows[:, i] - flows[:, j] + prefix[:, j] - prefix[:, i]
        valid_prefix = jnp.pad(jnp.cumsum(valid, axis=1), ((0, 0), (1, 0)))
        valid_pairs = (valid_prefix[:, j] - valid_prefix[:, i]) == (j - i)
        weights = jnp.where(valid_pairs, lmbd ** (j - i), 0.0)
        return jnp.mean((weights * errors**2).sum(axis=1) / weights.sum(axis=1))
    raise ValueError(f"Unknown balance objective: {objective}")
