"""Physical identities that detect reward counting and error normalization mistakes."""

import jax
import jax.numpy as jnp
import numpy as np

import gfnx
from gfnx.environment.sequence import EnvState
from gfnx.metrics import IsingPhysicsMetricsModule


def test_periodic_reward_counts_each_bond_once():
    length, beta = 4, 0.6
    env = gfnx.IsingBitseqEnvironment(
        gfnx.BitseqIsingRewardModule(L=length, beta=beta, k=1),
        L=length,
        beta=beta,
        k=1,
    )
    params = env.init(jax.random.PRNGKey(0))
    checkerboard = (np.indices((length, length)).sum(axis=0) % 2).reshape(-1)
    bits = jnp.array([np.zeros(length**2), np.ones(length**2), checkerboard], dtype=jnp.int32)
    state = EnvState.from_tokens(bits)
    actual = env.reward_module.log_reward(state, params)
    expected = beta * 2 * length**2 * np.array([1, 1, -1])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_collapsed_sampler_has_unit_magnetization_and_correlation_error(tmp_path):
    # Equal all-minus/all-plus reference states have mean zero and C(r)=1.
    # An all-plus sampler has mean one and connected C(r)=0.
    length = 4
    bits = np.array([np.zeros(length**2), np.ones(length**2)], dtype=np.int32)
    path = tmp_path / "reference.npz"
    np.savez(path, bits=bits, L=length, beta=0.6)
    env = gfnx.IsingBitseqEnvironment(
        gfnx.BitseqIsingRewardModule(L=length, beta=0.6, k=1),
        L=length,
        beta=0.6,
        k=1,
        gt_samples_path=str(path),
    )
    params = env.init(jax.random.PRNGKey(0))

    def policy(key, obs, unused):
        logits = jnp.tile(jnp.array([-1e9, 0.0]), (obs.shape[0], length**2))
        return logits, {
            "forward_logits": logits,
            "backward_logits": jnp.zeros((obs.shape[0], length**2)),
        }

    module = IsingPhysicsMetricsModule(
        env, length, 1, policy, n_rounds=1, batch_size=2, compute_sinkhorn=False
    )
    state = module.init(jax.random.PRNGKey(1), module.InitArgs())
    result = module.process(
        state, jax.random.PRNGKey(2), module.ProcessArgs(policy_params=None, env_params=params)
    )
    np.testing.assert_allclose(result.mag_error, 1.0, atol=1e-6)
    np.testing.assert_allclose(result.corr_error, 1.0, atol=1e-6)
