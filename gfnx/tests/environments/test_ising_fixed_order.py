import jax
import jax.numpy as jnp
import numpy as np

import gfnx
from gfnx.environment.sequence import EnvState


def test_fixed_order_rollouts_and_reverse_actions():
    env = gfnx.IsingBitseqEnvironment(
        gfnx.BitseqIsingRewardModule(L=2, beta=1.2, k=1),
        L=2,
        beta=1.2,
        k=1,
        fixed_order=True,
    )
    params = env.init(jax.random.PRNGKey(0))
    # Enumerate every terminal lattice; backward paths must visit prefixes only.
    tokens = (jnp.arange(16)[:, None] >> jnp.arange(4)) & 1
    terminal = EnvState.from_tokens(tokens).replace(
        is_terminal=jnp.ones(16, dtype=bool),
    )

    def backward_policy(key, obs, unused):
        return jnp.zeros((obs.shape[0], env.backward_action_space.n)), {}

    reverse, _ = gfnx.utils.backward_rollout(
        jax.random.PRNGKey(1),
        terminal,
        backward_policy,
        None,
        env,
        params,
    )
    for t in range(4):
        state = jax.tree.map(lambda a: a[:, t], reverse.state)
        expected = np.array(tokens).copy()
        expected[:, 4 - t :] = env.pad_token
        np.testing.assert_array_equal(state.tokens, expected)
        mask = env.get_invalid_backward_mask(state, params)
        np.testing.assert_array_equal((~mask).sum(axis=-1), 1)
        np.testing.assert_array_equal(reverse.action[:, t], 3 - t)
        assert np.all(
            np.isfinite(
                jax.nn.log_softmax(
                    gfnx.utils.mask_logits(jnp.zeros_like(mask, dtype=jnp.float32), mask)
                )
            )
        )
        previous = jax.tree.map(lambda a: a[:, t + 1], reverse.state)
        forward_action = env.get_forward_action(state, reverse.action[:, t], previous, params)
        np.testing.assert_array_equal(
            jnp.take_along_axis(
                env.get_invalid_mask(previous, params), forward_action[:, None], axis=1
            ),
            False,
        )
        restored, _, _ = jax.vmap(env._single_transition, in_axes=(0, 0, None))(
            previous,
            forward_action,
            params,
        )
        np.testing.assert_array_equal(restored.tokens, state.tokens)

    def forward_policy(key, obs, unused):
        logits = jnp.zeros((obs.shape[0], env.action_space.n))
        return logits, {"forward_logits": logits, "backward_logits": jnp.zeros((obs.shape[0], 4))}

    trajectory, _ = gfnx.utils.forward_rollout(
        jax.random.PRNGKey(2),
        16,
        forward_policy,
        None,
        env,
        params,
    )
    np.testing.assert_array_equal(trajectory.action[:, :4] // 2, np.tile(np.arange(4), (16, 1)))
    _, log_pb = gfnx.utils.forward_trajectory_log_probs(env, trajectory, params)
    np.testing.assert_allclose(log_pb, 0)
    # The default environment still permits all positions at the empty lattice.
    ordinary = gfnx.IsingBitseqEnvironment(env.reward_module, L=2, beta=1.2, k=1)
    assert not np.any(ordinary.get_invalid_mask(ordinary.get_init_state(1), params))
