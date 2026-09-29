"""Check TRPO's training cost against the trajectory log-density ratio."""

import importlib
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from omegaconf import OmegaConf
import pytest

import gfnx
from gfnx.metrics import MultiMetricsModule


class ToyReward:
    """A deterministic reward, so the test needs no proxy files or downloads."""

    def init(self, rng_key, dummy_state):
        return None

    def log_reward(self, state, env_params):
        return 0.8 + state.tokens.sum(axis=-1) / 50.0

    def reward(self, state, env_params):
        return jnp.exp(self.log_reward(state, env_params))


class PaddedQM9(gfnx.QM9SmallEnvironment):
    def __init__(self, extra_padding):
        super().__init__(ToyReward())
        self.extra_padding = extra_padding

    @property
    def max_steps_in_episode(self):
        return self.max_length + self.extra_padding


@pytest.fixture
def trpo(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "baselines"))
    return importlib.import_module("TRPO_qm9_small")


@pytest.mark.parametrize("extra_padding", [0, 2])
def test_train_step_uses_complete_trajectory_ratio(trpo, extra_padding):
    env = PaddedQM9(extra_padding)
    env_params = env.init(jax.random.key(1))
    encoder = dict(pad_id=env.pad_token, vocab_size=env.ntoken,
                   max_length=env.max_length, hidden_size=4, depth=1)
    model = trpo.MLPPolicy(env.action_space.n, env.backward_action_space.n,
                           False, encoder, key=jax.random.key(2))
    baseline = trpo.BaselineMLP(encoder, key=jax.random.key(3))
    config = OmegaConf.create({
        "num_envs": 2, "num_train_steps": 2,
        "agent": {
            "gae_lambda": 0.7, "baseline_epochs": 1, "baseline_num_splits": 1,
            "trpo_delta": 0.01, "cg_iters": 2, "cg_damping": 0.0,
            "line_search_iters": 2, "line_search_shrink": 0.5,
            "center_advantages": True,
        },
        "logging": {"eval_each": 10, "track_each": 10, "use_writer": False},
    })
    baseline_optimizer = optax.sgd(0.01)
    # A unit SGD step exposes the actual cost used for training logZ.
    logz_optimizer = optax.sgd(1.0)
    logz = jnp.array(0.4)
    metrics = MultiMetricsModule(metrics={})
    metrics_state = metrics.init(jax.random.key(4), metrics.InitArgs(metrics_args={}))
    state = trpo.TrainState(
        rng_key=jax.random.key(5), config=config, env=env, env_params=env_params,
        model=model, baseline=baseline, logZ=logz,
        baseline_optimizer=baseline_optimizer, logZ_optimizer=logz_optimizer,
        baseline_opt_state=baseline_optimizer.init(eqx.filter(baseline, eqx.is_array)),
        logZ_opt_state=logz_optimizer.init(logz), metrics_module=metrics,
        metrics_state=metrics_state, exploration_schedule=optax.constant_schedule(0.0),
        eval_info={},
    )

    # Independently evaluate the same sampled trajectories with the library's
    # trajectory-density evaluator, which includes every backward edge.
    def policy_fn(key, obs, params):
        outputs = jax.vmap(model)(obs)
        return outputs["forward_logits"], outputs

    _, sample_key = jax.random.split(state.rng_key)
    trajectories, info = gfnx.utils.forward_rollout(
        sample_key, config.num_envs, policy_fn, None, env, env_params,
    )
    log_pf, log_pb = gfnx.utils.forward_trajectory_log_probs(env, trajectories, env_params)
    np.testing.assert_allclose(log_pb, -env.max_length * np.log(2), rtol=1e-6)
    expected_cost = (log_pf - log_pb - info["log_gfn_reward"] + logz).mean()

    updated = trpo.train_step(0, state)
    np.testing.assert_allclose(updated.logZ, logz - expected_cost, rtol=1e-6, atol=1e-6)
    for leaf in jax.tree.leaves(eqx.filter((updated.model, updated.baseline), eqx.is_array)):
        assert np.isfinite(leaf).all()
