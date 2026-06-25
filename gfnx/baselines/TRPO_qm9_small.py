"""Single-file implementation for Trust Region Policy Optimization (TRPO) in QM9Small environment.

Run the script with the following command:
```bash
python baselines/TRPO_qm9_small.py
```

Also see https://jax.readthedocs.io/en/latest/gpu_performance_tips.html for
performance tips when running on GPU, i.e., XLA flags.

"""

import functools
import logging
import os
from typing import NamedTuple

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import optax
from jax_tqdm import loop_tqdm
from jaxtyping import Array, Int
from omegaconf import OmegaConf

from jax.lax import stop_gradient

import gfnx
from gfnx.metrics import (
    ApproxDistributionMetricsModule,
    ELBOMetricsModule,
    EUBOMetricsModule,
    ExactDistributionMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
    SWMeanRewardSWMetricsModule,
)

from utils.logger import Writer
from utils.checkpoint import save_checkpoint

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class MLPPolicy(eqx.Module):
    """
    A policy module that uses a Multi-Layer Perceptron (MLP) to generate
    forward and backward action logits as well as a flow.
    """

    encoder: gfnx.networks.Encoder
    pooler: eqx.nn.Linear
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int
    vocab_size: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        train_backward_policy: bool,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions
        self.vocab_size = encoder_params["vocab_size"]

        output_size = self.n_fwd_actions
        if train_backward_policy:
            output_size += n_bwd_actions

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            in_size=encoder_params["max_length"] * encoder_params["vocab_size"],
            out_size=encoder_params["hidden_size"],
            width_size=encoder_params["hidden_size"],
            depth=encoder_params["depth"],
            key=encoder_key,
        )
        self.pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=output_size,
            key=pooler_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        obs_ids = jax.nn.one_hot(obs_ids[1:], self.vocab_size).reshape(-1)
        encoded_obs = self.encoder(obs_ids)
        output = self.pooler(encoded_obs)
        if self.train_backward_policy:
            forward_logits, backward_logits = jnp.split(output, [self.n_fwd_actions], axis=-1)
        else:
            forward_logits= output
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }
    

class BaselineMLP(eqx.Module):
    encoder: gfnx.networks.Encoder
    pooler: eqx.nn.Linear
    vocab_size: int

    def __init__(
        self,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.vocab_size = encoder_params["vocab_size"]

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            in_size=encoder_params["max_length"] * encoder_params["vocab_size"],
            out_size=encoder_params["hidden_size"],
            width_size=encoder_params["hidden_size"],
            depth=encoder_params["depth"],
            key=encoder_key,
        )
        self.pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=1,
            key=pooler_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        obs_ids = jax.nn.one_hot(obs_ids[1:], self.vocab_size).reshape(-1)
        encoded_obs = self.encoder(obs_ids)
        output = self.pooler(encoded_obs)
        return output.squeeze(-1)


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.QM9SmallEnvironment
    env_params: chex.Array
    model: MLPPolicy
    baseline: BaselineMLP
    logZ: chex.Array
    baseline_optimizer: optax.GradientTransformation
    logZ_optimizer: optax.GradientTransformation
    baseline_opt_state: optax.OptState
    logZ_opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
    exploration_schedule: optax.Schedule
    eval_info: dict


def tree_dot(tree_a, tree_b):
    return sum(
        jnp.sum(a * b)
        for a, b in zip(jax.tree.leaves(tree_a), jax.tree.leaves(tree_b))
    )


def tree_mul(tree, scale):
    return jax.tree.map(lambda x: x * scale, tree)


def tree_add(tree_a, tree_b, scale=1.0):
    return jax.tree.map(lambda a, b: a + scale * b, tree_a, tree_b)


def tree_where(pred, tree_a, tree_b):
    return jax.tree.map(lambda a, b: jnp.where(pred, a, b), tree_a, tree_b)


def tree_zeros_like(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def conjugate_gradients(matvec, b, num_steps):
    x = tree_zeros_like(b)
    r = b
    p = b
    rdotr = tree_dot(r, r)

    def body_fn(_, carry):
        x, r, p, rdotr = carry
        Ap = matvec(p)
        alpha = rdotr / (tree_dot(p, Ap) + 1e-8)
        x = tree_add(x, p, alpha)
        r = tree_add(r, Ap, -alpha)
        new_rdotr = tree_dot(r, r)
        beta = new_rdotr / (rdotr + 1e-8)
        p = tree_add(r, p, beta)
        return x, r, p, new_rdotr

    x, _, _, _ = jax.lax.fori_loop(0, num_steps, body_fn, (x, r, p, rdotr))
    return x


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params
    # Step 1. Generate a batch of trajectories and split to transitions
    rng_key, sample_traj_key = jax.random.split(train_state.rng_key)
    # Split the model to pass into forward rollout
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
    baseline_params, baseline_static = eqx.partition(train_state.baseline, eqx.is_array)

    # Get epsilon exploration value from config
    cur_eps = train_state.exploration_schedule(idx)
    gae_lambda = train_state.config.agent.gae_lambda
    baseline_epochs = train_state.config.agent.baseline_epochs
    baseline_num_splits = train_state.config.agent.baseline_num_splits
    trpo_delta = train_state.config.agent.trpo_delta
    cg_iters = train_state.config.agent.cg_iters
    cg_damping = train_state.config.agent.cg_damping
    line_search_iters = train_state.config.agent.line_search_iters
    line_search_shrink = train_state.config.agent.line_search_shrink
    center_advantages = train_state.config.agent.center_advantages

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        current_policy_params,  # current_policy_params are network params
        train=True,
    ) -> chex.Array:
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)

        # Get forward logits
        fwd_logits = policy_outputs["forward_logits"]

        # Apply epsilon exploration to logits
        if train:
            rng_key, exploration_key = jax.random.split(fwd_rng_key)
            batch_size, _ = fwd_logits.shape
            exploration_mask = jax.random.bernoulli(exploration_key, cur_eps, (batch_size,))
            fwd_logits = jnp.where(exploration_mask[..., None], 0, fwd_logits)
        # Update policy outputs with modified logits
        policy_outputs = policy_outputs.copy()
        policy_outputs["forward_logits"] = fwd_logits

        return fwd_logits, policy_outputs

    # Generating the trajectory and splitting it into transitions
    traj_data, log_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,
        env=train_state.env,
        env_params=train_state.env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(
        env, traj_data, env_params
    )
    rl_reward = log_pb_traj + log_info["log_gfn_reward"] + log_info["entropy"]

    def extract_advantage_components(
        model: MLPPolicy,
        baseline: BaselineMLP,
        logZ_val: chex.Array,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.QM9SmallEnvironment,
        current_env_params: gfnx.QM9SmallEnvParams,
    ):
        policy_outputs_traj = jax.vmap(jax.vmap(model))(current_traj_data.obs)
        baseline_values = jax.vmap(jax.vmap(baseline))(current_traj_data.obs) 

        # Step 2.1 Compute forward actions and log probabilities
        fwd_logits_traj = policy_outputs_traj["forward_logits"]

        # Vmap get_fwd_masks_per_step over the time dimension. For each time
        # step t, it processes the batch of states state[:, t, ...].
        invalid_fwd_mask = jax.vmap(current_env.get_invalid_mask, in_axes=(1, None), out_axes=1)(
            current_traj_data.state, current_env_params
        )

        masked_fwd_logits_traj = gfnx.utils.mask_logits(fwd_logits_traj, invalid_fwd_mask)
        fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)

        fwd_logprobs_traj = jnp.take_along_axis(
            fwd_all_log_probs_traj,
            jnp.expand_dims(current_traj_data.action, axis=-1),
            axis=-1,
        ).squeeze(-1)

        fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)


        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state) # [B, T]
        fwd_actions = current_traj_data.action[:, :-1] # [B, T]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state) # [B, T]

        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action,
            in_axes=(1, 1, 1, None),
            out_axes=1,
        )(prev_states, fwd_actions, curr_states, current_env_params) # [B, T]

        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]
        # Vmap get_bwd_masks_per_step over the time dimension.
        invalid_bwd_mask = jax.vmap(
            current_env.get_invalid_backward_mask,
            in_axes=(1, None),
            out_axes=1,
        )(curr_states, current_env_params)

        masked_bwd_logits_traj = gfnx.utils.mask_logits(bwd_logits_for_pb, invalid_bwd_mask)
        bwd_all_log_probs_traj = jax.nn.log_softmax(masked_bwd_logits_traj, axis=-1)

        log_pb_selected = jnp.take_along_axis(
            bwd_all_log_probs_traj,
            jnp.expand_dims(bwd_actions_traj, axis=-1),
            axis=-1,
        ).squeeze(-1)

        pad_mask_for_bwd = current_traj_data.pad[:, :-1]
        log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

        log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
        masked_log_rewards_at_steps = jnp.where(pad_mask_for_bwd, 0.0, log_rewards_at_steps)
        terminal_step_mask = jnp.logical_and(
            current_traj_data.done[:, :-1],
            jnp.logical_not(pad_mask_for_bwd),
        )

        V_pred = baseline_values[:, :-1]
        masked_V_pred = jnp.where(pad_mask_for_bwd, 0.0, V_pred)

        pb_or_terminal = jnp.where(
            terminal_step_mask,
            masked_log_rewards_at_steps - logZ_val,
            log_pb_selected,
        )
        scores = fwd_logprobs_traj[:, :-1] - pb_or_terminal
        scores = jnp.where(pad_mask_for_bwd, 0.0, scores)

        return {
            "full_forward_logprobs": fwd_all_log_probs_traj[:, :-1],
            "gflow_logreward": masked_log_rewards_at_steps, # (B, T)
            "backward_logprobs": log_pb_selected, # (B, T)
            "scores": scores,
            "V_pred": masked_V_pred,
            "forward_logprobs": fwd_logprobs_traj[:, :-1],
            "pad_mask": pad_mask_for_bwd,
            "terminal_step_mask": terminal_step_mask,
            "obs": current_traj_data.obs,
        }
    
    def compute_gae(deltas, lam):
        def scan_fn(carry, delta):
            return delta + lam * carry, delta + lam * carry
        _, adv_rev = jax.lax.scan(scan_fn, init=0.0, xs=deltas[::-1])
        return adv_rev[::-1]
    
    def baseline_loss_fn(
        baseline_params: BaselineMLP,
        aux_info: dict
    ):
        pad_mask = aux_info['pad_mask']
        value_target = aux_info["value_target"]

        baseline = eqx.combine(baseline_params, baseline_static)
        baseline_values = jax.vmap(jax.vmap(baseline))(aux_info['obs'])

        V_current = baseline_values[:, :-1]
        V_current = jnp.where(pad_mask, 0.0, V_current)
        sq_error = (V_current - value_target) ** 2
        valid = jnp.logical_not(pad_mask)
        return jnp.where(valid, sq_error, 0.0).sum() / jnp.maximum(valid.sum(), 1)
    
    comp_old = extract_advantage_components(
        train_state.model,
        train_state.baseline,
        train_state.logZ,
        traj_data,
        env,
        env_params,
    )

    V_current = comp_old['V_pred']
    scores = stop_gradient(comp_old['scores'])
    pad_mask = stop_gradient(comp_old['pad_mask'])
    valid = jnp.logical_not(pad_mask)
    n_valid = jnp.maximum(valid.sum(), 1)

    V_next = jnp.roll(V_current, -1, axis=1).at[:, -1].set(0.0)
    deltas = scores + V_next - V_current
    deltas = jnp.where(pad_mask, 0.0, deltas)

    raw_advantages = jax.vmap(compute_gae, in_axes=(0, None))(deltas, gae_lambda)
    value_target = stop_gradient(raw_advantages + V_current)
    advantages_old = stop_gradient(raw_advantages)
    mean_advantage = jnp.where(valid, advantages_old, 0.0).sum() / n_valid
    advantages_old = jnp.where(
        center_advantages,
        advantages_old - mean_advantage,
        advantages_old,
    )
    advantages_old = jnp.where(pad_mask, 0.0, advantages_old)

    trpo_aux_info = {
        'advantages_old': advantages_old,
        'log_pf_old_full': stop_gradient(comp_old["full_forward_logprobs"]),
        'log_pf_old_sampled': stop_gradient(comp_old["forward_logprobs"]),
        'pad_mask': pad_mask,
    }

    baseline_aux_info = {
        'value_target': value_target,
        'pad_mask': pad_mask,
        'obs': traj_data.obs,
    }

    def baseline_epoch_body(epoch_i, carry):
        b_params, b_opt_state, _ = carry
        B = baseline_aux_info['pad_mask'].shape[0]
        split_size = B // baseline_num_splits
        total_loss = 0.0
 
        for i in range(baseline_num_splits):
            start = i * split_size
            end = (i + 1) * split_size
 
            chunk_aux = jax.tree.map(lambda x: x[start:end], baseline_aux_info)
 
            chunk_loss, chunk_grads = eqx.filter_value_and_grad(baseline_loss_fn)(
                b_params, chunk_aux
            )
            b_updates, b_opt_state = train_state.baseline_optimizer.update(
                chunk_grads, b_opt_state, b_params
            )
            b_params = optax.apply_updates(b_params, b_updates)
            total_loss += chunk_loss
 
        return (b_params, b_opt_state, total_loss / baseline_num_splits)

    final_b_params, final_b_opt_state, baseline_loss = jax.lax.fori_loop(
        lower=0,
        upper=baseline_epochs,
        body_fun=baseline_epoch_body,
        init_val=(baseline_params, train_state.baseline_opt_state, 0.0),
    )

    traj_score = scores.sum(axis=1)
    score_mean = traj_score.mean()

    def logZ_loss_fn(logZ_val):
        return logZ_val * stop_gradient(score_mean)

    logZ_loss, logZ_grad = eqx.filter_value_and_grad(logZ_loss_fn)(train_state.logZ)
    logZ_updates, final_logZ_opt_state = train_state.logZ_optimizer.update(
        logZ_grad, train_state.logZ_opt_state, train_state.logZ
    )
    final_logZ = optax.apply_updates(train_state.logZ, logZ_updates)

    def forward_log_probs_from_params(model_params, current_traj_data):
        model_to_call = eqx.combine(model_params, policy_static)
        policy_outputs_traj = jax.vmap(jax.vmap(model_to_call))(current_traj_data.obs)
        fwd_logits_traj = policy_outputs_traj["forward_logits"]
        invalid_fwd_mask = jax.vmap(env.get_invalid_mask, in_axes=(1, None), out_axes=1)(
            current_traj_data.state, env_params
        )
        masked_fwd_logits_traj = gfnx.utils.mask_logits(fwd_logits_traj, invalid_fwd_mask)
        fwd_all_log_probs_traj = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)
        fwd_logprobs_traj = jnp.take_along_axis(
            fwd_all_log_probs_traj,
            jnp.expand_dims(current_traj_data.action, axis=-1),
            axis=-1,
        ).squeeze(-1)
        fwd_logprobs_traj = jnp.where(current_traj_data.pad, 0.0, fwd_logprobs_traj)
        return fwd_all_log_probs_traj[:, :-1], fwd_logprobs_traj[:, :-1]

    def policy_surrogate_loss(model_params, aux_info):
        _, log_pf_new_sampled = forward_log_probs_from_params(model_params, traj_data)
        ratio = jnp.exp(log_pf_new_sampled - aux_info["log_pf_old_sampled"])
        surrogate = ratio * aux_info["advantages_old"]
        surrogate = jnp.where(aux_info["pad_mask"], 0.0, surrogate)
        return jnp.sum(surrogate, axis=1).mean()

    def policy_kl_old_new(model_params, aux_info):
        log_pf_new_full, _ = forward_log_probs_from_params(model_params, traj_data)
        log_pf_old_full = aux_info["log_pf_old_full"]
        old_probs = jnp.exp(log_pf_old_full)
        kl = old_probs * (log_pf_old_full - log_pf_new_full)
        kl = jnp.where(aux_info["pad_mask"][..., None], 0.0, kl).sum(axis=-1)
        return jnp.sum(kl, axis=1).mean()

    policy_loss, policy_grads = eqx.filter_value_and_grad(policy_surrogate_loss)(
        policy_params, trpo_aux_info
    )

    def hvp_fn(vector):
        def kl_loss_fn(model_params):
            return policy_kl_old_new(model_params, trpo_aux_info)

        _, hvp = jax.jvp(eqx.filter_grad(kl_loss_fn), (policy_params,), (vector,))
        return tree_add(hvp, vector, cg_damping)

    search_dir = conjugate_gradients(hvp_fn, policy_grads, cg_iters)
    shs = 0.5 * tree_dot(search_dir, hvp_fn(search_dir))
    step_scale = jnp.sqrt(trpo_delta / (shs + 1e-8))
    step_scale = jnp.where(jnp.isfinite(step_scale), step_scale, 0.0)
    full_step = tree_mul(search_dir, step_scale)

    def line_search_body(i, carry):
        best_params, best_loss, best_kl, best_fraction, accepted = carry
        fraction = jnp.power(line_search_shrink, i)
        candidate_params = tree_add(policy_params, full_step, -fraction)
        candidate_loss = policy_surrogate_loss(candidate_params, trpo_aux_info)
        candidate_kl = policy_kl_old_new(candidate_params, trpo_aux_info)
        improves = policy_loss - candidate_loss
        candidate_ok = (
            jnp.isfinite(candidate_loss)
            & jnp.isfinite(candidate_kl)
            & (improves > 0.0)
            & (candidate_kl <= trpo_delta)
        )
        use_candidate = jnp.logical_and(jnp.logical_not(accepted), candidate_ok)
        best_params = tree_where(use_candidate, candidate_params, best_params)
        best_loss = jnp.where(use_candidate, candidate_loss, best_loss)
        best_kl = jnp.where(use_candidate, candidate_kl, best_kl)
        best_fraction = jnp.where(use_candidate, fraction, best_fraction)
        accepted = jnp.logical_or(accepted, candidate_ok)
        return best_params, best_loss, best_kl, best_fraction, accepted

    final_p_params, final_policy_loss, final_policy_kl, step_fraction, trpo_accepted = (
        jax.lax.fori_loop(
            lower=0,
            upper=line_search_iters,
            body_fun=line_search_body,
            init_val=(policy_params, policy_loss, 0.0, 0.0, False),
        )
    )

    log_pf_new_full, log_pf_new_sampled = forward_log_probs_from_params(
        final_p_params, traj_data
    )
    log_ratio = log_pf_new_sampled - trpo_aux_info["log_pf_old_sampled"]
    ratio = jnp.exp(log_ratio)
    mean_ratio = jnp.where(valid, ratio, 0.0).sum() / n_valid
    mean_log_ratio = jnp.where(valid, log_ratio, 0.0).sum() / n_valid
    max_ratio = jnp.where(valid, ratio, 0.0).max()
    final_kl_per_step = jnp.exp(trpo_aux_info["log_pf_old_full"]) * (
        trpo_aux_info["log_pf_old_full"] - log_pf_new_full
    )
    final_kl_per_step = jnp.where(pad_mask[..., None], 0.0, final_kl_per_step).sum(axis=-1)
    mean_kl_per_step = final_kl_per_step.sum() / n_valid

    rewards = env.reward_module.reward(
        log_info["final_env_state"],
        env_params=env_params,
    )
    metrics_state = train_state.metrics_module.update(
        train_state.metrics_state,
        rng_key=jax.random.key(0),  # not used, but required by the API
        args=train_state.metrics_module.UpdateArgs(
            metrics_args={
                "approx_dist": ApproxDistributionMetricsModule.UpdateArgs(
                    states=log_info["final_env_state"]
                ),
                "exact_dist": ExactDistributionMetricsModule.UpdateArgs(),
                "elbo": ELBOMetricsModule.UpdateArgs(),
                "eubo": EUBOMetricsModule.UpdateArgs(),
                "rd": SWMeanRewardSWMetricsModule.UpdateArgs(
                    rewards=rewards,
                ),
            }
        ),
    )

    rng_key, eval_rng_key = jax.random.split(rng_key)
    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: train_state.metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": eval_rng_key,
            "args": train_state.metrics_module.ProcessArgs(
                metrics_args={
                    "approx_dist": ApproxDistributionMetricsModule.ProcessArgs(
                        env_params=env_params
                    ),
                    "exact_dist": ExactDistributionMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=env_params
                    ),
                    "elbo": ELBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=env_params
                    ),
                    "eubo": EUBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=env_params
                    ),
                    "rd": SWMeanRewardSWMetricsModule.ProcessArgs(),
                }
            ),
        },
    )
    eval_info = jax.lax.cond(
        is_eval_step,
        lambda metrics_state: train_state.metrics_module.get(metrics_state),
        lambda metrics_state: train_state.eval_info,  # Do nothing if not eval step
        metrics_state,
    )

    # Perform the logging via JAX debug callback
    def logging_callback(
        idx: int, train_info: dict, eval_info: dict, cfg
    ):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}

        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            eval_info = {f"eval/{key}": float(value) for key, value in eval_info.items()}
            log.info(eval_info)
            if cfg.logging.use_writer:
                writer.log(eval_info, step=idx)

        if cfg.logging.use_writer and idx % cfg.logging.track_each == 0:
            writer.log(train_info)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "mean_loss": final_policy_loss,
            "baseline_loss": baseline_loss,
            "logZ_loss": logZ_loss,
            "logZ": final_logZ,
            "score_mean": score_mean,
            "score_std": traj_score.std(),
            "trpo_accepted": trpo_accepted.astype(jnp.float32),
            "trpo_kl": final_policy_kl,
            "trpo_kl_per_step": mean_kl_per_step,
            "trpo_step_fraction": step_fraction,
            "entropy": log_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(policy_grads),
            "natural_step_norm": jnp.sqrt(2.0 * jnp.maximum(shs, 0.0)),
            "mean_reward": jnp.exp(log_info["log_gfn_reward"]).mean(),
            "mean_log_reward": log_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
            "mean_importance_weight": mean_ratio,
            "mean_log_importance_weight": mean_log_ratio,
            "max_importance_weight": max_ratio,
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    new_model = eqx.combine(final_p_params, policy_static)
    new_baseline = eqx.combine(final_b_params, baseline_static)

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        baseline=new_baseline,
        logZ=final_logZ,
        baseline_opt_state=final_b_opt_state,
        logZ_opt_state=final_logZ_opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="TRPO_qm9_small", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.agent.train_backward:
        raise ValueError("TRPO_qm9_small keeps P_B fixed; set agent.train_backward=false.")

    rng_key = jax.random.PRNGKey(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    # Define the reward function for the environment
    reward_module = gfnx.QM9SmallRewardModule()
    # Initialize the environment and its inner parameters
    env = gfnx.QM9SmallEnvironment(reward_module)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = MLPPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length,
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
    )

    rng_key, baseline_init_key = jax.random.split(rng_key)
    baseline = BaselineMLP(
        encoder_params={
            "pad_id": env.pad_token,
            "vocab_size": env.ntoken,
            "max_length": env.max_length,
            **OmegaConf.to_container(cfg.network),
        },
        key=baseline_init_key,
    )

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    baseline_params = eqx.filter(baseline, eqx.is_array)
    logZ = jnp.array(0.0)

    baseline_optimizer = optax.adam(learning_rate=cfg.agent.baseline_learning_rate)
    logZ_optimizer = optax.adam(learning_rate=cfg.agent.logZ_learning_rate)

    baseline_opt_state = baseline_optimizer.init(baseline_params)
    logZ_opt_state = logZ_optimizer.init(logZ)

    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def fwd_policy_fn(
        fwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        policy_params,  # current_policy_params are network params
    ) -> chex.Array:
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def bwd_policy_fn(
        bwd_rng_key: chex.PRNGKey,
        env_obs: gfnx.TObs,
        policy_params,  # current_policy_params are network params
    ) -> chex.Array:
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["backward_logits"], policy_outputs


    metrics_module = MultiMetricsModule(
        metrics={
            "approx_dist": ApproxDistributionMetricsModule(
                metrics=["tv", "kl"],
                env=env,
                buffer_size=cfg.logging.metric_buffer_size,
            ),
            "exact_dist": ExactDistributionMetricsModule(
                metrics=["tv", "kl"],
                env=env,
                fwd_policy_fn=fwd_policy_fn,
                batch_size=cfg.metrics.batch_size,
            ),
            "elbo": ELBOMetricsModule(
                env=env,
                env_params=env_params,
                fwd_policy_fn=fwd_policy_fn,
                n_rounds=cfg.metrics.n_rounds,
                batch_size=cfg.num_envs,
            ),
            "eubo": EUBOMetricsModule(
                env=env,
                env_params=env_params,
                bwd_policy_fn=bwd_policy_fn,
                n_rounds=cfg.metrics.n_rounds,
                batch_size=cfg.num_envs,
                rng_key=eval_init_key,
            ),
            "rd": SWMeanRewardSWMetricsModule(
                env=env,
                env_params=env_params,
                buffer_size=cfg.logging.metric_buffer_size,
            ),
        }
    )
    # Fill the initial states of metrics
    metrics_state = metrics_module.init(
        rng_key=eval_init_key,
        args=metrics_module.InitArgs(
            metrics_args={
                "approx_dist": ApproxDistributionMetricsModule.InitArgs(env_params=env_params),
                "exact_dist": ExactDistributionMetricsModule.InitArgs(env_params=env_params),
                "elbo": ELBOMetricsModule.InitArgs(),
                "eubo": EUBOMetricsModule.InitArgs(),
                "rd": SWMeanRewardSWMetricsModule.InitArgs(),
            }
        ),
    )
    eval_info = metrics_module.get(metrics_state)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        baseline=baseline,
        logZ=logZ,
        baseline_optimizer=baseline_optimizer,
        logZ_optimizer=logZ_optimizer,
        baseline_opt_state=baseline_opt_state,
        logZ_opt_state=logZ_opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        exploration_schedule=exploration_schedule,
        eval_info=eval_info,
    )
    # Split train state into parameters and static parts to make jit work.
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, train_state_params):
        # Wrapper to use a usual jit in jax, since it is required by fori_loop.
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

    if cfg.logging.use_writer:
        log.info("Initialize writer")
        log_dir = cfg.logging.log_dir if cfg.logging.log_dir else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"run_{os.getpid()}/"
        )
        writer.init(
            writer_type=cfg.writer.writer_type,
            save_locally=cfg.writer.save_locally,
            log_dir=log_dir,
            entity=cfg.writer.entity,
            project=cfg.writer.project,
            offline_directory=cfg.writer.get("offline_directory", "./comet_offline_logs"),
            tags=["TRPO", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    # Run the training loop via jax lax.fori_loop
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )
    jax.block_until_ready(train_state_params)

    # Save the final model
    train_state = eqx.combine(train_state_params, train_state_static)
    dir = cfg.logging.checkpoint_dir if cfg.logging.checkpoint_dir else os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        f"checkpoints_{os.getpid()}/",
    )
    save_checkpoint(
        os.path.join(dir, "model_and_baseline"),
        {
            "model": train_state.model,
            "baseline": train_state.baseline,
            "logZ": train_state.logZ,
        },
    )


if __name__ == "__main__":
    run_experiment()
