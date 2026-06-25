"""Single-file implementation for GFN-PG/TRPO in the hypergrid environment.

Run the script with the following command:
```bash
python baselines/TRPO_hypergrid.py
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
from omegaconf import OmegaConf

import gfnx
from gfnx.metrics import (
    ApproxDistributionMetricsModule,
    ELBOMetricsModule,
    EUBOMetricsModule,
    ExactDistributionMetricsModule,
    MultiMetricsModule,
    MultiMetricsState,
)

from jax.lax import stop_gradient

import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL.Image import open as pil_open

from utils.logger import Writer
from utils.checkpoint import save_checkpoint

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class MLPPolicy(eqx.Module):
    """
    A policy module that uses a Multi-Layer Perceptron (MLP) to generate
    forward and backward action logits.

    Args:
        input_size (int): The size of the input features.
        n_fwd_actions (int): Number of forward actions.
        n_bwd_actions (int): Number of backward actions.
        hidden_size (int): The size of the hidden layers in the MLP.
        train_backward_policy (bool): Flag indicating whether to train
            the backward policy.
        depth (int): The number of layers in the MLP.
        rng_key (chex.PRNGKey): Random key for initializing the MLP.

    Methods:
        __call__(x: chex.Array) -> chex.Array:
            Forward pass through the MLP network. Returns a dictionary
            containing forward logits and backward logits.
    """

    network: eqx.nn.MLP
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(
        self,
        input_size: int,
        n_fwd_actions: int,
        n_bwd_actions: int,
        hidden_size: int,
        train_backward_policy: bool,
        depth: int,
        rng_key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        output_size = self.n_fwd_actions
        if train_backward_policy:
            output_size += n_bwd_actions
        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.network(x)
        if self.train_backward_policy:
            forward_logits, backward_logits = jnp.split(x, [self.n_fwd_actions], axis=-1)
        else:
            forward_logits = x
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }
    
class BaselineMLP(eqx.Module):
    network: eqx.nn.MLP

    def __init__(self, input_size: int, hidden_size: int, depth: int, rng_key: chex.PRNGKey):
        self.network = eqx.nn.MLP(
            in_size=input_size,
            out_size=1,
            width_size=hidden_size,
            depth=depth,
            key=rng_key,
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        return self.network(x).squeeze(-1)

# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.HypergridEnvironment
    env_params: chex.Array
    model: MLPPolicy
    baseline: BaselineMLP
    logZ: chex.Array
    exploration_schedule: optax.Schedule
    baseline_optimizer: optax.GradientTransformation
    logZ_optimizer: optax.GradientTransformation
    baseline_opt_state: optax.OptState
    logZ_opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
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


def compute_gae(deltas, lam):
    def scan_fn(carry, delta):
        advantage = delta + lam * carry
        return advantage, advantage

    _, adv_rev = jax.lax.scan(scan_fn, init=0.0, xs=deltas[::-1])
    return adv_rev[::-1]


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params

    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
    baseline_params, baseline_static = eqx.partition(train_state.baseline, eqx.is_array)

    rng_key, sample_traj_key = jax.random.split(rng_key)

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

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        current_model = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        fwd_logits = policy_outputs["forward_logits"]

        rng_key, exploration_key = jax.random.split(rng_key)
        batch_size, _ = fwd_logits.shape
        exploration_mask = jax.random.bernoulli(exploration_key, cur_eps, (batch_size,))
        fwd_logits = jnp.where(exploration_mask[..., None], 0, fwd_logits)
        return fwd_logits, policy_outputs

    traj_data, aux_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,  # Pass only network parameters
        env=env,
        env_params=env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(
        env, traj_data, env_params
    )
    rl_reward = log_pb_traj + aux_info["log_gfn_reward"] + aux_info["entropy"]

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

    def extract_advantage_components(
        model_params,
        baseline_params,
        logZ_val,
        current_traj_data: gfnx.utils.TrajectoryData,
    ):
        full_fwd_logprobs, fwd_logprobs = forward_log_probs_from_params(
            model_params, current_traj_data
        )
        baseline_to_call = eqx.combine(baseline_params, baseline_static)
        baseline_values = jax.vmap(jax.vmap(baseline_to_call))(current_traj_data.obs)

        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

        invalid_bwd_mask = jax.vmap(
            env.get_invalid_backward_mask,
            in_axes=(1, None),
            out_axes=1,
        )(curr_states, env_params)

        num_valid_bwd = jnp.logical_not(invalid_bwd_mask).sum(axis=-1).astype(jnp.float32)
        log_pb_selected = -jnp.log(jnp.maximum(num_valid_bwd, 1.0))
        pad_mask_for_bwd = current_traj_data.pad[:, :-1]
        log_pb_selected = jnp.where(pad_mask_for_bwd, 0.0, log_pb_selected)

        log_rewards_at_steps = current_traj_data.log_gfn_reward[:, :-1]
        masked_log_rewards_at_steps = jnp.where(pad_mask_for_bwd, 0.0, log_rewards_at_steps)
        terminal_step_mask = jnp.logical_and(curr_states.is_terminal, jnp.logical_not(pad_mask_for_bwd))

        V_pred = baseline_values[:, :-1]
        masked_V_pred = jnp.where(pad_mask_for_bwd, 0.0, V_pred)

        pb_or_terminal = jnp.where(
            terminal_step_mask,
            masked_log_rewards_at_steps - logZ_val,
            log_pb_selected,
        )
        scores = fwd_logprobs - pb_or_terminal
        scores = jnp.where(pad_mask_for_bwd, 0.0, scores)

        return {
            "full_forward_logprobs": full_fwd_logprobs,
            "gflow_logreward": masked_log_rewards_at_steps,
            "backward_logprobs": log_pb_selected,
            "scores": scores,
            "V_pred": masked_V_pred,
            "forward_logprobs": fwd_logprobs,
            "pad_mask": pad_mask_for_bwd,
            "terminal_step_mask": terminal_step_mask,
            "obs": current_traj_data.obs,
        }

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
        policy_params, baseline_params, train_state.logZ, traj_data
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

    # Perform all the required logging
    metrics_state = train_state.metrics_module.update(
        train_state.metrics_state,
        rng_key=jax.random.key(0),  # This key is not used in the update method
        args=train_state.metrics_module.UpdateArgs(
            metrics_args={
                "approx_dist": ApproxDistributionMetricsModule.UpdateArgs(
                    states=aux_info["final_env_state"]
                ),
                "exact_dist": ExactDistributionMetricsModule.UpdateArgs(),
                "elbo": ELBOMetricsModule.UpdateArgs(),
                "eubo": EUBOMetricsModule.UpdateArgs(),
            }
        ),
    )

    # Perform evaluation computations if needed
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: train_state.metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],  # Do nothing if not eval step
        {
            "metrics_state": metrics_state,
            "rng_key": jax.random.key(0),  # This key is not used in the process method
            "args": train_state.metrics_module.ProcessArgs(
                metrics_args={
                    "approx_dist": ApproxDistributionMetricsModule.ProcessArgs(
                        env_params=env_params
                    ),
                    "exact_dist": ExactDistributionMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=train_state.env_params
                    ),
                    "elbo": ELBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=train_state.env_params
                    ),
                    "eubo": EUBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=train_state.env_params
                    ),
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
        idx: int,
        train_info: dict,
        eval_info: dict,
        cfg,
    ):
        train_info = {
            f"train/{k}": float(v) for k, v in train_info.items()
        }

        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            # Get the evaluation metrics
            eval_info_for_log = {
                f"eval/{key}": float(value)
                for key, value in eval_info.items()
                if "2d_marginal_distribution" not in key
            }
            log.info({
                key: value
                for key, value in eval_info_for_log.items()
                if "2d_marginal_distribution" not in key
            })

            if cfg.logging.use_writer:
                marginal_dist = eval_info["approx_dist/2d_marginal_distribution"]
                marginal_dist = (marginal_dist - marginal_dist.min()) / (
                    marginal_dist.max() - marginal_dist.min()
                )

                plt.figure(figsize=(4, 4))
                im = plt.imshow(marginal_dist, cmap='viridis')
                plt.colorbar(im)
                plt.title(f"2D Marginal Distribution (Step {idx})")
                
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                plt.close()
                buf.seek(0)
                pil_img = pil_open(buf)

                writer.Image(
                    pil_img,
                    caption="approx_dist/marginal_dist",
                )

                writer.log(eval_info_for_log, step=idx)

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
            "entropy": aux_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(policy_grads),
            "natural_step_norm": jnp.sqrt(2.0 * jnp.maximum(shs, 0.0)),
            "mean_reward": jnp.exp(aux_info["log_gfn_reward"]).mean(),
            "mean_log_reward": aux_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
            "mean_importance_weight": mean_ratio,
            "mean_log_importance_weight": mean_log_ratio,
            "max_importance_weight": max_ratio,
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    new_model = eqx.combine(final_p_params, policy_static)
    new_baseline = eqx.combine(final_b_params, baseline_static)

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


@hydra.main(config_path="configs/", config_name="TRPO_hypergrid", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))
    if cfg.agent.train_backward:
        raise ValueError("TRPO_hypergrid keeps P_B fixed; set agent.train_backward=false.")

    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    reward_module_factory = {
        "easy": gfnx.EasyHypergridRewardModule,
        "hard": gfnx.HardHypergridRewardModule,
    }[cfg.environment.reward]
    reward_module = reward_module_factory()

    env = gfnx.environment.HypergridEnvironment(
        reward_module, dim=cfg.environment.dim, side=cfg.environment.side
    )
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    model = MLPPolicy(
        input_size=env.observation_space.shape[0],
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        hidden_size=cfg.network.hidden_size,
        train_backward_policy=cfg.agent.train_backward,
        depth=cfg.network.depth,
        rng_key=net_init_key,
    )

    rng_key, net_init_key = jax.random.split(rng_key)
    baseline = BaselineMLP(
        input_size=env.observation_space.shape[0],
        hidden_size=cfg.network.hidden_size,
        depth=cfg.network.depth,
        rng_key=net_init_key,
    )

    # Partition the model into its learnable parameters and static parts
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)
    def fwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, current_policy_params
    ) -> chex.Array:
        del rng_key
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def bwd_policy_fn(
        rng_key: chex.PRNGKey, env_obs: gfnx.TObs, current_policy_params
    ) -> chex.Array:
        del rng_key
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["backward_logits"], policy_outputs
    
    baseline_params = eqx.filter(baseline, eqx.is_array)
    logZ = jnp.array(0.0)

    baseline_optimizer = optax.adam(learning_rate=cfg.agent.baseline_learning_rate)
    logZ_optimizer = optax.adam(learning_rate=cfg.agent.logZ_learning_rate)

    baseline_opt_state = baseline_optimizer.init(baseline_params)
    logZ_opt_state = logZ_optimizer.init(logZ)

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    metrics_module = MultiMetricsModule({
        "approx_dist": ApproxDistributionMetricsModule(
            metrics=["tv", "kl", "2d_marginal_distribution"],
            env=env,
            buffer_size=cfg.logging.metric_buffer_size,
        ),
        "exact_dist": ExactDistributionMetricsModule(
            metrics=["tv", "kl", "2d_marginal_distribution"],
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
    })
    # Initialize the metrics state
    eval_init_key, new_eval_init_key = jax.random.split(eval_init_key)
    metrics_state = metrics_module.init(
        new_eval_init_key,
        metrics_module.InitArgs(
            metrics_args={
                "approx_dist": ApproxDistributionMetricsModule.InitArgs(env_params=env_params),
                "exact_dist": ExactDistributionMetricsModule.InitArgs(env_params=env_params),
                "elbo": ELBOMetricsModule.InitArgs(),
                "eubo": EUBOMetricsModule.InitArgs(),
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
        baseline = baseline,
        logZ=logZ,
        baseline_optimizer=baseline_optimizer,
        logZ_optimizer=logZ_optimizer,
        baseline_opt_state=baseline_opt_state,
        logZ_opt_state=logZ_opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        eval_info=eval_info,
        exploration_schedule=exploration_schedule,
    )

    # Partition the initial TrainState into dynamic (jittable) and static parts
    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))  # train_state_params is arg 1 (0-indexed)
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, current_train_state_params) -> TrainState:  # Input is params
        # Recombine static and dynamic parts to get the full TrainState
        current_train_state = eqx.combine(current_train_state_params, train_state_static)
        # Call the original JITted train_step
        updated_train_state = train_step(idx, current_train_state)
        # Partition again before returning for the next iteration of the loop
        new_train_state_params, _ = eqx.partition(updated_train_state, eqx.is_array)
        return new_train_state_params

    # Initial train_state_params for the loop
    loop_init_val = train_state_params

    if cfg.logging.use_writer:
        log.info("Initialize writer")
        log_dir = (
            cfg.logging.log_dir
            if cfg.logging.log_dir
            else os.path.join(
                hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, f"run_{os.getpid()}/"
            )
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
    final_train_state_params = jax.lax.fori_loop(  # Result will be params
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,  # body_fun now expects and returns params
        init_val=loop_init_val,  # Pass only the JAX array parts
    )
    final_train_state_params = jax.block_until_ready(final_train_state_params)

    # Save the final model
    final_train_state = eqx.combine(final_train_state_params, train_state_static)
    dir = (
        cfg.logging.checkpoint_dir
        if cfg.logging.checkpoint_dir
        else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            f"checkpoints_{os.getpid()}/",
        )
    )
    save_checkpoint(
        os.path.join(dir, "model_and_baseline"),
        {
            "model": final_train_state.model,
            "baseline": final_train_state.baseline,
            "logZ": final_train_state.logZ,
        },
    )

if __name__ == "__main__":
    run_experiment()
