"""Single-file implementation for Proximal Policy Optimization (PPO) in QM9Small environment.

Run the script with the following command:
```bash
python baselines/PPO_qm9_small.py
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
    policy_optimizer: optax.GradientTransformation
    baseline_optimizer: optax.GradientTransformation
    policy_opt_state: optax.OptState
    baseline_opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
    eval_info: dict
    # TLM
    tlm_backward_optimizer: optax.GradientTransformation
    tlm_backward_opt_state: optax.OptState


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

    gae_lambda = train_state.config.agent.gae_lambda
    ppo_policy_epochs = train_state.config.agent.ppo_policy_epochs
    ppo_baseline_epochs = train_state.config.agent.ppo_baseline_epochs
    ppo_baseline_num_splits = train_state.config.agent.ppo_baseline_num_splits
    clip_eps = train_state.config.agent.clip_eps

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
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.HypergridEnvironment,
        current_env_params: gfnx.HypergridEnvParams,
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
        sum_log_pf_along_traj = fwd_logprobs_traj.sum(axis=1)


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

        V_pred = baseline_values[:, :-1]
        masked_V_pred = jnp.where(pad_mask_for_bwd, 0.0, V_pred)

        return {
            "full_forward_logprobs": fwd_all_log_probs_traj[:, :-1],
            "gflow_logreward": masked_log_rewards_at_steps, # (B, T)
            "backward_logprobs": log_pb_selected, # (B, T)
            "V_pred": masked_V_pred,
            "forward_logprobs": fwd_logprobs_traj[:, :-1],
            "pad_mask": pad_mask_for_bwd,
        }
    
    def compute_gae(deltas, lam):
        def scan_fn(carry, delta):
            return delta + lam * carry, delta + lam * carry
        _, adv_rev = jax.lax.scan(scan_fn, init=0.0, xs=deltas[::-1])
        return adv_rev[::-1]
    
    def policy_loss_fn(
        model_params: MLPPolicy,
        aux_info: dict
    ):
        pad_mask = aux_info['pad_mask']
        advantages_old = aux_info['advantages_old']
        log_pf_old_full = aux_info["log_pf_old_full"]
        log_pf_old_sampled = aux_info["log_pf_old_sampled"]

        model_to_call = eqx.combine(model_params, policy_static)
        policy_outputs = jax.vmap(jax.vmap(model_to_call))(traj_data.obs)
        invalid_fwd_mask = jax.vmap(env.get_invalid_mask, in_axes=(1, None), out_axes=1)(
            traj_data.state, env_params
        )
        masked_fwd_logits_traj = gfnx.utils.mask_logits(policy_outputs["forward_logits"], invalid_fwd_mask)
        log_pf_new_full = jax.nn.log_softmax(masked_fwd_logits_traj, axis=-1)

        fwd_logprobs_traj = jnp.take_along_axis(
            log_pf_new_full,
            jnp.expand_dims(traj_data.action, axis=-1),
            axis=-1,
        ).squeeze(-1)
        log_pf_new_sampled = jnp.where(traj_data.pad, 0.0, fwd_logprobs_traj)[:, :-1]

        ratio = jnp.exp(log_pf_new_sampled - log_pf_old_sampled)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        ppo_clip = jnp.minimum(ratio * advantages_old, clipped_ratio * advantages_old)
        ppo_clip = jnp.where(pad_mask, 0.0, ppo_clip) # (batch, T)
        
        log_pf_new_full = log_pf_new_full[:, :-1]
        kl = jnp.exp(log_pf_new_full) * (log_pf_new_full - log_pf_old_full) # (batch, T, A)
        kl = jnp.where(pad_mask[..., None], 0.0, kl).sum(axis=-1) # (batch, T)

        loss_per_traj = jnp.sum(ppo_clip, axis=1) - jnp.sum(kl, axis=1)
        policy_loss = -jnp.mean(loss_per_traj)


        # For logging
        log_ratio = log_pf_new_sampled - log_pf_old_sampled
        valid = ~pad_mask
        n_valid = valid.sum()
        mean_ratio = jnp.where(valid, ratio, 0.0).sum() / n_valid
        mean_log_ratio = jnp.where(valid, log_ratio, 0.0).sum() / n_valid
        max_ratio = jnp.where(valid, ratio, 0.0).max()

        is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
        clip_fraction = jnp.where(valid, is_clipped, False).sum() / n_valid

        ratio_metrics = {
            "mean_importance_weight": mean_ratio,
            "mean_log_importance_weight": mean_log_ratio,
            "max_importance_weight": max_ratio,
            "clip_fraction": clip_fraction,
        }

        return policy_loss, ratio_metrics

    def baseline_loss_fn(
        baseline_params: BaselineMLP,
        aux_info: dict
    ):
        deltas_old = aux_info['deltas_old']
        pad_mask = aux_info['pad_mask']

        baseline = eqx.combine(baseline_params, baseline_static)
        baseline_values = jax.vmap(jax.vmap(baseline))(aux_info['obs'])

        V_current = baseline_values[:, :-1]
        V_current = jnp.where(pad_mask, 0.0, V_current)
        V_next = jnp.roll(V_current, -1, axis=1).at[:, -1].set(0.0)

        deltas = deltas_old + V_next - V_current
        deltas = jnp.where(pad_mask, 0.0, deltas)

        advantages = jax.vmap(compute_gae, in_axes=(0, None))(deltas, gae_lambda)

        value_target = stop_gradient(advantages + V_current)
        return jnp.mean((V_current - value_target) ** 2)

    def tlm_backward_loss_fn(model_params):
        model_to_call = eqx.combine(model_params, policy_static)
        baseline_to_call = eqx.combine(stop_gradient(baseline_params), baseline_static)
        
        comp = extract_advantage_components(
            model_to_call,
            baseline_to_call,
            stop_gradient(traj_data),
            env,
            env_params,
        )
        
        log_pb = comp['backward_logprobs']
        loss_per_traj = -log_pb.sum(axis=1)
        return loss_per_traj.mean()   

    use_tlm = train_state.config.agent.get("train_backward", True)

    def do_tlm_update(args):
        p_params, tlm_opt_state = args

        def tlm_epoch_body(epoch_i, carry):
            p_params, tlm_opt_state = carry
            tlm_loss, tlm_grads = eqx.filter_value_and_grad(tlm_backward_loss_fn)(p_params)
            tlm_updates, new_tlm_opt_state = train_state.tlm_backward_optimizer.update(
                tlm_grads, tlm_opt_state, p_params
            )
            new_p_params = optax.apply_updates(p_params, tlm_updates)
            return (new_p_params, new_tlm_opt_state)

        tlm_epochs = train_state.config.agent.get("tlm_epochs", 1)
        final_p_params, final_tlm_opt_state = jax.lax.fori_loop(
            lower=0,
            upper=tlm_epochs,
            body_fun=tlm_epoch_body,
            init_val=(p_params, tlm_opt_state),
        )

        tlm_loss = tlm_backward_loss_fn(final_p_params)
        return final_p_params, final_tlm_opt_state, tlm_loss

    def skip_tlm_update(args):
        p_params, tlm_opt_state = args
        return p_params, tlm_opt_state, jnp.array(0.0)

    policy_params_after_tlm, tlm_opt_state_after, tlm_loss = jax.lax.cond(
        use_tlm,
        do_tlm_update,
        skip_tlm_update,
        (policy_params, train_state.tlm_backward_opt_state),
    )

    model_after_tlm = eqx.combine(policy_params_after_tlm, policy_static)
    comp_old = extract_advantage_components(
        model_after_tlm,
        train_state.baseline,
        traj_data,
        env,
        env_params,
    )
 
    V_current = stop_gradient(comp_old['V_pred'])
    gflow_logreward = stop_gradient(comp_old['gflow_logreward'])
    backward_logprobs = stop_gradient(comp_old['backward_logprobs'])
    forward_logprobs = stop_gradient(comp_old['forward_logprobs'])
    pad_mask = stop_gradient(comp_old['pad_mask'])
 
    V_next = jnp.roll(V_current, -1, axis=1).at[:, -1].set(0.0)
 
    deltas_reward_old = stop_gradient(gflow_logreward + backward_logprobs - forward_logprobs)
    deltas_reward_old = jnp.where(pad_mask, 0.0, deltas_reward_old)
 
    advantages_old = stop_gradient(jax.vmap(compute_gae, in_axes=(0, None))(deltas_reward_old + V_next - V_current, gae_lambda))
    log_pf_old = stop_gradient(comp_old["full_forward_logprobs"])
 
    ppo_aux_info = {
        'advantages_old': advantages_old,
        'log_pf_old_full': log_pf_old,
        'log_pf_old_sampled': forward_logprobs,
        'deltas_old': deltas_reward_old,
        'pad_mask': pad_mask,
        'obs': traj_data.obs,
    }

    def policy_epoch_body(epoch_i, carry):
        p_params, p_opt_state = carry
        (policy_loss, ratio_metrics), policy_grads = eqx.filter_value_and_grad(
            policy_loss_fn, has_aux=True
        )(p_params, ppo_aux_info)
        p_updates, new_p_opt_state = train_state.policy_optimizer.update(
            policy_grads, p_opt_state, p_params
        )
        new_p_params = optax.apply_updates(p_params, p_updates)
        return (new_p_params, new_p_opt_state)

    def baseline_epoch_body(epoch_i, carry):
        b_params, b_opt_state, _ = carry
        B = ppo_aux_info['pad_mask'].shape[0]
        split_size = B // ppo_baseline_num_splits
        total_loss = 0.0
 
        for i in range(ppo_baseline_num_splits):
            start = i * split_size
            end = (i + 1) * split_size
 
            chunk_aux = jax.tree.map(lambda x: x[start:end], ppo_aux_info)
 
            chunk_loss, chunk_grads = eqx.filter_value_and_grad(baseline_loss_fn)(
                b_params, chunk_aux
            )
            b_updates, b_opt_state = train_state.baseline_optimizer.update(
                chunk_grads, b_opt_state, b_params
            )
            b_params = optax.apply_updates(b_params, b_updates)
            total_loss += chunk_loss
 
        return (b_params, b_opt_state, total_loss / ppo_baseline_num_splits)

    final_p_params, final_p_opt_state = jax.lax.fori_loop(
        lower=0,
        upper=ppo_policy_epochs,
        body_fun=policy_epoch_body,
        init_val=(policy_params, train_state.policy_opt_state),
    )

    final_b_params, final_b_opt_state, baseline_loss = jax.lax.fori_loop(
        lower=0,
        upper=ppo_baseline_epochs,
        body_fun=baseline_epoch_body,
        init_val=(baseline_params, train_state.baseline_opt_state, 0.0),
    )

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
                        policy_params=policy_params, env_params=env_params
                    ),
                    "elbo": ELBOMetricsModule.ProcessArgs(
                        policy_params=policy_params, env_params=env_params
                    ),
                    "eubo": EUBOMetricsModule.ProcessArgs(
                        policy_params=policy_params, env_params=env_params
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

    (last_policy_loss, last_ratio_metrics), last_policy_grads = eqx.filter_value_and_grad(
        policy_loss_fn, has_aux=True
    )(final_p_params, ppo_aux_info)

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
            "mean_loss": last_policy_loss,
            "baseline_loss": baseline_loss,
            "tlm_backward_loss": tlm_loss,
            "entropy": log_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(last_policy_grads),
            "mean_reward": jnp.exp(log_info["log_gfn_reward"]).mean(),
            "mean_log_reward": log_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
            **last_ratio_metrics,
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
        policy_opt_state=final_p_opt_state,
        baseline_opt_state=final_b_opt_state,
        tlm_backward_opt_state=tlm_opt_state_after,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="PPO_qm9_small", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

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

    model_params = eqx.filter(model, eqx.is_array)
    baseline_params = eqx.filter(baseline, eqx.is_array)

    policy_optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    baseline_optimizer = optax.adam(learning_rate=cfg.agent.baseline_learning_rate)

    tlm_backward_lr = cfg.agent.get("tlm_backward_learning_rate", cfg.agent.learning_rate)
    tlm_backward_schedule = optax.exponential_decay(
        init_value=tlm_backward_lr,
        transition_steps=1,
        decay_rate=0.999,
        staircase=False,
    )

    tlm_backward_optimizer = optax.adam(
        learning_rate=tlm_backward_schedule
    )

    policy_opt_state = policy_optimizer.init(model_params)
    baseline_opt_state = baseline_optimizer.init(baseline_params)
    tlm_backward_opt_state = tlm_backward_optimizer.init(model_params)

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
        policy_optimizer=policy_optimizer,
        baseline_optimizer=baseline_optimizer,
        policy_opt_state=policy_opt_state,
        baseline_opt_state=baseline_opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        eval_info=eval_info,
        tlm_backward_optimizer=tlm_backward_optimizer,
        tlm_backward_opt_state=tlm_backward_opt_state,
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
            tags=["PPO", env.name.upper()],
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
        },
    )


if __name__ == "__main__":
    run_experiment()
