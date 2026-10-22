"""Single-file implementation of entropy-regularized PPO (ent-PPO) for the
Ising-Bitseq environment.

Run the script with the following command:
```bash
python baselines/PPO_ising_bitseq.py
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
from jax.lax import stop_gradient
from jax_tqdm import loop_tqdm
from jaxtyping import Array, Int
from omegaconf import OmegaConf
from utils.checkpoint import save_checkpoint
from utils.logger import Writer

import gfnx
from gfnx.utils.bitseq import detokenize
from gfnx.metrics import (
    MultiMetricsModule,
    MultiMetricsState,
    ELBOMetricsModule,
    EUBOMetricsModule,
    IsingPhysicsMetricsModule,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
writer = Writer()


class TransformerPolicy(eqx.Module):
    """
    A policy module that uses a simple transformer model to generate
    forward and backward action logits.
    """

    encoder: gfnx.networks.Encoder
    forward_pooler: eqx.nn.Linear
    backward_pooler: eqx.nn.Linear
    train_backward_policy: bool
    n_fwd_actions: int
    n_bwd_actions: int
    env_max_length: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        env_max_length: int,
        train_backward_policy: bool,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.train_backward_policy = train_backward_policy
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions
        self.env_max_length = env_max_length

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = gfnx.networks.Encoder(key=encoder_key, **encoder_params)

        self.forward_pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=self.n_fwd_actions // self.env_max_length,
            key=pooler_key,
        )

        self.backward_pooler = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=self.n_bwd_actions // self.env_max_length,
            key=pooler_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        pos_ids = jnp.arange(obs_ids.shape[0])
        encoded_obs = self.encoder(obs_ids, pos_ids, enable_dropout=enable_dropout, key=key)[
            "layers_out"
        ][-1]  # [seq_len, hidden_size]

        forward_logits = jnp.ravel(jax.vmap(self.forward_pooler)(encoded_obs[1:]))

        if self.train_backward_policy:
            backward_logits = jnp.ravel(jax.vmap(self.backward_pooler)(encoded_obs[1:]))
        else:
            backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)

        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }


class TransformerBaseline(eqx.Module):
    """
    A state-value baseline that reuses the same transformer encoder backbone
    as the policy, pooling the BOS-token representation into a scalar value
    estimate V(s).
    """

    encoder: gfnx.networks.Encoder
    value_head: eqx.nn.Linear

    def __init__(self, encoder_params: dict, *, key: chex.PRNGKey):
        encoder_key, head_key = jax.random.split(key)
        self.encoder = gfnx.networks.Encoder(key=encoder_key, **encoder_params)
        self.value_head = eqx.nn.Linear(
            in_features=encoder_params["hidden_size"],
            out_features=1,
            key=head_key,
        )

    def __call__(
        self,
        obs_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> chex.Array:
        pos_ids = jnp.arange(obs_ids.shape[0])
        encoded_obs = self.encoder(obs_ids, pos_ids, enable_dropout=enable_dropout, key=key)[
            "layers_out"
        ][-1]  # [seq_len, hidden_size]
        # Use the BOS-token (position 0) representation as a pooled summary
        # of the whole sequence, analogous to a [CLS] token.
        pooled = encoded_obs[0]
        return self.value_head(pooled).squeeze(-1)


class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.IsingBitseqEnvironment
    env_params: chex.Array
    model: TransformerPolicy
    baseline: TransformerBaseline
    policy_optimizer: optax.GradientTransformation
    baseline_optimizer: optax.GradientTransformation
    policy_opt_state: optax.OptState
    baseline_opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
    exploration_schedule: optax.Schedule
    beta_schedule: optax.Schedule
    eval_info: dict
    # Trajectory (log-matching) backward-policy training.
    tlm_backward_optimizer: optax.GradientTransformation
    tlm_backward_opt_state: optax.OptState


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params

    # Get model / baseline parameters and static parts.
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)
    baseline_params, baseline_static = eqx.partition(train_state.baseline, eqx.is_array)

    gae_lambda = train_state.config.agent.gae_lambda
    ppo_policy_epochs = train_state.config.agent.ppo_policy_epochs
    ppo_baseline_epochs = train_state.config.agent.ppo_baseline_epochs
    ppo_baseline_num_splits = train_state.config.agent.ppo_baseline_num_splits
    clip_eps = train_state.config.agent.clip_eps

    # Step 1. Generate a batch of trajectories with epsilon-exploration and
    # annealed inverse temperature (as in the TB Ising baseline).
    rng_key, sample_traj_key = jax.random.split(rng_key)

    cur_eps = train_state.exploration_schedule(idx)

    cur_beta_frac = train_state.beta_schedule(idx)
    beta_final = train_state.env_params.reward_params["beta"]
    cur_beta = cur_beta_frac * beta_final

    train_env_params = train_state.env_params.replace(
        reward_params={**train_state.env_params.reward_params, "beta": cur_beta}
    )

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        rng_key, explore_key = jax.random.split(rng_key)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(
            lambda obs, key: policy(obs, enable_dropout=True, key=key), in_axes=(0, 0)
        )(env_obs, jax.random.split(rng_key, env_obs.shape[0]))
        # With probability cur_eps, act uniformly at random (zero logits).
        explore_mask = jax.random.bernoulli(explore_key, cur_eps, (env_obs.shape[0],))
        forward_logits = jnp.where(explore_mask[..., None], 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    traj_data, aux_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,
        env=env,
        env_params=train_env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes only).
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(env, traj_data, env_params)
    rl_reward = log_pb_traj + aux_info["log_gfn_reward"] + aux_info["entropy"]
    # Sign fractions of terminal magnetization in the current training batch.
    # Keep zero magnetization separate; these diagnostics do not change sampling.
    terminal_bits = jax.vmap(
        lambda tokens: detokenize(tokens, train_state.config.environment.k)
    )(aux_info["final_env_state"].tokens)
    terminal_magnetization = (2.0 * terminal_bits.astype(jnp.float32) - 1.0).mean(axis=-1)

    # ----------------------------------------------------------------- #
    # Helpers shared across the TLM / advantage / policy / baseline losses
    # ----------------------------------------------------------------- #
    def extract_advantage_components(
        model: TransformerPolicy,
        baseline: TransformerBaseline,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.IsingBitseqEnvironment,
        current_env_params: chex.Array,
    ):
        dropout_keys = jax.random.split(rng_key, current_traj_data.obs.shape[:2])
        policy_outputs_traj = jax.vmap(
            jax.vmap(lambda obs, key: model(obs, enable_dropout=False, key=key)),
        )(current_traj_data.obs, dropout_keys)
        baseline_values = jax.vmap(
            jax.vmap(lambda obs, key: baseline(obs, enable_dropout=False, key=key)),
        )(current_traj_data.obs, dropout_keys)

        # Step 2.1 Compute forward actions and log probabilities.
        fwd_logits_traj = policy_outputs_traj["forward_logits"]

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

        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)  # [B, T]
        fwd_actions = current_traj_data.action[:, :-1]  # [B, T]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)  # [B, T]

        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action,
            in_axes=(1, 1, 1, None),
            out_axes=1,
        )(prev_states, fwd_actions, curr_states, current_env_params)  # [B, T]

        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]
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
            "gflow_logreward": masked_log_rewards_at_steps,  # (B, T)
            "backward_logprobs": log_pb_selected,  # (B, T)
            "V_pred": masked_V_pred,
            "forward_logprobs": fwd_logprobs_traj[:, :-1],
            "pad_mask": pad_mask_for_bwd,
        }

    def compute_gae(deltas, lam):
        def scan_fn(carry, delta):
            return delta + lam * carry, delta + lam * carry

        _, adv_rev = jax.lax.scan(scan_fn, init=0.0, xs=deltas[::-1])
        return adv_rev[::-1]

    def policy_loss_fn(model_params: TransformerPolicy, aux_info: dict):
        pad_mask = aux_info["pad_mask"]
        advantages_old = aux_info["advantages_old"]
        log_pf_old_full = aux_info["log_pf_old_full"]
        log_pf_old_sampled = aux_info["log_pf_old_sampled"]

        model_to_call = eqx.combine(model_params, policy_static)
        dropout_keys = jax.random.split(rng_key, traj_data.obs.shape[:2])
        policy_outputs = jax.vmap(
            jax.vmap(lambda obs, key: model_to_call(obs, enable_dropout=False, key=key)),
        )(traj_data.obs, dropout_keys)
        invalid_fwd_mask = jax.vmap(env.get_invalid_mask, in_axes=(1, None), out_axes=1)(
            traj_data.state, env_params
        )
        masked_fwd_logits_traj = gfnx.utils.mask_logits(
            policy_outputs["forward_logits"], invalid_fwd_mask
        )
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
        ppo_clip = jnp.where(pad_mask, 0.0, ppo_clip)  # (batch, T)

        log_pf_new_full = log_pf_new_full[:, :-1]
        kl = jnp.exp(log_pf_new_full) * (log_pf_new_full - log_pf_old_full)  # (batch, T, A)
        kl = jnp.where(pad_mask[..., None], 0.0, kl).sum(axis=-1)  # (batch, T)

        loss_per_traj = jnp.sum(ppo_clip, axis=1) - jnp.sum(kl, axis=1)
        policy_loss = -jnp.mean(loss_per_traj)

        # For logging.
        log_ratio = log_pf_new_sampled - log_pf_old_sampled
        valid = ~pad_mask
        n_valid = valid.sum()
        mean_ratio = jnp.where(valid, ratio, 0.0).sum() / n_valid
        mean_log_ratio = jnp.where(valid, log_ratio, 0.0).sum() / n_valid
        max_ratio = jnp.where(valid, ratio, 0.0).max()

        is_clipped = (ratio < 1.0 - clip_eps) | (ratio > 1.0 + clip_eps)
        clip_fraction = jnp.where(valid, is_clipped, False).sum() / n_valid

        ratio_metrics = {
            # KL(new || rollout policy), measured after all policy updates.
            "policy_kl_per_step": kl.sum() / n_valid,
            "policy_kl_per_trajectory": kl.sum(axis=1).mean(),
            "mean_importance_weight": mean_ratio,
            "mean_log_importance_weight": mean_log_ratio,
            "max_importance_weight": max_ratio,
            "clip_fraction": clip_fraction,
        }

        return policy_loss, ratio_metrics

    def baseline_loss_fn(baseline_params: TransformerBaseline, aux_info: dict):
        deltas_old = aux_info["deltas_old"]
        pad_mask = aux_info["pad_mask"]

        baseline = eqx.combine(baseline_params, baseline_static)
        dropout_keys = jax.random.split(rng_key, aux_info["obs"].shape[:2])
        baseline_values = jax.vmap(
            jax.vmap(lambda obs, key: baseline(obs, enable_dropout=False, key=key)),
        )(aux_info["obs"], dropout_keys)

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

        log_pb = comp["backward_logprobs"]
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

    V_current = stop_gradient(comp_old["V_pred"])
    gflow_logreward = stop_gradient(comp_old["gflow_logreward"])
    backward_logprobs = stop_gradient(comp_old["backward_logprobs"])
    forward_logprobs = stop_gradient(comp_old["forward_logprobs"])
    pad_mask = stop_gradient(comp_old["pad_mask"])

    V_next = jnp.roll(V_current, -1, axis=1).at[:, -1].set(0.0)

    # Entropy-regularized (soft) TD residual: reward + backward logprob
    # (i.e. the reversed/entropy term) minus forward logprob, exactly as
    # in the TB objective but used here as a one-step Bellman residual.
    deltas_reward_old = stop_gradient(gflow_logreward + backward_logprobs - forward_logprobs)
    deltas_reward_old = jnp.where(pad_mask, 0.0, deltas_reward_old)

    advantages_old = stop_gradient(
        jax.vmap(compute_gae, in_axes=(0, None))(deltas_reward_old + V_next - V_current, gae_lambda)
    )
    log_pf_old = stop_gradient(comp_old["full_forward_logprobs"])

    ppo_aux_info = {
        "advantages_old": advantages_old,
        "log_pf_old_full": log_pf_old,
        "log_pf_old_sampled": forward_logprobs,
        "deltas_old": deltas_reward_old,
        "pad_mask": pad_mask,
        "obs": traj_data.obs,
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
        B = ppo_aux_info["pad_mask"].shape[0]
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
        init_val=(policy_params_after_tlm, train_state.policy_opt_state),
    )

    final_b_params, final_b_opt_state, baseline_loss = jax.lax.fori_loop(
        lower=0,
        upper=ppo_baseline_epochs,
        body_fun=baseline_epoch_body,
        init_val=(baseline_params, train_state.baseline_opt_state, 0.0),
    )

    metrics_state = train_state.metrics_state

    rng_key, eval_rng_key = jax.random.split(rng_key)
    is_eval_step = idx % train_state.config.logging.eval_each == 0
    is_eval_step = is_eval_step | (idx + 1 == train_state.config.num_train_steps)

    metrics_state = jax.lax.cond(
        is_eval_step,
        lambda kwargs: train_state.metrics_module.process(**kwargs),
        lambda kwargs: kwargs["metrics_state"],
        {
            "metrics_state": metrics_state,
            "rng_key": eval_rng_key,
            "args": train_state.metrics_module.ProcessArgs(
                metrics_args={
                    "elbo": ELBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=train_state.env_params,
                    ),
                    "eubo": EUBOMetricsModule.ProcessArgs(
                        policy_params=final_p_params, env_params=train_state.env_params
                    ),
                    "physics": IsingPhysicsMetricsModule.ProcessArgs(
                        policy_params=final_p_params,
                        env_params=train_state.env_params,
                    ),
                }
            ),
        },
    )

    eval_info = jax.lax.cond(
        is_eval_step,
        lambda metrics_state: train_state.metrics_module.get(metrics_state),
        lambda metrics_state: train_state.eval_info,
        metrics_state,
    )

    # Losses for logging.
    (last_policy_loss, last_ratio_metrics), last_policy_grads = eqx.filter_value_and_grad(
        policy_loss_fn, has_aux=True
    )(final_p_params, ppo_aux_info)

    def logging_callback(idx: int, train_info: dict, eval_info: dict, cfg):
        train_info = {f"train/{key}": float(value) for key, value in train_info.items()}

        if idx % cfg.logging.eval_each == 0 or idx + 1 == cfg.num_train_steps:
            log.info(f"Step {idx}")
            log.info(train_info)
            eval_info = {f"eval/{key}": float(value) for key, value in eval_info.items()}
            log.info(eval_info)
            if cfg.logging.use_writer:
                writer.log(eval_info, step=idx)

        if cfg.logging.use_writer and idx % cfg.logging.track_each == 0:
            writer.log(train_info, step=idx)

    jax.debug.callback(
        logging_callback,
        idx,
        {
            "policy_loss": last_policy_loss,
            "baseline_loss": baseline_loss,
            "tlm_backward_loss": tlm_loss,
            "entropy": aux_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(last_policy_grads),
            "rl_reward": rl_reward.mean(),
            "cur_beta": cur_beta,
            "cur_eps": cur_eps,
            "positive_magnetization_fraction": (terminal_magnetization > 0).mean(),
            "negative_magnetization_fraction": (terminal_magnetization < 0).mean(),
            "zero_magnetization_fraction": (terminal_magnetization == 0).mean(),
            "mean_magnetization": terminal_magnetization.mean(),
            **last_ratio_metrics,
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    new_model = eqx.combine(final_p_params, policy_static)
    new_baseline = eqx.combine(final_b_params, baseline_static)

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


# --------------------------------------------------------------------------- #
# Experiment entry point
# --------------------------------------------------------------------------- #
@hydra.main(config_path="configs/", config_name="ppo_ising_bitseq", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    eval_init_key = jax.random.PRNGKey(cfg.eval_init_seed)

    GT_PATHS = {
        0.6: "src/gfnx/ising_target_set/ising_L16_beta0.6_seed0.npz",
        0.4407: "src/gfnx/ising_target_set/ising_L16_beta0.4407_seed0.npz",
        1.2: "src/gfnx/ising_target_set/ising_L16_beta1.2_seed0.npz",
    }
    gt_samples_path = GT_PATHS[float(cfg.environment.beta)]

    reward_module = gfnx.BitseqIsingRewardModule(
        L=cfg.environment.L,
        beta=cfg.environment.beta,
        k=cfg.environment.k,
    )

    env = gfnx.IsingBitseqEnvironment(
        reward_module,
        L=cfg.environment.L,
        k=cfg.environment.k,
        beta=cfg.environment.beta,
        gt_samples_path=gt_samples_path,
        fixed_order=cfg.environment.get("fixed_order", False),
    )

    env_params = env.init(env_init_key)

    encoder_params = {
        "pad_id": -1,  # No masking in this setup
        "vocab_size": env.ntoken,
        "max_length": env.max_length + 1,  # +1 for BOS token
        **OmegaConf.to_container(cfg.network),
    }

    rng_key, net_init_key = jax.random.split(rng_key)
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        env_max_length=env.max_length,
        train_backward_policy=cfg.agent.train_backward,
        encoder_params=encoder_params,
        key=net_init_key,
    )

    rng_key, baseline_init_key = jax.random.split(rng_key)
    baseline = TransformerBaseline(
        encoder_params=encoder_params,
        key=baseline_init_key,
    )

    exploration_schedule = optax.linear_schedule(
        init_value=cfg.agent.start_eps,
        end_value=cfg.agent.end_eps,
        transition_steps=cfg.agent.exploration_steps,
    )

    beta_final = env_params.reward_params["beta"]
    beta_anneal_steps = cfg.agent.get("beta_anneal_steps", None) or max(1, cfg.num_train_steps // 2)

    if cfg.agent.anneal_beta:
        beta_schedule = optax.linear_schedule(
            init_value=0.0,
            end_value=1.0,
            transition_steps=beta_anneal_steps,
        )
    else:
        beta_schedule = optax.constant_schedule(1.0)

    policy_static = eqx.filter(model, eqx.is_array, inverse=True)
    baseline_static = eqx.filter(baseline, eqx.is_array, inverse=True)

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, current_policy_params) -> chex.Array:
        del rng_key
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def bwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
        return policy_outputs["backward_logits"], policy_outputs

    model_params = eqx.filter(model, eqx.is_array)
    baseline_params = eqx.filter(baseline, eqx.is_array)

    policy_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=cfg.agent.learning_rate, weight_decay=cfg.agent.weight_decay),
    )
    baseline_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=cfg.agent.baseline_learning_rate),
    )

    tlm_backward_lr = cfg.agent.get("tlm_backward_learning_rate", cfg.agent.learning_rate)
    tlm_backward_schedule = optax.exponential_decay(
        init_value=tlm_backward_lr,
        transition_steps=1,
        decay_rate=0.999,
        staircase=False,
    )
    tlm_backward_optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=tlm_backward_schedule),
    )

    policy_opt_state = policy_optimizer.init(model_params)
    baseline_opt_state = baseline_optimizer.init(baseline_params)
    tlm_backward_opt_state = tlm_backward_optimizer.init(model_params)

    # ELBO / EUBO / physics metrics; ground truth is loaded from
    # gt_samples_path (same setup as in the TB Ising baseline).
    metrics_module = MultiMetricsModule({
        "elbo": ELBOMetricsModule(
            env=env, env_params=env_params, fwd_policy_fn=fwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.batch_size,
        ),
        "eubo": EUBOMetricsModule(
            env=env, env_params=env_params, bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.batch_size,
            rng_key=eval_init_key,
        ),
        "physics": IsingPhysicsMetricsModule(
            env=env, L=cfg.environment.L, k=cfg.environment.k,
            fwd_policy_fn=fwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.batch_size,
            sinkhorn_reg=cfg.metrics.sinkhorn_reg, sinkhorn_iters=cfg.metrics.sinkhorn_iters,
            compute_sinkhorn=cfg.metrics.get("compute_sinkhorn", True),
            sinkhorn_chunk_size=cfg.metrics.get("sinkhorn_chunk_size", None),
        ),
    })

    eval_init_key, metrics_init_key = jax.random.split(eval_init_key)
    metrics_state = metrics_module.init(
        metrics_init_key,
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
        exploration_schedule=exploration_schedule,
        beta_schedule=beta_schedule,
        eval_info=eval_info,
        tlm_backward_optimizer=tlm_backward_optimizer,
        tlm_backward_opt_state=tlm_backward_opt_state,
    )

    train_state_params, train_state_static = eqx.partition(train_state, eqx.is_array)

    @functools.partial(jax.jit, donate_argnums=(1,))
    @loop_tqdm(cfg.num_train_steps, print_rate=cfg.logging["tqdm_print_rate"])
    def train_step_wrapper(idx: int, train_state_params):
        train_state = eqx.combine(train_state_params, train_state_static)
        train_state = train_step(idx, train_state)
        train_state_params, _ = eqx.partition(train_state, eqx.is_array)
        return train_state_params

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
            tags=["ent-PPO", env.name.upper()],
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        )

    log.info("Start training")
    train_state_params = jax.lax.fori_loop(
        lower=0,
        upper=cfg.num_train_steps,
        body_fun=train_step_wrapper,
        init_val=train_state_params,
    )
    train_state_params = jax.block_until_ready(train_state_params)

    train_state = eqx.combine(train_state_params, train_state_static)
    dir = (
        cfg.logging.checkpoint_dir
        if cfg.logging.checkpoint_dir
        else os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            f"checkpoints_{os.getpid()}/",
        )
    )

    log.info(
        "Running final evaluation: %d rounds x %d samples",
        cfg.metrics.n_rounds, cfg.metrics.final_batch_size,
    )

    policy_params, _ = eqx.partition(train_state.model, eqx.is_array)

    final_metrics_module = MultiMetricsModule({
        "elbo": ELBOMetricsModule(
            env=env, env_params=train_state.env_params, fwd_policy_fn=fwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.final_batch_size,
        ),
        "eubo": EUBOMetricsModule(
            env=env, env_params=train_state.env_params, bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.final_batch_size,
            rng_key=eval_init_key,
        ),
        "physics": IsingPhysicsMetricsModule(
            env=env, L=cfg.environment.L, k=cfg.environment.k,
            fwd_policy_fn=fwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.final_batch_size,
            sinkhorn_reg=cfg.metrics.sinkhorn_reg, sinkhorn_iters=cfg.metrics.sinkhorn_iters,
            compute_sinkhorn=cfg.metrics.get("compute_sinkhorn", True),
            sinkhorn_chunk_size=cfg.metrics.get("sinkhorn_chunk_size", None),
        ),
    })

    eval_init_key, final_init_key = jax.random.split(eval_init_key)
    final_metrics_state = final_metrics_module.init(
        final_init_key,
    )

    eval_init_key, final_process_key = jax.random.split(eval_init_key)
    final_metrics_state = final_metrics_module.process(
        metrics_state=final_metrics_state,
        rng_key=final_process_key,
        args=final_metrics_module.ProcessArgs(metrics_args={
            "elbo": ELBOMetricsModule.ProcessArgs(
                policy_params=policy_params, env_params=train_state.env_params,
            ),
            "eubo": EUBOMetricsModule.ProcessArgs(
                policy_params=policy_params, env_params=train_state.env_params,
            ),
            "physics": IsingPhysicsMetricsModule.ProcessArgs(
                policy_params=policy_params, env_params=train_state.env_params,
            ),
        }),
    )
    final_eval_info = {f"final/{k}": float(v) for k, v in final_metrics_module.get(final_metrics_state).items()}
    log.info(final_eval_info)
    if cfg.logging.use_writer:
        writer.log(final_eval_info, step=cfg.num_train_steps)

    save_checkpoint(
        os.path.join(dir, "model_and_baseline"),
        {
            "model": train_state.model,
            "baseline": train_state.baseline,
        },
    )


if __name__ == "__main__":
    run_experiment()
