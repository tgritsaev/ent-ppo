"""Single-file implementation for SUBEB-GAE in TFBind-8 environment.

Run the script with the following command:
```bash
python baselines/GAE_subeb_tfbind.py
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
    forward and backward action logits.
    """

    encoder: gfnx.networks.Encoder
    pooler: eqx.nn.Linear
    n_fwd_actions: int
    n_bwd_actions: int

    def __init__(
        self,
        n_fwd_actions: int,
        n_bwd_actions: int,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        self.n_fwd_actions = n_fwd_actions
        self.n_bwd_actions = n_bwd_actions

        output_size = self.n_fwd_actions

        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            in_size=5 * 8,
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
        obs_ids = jax.nn.one_hot(obs_ids[1:], 5).reshape(-1)
        encoded_obs = self.encoder(obs_ids)
        forward_logits = self.pooler(encoded_obs)
        backward_logits = jnp.zeros(shape=(self.n_bwd_actions,), dtype=jnp.float32)
        return {
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }

class BaselineMLP(eqx.Module):
    encoder: gfnx.networks.Encoder
    pooler: eqx.nn.Linear

    def __init__(
        self,
        encoder_params: dict,
        *,
        key: chex.PRNGKey,
    ):
        encoder_key, pooler_key = jax.random.split(key)
        self.encoder = eqx.nn.MLP(
            in_size=5 * 8,
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
        obs_ids = jax.nn.one_hot(obs_ids[1:], 5).reshape(-1)
        encoded_obs = self.encoder(obs_ids)
        output = self.pooler(encoded_obs)
        return output.squeeze(-1)


# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.TFBind8Environment
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

        V_pred = jnp.where(pad_mask_for_bwd, 0.0, baseline_values[:, :-1])

        return {
            "gflow_logreward": masked_log_rewards_at_steps, # (B, T)
            "backward_logprobs": log_pb_selected, # (B, T)
            "V_pred": V_pred,
            "forward_logprobs": fwd_logprobs_traj[:, :-1],
            "pad_mask": pad_mask_for_bwd,
            "obs": current_traj_data.obs,
        }

    def compute_gae(deltas, lam):
        def scan_fn(carry, delta):
            return delta + lam * carry, delta + lam * carry
        _, adv_rev = jax.lax.scan(scan_fn, init=0.0, xs=deltas[::-1])
        return adv_rev[::-1]
    
    def policy_loss_fn(
        model_params: MLPPolicy,
        baseline_params: BaselineMLP,
        static_model_parts: MLPPolicy,
        static_baseline_parts: BaselineMLP,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.HypergridEnvironment,
        current_env_params: gfnx.HypergridEnvParams,
        gae_lambda: float,
    ):
        model_to_call = eqx.combine(model_params, static_model_parts)
        baseline_to_call = eqx.combine(baseline_params, static_baseline_parts)

        comp = extract_advantage_components(model_to_call, baseline_to_call, current_traj_data, current_env, current_env_params)
    
        baseline_values = comp['V_pred']
        gflow_logreward = comp['gflow_logreward']
        backward_logprobs = comp['backward_logprobs']
        forward_logprobs = comp['forward_logprobs']
        pad_mask = comp['pad_mask']
                                           
        V_current = baseline_values
        V_next = jnp.roll(V_current, -1, axis=1)
        V_next = V_next.at[:, -1].set(0.0)

        deltas = gflow_logreward + backward_logprobs - stop_gradient(forward_logprobs) + V_next - V_current
        deltas = jnp.where(pad_mask, 0.0, deltas)

        advantages = jax.vmap(compute_gae, in_axes=(0, None))(deltas, gae_lambda)

        loss_per_traj = jnp.sum(forward_logprobs * stop_gradient(advantages), axis = 1)
        policy_loss = -jnp.mean(loss_per_traj)

        return policy_loss

    def baseline_loss_fn(
        baseline_params,
        static_baseline_parts,
        model_params,
        static_model_parts,
        current_traj_data,
        current_env,
        current_env_params,
    ):
        model_to_call = eqx.combine(model_params, static_model_parts)
        baseline_to_call = eqx.combine(baseline_params, static_baseline_parts)
        comp = extract_advantage_components(model_to_call, baseline_to_call, current_traj_data, current_env, current_env_params)

        log_pf = stop_gradient(comp['forward_logprobs'])
        log_pb = stop_gradient(comp['backward_logprobs'])
    
        V = jax.vmap(jax.vmap(baseline_to_call))(current_traj_data.obs)

        batch_size, traj_len_plus1 = current_traj_data.action.shape
        traj_len = traj_len_plus1 - 1

        pad_mask = current_traj_data.pad[:, :-1]
        done_mask = current_traj_data.done[:, :-1]

        # log_flow
        V = V.at[:, 1:].set(
            jnp.where(done_mask, current_traj_data.log_gfn_reward[:, :-1], V[:, 1:])
        )
        V = V.at[:, 1:].set(
            jnp.where(pad_mask, 0.0, V[:, 1:])
        )

        def process_one_traj(log_pf, log_pb, log_flow, done, pad):
            def process_pair_idx(i, j, log_pf, log_pb, log_flow, done, pad):
                def fn():
                    mask = jnp.logical_and(i <= jnp.arange(traj_len), jnp.arange(traj_len) < j)
                    weight = jnp.power(train_state.config.agent.lmbd, j - i)
                    log_pf_subtraj = log_flow[i] + (log_pf * mask).sum()
                    log_pb_subtraj = log_flow[j] + (log_pb * mask).sum()
                    loss = optax.losses.squared_error(log_pf_subtraj, log_pb_subtraj)
                    return weight * loss, weight

                return jax.lax.cond(pad[j - 1], lambda: (0.0, 0.0), fn)

            i, j = jnp.triu_indices(traj_len + 1, k=1)
            weighted_loss, weighted_norm = jax.vmap(
                process_pair_idx, in_axes=(0, 0, None, None, None, None, None)
            )(i, j, log_pf, log_pb, log_flow, done, pad)
            return weighted_loss.sum() / weighted_norm.sum()

        loss = jax.vmap(process_one_traj)(
            log_pf,
            log_pb,
            V,
            done_mask,
            pad_mask,
        ).mean()
        return loss
    
    policy_loss, policy_grads = eqx.filter_value_and_grad(policy_loss_fn)(
        policy_params, baseline_params, policy_static, baseline_static, traj_data, env, env_params, gae_lambda
    )
    
    policy_updates, policy_new_opt_state = train_state.policy_optimizer.update(
        policy_grads, train_state.policy_opt_state, policy_params
    )

    baseline_loss, baseline_grads = eqx.filter_value_and_grad(baseline_loss_fn)(
        baseline_params, baseline_static, policy_params, policy_static, traj_data, env, env_params
    )
    baseline_updates, baseline_new_opt_state = train_state.baseline_optimizer.update(
        baseline_grads, train_state.baseline_opt_state, baseline_params
    )

    new_model = eqx.apply_updates(train_state.model, policy_updates)
    new_baseline = eqx.apply_updates(train_state.baseline, baseline_updates)
    # Peform all the requied logging
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

    # Perform the logging via JAX debug callback
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
            "mean_loss": policy_loss,
            "baseline_loss": baseline_loss,
            "entropy": log_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(policy_grads),
            "mean_reward": jnp.exp(log_info["log_gfn_reward"]).mean(),
            "mean_log_reward": log_info["log_gfn_reward"].mean(),
            "rl_reward": rl_reward.mean(),
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        baseline=new_baseline,
        policy_opt_state=policy_new_opt_state,
        baseline_opt_state=baseline_new_opt_state,
        metrics_state=metrics_state,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="GAE_subeb_tfbind", version_base=None)
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
    reward_module = gfnx.TFBind8RewardModule()
    # Initialize the environment and its inner parameters
    env = gfnx.TFBind8Environment(reward_module)
    env_params = env.init(env_init_key)

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = MLPPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
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

    # Prepare parameters for Optax
    model_params = eqx.filter(model, eqx.is_array)
    baseline_params = eqx.filter(baseline, eqx.is_array)

    policy_optimizer = optax.adam(learning_rate=cfg.agent.learning_rate)
    baseline_optimizer = optax.adam(learning_rate=cfg.agent.baseline_learning_rate)

    policy_opt_state = policy_optimizer.init(model_params)
    baseline_opt_state = baseline_optimizer.init(baseline_params)

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
            tags=["GAE", env.name.upper()],
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
            "model": train_state.model,
            "baseline": train_state.baseline,
        },
    )


if __name__ == "__main__":
    run_experiment()
