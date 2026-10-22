"""TB, DB, SubTB and VarGrad for Ising, sharing the established Ising evaluation harness.

Run the script with the following command:
```bash
python baselines/tb_ising_bitseq.py
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
import numpy as np
import optax
from jax_tqdm import loop_tqdm
from jaxtyping import Array, Int
from omegaconf import OmegaConf
from utils.checkpoint import save_checkpoint
from utils.logger import Writer

import gfnx
from ising_balance_losses import balance_loss
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
    forward and backward action logits as well as a flow.
    """

    encoder: gfnx.networks.Encoder
    forward_pooler: eqx.nn.Linear
    backward_pooler: eqx.nn.Linear
    flow_pooler: eqx.nn.Linear | None
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
        learn_flow: bool = False,
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

        self.flow_pooler = (
            eqx.nn.Linear(encoder_params["hidden_size"], 1, key=pooler_key)
            if learn_flow else None
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
            "log_flow": self.flow_pooler(encoded_obs[0]).squeeze(-1) if self.flow_pooler is not None else jnp.array(0.0),
            "forward_logits": forward_logits,
            "backward_logits": backward_logits,
        }



# Define the train state that will be used in the training loop
class TrainState(NamedTuple):
    rng_key: chex.PRNGKey
    config: OmegaConf
    env: gfnx.BitseqEnvironment
    env_params: chex.Array
    model: TransformerPolicy
    logZ: chex.Array  # Added logZ for TB
    optimizer: optax.GradientTransformation
    opt_state: optax.OptState
    metrics_module: MultiMetricsModule
    metrics_state: MultiMetricsState
    exploration_schedule: optax.Schedule
    eval_info: dict
    beta_schedule: optax.Schedule


@eqx.filter_jit
def train_step(idx: int, train_state: TrainState) -> TrainState:
    rng_key = train_state.rng_key
    num_envs = train_state.config.num_envs
    env = train_state.env
    env_params = train_state.env_params

    # Get model parameters and static parts
    policy_params, policy_static = eqx.partition(train_state.model, eqx.is_array)

    # Step 1. Generate a batch of trajectories
    rng_key, sample_traj_key = jax.random.split(rng_key)

    # Get epsilon exploration value from config
    cur_eps = train_state.exploration_schedule(idx)

    # Reference-style inverse-temperature annealing:
    # keep the physical target fixed at beta_final and anneal only its
    # contribution in the objective from 0 -> 1 over the configured annealing phase.
    invtemp = train_state.beta_schedule(idx)
    beta_final = train_state.env_params.reward_params["beta"]
    effective_beta = invtemp * beta_final

    # The environment always evaluates the true target reward at beta_final.
    train_env_params = train_state.env_params

    # Define the policy function suitable for gfnx.utils.forward_rollout
    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        rng_key, explore_key = jax.random.split(rng_key)
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(
            lambda obs, key: policy(obs, enable_dropout=True, key=key), in_axes=(0, 0)
        )(env_obs, jax.random.split(rng_key, env_obs.shape[0]))
        # With probability cur_eps, return zero logits and the same policy outputs
        explore_mask = jax.random.bernoulli(explore_key, cur_eps, (env_obs.shape[0],))
        forward_logits = jnp.where(explore_mask[..., None], 0, policy_outputs["forward_logits"])
        return forward_logits, policy_outputs

    traj_data, aux_info = gfnx.utils.forward_rollout(
        rng_key=sample_traj_key,
        num_envs=num_envs,
        policy_fn=fwd_policy_fn,
        policy_params=policy_params,  # Pass only network parameters
        env=train_state.env,
        env_params=train_env_params,
    )
    # Compute the RL reward / ELBO (for logging purposes)
    _, log_pb_traj = gfnx.utils.forward_trajectory_log_probs(
        env, traj_data, env_params
    )
    rl_reward = log_pb_traj + aux_info["log_gfn_reward"] + aux_info["entropy"]

    # Step 2. Compute the TB loss
    def loss_fn(
        current_all_params: dict,
        static_model_parts: TransformerPolicy,
        current_traj_data: gfnx.utils.TrajectoryData,
        current_env: gfnx.BitseqEnvironment,
        current_env_params: chex.Array,
    ):
        # Extract model's learnable parameters and logZ from the input
        model_learnable_params = current_all_params["model_params"]
        logZ_val = current_all_params["logZ"]

        # Reconstruct the callable model using its learnable parameters
        model_to_call = eqx.combine(model_learnable_params, static_model_parts)
        dropout_keys = jax.random.split(rng_key, current_traj_data.obs.shape[:2])
        # Get policy outputs for the entire trajectory
        policy_outputs_traj = jax.vmap(
            jax.vmap(
                lambda obs, key: model_to_call(obs, enable_dropout=True, key=key),
            ),
        )(current_traj_data.obs, dropout_keys)

        # Step 2.1 Compute forward actions and log probabilities
        fwd_logits_traj = policy_outputs_traj["forward_logits"]

        # Vmap get_invalid_mask over the time dimension
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
        # Use extracted logZ_val
        log_pf_traj = logZ_val + sum_log_pf_along_traj

        # Step 2.2 Compute backward actions and log probabilities
        prev_states = jax.tree.map(lambda x: x[:, :-1], current_traj_data.state)
        fwd_actions = current_traj_data.action[:, :-1]
        curr_states = jax.tree.map(lambda x: x[:, 1:], current_traj_data.state)

        bwd_actions_traj = jax.vmap(
            current_env.get_backward_action,
            in_axes=(1, 1, 1, None),
            out_axes=1,
        )(prev_states, fwd_actions, curr_states, current_env_params)

        bwd_logits_traj = policy_outputs_traj["backward_logits"]
        bwd_logits_for_pb = bwd_logits_traj[:, 1:]
        # Vmap get_invalid_backward_mask over the time dimension
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

        # Temperature annealing is applied to log R(x) only.
        # At invtemp=0 the target contribution is zero; at invtemp=1
        # this is the true target used for training/evaluation.
        annealed_log_rewards_at_steps = invtemp * log_rewards_at_steps
        masked_annealed_log_rewards_at_steps = jnp.where(
            pad_mask_for_bwd,
            0.0,
            annealed_log_rewards_at_steps,
        )

        log_pb_plus_rewards_along_traj = (
            log_pb_selected + masked_annealed_log_rewards_at_steps
        )
        target = jnp.sum(log_pb_plus_rewards_along_traj, axis=1)

        return balance_loss(
            train_state.config.agent.objective,
            fwd_logprobs_traj[:, :-1], log_pb_selected,
            policy_outputs_traj["log_flow"],
            masked_annealed_log_rewards_at_steps,
            current_traj_data.done[:, :-1], pad_mask_for_bwd,
            logZ_val, train_state.config.agent.get("lmbd", 1.9),
        )

    # Prepare parameters for the loss function and gradient calculation
    params_for_loss = {"model_params": policy_params, "logZ": train_state.logZ}

    mean_loss, grads = eqx.filter_value_and_grad(loss_fn)(
        params_for_loss, policy_static, traj_data, env, env_params
    )

    # Step 3. Update parameters (model network and logZ)
    optax_params_for_update = {
        "model_params": policy_params,
        "logZ": train_state.logZ,
    }
    updates, new_opt_state = train_state.optimizer.update(
        grads, train_state.opt_state, optax_params_for_update
    )

    # Apply updates
    new_model = eqx.apply_updates(train_state.model, updates["model_params"])
    new_logZ = eqx.apply_updates(train_state.logZ, updates["logZ"])

    metrics_state = train_state.metrics_state

    rng_key, eval_rng_key = jax.random.split(rng_key)
    # Perform evaluation computations if needed
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
                        policy_params=eqx.filter(new_model, eqx.is_array), env_params=train_state.env_params,
                    ),
                    "eubo": EUBOMetricsModule.ProcessArgs(
                        policy_params=eqx.filter(new_model, eqx.is_array), env_params=train_state.env_params
                    ),
                    "physics": IsingPhysicsMetricsModule.ProcessArgs(
                        policy_params=eqx.filter(new_model, eqx.is_array),
                        env_params=train_state.env_params,
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
            "mean_loss": mean_loss,
            "entropy": aux_info["entropy"].mean(),
            "grad_norm": optax.tree_utils.tree_l2_norm(grads),
            "logZ": new_logZ,
            "rl_reward": rl_reward.mean(),
            "cur_beta": effective_beta,
        },
        eval_info,
        train_state.config,
        ordered=True,
    )

    # Return the updated train state
    return train_state._replace(
        rng_key=rng_key,
        model=new_model,
        opt_state=new_opt_state,
        metrics_state=metrics_state,
        logZ=new_logZ,
        eval_info=eval_info,
    )


@hydra.main(config_path="configs/", config_name="balance_ising_bitseq", version_base=None)
def run_experiment(cfg: OmegaConf) -> None:
    # Log the configuration
    log.info(OmegaConf.to_yaml(cfg))

    rng_key = jax.random.PRNGKey(cfg.seed)
    # This key is needed to initialize the environment
    env_init_key = jax.random.PRNGKey(cfg.env_init_seed)
    # This key is needed to initialize the evaluation process
    # i.e., generate random test set.
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

    rng_key, net_init_key = jax.random.split(rng_key)
    # Initialize the network
    model = TransformerPolicy(
        n_fwd_actions=env.action_space.n,
        n_bwd_actions=env.backward_action_space.n,
        env_max_length=env.max_length,
        train_backward_policy=cfg.agent.train_backward,
        learn_flow=cfg.agent.objective in ("db", "subtb"),
        encoder_params={
            "pad_id": -1,  # No masking in this setup
            "vocab_size": env.ntoken,
            "max_length": env.max_length + 1,  # +1 for BOS token
            **OmegaConf.to_container(cfg.network),
        },
        key=net_init_key,
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

    # Initialize logZ separately
    logZ = jnp.array(0.0)

    # Prepare parameters for Optax
    model_params_init = eqx.filter(model, eqx.is_array)
    initial_optax_params = {"model_params": model_params_init, "logZ": logZ}

    # Define parameter labels for multi_transform
    param_labels = {
        "model_params": jax.tree.map(lambda _: "network_lr", model_params_init),
        "logZ": "logZ_lr",
    }

    # optimizer_defs = {
    #     "network_lr": optax.adamw(
    #         learning_rate=cfg.agent.learning_rate,
    #         weight_decay=cfg.agent.weight_decay
    #     ),
    #     "logZ_lr": optax.adam(learning_rate=cfg.agent.logZ_learning_rate),
    # }

    optimizer_defs = {
        "network_lr": optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=cfg.agent.learning_rate, weight_decay=cfg.agent.weight_decay),
        ),
        "logZ_lr": optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=cfg.agent.logZ_learning_rate),
        ),
    }
    optimizer = optax.multi_transform(optimizer_defs, param_labels)
    opt_state = optimizer.init(initial_optax_params)

    # Initialize the backward policy function for correlation computation
    policy_static = eqx.filter(model, eqx.is_array, inverse=True)

    def fwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, current_policy_params) -> chex.Array:
        del rng_key
        # Recombine the network parameters with the static parts of the model
        current_model = eqx.combine(current_policy_params, policy_static)
        policy_outputs = jax.vmap(current_model, in_axes=(0,))(env_obs)
        return policy_outputs["forward_logits"], policy_outputs

    def bwd_policy_fn(rng_key: chex.PRNGKey, env_obs: gfnx.TObs, policy_params) -> chex.Array:
        del rng_key
        policy = eqx.combine(policy_params, policy_static)
        policy_outputs = jax.vmap(policy, in_axes=(0,))(env_obs)
        return policy_outputs["backward_logits"], policy_outputs


    metrics_module = MultiMetricsModule({
        "elbo": ELBOMetricsModule(
            env=env, env_params=env_params, fwd_policy_fn=fwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds, batch_size=cfg.metrics.batch_size,
        ),
        "eubo": EUBOMetricsModule(
            env=env,
            env_params=env_params,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.batch_size,
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

    eval_init_key, correlation_init_key = jax.random.split(eval_init_key)
    metrics_state = metrics_module.init(
        correlation_init_key,
    )
    eval_info = metrics_module.get(metrics_state)

    train_state = TrainState(
        rng_key=rng_key,
        config=cfg,
        env=env,
        env_params=env_params,
        model=model,
        optimizer=optimizer,
        opt_state=opt_state,
        metrics_module=metrics_module,
        metrics_state=metrics_state,
        exploration_schedule=exploration_schedule,
        logZ=logZ,
        eval_info=eval_info,
        beta_schedule = beta_schedule,
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
            tags=[cfg.agent.objective.upper(), env.name.upper()],
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
            env=env,
            env_params=env_params,
            bwd_policy_fn=bwd_policy_fn,
            n_rounds=cfg.metrics.n_rounds,
            batch_size=cfg.metrics.final_batch_size,
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
            "elbo": ELBOMetricsModule.ProcessArgs(policy_params=policy_params, env_params=train_state.env_params),
            "eubo": EUBOMetricsModule.ProcessArgs(policy_params=policy_params, env_params=train_state.env_params),
            "physics": IsingPhysicsMetricsModule.ProcessArgs(policy_params=policy_params, env_params=train_state.env_params),
        }),
    )
    final_eval_info = {f"final/{k}": float(v) for k, v in final_metrics_module.get(final_metrics_state).items()}
    log.info(final_eval_info)
    if cfg.logging.use_writer:
        writer.log(final_eval_info, step=cfg.num_train_steps)

    save_checkpoint(
        os.path.join(dir, "model_and_logZ"),
        {
            "model": train_state.model,
            "logZ": train_state.logZ,
        },
    )


if __name__ == "__main__":
    run_experiment()
