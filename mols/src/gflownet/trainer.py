import copy
import gc
import logging
import os
import pathlib
import shutil
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard
import torch_geometric.data as gd
import git
import wandb
from omegaconf import MISSING, OmegaConf
from rdkit import RDLogger
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from gflownet import GFNAlgorithm, GFNTask
from gflownet.algo.entropy_ppo import EntPPO
from gflownet.algo.trajectory_balance import TrajectoryBalance
from gflownet.data.data_source import DataSource
from gflownet.data.replay_buffer import ReplayBuffer
from gflownet.models.graph_transformer import GraphTransformerGFN
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from gflownet.utils.misc import create_logger, set_main_process_device, set_worker_rng_seed
from gflownet.utils.multiprocessing_proxy import mp_object_wrapper
from gflownet.utils.sqlite_log import SQLiteLogHook

from .config import Config


class Closable(Protocol):
    def close(self):
        pass


class GFNTrainer:
    def __init__(self, config: Config, print_config=True):
        """A GFlowNet trainer. Contains the main training loop in `run` and should be subclassed.

        Parameters
        ----------
        config: Config
            The hyperparameters for the trainer.
        """
        self.print_config = print_config
        self.to_terminate: List[Closable] = []
        # self.setup should at least set these up:
        self.training_data: Dataset
        self.test_data: Dataset
        self.model: nn.Module
        # `sampling_model` is used by the data workers to sample new objects from the model. Can be
        # the same as `model`.
        self.sampling_model: nn.Module
        # `validation_model` is used for validation-time sampling and metric evaluation. It can be an EMA model.
        self.validation_model: nn.Module
        self.replay_buffer: Optional[ReplayBuffer]
        self.env: GraphBuildingEnv
        self.ctx: GraphBuildingEnvContext
        self.task: GFNTask
        self.algo: GFNAlgorithm

        # There are three sources of config values
        #   - The default values specified in individual config classes
        #   - The default values specified in the `default_hps` method, typically what is defined by a task
        #   - The values passed in the constructor, typically what is called by the user
        # The final config is obtained by merging the three sources with the following precedence:
        #   config classes < default_hps < constructor (i.e. the constructor overrides the default_hps, and so on)
        self.default_cfg: Config = Config()
        self.set_default_hps(self.default_cfg)
        assert isinstance(self.default_cfg, Config) and isinstance(
            config, Config
        )  # make sure the config is a Config object, and not the Config class itself
        self.cfg: Config = OmegaConf.merge(self.default_cfg, config)

        self.device = torch.device(self.cfg.device)
        set_main_process_device(self.device)
        # Print the loss every `self.print_every` iterations
        self.print_every = self.cfg.print_every
        # These hooks allow us to compute extra quantities when sampling data
        self.sampling_hooks: List[Callable] = []
        self.valid_sampling_hooks: List[Callable] = []
        # Will check if parameters are finite at every iteration (can be costly)
        self._validate_parameters = False

        self.setup()

    def set_default_hps(self, base: Config):
        raise NotImplementedError()

    def setup_env_context(self):
        raise NotImplementedError()

    def setup_task(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_algo(self):
        raise NotImplementedError()

    def setup_data(self):
        pass

    def step(self, loss: Tensor):
        raise NotImplementedError()

    def setup(self):
        if os.path.exists(self.cfg.log_dir):
            if self.cfg.overwrite_existing_exp:
                shutil.rmtree(self.cfg.log_dir)
            else:
                raise ValueError(
                    f"Log dir {self.cfg.log_dir} already exists. Set overwrite_existing_exp=True to delete it."
                )
        os.makedirs(self.cfg.log_dir)

        RDLogger.DisableLog("rdApp.*")
        set_worker_rng_seed(self.cfg.seed)
        self.env = GraphBuildingEnv()
        self.setup_data()
        self.setup_task()
        self.setup_env_context()
        self.setup_algo()
        self.setup_model()

    def _wrap_for_mp(self, obj):
        """Wraps an object in a placeholder whose reference can be sent to a
        data worker process (only if the number of workers is non-zero)."""
        if self.cfg.num_workers > 0 and obj is not None:
            wrapper = mp_object_wrapper(
                obj,
                self.cfg.num_workers,
                cast_types=(gd.Batch, GraphActionCategorical),
                pickle_messages=self.cfg.pickle_mp_messages,
            )
            self.to_terminate.append(wrapper.terminate)
            return wrapper.placeholder
        else:
            return obj

    def _cfg_value(self, name: str, default: Any) -> Any:
        value = getattr(self.cfg, name, default)
        return default if value == MISSING else value

    def _debug_timing_enabled(self) -> bool:
        return bool(self._cfg_value("debug_timing", False)) or os.environ.get("GFN_DEBUG_TIMING", "0") == "1"

    def _slow_stage_seconds(self) -> float:
        return float(self._cfg_value("slow_stage_seconds", 60.0))

    def _should_log_generated_objs(self) -> bool:
        return bool(self._cfg_value("log_generated_objs", True))

    def build_callbacks(self):
        return {}

    def _make_data_loader(self, src):
        return torch.utils.data.DataLoader(
            src,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            prefetch_factor=1 if self.cfg.num_workers else None,
        )

    def build_training_data_loader(self) -> DataLoader:
        # Since the model may be used by a worker in a different process, we need to wrap it.
        # See implementation_notes.md for more details.
        model = self._wrap_for_mp(self.sampling_model)
        replay_buffer = self._wrap_for_mp(self.replay_buffer)

        if self.cfg.replay.use:
            # None is fine for either value, it will be replaced by num_from_policy, but 0 is not
            assert self.cfg.replay.num_from_replay != 0, "Replay is enabled but no samples are being drawn from it"
            assert self.cfg.replay.num_new_samples != 0, "Replay is enabled but no new samples are being added to it"

        n_drawn = self.cfg.algo.num_from_policy
        n_replayed = self.cfg.replay.num_from_replay or n_drawn if self.cfg.replay.use else 0
        n_new_replay_samples = self.cfg.replay.num_new_samples or n_drawn if self.cfg.replay.use else None
        n_from_dataset = self.cfg.algo.num_from_dataset
        assert n_from_dataset == 0, "Dataset trajectories are reserved for proxy-EUBO validation, not training."

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, replay_buffer=replay_buffer)
        if n_drawn:
            src.do_sample_model(model, n_drawn, n_new_replay_samples)
        if n_replayed and replay_buffer is not None:
            src.do_sample_replay(n_replayed)
        if self.cfg.log_dir and self._should_log_generated_objs():
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "train"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_validation_data_loader(self) -> DataLoader:
        validation_model = getattr(self, "validation_model", self.model)
        model = self._wrap_for_mp(validation_model)
        n_drawn = self.cfg.algo.valid_num_from_policy
        n_from_dataset = self.cfg.algo.valid_num_from_dataset
        n_dataset_eval = self.cfg.algo.valid_num_eval_dataset_trajectories
        if n_dataset_eval is None:
            n_dataset_eval = self.cfg.algo.valid_num_eval_trajectories

        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True, allow_uneven_iterators=True)
        if n_from_dataset:
            validation_dataset_specs = getattr(self, "validation_dataset_specs", None)
            if validation_dataset_specs is None:
                src.do_dataset_in_order(self.test_data, n_from_dataset, backwards_model=model, num_total=n_dataset_eval)
            else:
                for label, dataset in validation_dataset_specs:
                    if len(dataset):
                        src.do_dataset_in_order(
                            dataset,
                            n_from_dataset,
                            backwards_model=model,
                            proxy_eubo_label=label,
                            num_total=n_dataset_eval,
                        )
        if n_drawn:
            num_eval = self.cfg.algo.valid_num_eval_trajectories
            if num_eval is None:
                assert self.cfg.num_validation_gen_steps is not None
                num_eval = self.cfg.num_validation_gen_steps * n_drawn
            src.do_sample_model_n_times(model, n_drawn, num_total=num_eval)

        if self.cfg.log_dir and self._should_log_generated_objs():
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "valid"), self.ctx))
        for hook in self.valid_sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def build_final_data_loader(self) -> DataLoader:
        model = self._wrap_for_mp(self.model)

        n_drawn = self.cfg.algo.num_from_policy
        src = DataSource(self.cfg, self.ctx, self.algo, self.task, is_algo_eval=True)
        assert self.cfg.num_final_gen_steps is not None
        # TODO: might be better to change total steps to total trajectories drawn
        src.do_sample_model_n_times(model, n_drawn, num_total=self.cfg.num_final_gen_steps * n_drawn)

        if self.cfg.log_dir and self._should_log_generated_objs():
            src.add_sampling_hook(SQLiteLogHook(str(pathlib.Path(self.cfg.log_dir) / "final"), self.ctx))
        for hook in self.sampling_hooks:
            src.add_sampling_hook(hook)
        return self._make_data_loader(src)

    def train_batch(self, batch: gd.Batch, epoch_idx: int, batch_idx: int, train_it: int) -> Dict[str, Any]:
        tick = time.time()
        self.model.train()
        loss = None
        info = {}
        try:
            backward_approach = getattr(self.cfg.algo, "backward_approach", "uniform")
            backward_model = self.sampling_model if self.cfg.algo.use_backward_ema else None
            if hasattr(self.algo, "update_batch"):
                info = self.algo.update_batch(
                    self.model,
                    batch,
                    self.step,
                    backward_step=self.backward_step if backward_approach == "tlm" else None,
                    backward_model=backward_model,
                )
                step_info = None
            else:
                gradient_steps = int(getattr(getattr(self.algo, "cfg", None), "gradient_steps", 1))
                backward_info = {}
                if backward_approach == "tlm":
                    backward_losses = []
                    backward_step_info = {}
                    for _ in range(max(gradient_steps, 1)):
                        backward_loss, backward_info = self.algo.compute_backward_policy_loss(self.model, batch)
                        if not torch.isfinite(backward_loss):
                            raise ValueError("backward policy loss is not finite")
                        backward_step_info = self.backward_step(backward_loss)
                        backward_losses.append(backward_loss.detach())
                    backward_info.update(backward_step_info)
                    backward_info["tlm_loss_mean"] = torch.stack(backward_losses).mean()
                    backward_info["backward_updates"] = max(gradient_steps, 1)
                for _ in range(max(gradient_steps, 1)):
                    loss, info = self.algo.compute_batch_losses(self.model, batch, backward_model=backward_model)
                    if not torch.isfinite(loss):
                        raise ValueError("loss is not finite")
                    step_info = self.step(loss)
                info.update(backward_info)
                info["gradient_steps"] = gradient_steps
            self.algo.step()  # This also isn't used anywhere?
            if self._validate_parameters and not all([torch.isfinite(i).all() for i in self.model.parameters()]):
                raise ValueError("parameters are not finite")
        except ValueError as e:
            os.makedirs(self.cfg.log_dir, exist_ok=True)
            torch.save([self.model.state_dict(), batch, loss, info], open(self.cfg.log_dir + "/dump.pkl", "wb"))
            raise e

        if step_info is not None:
            info.update(step_info)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["train_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def evaluate_batch(
        self,
        batch: gd.Batch,
        epoch_idx: int = 0,
        batch_idx: int = 0,
        model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        tick = time.time()
        model = model or self.model
        model.eval()
        with torch.no_grad():
            loss, info = self.algo.compute_batch_losses(model, batch)
        if hasattr(batch, "extra_info"):
            info.update(batch.extra_info)
        info["eval_time"] = time.time() - tick
        return {k: v.item() if hasattr(v, "item") else v for k, v in info.items()}

    def _summarize_validation_infos(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not infos:
            return {}
        summary: Dict[str, Any] = {}
        stat_keys = {
            key
            for info in infos
            for key in info
            if "_stat_" in key or key.endswith("_stat_sum") or key.endswith("_stat_count")
        }
        metric_keys = {
            key
            for info in infos
            for key, value in info.items()
            if key not in stat_keys and isinstance(value, (int, float, np.number))
        }
        for key in sorted(metric_keys):
            values = [float(info[key]) for info in infos if key in info and np.isfinite(float(info[key]))]
            if values:
                summary[key] = float(np.mean(values))

        sum_suffix = "_stat_sum"
        count_suffix = "_stat_count"
        for key in sorted(stat_keys):
            if not key.endswith(sum_suffix):
                continue
            base = key[: -len(sum_suffix)]
            count_key = f"{base}{count_suffix}"
            total = sum(float(info.get(key, 0.0)) for info in infos)
            count = sum(float(info.get(count_key, 0.0)) for info in infos)
            if count > 0:
                summary[base] = total / count

        weighted_suffix = "_stat_weighted_sum"
        weight_suffix = "_stat_weight_sum"
        for key in sorted(stat_keys):
            if not key.endswith(weighted_suffix):
                continue
            base = key[: -len(weighted_suffix)]
            weight_key = f"{base}{weight_suffix}"
            weighted_sum = sum(float(info.get(key, 0.0)) for info in infos)
            weight_sum = sum(float(info.get(weight_key, 0.0)) for info in infos)
            if weight_sum > 0:
                summary[base] = weighted_sum / weight_sum

        corr_suffix = "_stat_n"
        for key in sorted(stat_keys):
            if not key.endswith(corr_suffix):
                continue
            base = key[: -len(corr_suffix)]
            n = sum(float(info.get(f"{base}_stat_n", 0.0)) for info in infos)
            if n < 2:
                continue
            x_sum = sum(float(info.get(f"{base}_stat_x_sum", 0.0)) for info in infos)
            y_sum = sum(float(info.get(f"{base}_stat_y_sum", 0.0)) for info in infos)
            x2_sum = sum(float(info.get(f"{base}_stat_x2_sum", 0.0)) for info in infos)
            y2_sum = sum(float(info.get(f"{base}_stat_y2_sum", 0.0)) for info in infos)
            xy_sum = sum(float(info.get(f"{base}_stat_xy_sum", 0.0)) for info in infos)
            x_var = x2_sum - x_sum * x_sum / n
            y_var = y2_sum - y_sum * y_sum / n
            denom = np.sqrt(max(x_var * y_var, 0.0))
            if denom > 0:
                summary[base] = (xy_sum - x_sum * y_sum / n) / denom
        return summary

    def run(self, logger=None):
        """Trains the GFN for `num_training_steps` minibatches, performing
        validation every `validate_every` minibatches.
        """
        if logger is None:
            logger = create_logger(logfile=self.cfg.log_dir + "/train.log")
        self.model.to(self.device)
        self.sampling_model.to(self.device)
        getattr(self, "validation_model", self.model).to(self.device)
        epoch_length = max(len(self.training_data), 1)
        valid_freq = self.cfg.validate_every
        # If checkpoint_every is not specified, checkpoint at every validation epoch
        ckpt_freq = self.cfg.checkpoint_every if self.cfg.checkpoint_every is not None else valid_freq
        train_dl = self.build_training_data_loader()
        valid_dl = self.build_validation_data_loader()
        if self.cfg.num_final_gen_steps:
            final_dl = self.build_final_data_loader()
        callbacks = self.build_callbacks()
        start = self.cfg.start_at_step + 1
        num_training_steps = self.cfg.num_training_steps
        logger.info("Starting training")
        start_time = time.time()
        train_iter = cycle(train_dl)
        debug_timing = self._debug_timing_enabled()
        slow_stage_seconds = self._slow_stage_seconds()
        for it in range(start, 1 + num_training_steps):
            # the memory fragmentation or allocation keeps growing, how often should we clean up?
            # is changing the allocation strategy helpful?
            if debug_timing:
                logger.info(f"[timing][trainer] fetch_train_batch:start iteration={it}")
            fetch_start = time.time()
            batch = next(train_iter)
            fetch_elapsed = time.time() - fetch_start
            if debug_timing:
                logger.info(f"[timing][trainer] fetch_train_batch:done iteration={it} elapsed={fetch_elapsed:.3f}s")
            elif fetch_elapsed >= slow_stage_seconds:
                logger.warning(f"[slow][trainer] fetch_train_batch iteration={it} elapsed={fetch_elapsed:.3f}s")

            if it % 1024 == 0:
                gc.collect()
                torch.cuda.empty_cache()
            epoch_idx = it // epoch_length
            batch_idx = it % epoch_length
            if self.replay_buffer is not None and len(self.replay_buffer) < self.replay_buffer.warmup:
                logger.info(
                    f"iteration {it} : warming up replay buffer {len(self.replay_buffer)}/{self.replay_buffer.warmup}"
                )
                continue
            if debug_timing:
                logger.info(f"[timing][trainer] train_batch:start iteration={it}")
            train_start = time.time()
            info = self.train_batch(batch.to(self.device), epoch_idx, batch_idx, it)
            train_elapsed = time.time() - train_start
            if debug_timing:
                logger.info(f"[timing][trainer] train_batch:done iteration={it} elapsed={train_elapsed:.3f}s")
            elif train_elapsed >= slow_stage_seconds:
                logger.warning(f"[slow][trainer] train_batch iteration={it} elapsed={train_elapsed:.3f}s")
            info["time_spent"] = time.time() - start_time
            start_time = time.time()
            self.log(info, it, "train")
            if it % self.print_every == 0:
                logger.info(f"iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))

            if valid_freq > 0 and it % valid_freq == 0:
                if debug_timing:
                    logger.info(f"[timing][trainer] validation:start iteration={it}")
                validation_start = time.time()
                validation_model = getattr(self, "validation_model", self.model)
                validation_infos = []
                for validation_batch_idx, batch in enumerate(valid_dl):
                    if debug_timing:
                        logger.info(
                            f"[timing][trainer] validation_batch:start iteration={it} batch={validation_batch_idx}"
                        )
                    validation_batch_start = time.time()
                    info = self.evaluate_batch(batch.to(self.device), epoch_idx, batch_idx, model=validation_model)
                    validation_batch_elapsed = time.time() - validation_batch_start
                    if debug_timing:
                        logger.info(
                            f"[timing][trainer] validation_batch:done iteration={it} "
                            f"batch={validation_batch_idx} elapsed={validation_batch_elapsed:.3f}s"
                        )
                    elif validation_batch_elapsed >= slow_stage_seconds:
                        logger.warning(
                            f"[slow][trainer] validation_batch iteration={it} "
                            f"batch={validation_batch_idx} elapsed={validation_batch_elapsed:.3f}s"
                        )
                    validation_infos.append(info)
                    log_info = {k: v for k, v in info.items() if "_stat_" not in k}
                    logger.info(
                        f"validation batch - iteration {it} : "
                        + " ".join(f"{k}:{v:.2f}" for k, v in log_info.items())
                    )
                info = self._summarize_validation_infos(validation_infos)
                self.log(info, it, "valid")
                logger.info(f"validation - iteration {it} : " + " ".join(f"{k}:{v:.2f}" for k, v in info.items()))
                end_metrics = {}
                for c in callbacks.values():
                    if hasattr(c, "on_validation_end"):
                        c.on_validation_end(end_metrics)
                self.log(end_metrics, it, "valid_end")
                validation_elapsed = time.time() - validation_start
                if debug_timing:
                    logger.info(f"[timing][trainer] validation:done iteration={it} elapsed={validation_elapsed:.3f}s")
                elif validation_elapsed >= slow_stage_seconds:
                    logger.warning(f"[slow][trainer] validation iteration={it} elapsed={validation_elapsed:.3f}s")
            if ckpt_freq > 0 and it % ckpt_freq == 0:
                if debug_timing:
                    logger.info(f"[timing][trainer] checkpoint:start iteration={it}")
                checkpoint_start = time.time()
                self._save_state(it)
                checkpoint_elapsed = time.time() - checkpoint_start
                if debug_timing:
                    logger.info(f"[timing][trainer] checkpoint:done iteration={it} elapsed={checkpoint_elapsed:.3f}s")
                elif checkpoint_elapsed >= slow_stage_seconds:
                    logger.warning(f"[slow][trainer] checkpoint iteration={it} elapsed={checkpoint_elapsed:.3f}s")
        self._save_state(num_training_steps)

        num_final_gen_steps = self.cfg.num_final_gen_steps
        final_info = {}
        if num_final_gen_steps:
            logger.info(f"Generating final {num_final_gen_steps} batches ...")
            for it, batch in zip(
                range(num_training_steps + 1, num_training_steps + num_final_gen_steps + 1),
                cycle(final_dl),
            ):
                if hasattr(batch, "extra_info"):
                    for k, v in batch.extra_info.items():
                        if k not in final_info:
                            final_info[k] = []
                        if hasattr(v, "item"):
                            v = v.item()
                        final_info[k].append(v)
                if it % self.print_every == 0:
                    logger.info(f"Generating objs {it - num_training_steps}/{num_final_gen_steps}")
            final_info = {k: np.mean(v) for k, v in final_info.items()}

            logger.info("Final generation steps completed - " + " ".join(f"{k}:{v:.2f}" for k, v in final_info.items()))
            self.log(final_info, num_training_steps, "final")

        # for pypy and other GC having implementations, we need to manually clean up
        del train_dl
        del valid_dl
        if self.cfg.num_final_gen_steps:
            del final_dl

    def terminate(self):
        logger = logging.getLogger("logger")
        for handler in logger.handlers:
            handler.close()

        for hook in self.sampling_hooks:
            if hasattr(hook, "terminate") and hook.terminate not in self.to_terminate:
                hook.terminate()

        for terminate in self.to_terminate:
            terminate()

    def _save_state(self, it):
        state = {
            "models_state_dict": [self.model.state_dict()],
            "cfg": self.cfg,
            "step": it,
        }
        if self.sampling_model is not self.model:
            state["sampling_model_state_dict"] = [self.sampling_model.state_dict()]
        validation_model = getattr(self, "validation_model", self.model)
        if validation_model is not self.model and validation_model is not self.sampling_model:
            state["validation_model_state_dict"] = [validation_model.state_dict()]
        fn = pathlib.Path(self.cfg.log_dir) / "model_state.pt"
        with open(fn, "wb") as fd:
            torch.save(
                state,
                fd,
            )
        if self.cfg.store_all_checkpoints:
            shutil.copy(fn, pathlib.Path(self.cfg.log_dir) / f"model_state_{it}.pt")

    def log(self, info, index, key):
        if not hasattr(self, "_summary_writer"):
            self._summary_writer = torch.utils.tensorboard.SummaryWriter(self.cfg.log_dir)
        for k, v in info.items():
            self._summary_writer.add_scalar(f"{key}_{k}", v, index)
        if wandb.run is not None:
            wandb.log({f"{key}_{k}": v for k, v in info.items()}, step=index)

    def __del__(self):
        self.terminate()


def cycle(it):
    while True:
        for i in it:
            yield i



def model_grad_norm(model):
    x = 0
    for param in model.parameters():
        if param.grad is not None:
            x += (param.grad * param.grad).sum()
    return torch.sqrt(x)


class AvgRewardHook:
    def __call__(self, trajs, rewards, obj_props, extra_info):
        return {"sampled_reward_avg": rewards.mean().item()}


class StandardOnlineTrainer(GFNTrainer):
    def _backward_approach(self):
        return getattr(self.cfg.algo, "backward_approach", "uniform")

    def _uses_parameterized_backward(self):
        return self._backward_approach() != "uniform" or self.cfg.algo.tb.do_parameterize_p_b

    def setup_model(self):
        self.model = GraphTransformerGFN(
            self.ctx,
            self.cfg,
            do_bck=self._uses_parameterized_backward(),
            num_graph_out=self.cfg.algo.tb.do_predict_n + 1,
            unif_init=self.cfg.model.unif_init,
        )

    def setup_algo(self):
        if self.cfg.algo.method == "TB":
            algo_cls = TrajectoryBalance
        elif self.cfg.algo.method in {"EntPPO", "ENTPPO", "PPO"}:
            algo_cls = EntPPO
        else:
            raise ValueError(f"Unsupported algorithm in this publication snapshot: {self.cfg.algo.method}")
        self.algo = algo_cls(self.env, self.ctx, self.cfg)

    def setup_data(self):
        self.training_data = []
        self.test_data = []

    def _opt(self, params, lr=None, momentum=None):
        params = list(params)
        if not params:
            return None
        if lr is None:
            lr = self.cfg.opt.learning_rate
        if momentum is None:
            momentum = self.cfg.opt.momentum
        if self.cfg.opt.opt == "adam":
            return torch.optim.Adam(
                params,
                lr,
                (momentum, 0.999),
                weight_decay=self.cfg.opt.weight_decay,
                eps=self.cfg.opt.adam_eps,
            )
        raise NotImplementedError(f"{self.cfg.opt.opt} is not implemented")

    def setup(self):
        super().setup()
        self.offline_ratio = 0
        self.replay_buffer = ReplayBuffer(self.cfg) if self.cfg.replay.use else None
        self.sampling_hooks.append(AvgRewardHook())
        self.valid_sampling_hooks.append(AvgRewardHook())

        def is_backward_head(name):
            return ".remove_" in name or name.endswith(".remove_node") or "mlps.remove" in name

        def is_forward_head(name):
            return "mlps.stop" in name or "mlps.add_" in name or "mlps.set_" in name

        if hasattr(self.model, "_logZ"):
            z_params = list(self.model._logZ.parameters())
            non_z_params = [
                p
                for n, p in self.model.named_parameters()
                if all(id(p) != id(j) for j in z_params)
                and not (self._backward_approach() == "tlm" and is_backward_head(n))
            ]
        else:
            z_params = []
            non_z_params = list(self.model.parameters())
        self.opt = self._opt(non_z_params)
        self.opt_Z = self._opt(z_params, self.cfg.algo.tb.Z_learning_rate, 0.9)
        self.lr_sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda steps: 2 ** (-steps / self.cfg.opt.lr_decay))
        self.lr_sched_Z = (
            torch.optim.lr_scheduler.LambdaLR(self.opt_Z, lambda steps: 2 ** (-steps / self.cfg.algo.tb.Z_lr_decay))
            if self.opt_Z is not None
            else None
        )

        if self._backward_approach() == "tlm":
            backward_params = [p for n, p in self.model.named_parameters() if "logZ" not in n and not is_forward_head(n)]
            self.b_opt = self._opt(
                backward_params,
                self.cfg.opt.learning_rate * self.cfg.algo.backward_learning_rate_multiplier,
            )
            self.b_lr_sched = torch.optim.lr_scheduler.LambdaLR(
                self.b_opt, lambda steps: 2 ** (-steps / self.cfg.algo.backward_lr_decay)
            )
        else:
            self.b_opt = None
            self.b_lr_sched = None

        self.sampling_tau = self.cfg.algo.backward_ema_tau if self.cfg.algo.use_backward_ema else self.cfg.algo.sampling_tau
        self.sampling_model = copy.deepcopy(self.model) if self.sampling_tau > 0 else self.model
        self.validation_model = copy.deepcopy(self.model) if self.cfg.algo.valid_use_ema else self.model

        self.clip_grad_callback = {
            "value": lambda params: torch.nn.utils.clip_grad_value_(params, self.cfg.opt.clip_grad_param),
            "norm": lambda params: [torch.nn.utils.clip_grad_norm_(p, self.cfg.opt.clip_grad_param) for p in params],
            "total_norm": lambda params: torch.nn.utils.clip_grad_norm_(params, self.cfg.opt.clip_grad_param),
            "none": lambda x: None,
        }[self.cfg.opt.clip_grad_type]

        try:
            self.cfg.git_hash = git.Repo(__file__, search_parent_directories=True).head.object.hexsha[:7]
        except (git.InvalidGitRepositoryError, ValueError):
            self.cfg.git_hash = "unknown"

        yaml_cfg = OmegaConf.to_yaml(self.cfg)
        if self.print_config:
            print("\n\nHyperparameters:\n")
            print(yaml_cfg)
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        with open(pathlib.Path(self.cfg.log_dir) / "config.yaml", "w", encoding="utf8") as handle:
            handle.write(yaml_cfg)

    def step(self, loss: Tensor):
        loss.backward()
        with torch.no_grad():
            g0 = model_grad_norm(self.model)
            self.clip_grad_callback(self.model.parameters())
            g1 = model_grad_norm(self.model)
        self.opt.step()
        self.opt.zero_grad()
        if self.opt_Z is not None:
            self.opt_Z.step()
            self.opt_Z.zero_grad()
        self.lr_sched.step()
        if self.lr_sched_Z is not None:
            self.lr_sched_Z.step()
        if self.sampling_tau > 0:
            self._update_sampling_model()
        if self.cfg.algo.valid_use_ema:
            self._update_validation_model()
        return {"grad_norm": g0, "grad_norm_clip": g1}

    def backward_step(self, loss: Tensor):
        if self.b_opt is None:
            raise ValueError("backward_step requires cfg.algo.backward_approach='tlm'")
        loss.backward()
        with torch.no_grad():
            g0 = model_grad_norm(self.model)
            self.clip_grad_callback(self.model.parameters())
            g1 = model_grad_norm(self.model)
        self.b_opt.step()
        self.b_opt.zero_grad()
        self.b_lr_sched.step()
        if self.sampling_tau > 0:
            self._update_sampling_model()
        if self.cfg.algo.valid_use_ema:
            self._update_validation_model()
        return {"backward_grad_norm": g0, "backward_grad_norm_clip": g1}

    def _update_sampling_model(self):
        for current, ema in zip(self.model.parameters(), self.sampling_model.parameters()):
            ema.data.mul_(self.sampling_tau).add_(current.data * (1 - self.sampling_tau))

    def _update_validation_model(self):
        tau = self.cfg.algo.valid_ema_tau
        for current, ema in zip(self.model.parameters(), self.validation_model.parameters()):
            ema.data.mul_(tau).add_(current.data * (1 - tau))
