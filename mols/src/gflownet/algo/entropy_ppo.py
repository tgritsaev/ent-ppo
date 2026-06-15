import copy
import os
import time
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
import torch_geometric.data as gd
from torch import Tensor
from torch_scatter import scatter

from gflownet import GFNAlgorithm
from gflownet.algo.backward_policy import shifted_model_log_p_b, tlm_loss_from_log_p_b, unpack_model_outputs
from gflownet.config import Config
from gflownet.envs.graph_building_env import GraphActionCategorical, GraphBuildingEnv, GraphBuildingEnvContext
from gflownet.utils.eval_metrics import add_bound_metrics, add_correlation_metric
from gflownet.utils.misc import get_worker_device

from .graph_sampling import GraphSampler


class EntPPO(GFNAlgorithm):
    def __init__(
        self,
        env: GraphBuildingEnv,
        ctx: GraphBuildingEnvContext,
        cfg: Config,
    ):
        """Entropy-regularized PPO with GAE for on-policy GFlowNet training.

        The soft-MDP reward is r(s_t, s_{t+1}) = log P_B(s_t | s_{t+1}) for
        intermediate transitions, with the terminal log-reward added on the
        terminating transition. GAE is computed over g_t = r_t - log P_F_old(a_t | s_t).
        """
        self.ctx = ctx
        self.env = env
        self.global_cfg = cfg
        self.cfg = cfg.algo.ent_ppo
        self.max_len = cfg.algo.max_len
        self.max_nodes = cfg.algo.max_nodes
        self.clip_eps = self.cfg.clip_eps
        self.gae_lambda = self.cfg.gae_lambda
        self.policy_updates = self.cfg.policy_updates
        self.backward_updates = self.cfg.backward_updates or self.policy_updates
        self.value_updates = self.cfg.value_updates
        self.value_num_splits = self.cfg.value_num_splits
        self.value_loss_multiplier = self.cfg.value_loss_multiplier
        self.kl_coeff = self.cfg.kl_coeff
        self.normalize_advantages = self.cfg.normalize_advantages
        self.gamma = self.cfg.gamma
        self.do_parameterize_p_b = self.global_cfg.algo.backward_approach == "tlm"
        assert self.gamma == 1.0, "EntPPO currently implements the GFlowNet gamma=1 setting."
        assert self.policy_updates > 0, "EntPPO requires at least one policy update."
        assert self.backward_updates > 0, "EntPPO requires at least one backward update."
        assert self.value_updates > 0, "EntPPO requires at least one value update."
        assert self.value_num_splits > 0, "EntPPO requires at least one value mini-batch split."
        self.bootstrap_own_reward = False
        self.sample_temp = 1
        self.graph_sampler = GraphSampler(
            ctx,
            env,
            self.max_len,
            self.max_nodes,
            self.sample_temp,
            pad_with_terminal_state=self.do_parameterize_p_b,
            max_backward_steps=cfg.algo.max_backward_steps,
        )
        self.is_eval = False
        self.value_model = None
        self.value_opt = None

    def _timing_enabled(self):
        return os.environ.get("GFN_DEBUG_TIMING", "0") == "1"

    def _sync(self, batch: gd.Batch = None):
        if torch.cuda.is_available():
            if batch is None or batch.x.device.type == "cuda":
                torch.cuda.synchronize()

    def _timing(self, message, batch: gd.Batch = None):
        if self._timing_enabled():
            self._sync(batch)
            print(f"[timing][ent_ppo] {message}", flush=True)

    def set_is_eval(self, is_eval: bool):
        self.is_eval = is_eval

    def create_training_data_from_own_samples(
        self, model: nn.Module, n: int, cond_info: Tensor, random_action_prob: float
    ):
        dev = get_worker_device()
        cond_info = cond_info.to(dev)
        return self.graph_sampler.sample_from_model(model, n, cond_info, random_action_prob)

    def create_training_data_from_graphs(
        self,
        graphs,
        model: nn.Module = None,
        cond_info: Tensor = None,
        random_action_prob: float = None,
    ):
        if not self.cfg.do_sample_p_b:
            raise ValueError("EntPPO dataset metrics require cfg.algo.ent_ppo.do_sample_p_b=True.")
        if cond_info is not None:
            cond_info = cond_info.to(get_worker_device())
        return self.graph_sampler.sample_backward_from_graphs(
            graphs,
            model if self.do_parameterize_p_b else None,
            cond_info,
            random_action_prob or 0.0,
        )

    def construct_batch(self, trajs, cond_info, log_rewards):
        torch_graphs = [self.ctx.graph_to_Data(i[0]) for traj in trajs for i in traj["traj"]]
        actions = [
            self.ctx.GraphAction_to_ActionIndex(g, a)
            for g, a in zip(torch_graphs, [i[1] for traj in trajs for i in traj["traj"]])
        ]
        batch = self.ctx.collate(torch_graphs)
        batch.traj_lens = torch.tensor([len(i["traj"]) for i in trajs])
        batch.actions = torch.tensor(actions)
        batch.log_rewards = log_rewards
        batch.cond_info = cond_info
        batch.is_valid = torch.tensor([i.get("is_valid", True) for i in trajs]).float()
        batch.log_p_B = torch.cat([i["bck_logprobs"] for i in trajs], dim=0).float()
        if self.do_parameterize_p_b:
            batch.bck_actions = torch.tensor(
                [
                    self.ctx.GraphAction_to_ActionIndex(g, a)
                    for g, a in zip(torch_graphs, [i for traj in trajs for i in traj["bck_a"]])
                ]
            )
            batch.is_sink = torch.tensor(sum([i["is_sink"] for i in trajs], []))
        batch.ppo_traj_mask = torch.tensor(["fwd_logprobs" in i and "fwd_full_logprobs" in i for i in trajs]).bool()
        batch.ppo_step_mask = batch.ppo_traj_mask.repeat_interleave(batch.traj_lens)
        if self.do_parameterize_p_b:
            final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
            batch.ppo_step_mask[final_graph_idx] = False
        batch.old_log_p_F = torch.cat(
            [
                torch.cat([i["fwd_logprobs"], torch.zeros(len(i["traj"]) - len(i["fwd_logprobs"]))])
                if "fwd_logprobs" in i
                else torch.zeros(len(i["traj"]))
                for i in trajs
            ],
            dim=0,
        ).float()
        batch.old_full_log_p_F_steps = [
            step if "fwd_full_logprobs" in traj else None
            for traj in trajs
            for step in traj.get("fwd_full_logprobs", [])
            + [None] * (len(traj["traj"]) - len(traj.get("fwd_full_logprobs", [])))
        ]
        return batch

    def _compute_gae(self, deltas: Tensor, traj_lens: Tensor) -> Tensor:
        advantages = torch.zeros_like(deltas)
        start = 0
        for length in traj_lens.tolist():
            gae = deltas.new_zeros(())
            end = start + int(length)
            for idx in range(end - 1, start - 1, -1):
                gae = deltas[idx] + self.gae_lambda * gae
                advantages[idx] = gae
            start = end
        return advantages

    def _analytic_kl_to_old_steps(
        self, policy: GraphActionCategorical, old_logprobs_steps: List[List[Tensor]], step_mask: Tensor
    ) -> Tensor:
        new_logprobs = policy.logsoftmax()
        kl = new_logprobs[0].new_zeros(policy.num_graphs)
        for graph_idx in torch.nonzero(step_mask, as_tuple=False).flatten().tolist():
            old_step = old_logprobs_steps[graph_idx]
            if old_step is None:
                continue
            for action_type, new in enumerate(new_logprobs):
                old = old_step[action_type].to(new.device)
                new_for_graph = new[policy.slice[action_type][graph_idx] : policy.slice[action_type][graph_idx + 1]]
                valid = torch.isfinite(new_for_graph) & torch.isfinite(old)
                if valid.any():
                    kl[graph_idx] = kl[graph_idx] + (
                        new_for_graph[valid].exp() * (new_for_graph[valid] - old[valid])
                    ).sum()
        return kl

    def _safe_entropy(self, policy: GraphActionCategorical) -> Tensor:
        per_action_type_entropy = []
        for log_probs, batch_idx in zip(policy.logsoftmax(), policy.batch):
            valid = torch.isfinite(log_probs)
            terms = torch.zeros_like(log_probs)
            terms[valid] = -log_probs[valid].exp() * log_probs[valid]
            per_action_type_entropy.append(
                scatter(terms, batch_idx, dim=0, dim_size=policy.num_graphs, reduce="sum").sum(1)
            )
        return sum(per_action_type_entropy)

    def _ensure_value_model(self, model: nn.Module):
        if self.value_model is not None:
            return
        self.value_model = copy.deepcopy(model)
        self.value_model.to(next(model.parameters()).device)
        lr = (
            self.cfg.value_learning_rate
            if self.cfg.value_learning_rate is not None
            else self.global_cfg.opt.learning_rate * self.cfg.value_learning_rate_multiplier
        )
        self.value_opt = torch.optim.Adam(
            self.value_model.parameters(),
            lr=lr,
            betas=(self.global_cfg.opt.momentum, 0.999),
            weight_decay=self.global_cfg.opt.weight_decay,
            eps=self.global_cfg.opt.adam_eps,
        )

    def _forward_policy_and_graph_out(self, model: nn.Module, batch: gd.Batch, cond_info: Tensor):
        fwd_cat, _, per_graph_out = unpack_model_outputs(model(batch, cond_info))
        return fwd_cat, per_graph_out

    def _batch_index(self, batch: gd.Batch) -> Tuple[Tensor, Tensor]:
        dev = batch.x.device
        num_trajs = int(batch.traj_lens.shape[0])
        batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
        final_graph_idx = torch.cumsum(batch.traj_lens, 0) - 1
        return batch_idx, final_graph_idx

    def _batch_log_p_B(self, model: nn.Module, batch: gd.Batch, batch_idx: Tensor) -> Tensor:
        if self.do_parameterize_p_b:
            log_p_B, _ = shifted_model_log_p_b(model, batch, batch.cond_info[batch_idx])
            return log_p_B
        return batch.log_p_B.to(batch.x.device)

    def _compute_fixed_targets(self, model: nn.Module, batch: gd.Batch, backward_model: nn.Module = None) -> Dict[str, Tensor]:
        self._ensure_value_model(model)
        self.value_model.eval()
        with torch.no_grad():
            batch_idx, final_graph_idx = self._batch_index(batch)
            _, per_state_preds = self._forward_policy_and_graph_out(self.value_model, batch, batch.cond_info[batch_idx])
            values = per_state_preds[:, 0]
            terminal_graph_idx = final_graph_idx - 1 if self.do_parameterize_p_b else final_graph_idx

            ppo_step_mask = batch.ppo_step_mask.to(batch.x.device).bool()
            log_p_B = self._batch_log_p_B(backward_model or model, batch, batch_idx)
            old_log_p_F = batch.old_log_p_F.to(batch.x.device)
            step_rewards = log_p_B.clone()
            step_rewards[terminal_graph_idx] = step_rewards[terminal_graph_idx] + batch.log_rewards.float()
            deltas_reward = step_rewards - old_log_p_F

            next_values = torch.roll(values, -1)
            next_values[terminal_graph_idx] = 0
            next_values[final_graph_idx] = 0
            deltas = deltas_reward + next_values - values
            advantages = self._compute_gae(deltas, batch.traj_lens)
            if self.normalize_advantages and ppo_step_mask.sum() > 1:
                advantages = advantages.clone()
                ppo_advantages = advantages[ppo_step_mask]
                advantages[ppo_step_mask] = (ppo_advantages - ppo_advantages.mean()) / (
                    ppo_advantages.std(unbiased=False) + 1e-8
                )
            value_targets = advantages + values

        return {
            "advantages": advantages.detach(),
            "value_targets": value_targets.detach(),
            "deltas_reward": deltas_reward.detach(),
            "log_p_B": log_p_B.detach(),
        }

    def _policy_loss_and_info(
        self, model: nn.Module, batch: gd.Batch, targets: Dict[str, Tensor], include_metrics: bool = True
    ):
        dev = batch.x.device
        num_trajs = int(batch.traj_lens.shape[0])
        batch_idx, _ = self._batch_index(batch)

        policy, _ = self._forward_policy_and_graph_out(model, batch, batch.cond_info[batch_idx])
        log_p_F = policy.log_prob(batch.actions)
        _, final_graph_idx = self._batch_index(batch)
        if self.do_parameterize_p_b:
            log_p_F[final_graph_idx] = 0
        log_p_B = targets["log_p_B"].to(dev)
        old_log_p_F = batch.old_log_p_F.to(dev)
        ppo_step_mask = batch.ppo_step_mask.to(dev).bool()
        ppo_traj_mask = batch.ppo_traj_mask.to(dev).bool()
        advantages = targets["advantages"].to(dev)

        ratio = (log_p_F - old_log_p_F).exp()
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        clipped_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        kl_to_old = self._analytic_kl_to_old_steps(policy, batch.old_full_log_p_F_steps, ppo_step_mask)

        traj_objective = scatter(
            (clipped_objective - self.kl_coeff * kl_to_old) * ppo_step_mask.float(),
            batch_idx,
            dim=0,
            dim_size=num_trajs,
            reduce="sum",
        )
        if ppo_traj_mask.any():
            policy_loss = -traj_objective[ppo_traj_mask].mean()
        else:
            policy_loss = log_p_F.sum() * 0

        if not include_metrics:
            return policy_loss, {}

        traj_log_p_F = scatter(log_p_F, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        traj_log_p_B = scatter(log_p_B, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
        bound_metric = batch.log_rewards.float() + traj_log_p_B - traj_log_p_F

        is_clipped = (ratio < 1.0 - self.clip_eps) | (ratio > 1.0 + self.clip_eps)
        invalid_mask = 1 - batch.is_valid
        ppo_ratios = ratio[ppo_step_mask]
        ppo_log_ratios = (log_p_F - old_log_p_F)[ppo_step_mask]
        ppo_advantages = advantages[ppo_step_mask]
        ppo_soft_returns = targets["deltas_reward"].to(dev)[ppo_step_mask]
        zero = policy_loss.new_zeros(())
        ppo_clip_fraction = is_clipped[ppo_step_mask].float().mean() if ppo_step_mask.any() else zero
        info = {
            "policy_loss": policy_loss,
            "advantages": ppo_advantages.mean() if ppo_advantages.numel() else zero,
            "soft_returns": ppo_soft_returns.mean() if ppo_soft_returns.numel() else zero,
            "mean_importance_weight": ppo_ratios.mean() if ppo_ratios.numel() else zero,
            "mean_log_importance_weight": ppo_log_ratios.mean() if ppo_log_ratios.numel() else zero,
            "max_importance_weight": ppo_ratios.max() if ppo_ratios.numel() else zero,
            "clip_fraction": ppo_clip_fraction,
            "kl": kl_to_old[ppo_step_mask].mean() if ppo_step_mask.any() else zero,
            "traj_lens": batch.traj_lens.float().mean(),
            "batch_entropy": self._safe_entropy(policy).mean(),
            "invalid_trajectories": invalid_mask.sum() / batch.num_online if batch.num_online > 0 else 0,
        }
        elbo_mask = getattr(batch, "elbo_mask", torch.zeros(num_trajs, dtype=torch.bool, device=dev)).bool()
        proxy_eubo_mask = getattr(
            batch, "proxy_eubo_mask", torch.zeros(num_trajs, dtype=torch.bool, device=dev)
        ).bool()
        if elbo_mask.any():
            info["elbo"] = bound_metric[elbo_mask].mean()
            info["elbo_stat_sum"] = bound_metric[elbo_mask].double().sum()
            info["elbo_stat_count"] = torch.tensor(elbo_mask.sum().item(), dtype=torch.float64, device=dev)
            add_correlation_metric(
                info,
                "train_correlation",
                batch.log_rewards.float(),
                traj_log_p_F - traj_log_p_B,
                elbo_mask,
            )
        if self.cfg.do_sample_p_b and proxy_eubo_mask.any():
            masks = {"": proxy_eubo_mask}
            for attr_name in batch.keys():
                if attr_name.startswith("proxy_eubo_") and attr_name.endswith("_mask"):
                    suffix = attr_name.removeprefix("proxy_eubo").removesuffix("_mask")
                    masks[suffix] = getattr(batch, attr_name)
            add_bound_metrics(info, bound_metric, batch.log_rewards, traj_log_p_F - traj_log_p_B, masks)

        return policy_loss, info

    def _value_loss(self, model: nn.Module, batch: gd.Batch, targets: Dict[str, Tensor], train: bool = True):
        self._ensure_value_model(model)
        self.value_model.train(train)
        batch_idx, _ = self._batch_index(batch)
        ppo_step_mask = batch.ppo_step_mask.to(batch.x.device).bool()
        _, per_state_preds = self._forward_policy_and_graph_out(self.value_model, batch, batch.cond_info[batch_idx])
        values = per_state_preds[:, 0]
        if ppo_step_mask.any():
            return 0.5 * (values[ppo_step_mask] - targets["value_targets"].to(batch.x.device)[ppo_step_mask]).pow(2).mean()
        return values.sum() * 0

    def _trajectory_step_ranges(self, batch: gd.Batch) -> Tuple[Tensor, Tensor]:
        lens = batch.traj_lens.to(batch.x.device)
        ends = torch.cumsum(lens, 0)
        starts = ends - lens
        return starts, ends

    def _value_split_step_indices(self, batch: gd.Batch) -> List[Tuple[Tensor, Tensor]]:
        dev = batch.x.device
        ppo_traj_mask = batch.ppo_traj_mask.to(dev).bool()
        traj_indices = torch.nonzero(ppo_traj_mask, as_tuple=False).flatten()
        if traj_indices.numel() == 0:
            return []
        num_splits = min(self.value_num_splits, int(traj_indices.numel()))
        starts, ends = self._trajectory_step_ranges(batch)
        out = []
        for traj_chunk in torch.tensor_split(traj_indices, num_splits):
            if traj_chunk.numel() == 0:
                continue
            step_indices = torch.cat(
                [
                    torch.arange(int(starts[i].item()), int(ends[i].item()), device=dev, dtype=torch.long)
                    for i in traj_chunk.tolist()
                ]
            )
            step_traj_indices = torch.repeat_interleave(traj_chunk, batch.traj_lens.to(dev)[traj_chunk])
            out.append((step_indices, step_traj_indices))
        return out

    def _value_loss_on_steps(
        self,
        model: nn.Module,
        batch: gd.Batch,
        targets: Dict[str, Tensor],
        step_indices: Tensor,
        step_traj_indices: Tensor,
        train: bool = True,
    ):
        self._ensure_value_model(model)
        self.value_model.train(train)
        dev = batch.x.device
        graphs = [graph.cpu() for graph in batch.index_select(step_indices.detach().cpu())]
        sub_batch = self.ctx.collate(graphs).to(dev)
        _, per_state_preds = self._forward_policy_and_graph_out(
            self.value_model, sub_batch, batch.cond_info[step_traj_indices]
        )
        values = per_state_preds[:, 0]
        ppo_step_mask = batch.ppo_step_mask.to(dev).bool()[step_indices]
        if ppo_step_mask.any():
            value_targets = targets["value_targets"].to(dev)[step_indices]
            return 0.5 * (values[ppo_step_mask] - value_targets[ppo_step_mask]).pow(2).mean()
        return values.sum() * 0

    def _value_grad_norm(self):
        grad_sq = 0
        for param in self.value_model.parameters():
            if param.grad is not None:
                grad_sq = grad_sq + (param.grad * param.grad).sum()
        return torch.sqrt(grad_sq)

    def _step_value_loss(self, value_loss: Tensor):
        value_loss.backward()
        with torch.no_grad():
            if self.global_cfg.opt.clip_grad_type == "value":
                grad_norm = self._value_grad_norm()
                torch.nn.utils.clip_grad_value_(self.value_model.parameters(), self.global_cfg.opt.clip_grad_param)
            elif self.global_cfg.opt.clip_grad_type in {"norm", "total_norm"}:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.value_model.parameters(),
                    self.global_cfg.opt.clip_grad_param,
                )
            else:
                grad_norm = self._value_grad_norm()
            grad_norm_clip = self._value_grad_norm()
        self.value_opt.step()
        self.value_opt.zero_grad()
        return {"value_grad_norm": grad_norm, "value_grad_norm_clip": grad_norm_clip}

    def compute_backward_policy_loss(self, model: nn.Module, batch: gd.Batch):
        if not self.do_parameterize_p_b:
            raise ValueError("EntPPO TLM requires cfg.algo.backward_approach='tlm'.")
        batch_idx, _ = self._batch_index(batch)
        log_p_B, mask = shifted_model_log_p_b(model, batch, batch.cond_info[batch_idx])
        loss = tlm_loss_from_log_p_b(log_p_B, mask)
        return loss, {"tlm_loss": loss, "backward_log_p_B": log_p_B[mask].mean() if mask.any() else loss.new_zeros(())}

    def update_batch(
        self,
        model: nn.Module,
        batch: gd.Batch,
        policy_step: Callable[[Tensor], Dict[str, Tensor]],
        backward_step: Callable[[Tensor], Dict[str, Tensor]] = None,
        backward_model: nn.Module = None,
    ):
        self._timing(
            f"update_batch:start num_trajs={int(batch.traj_lens.shape[0])} num_steps={int(batch.traj_lens.sum())}",
            batch,
        )
        backward_info = {}
        if self.do_parameterize_p_b:
            if backward_step is None:
                raise ValueError("EntPPO TLM requires a backward_step callback.")
            backward_losses = []
            backward_step_info = {}
            for update_idx in range(self.backward_updates):
                start_time = time.perf_counter()
                backward_loss, backward_info = self.compute_backward_policy_loss(model, batch)
                if not torch.isfinite(backward_loss):
                    raise ValueError("backward policy loss is not finite")
                backward_step_info = backward_step(backward_loss)
                backward_losses.append(backward_loss.detach())
                self._timing(
                    f"backward_update:{update_idx + 1}/{self.backward_updates}:done elapsed={time.perf_counter() - start_time:.3f}s",
                    batch,
                )
            backward_info.update(backward_step_info)
            backward_info["tlm_loss_mean"] = torch.stack(backward_losses).mean()
            backward_info["backward_updates"] = self.backward_updates
        start_time = time.perf_counter()
        targets = self._compute_fixed_targets(model, batch, backward_model=backward_model)
        self._timing(f"compute_fixed_targets:done elapsed={time.perf_counter() - start_time:.3f}s", batch)
        value_splits = self._value_split_step_indices(batch)

        policy_info = {}
        policy_step_info = {}
        policy_loss = None
        for update_idx in range(self.policy_updates):
            start_time = time.perf_counter()
            policy_loss, policy_info = self._policy_loss_and_info(
                model,
                batch,
                targets,
                include_metrics=update_idx == self.policy_updates - 1,
            )
            if not torch.isfinite(policy_loss):
                raise ValueError("policy loss is not finite")
            policy_step_info = policy_step(policy_loss)
            self._timing(
                f"policy_update:{update_idx + 1}/{self.policy_updates}:done elapsed={time.perf_counter() - start_time:.3f}s",
                batch,
            )

        value_step_info = {}
        value_loss = None
        for update_idx in range(self.value_updates):
            split_losses = []
            for split_idx, (step_indices, step_traj_indices) in enumerate(value_splits):
                start_time = time.perf_counter()
                value_loss = self._value_loss_on_steps(
                    model,
                    batch,
                    targets,
                    step_indices,
                    step_traj_indices,
                    train=True,
                )
                if not torch.isfinite(value_loss):
                    raise ValueError("value loss is not finite")
                value_step_info = self._step_value_loss(value_loss)
                split_losses.append(value_loss.detach())
                self._timing(
                    f"value_update:{update_idx + 1}/{self.value_updates}:split:{split_idx + 1}/{self.value_num_splits}:done elapsed={time.perf_counter() - start_time:.3f}s",
                    batch,
                )
            if split_losses:
                value_loss = torch.stack(split_losses).mean()
            else:
                value_loss = self._value_loss(model, batch, targets, train=True)

        info = dict(policy_info)
        info["value_loss"] = value_loss
        info["loss"] = policy_loss + self.value_loss_multiplier * value_loss
        info["policy_updates"] = self.policy_updates
        info["backward_updates"] = self.backward_updates if self.do_parameterize_p_b else 0
        info["value_updates"] = self.value_updates
        info["value_num_splits"] = self.value_num_splits
        info.update(policy_step_info)
        info.update(value_step_info)
        info.update(backward_info)
        return info

    def compute_batch_losses(self, model: nn.Module, batch: gd.Batch, num_bootstrap: int = 0):
        targets = self._compute_fixed_targets(model, batch)
        policy_loss, info = self._policy_loss_and_info(model, batch, targets)
        value_loss = self._value_loss(model, batch, targets, train=False)
        loss = policy_loss + self.value_loss_multiplier * value_loss
        info["value_loss"] = value_loss
        info["loss"] = loss.item()
        info["policy_updates"] = 0
        info["value_updates"] = 0
        if not torch.isfinite(loss).all():
            raise ValueError("loss is not finite")
        return loss, info
