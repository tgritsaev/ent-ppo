"""Graph-molecular TRPO: gfnx cost convention and old-to-new trust region."""
import copy
import time

import torch
from torch import nn
from torch_scatter import scatter

from gflownet.algo.config import resolve_trpo_regime
from gflownet.algo.entropy_ppo import EntPPO
from gflownet.utils.eval_metrics import add_bound_metrics, add_correlation_metric


def flat_grad(output, params, create_graph=False, retain_graph=False):
    grads = torch.autograd.grad(output, params, allow_unused=True,
                                create_graph=create_graph, retain_graph=retain_graph or create_graph)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                      for p, g in zip(params, grads)])


def conjugate_gradients(matvec, b, iterations=10):
    """gfnx's fixed-iteration CG, with safe rejection of degenerate curvature."""
    x = torch.zeros_like(b)
    r = b.detach().clone()
    p = r.clone()
    rr = r.dot(r)
    for _ in range(iterations):
        if rr == 0:
            break
        ap = matvec(p)
        curvature = p.dot(ap)
        if not torch.isfinite(curvature) or curvature <= 0:
            break
        alpha = rr / (curvature + 1e-8)
        x += alpha * p
        r -= alpha * ap
        next_rr = r.dot(r)
        p = r + (next_rr / (rr + 1e-8)) * p
        rr = next_rr
    return x


def set_parameters(params, flat):
    with torch.no_grad():
        offset = 0
        for p in params:
            p.copy_(flat[offset:offset + p.numel()].view_as(p))
            offset += p.numel()


def backtracking(params, full_step, evaluate, old_loss, delta, iterations=10, shrink=0.5):
    """Minimize the surrogate, restoring *exactly* on failure or exception."""
    old = torch.cat([p.detach().reshape(-1) for p in params]).clone()
    accepted = False
    try:
        for i in range(iterations):
            fraction = shrink ** i
            set_parameters(params, old - fraction * full_step)
            with torch.no_grad():
                loss, kl = evaluate()
            if torch.isfinite(loss) and torch.isfinite(kl) and loss < old_loss and kl <= delta:
                accepted = True
                return True, fraction, loss.detach(), kl.detach()
        return False, 0.0, old_loss.detach(), old_loss.new_zeros(())
    finally:
        if not accepted:
            set_parameters(params, old)


def gae(deltas, lens, lam):
    out = torch.zeros_like(deltas)
    offset = 0
    for length in lens.tolist():
        carry = deltas.new_zeros(())
        for i in range(offset + length - 1, offset - 1, -1):
            carry = deltas[i] + lam * carry
            out[i] = carry
        offset += length
    return out


def old_to_new_kl(policy, old_logprobs):
    """Exact action KL per graph, including all legal node/edge/stop actions."""
    new_logprobs = policy.logsoftmax()
    result = new_logprobs[0].new_zeros(policy.num_graphs)
    for old, new, indices in zip(old_logprobs, new_logprobs, policy.batch):
        if old.shape != new.shape or not torch.equal(torch.isfinite(old), torch.isfinite(new)):
            raise ValueError("TRPO old/new action supports differ")
        valid = torch.isfinite(old)
        # Mask before subtraction to avoid 0 * (inf - inf) in either derivative.
        safe_old = torch.where(valid, old.detach(), 0.)
        safe_new = torch.where(valid, new, 0.)
        terms = torch.where(valid, safe_old.exp(), 0.) * (safe_old - safe_new)
        result = result + scatter(terms.sum(1), indices, dim=0,
                                  dim_size=policy.num_graphs, reduce="sum")
    return result


class TRPO(EntPPO):
    manual_policy_update = True

    def __init__(self, env, ctx, cfg):
        tcfg = resolve_trpo_regime(cfg.algo.trpo)
        if cfg.algo.backward_approach != "uniform" or cfg.algo.tb.do_parameterize_p_b:
            raise ValueError("Molecular TRPO requires fixed uniform Pb")
        if cfg.algo.train_random_action_prob or cfg.algo.sampling_tau or cfg.algo.use_backward_ema:
            raise ValueError("TRPO requires on-policy sampling without exploration or sampling EMA")
        if cfg.cond.temperature.sample_dist != "constant":
            raise ValueError("TRPO scalar logZ requires fixed temperature")
        if tcfg.delta <= 0 or tcfg.cg_iters < 1 or tcfg.cg_damping < 0:
            raise ValueError("Invalid TRPO trust-region/CG settings")
        if tcfg.line_search_iters < 1 or not 0 < tcfg.line_search_shrink < 1:
            raise ValueError("Invalid TRPO line search settings")
        # Initialize the shared sampler and independent critic without changing PPO config.
        shared = copy.deepcopy(cfg)
        shared.algo.ent_ppo = copy.deepcopy(tcfg)
        super().__init__(env, ctx, shared)
        self.global_cfg = cfg
        self.cfg = tcfg
        self.logZ = None
        self.logZ_opt = None

    def _ensure_value_model(self, model):
        super()._ensure_value_model(model)
        if self.logZ is None:
            self.logZ = nn.Parameter(torch.zeros((), device=next(model.parameters()).device))
            self.logZ_opt = torch.optim.Adam([self.logZ], lr=self.cfg.logZ_learning_rate)

    def _compute_fixed_targets(self, model, batch, backward_model=None):
        self._ensure_value_model(model)
        with torch.no_grad():
            self.value_model.eval()
            indices, terminal = self._batch_index(batch)
            _, preds = self._forward_policy_and_graph_out(self.value_model, batch, batch.cond_info[indices])
            values = preds[:, 0]
            costs = batch.old_log_p_F - batch.log_p_B
            costs = costs.clone()
            # Terminal Pb is 1 in this fixed-Pb graph sampler.
            costs[terminal] = batch.old_log_p_F[terminal] - batch.log_rewards + self.logZ
            next_values = torch.roll(values, -1)
            next_values[terminal] = 0
            deltas = costs + next_values - values
            raw_advantages = gae(deltas, batch.traj_lens, self.cfg.gae_lambda)
            value_targets = values + gae(deltas, batch.traj_lens, self.cfg.critic_lambda)
            mask = batch.ppo_step_mask.bool()
            advantages = raw_advantages.clone()
            advantages[mask] -= advantages[mask].mean()
            advantages[~mask] = 0
        return {"advantages": advantages, "value_targets": value_targets,
                "cost_mean": scatter(costs, indices, dim=0, reduce="sum").mean()}

    def _policy_parameters(self, model):
        return [p for name, p in model.named_parameters() if p.requires_grad
                and not name.startswith(("emb2graph_out.", "_logZ."))]

    def _policy_terms(self, model, batch, targets, old_full):
        indices, _ = self._batch_index(batch)
        policy, _ = self._forward_policy_and_graph_out(model, batch, batch.cond_info[indices])
        logpf = policy.log_prob(batch.actions)
        mask = batch.ppo_step_mask.bool()
        n = batch.ppo_traj_mask.sum()
        loss = ((logpf[mask] - batch.old_log_p_F[mask]).exp() * targets["advantages"][mask]).sum() / n
        kl = old_to_new_kl(policy, old_full)[mask].sum() / n
        return loss, kl

    def update_batch(self, model, batch, policy_step=None, backward_step=None, backward_model=None):
        if not batch.ppo_traj_mask.all():
            raise ValueError("TRPO updates require only fresh on-policy trajectories")
        start = time.perf_counter()
        model.eval()  # Deterministic policy for curvature and candidate comparisons.
        targets = self._compute_fixed_targets(model, batch)
        indices, _ = self._batch_index(batch)
        with torch.no_grad():
            initial, _ = self._forward_policy_and_graph_out(model, batch, batch.cond_info[indices])
            old_full = [x.detach().clone() for x in initial.logsoftmax()]
            if not torch.allclose(initial.log_prob(batch.actions), batch.old_log_p_F, atol=2e-4, rtol=2e-4):
                raise ValueError("TRPO rollout policy differs from current policy")
        params = self._policy_parameters(model)
        loss, kl = self._policy_terms(model, batch, targets, old_full)
        gradient = flat_grad(loss, params, retain_graph=True).detach()
        kl_gradient = flat_grad(kl, params, create_graph=True)
        def hvp(vector):
            return flat_grad((kl_gradient * vector).sum(), params, retain_graph=True).detach() + self.cfg.cg_damping * vector
        accepted, fraction = False, 0.
        old_loss = loss.detach().clone()
        final_loss, final_kl = old_loss, kl.detach()
        curvature = loss.new_zeros(())
        residual = gradient.norm()
        if torch.isfinite(gradient).all() and gradient.norm() > 0:
            direction = conjugate_gradients(hvp, gradient, self.cfg.cg_iters)
            hd = hvp(direction)
            curvature = direction.dot(hd)
            residual = (hd - gradient).norm()
            if torch.isfinite(curvature) and curvature > 0:
                scale = torch.sqrt(self.cfg.delta / (0.5 * curvature + 1e-8))
                accepted, fraction, final_loss, final_kl = backtracking(
                    params, scale * direction,
                    lambda: self._policy_terms(model, batch, targets, old_full),
                    loss.detach(), self.cfg.delta, self.cfg.line_search_iters, self.cfg.line_search_shrink)
        # Release second-order graphs before fitting the separate critic.
        del kl_gradient, loss, kl, hvp
        policy_seconds = time.perf_counter() - start
        value_losses = []
        splits = self._value_split_step_indices(batch)
        for _ in range(self.value_updates):
            for steps, trajs in splits:
                vl = self._value_loss_on_steps(model, batch, targets, steps, trajs)
                if not torch.isfinite(vl):
                    raise ValueError("TRPO critic loss is not finite")
                self._step_value_loss(vl)
                value_losses.append(vl.detach())
        self.logZ_opt.zero_grad()
        (self.logZ * targets["cost_mean"].detach()).backward()
        self.logZ_opt.step()
        with torch.no_grad():
            _, info = self.compute_batch_losses(model, batch)
        info.update(loss=final_loss, policy_loss_before=old_loss,
                    policy_loss=final_loss, trpo_accepted=float(accepted), trpo_kl=final_kl,
                    trpo_kl_per_step=final_kl * len(batch.traj_lens) / batch.ppo_step_mask.sum(),
                    trpo_step_fraction=fraction, trpo_cg_residual=residual,
                    trpo_curvature=curvature, grad_norm=gradient.norm(),
                    value_loss=torch.stack(value_losses).mean(), value_updates=self.value_updates,
                    value_num_splits=self.value_num_splits, value_optimizer_steps=len(value_losses),
                    policy_updates=int(accepted), logZ=self.logZ.detach().clone(),
                    policy_seconds=policy_seconds)
        return info

    def compute_batch_losses(self, model, batch, num_bootstrap=0):
        """Evaluate bounds with the supplied (EMA at validation) model, no critic mutations."""
        indices, _ = self._batch_index(batch)
        policy, _ = self._forward_policy_and_graph_out(model, batch, batch.cond_info[indices])
        logpf = scatter(policy.log_prob(batch.actions), indices, dim=0, reduce="sum")
        logpb = scatter(batch.log_p_B, indices, dim=0, reduce="sum")
        bound = batch.log_rewards + logpb - logpf
        info = {"loss": bound.new_zeros(()), "traj_lens": batch.traj_lens.float().mean(),
                "batch_entropy": self._safe_entropy(policy).mean()}
        elbo_mask = getattr(batch, "elbo_mask", torch.zeros_like(bound, dtype=torch.bool)).bool()
        if elbo_mask.any():
            info.update(elbo=bound[elbo_mask].mean(), elbo_stat_sum=bound[elbo_mask].double().sum(),
                        elbo_stat_count=elbo_mask.sum().double())
            add_correlation_metric(info, "train_correlation", batch.log_rewards, logpf-logpb, elbo_mask)
        mask = getattr(batch, "proxy_eubo_mask", torch.zeros_like(bound, dtype=torch.bool)).bool()
        if mask.any():
            masks = {"": mask}
            for name in batch.keys():
                if name.startswith("proxy_eubo_") and name.endswith("_mask"):
                    masks[name.removeprefix("proxy_eubo").removesuffix("_mask")] = getattr(batch, name)
            add_bound_metrics(info, bound, batch.log_rewards, logpf-logpb, masks)
        return info["loss"], info

    def state_dict(self):
        return {"value_model": self.value_model.state_dict(), "value_opt": self.value_opt.state_dict(),
                "logZ": self.logZ.detach().clone(), "logZ_opt": self.logZ_opt.state_dict()}

    def load_state_dict(self, state, model):
        self._ensure_value_model(model)
        self.value_model.load_state_dict(state["value_model"])
        self.value_opt.load_state_dict(state["value_opt"])
        with torch.no_grad():
            self.logZ.copy_(state["logZ"])
        self.logZ_opt.load_state_dict(state["logZ_opt"])
