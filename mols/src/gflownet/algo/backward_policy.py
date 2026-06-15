from typing import Tuple

import torch
from torch import Tensor
from torch_scatter import scatter


def unpack_model_outputs(outputs):
    if len(outputs) == 3:
        return outputs
    fwd_cat, per_graph_out = outputs
    return fwd_cat, None, per_graph_out


def batch_index_for_trajectories(batch) -> Tuple[Tensor, Tensor, Tensor]:
    dev = batch.x.device
    num_trajs = int(batch.traj_lens.shape[0])
    batch_idx = torch.arange(num_trajs, device=dev).repeat_interleave(batch.traj_lens)
    traj_cumlen = torch.cumsum(batch.traj_lens, 0)
    final_graph_idx = traj_cumlen - 1
    return batch_idx, traj_cumlen, final_graph_idx


def shifted_model_log_p_b(model, batch, cond_info: Tensor) -> Tuple[Tensor, Tensor]:
    _, bck_cat, _ = unpack_model_outputs(model(batch, cond_info))
    if bck_cat is None or not hasattr(batch, "bck_actions") or not hasattr(batch, "is_sink"):
        raise ValueError("A parameterized backward policy requires bck_actions and is_sink in the batch.")
    raw_log_p_b = bck_cat.log_prob(batch.bck_actions)
    is_sink = batch.is_sink.to(raw_log_p_b.device).bool()
    log_p_b = torch.roll(raw_log_p_b, -1, 0)
    log_p_b[is_sink] = 0
    log_p_b[torch.isnan(log_p_b)] = 0
    mask = ~is_sink
    return log_p_b, mask


def tlm_loss_from_log_p_b(log_p_b: Tensor, mask: Tensor) -> Tensor:
    if mask.any():
        return -log_p_b[mask].mean()
    return log_p_b.sum() * 0


def trajectory_log_probs(log_p: Tensor, batch_idx: Tensor, num_trajs: int) -> Tensor:
    return scatter(log_p, batch_idx, dim=0, dim_size=num_trajs, reduce="sum")
