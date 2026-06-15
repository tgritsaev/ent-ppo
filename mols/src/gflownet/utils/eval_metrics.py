from typing import Dict

import torch
from torch import Tensor

from gflownet.utils.correlation import pearson_corr


def add_correlation_metric(
    info: Dict[str, Tensor],
    name: str,
    x: Tensor,
    y: Tensor,
    mask: Tensor,
) -> None:
    mask = mask.to(x.device).bool()
    if not mask.any():
        return
    masked_x = x.float()[mask]
    masked_y = y[mask]
    corr = pearson_corr(masked_x, masked_y)
    if corr is not None:
        info[name] = corr
    finite = torch.isfinite(masked_x) & torch.isfinite(masked_y)
    x_stat = masked_x[finite].double()
    y_stat = masked_y[finite].double()
    if x_stat.numel():
        info[f"{name}_stat_n"] = torch.tensor(x_stat.numel(), dtype=torch.float64, device=x.device)
        info[f"{name}_stat_x_sum"] = x_stat.sum()
        info[f"{name}_stat_y_sum"] = y_stat.sum()
        info[f"{name}_stat_x2_sum"] = (x_stat * x_stat).sum()
        info[f"{name}_stat_y2_sum"] = (y_stat * y_stat).sum()
        info[f"{name}_stat_xy_sum"] = (x_stat * y_stat).sum()


def add_bound_metrics(
    info: Dict[str, Tensor],
    bound_metric: Tensor,
    log_rewards: Tensor,
    log_policy_ratio: Tensor,
    masks: Dict[str, Tensor],
) -> None:
    for suffix, mask in masks.items():
        mask = mask.to(bound_metric.device).bool()
        if not mask.any():
            continue

        proxy_name = f"proxy_eubo{suffix}"
        iw_name = f"iw_eubo{suffix}"
        corr_name = f"corr{suffix}"

        proxy_bound_metric = bound_metric[mask]
        masked_log_rewards = log_rewards.float()[mask]
        info[proxy_name] = proxy_bound_metric.mean()
        info[f"{proxy_name}_stat_sum"] = proxy_bound_metric.double().sum()
        info[f"{proxy_name}_stat_count"] = torch.tensor(
            proxy_bound_metric.numel(),
            dtype=torch.float64,
            device=bound_metric.device,
        )

        weights = torch.exp(masked_log_rewards.double())
        finite_weights = torch.isfinite(weights) & torch.isfinite(proxy_bound_metric.double())
        weights = weights[finite_weights]
        weighted_bound_metric = proxy_bound_metric.double()[finite_weights]
        if weights.numel() and weights.sum() > 0:
            weighted_sum = (weights * weighted_bound_metric).sum()
            weight_sum = weights.sum()
            info[iw_name] = weighted_sum / weight_sum
            info[f"{iw_name}_stat_weighted_sum"] = weighted_sum
            info[f"{iw_name}_stat_weight_sum"] = weight_sum

        add_correlation_metric(info, corr_name, log_rewards.float(), log_policy_ratio, mask)
