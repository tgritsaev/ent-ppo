import torch
from torch import Tensor


def pearson_corr(x: Tensor, y: Tensor, mask: Tensor | None = None) -> Tensor | None:
    """Compute a finite-sample Pearson correlation as a scalar tensor."""
    if mask is not None:
        mask = mask.to(x.device).bool()
        x = x[mask]
        y = y[mask]
    finite = torch.isfinite(x) & torch.isfinite(y)
    x = x[finite].float()
    y = y[finite].float()
    if x.numel() < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if denom <= 0:
        return None
    return torch.sum(x * y) / denom
