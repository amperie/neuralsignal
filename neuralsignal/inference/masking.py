from __future__ import annotations

import torch


def sequence_mask(attention_mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(dtype=target.dtype, device=target.device)
    while mask.ndim < target.ndim:
        mask = mask.unsqueeze(-1)
    return mask


def masked_mean(values: torch.Tensor, attention_mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mask = sequence_mask(attention_mask, values)
    denom = mask.sum(dim=dim).clamp_min(1)
    return (values * mask).sum(dim=dim) / denom


def masked_var(values: torch.Tensor, attention_mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    mean = masked_mean(values, attention_mask, dim=dim).unsqueeze(dim)
    mask = sequence_mask(attention_mask, values)
    denom = mask.sum(dim=dim).clamp_min(1)
    return (((values - mean) ** 2) * mask).sum(dim=dim) / denom


def masked_std(values: torch.Tensor, attention_mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return masked_var(values, attention_mask, dim=dim).sqrt()

