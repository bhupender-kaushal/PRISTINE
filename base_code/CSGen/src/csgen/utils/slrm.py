from __future__ import annotations

import math

import torch


def equivalent_diameter(area: torch.Tensor | float) -> torch.Tensor:
    """Return d = 2 * sqrt(A / pi) from Eq. (5)."""

    area_tensor = torch.as_tensor(area)
    if torch.any(area_tensor < 0):
        raise ValueError("Foreground area must be non-negative")
    return 2.0 * torch.sqrt(area_tensor / math.pi)


def slrm_weight(
    diameter: torch.Tensor | float,
    *,
    gamma: float,
    dmax: float,
    epsilon: float,
) -> torch.Tensor:
    """Return W(d) = gamma * log(1 + Dmax / (d + epsilon))."""

    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    if dmax <= 0.0:
        raise ValueError("dmax must be positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    diameter_tensor = torch.as_tensor(diameter)
    if torch.any(diameter_tensor < 0):
        raise ValueError("Equivalent diameter must be non-negative")
    return float(gamma) * torch.log1p(float(dmax) / (diameter_tensor + float(epsilon)))


def weighted_velocity_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    diameter: torch.Tensor,
    *,
    gamma: float,
    dmax: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Eq. (6) without undocumented clipping or normalization."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have identical shapes")
    diameter = diameter.reshape(-1)
    if diameter.numel() != prediction.shape[0]:
        raise ValueError("diameter must contain exactly one value per batch sample")
    per_sample_mse = (prediction.float() - target.float()).square().flatten(1).mean(1)
    weights = slrm_weight(
        diameter.to(device=per_sample_mse.device, dtype=per_sample_mse.dtype),
        gamma=gamma,
        dmax=dmax,
        epsilon=epsilon,
    )
    return (per_sample_mse * weights).mean(), per_sample_mse, weights
