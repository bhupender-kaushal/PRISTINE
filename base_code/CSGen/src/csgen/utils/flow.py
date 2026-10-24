from __future__ import annotations

import torch


def sample_uniform_t(
    batch_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample the paper's t ~ Uniform(0, 1), one scalar per sample."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return torch.rand(batch_size, device=device, dtype=dtype, generator=generator)


def flow_interpolate(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Compute paper Eq. (1): z_t = (1-t) z_0 + t z_1."""

    if z0.shape != z1.shape:
        raise ValueError("z0 and z1 must have identical shapes")
    if t.ndim != 1 or t.shape[0] != z0.shape[0]:
        raise ValueError("t must have shape [batch]")
    t_broadcast = t.to(device=z0.device, dtype=z0.dtype).view(
        z0.shape[0], *([1] * (z0.ndim - 1))
    )
    return (1.0 - t_broadcast) * z0 + t_broadcast * z1


def velocity_target(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
    """Compute the constant straight-path target v = z_1 - z_0."""

    if z0.shape != z1.shape:
        raise ValueError("z0 and z1 must have identical shapes")
    return z1 - z0


def scheduler_timestep(t: torch.Tensor, num_train_timesteps: int) -> torch.Tensor:
    """Map normalized flow time to the SD3 model's timestep coordinate."""

    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    return t * float(num_train_timesteps)
