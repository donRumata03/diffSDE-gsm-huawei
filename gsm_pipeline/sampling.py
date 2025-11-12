from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch


@torch.no_grad()
def sample_sb_sde(
    model: torch.nn.Module,
    num_samples: int,
    shape: Tuple[int, int],
    steps: int = 40,
    sde_sigma: float = 0.2,
    schedule: str = "const",
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Euler-Maruyama sampler for SB-SDE in feature space."""

    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    x = torch.randn((num_samples,) + shape, device=device)
    dt = 1.0 / max(steps, 1)

    def g_of_t(t_scalar: float) -> float:
        if schedule == "const":
            return sde_sigma
        if schedule == "vp":
            return sde_sigma * float(np.cos(0.5 * np.pi * float(t_scalar)))
        return sde_sigma

    for k in range(steps):
        t_scalar = (k + 0.5) / steps
        t = torch.full((num_samples,), t_scalar, device=device)
        v = model(x, t)
        g = g_of_t(t_scalar)
        noise = torch.randn_like(x)
        x = x + dt * v + (g * np.sqrt(dt)) * noise
    return x

